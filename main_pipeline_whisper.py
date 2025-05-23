from llm_api import get_sentiment_and_emotion
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from prediction_pipeline import prediction_pipeline
import sys
import re
import emoji
import contractions
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import time
import signal
from datetime import datetime

# Load Keras models and tokenizer
sentiment_model = keras.models.load_model('imdb_gru.keras')
emotion_model = keras.models.load_model('emotion_model_transformer.keras')

with open("tokenizer1.pkl", "rb") as f:
    tokenizer = pickle.load(f)
max_len = 100  # Should match what was used in training

# Set the microphone device index here
MICROPHONE_DEVICE_INDEX = 0  # 0 = MacBook Air-Mikrofon

def signal_handler(sig, frame):
    print("\nStopping the recording process...")
    sys.exit(0)

def record_audio(duration=10, sample_rate=16000):
    """Record audio for a specified duration."""
    print(f"\nRecording for {duration} seconds...")
    
    # Initialize recording array
    recording = np.zeros((int(duration * sample_rate), 1), dtype='float32')
    
    # Start recording
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', device=MICROPHONE_DEVICE_INDEX) as stream:
        for i in range(0, int(duration * sample_rate), sample_rate):
            chunk, _ = stream.read(sample_rate)
            recording[i:i+sample_rate] = chunk
            
            # Calculate and print audio level
            level = np.abs(chunk).mean()
            print(f"Audio level: {level:.4f}", end='\r')
            
    print("\nRecording finished!")
    
    # Check if we got any audio
    avg_level = np.abs(recording).mean()
    if avg_level < 0.01:
        print("\n⚠️  WARNING: Very low audio levels detected!")
        print("Please try:")
        print("1. Speaking louder")
        print("2. Moving closer to the microphone")
        print("3. Checking your system's microphone settings")
        print(f"Current audio level: {avg_level:.4f}")
    
    return recording

def save_audio(recording, sample_rate=16000):
    """Save the recording to a temporary WAV file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"temp_recording_{timestamp}.wav"
    # Convert float32 to int16
    recording = (recording * 32767).astype(np.int16)
    wav.write(filename, sample_rate, recording)
    return filename

def transcribe_audio(model, audio_file):
    """Transcribe the audio file using Whisper."""
    print("Transcribing...")
    result = model.transcribe(audio_file)
    # Clean up the temporary file
    os.remove(audio_file)
    return result["text"]

def preprocess_text(text):
    """
    Applies pre-processing steps as described in the paper:
    1. Convert Emojis to text
    2. Expand Contractions
    3. Fix specific Acronyms and Misspellings
    4. Lowercase text
    5. Normalize repeated characters
    """
    if not isinstance(text, str):
        return ""  # Return empty string for non-string inputs

    # 1. Convert Emojis to text
    text = emoji.demojize(text, delimiters=(" ", " "))

    # 2. Expand Contractions
    text = contractions.fix(text)

    # 3. Fix specific Acronyms and Misspellings
    text = re.sub(r'\b(Cuz|coz)\b', 'because', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(Ikr)\b', 'I know right', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(Faux pas)\b', 'mistake', text, flags=re.IGNORECASE)

    # 4. Lowercase text
    text = text.lower()

    # 5. Normalize repeated characters
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_emotion(text_input, model, tokenizer_instance, max_len_sequences):
    """Predicts top 3 emotions for a given text."""
    # Pre-process the input text
    processed_text_input = preprocess_text(text_input)
    
    # Tokenize and pad the input text
    sequence = tokenizer_instance.texts_to_sequences([processed_text_input])
    padded_sequence = pad_sequences(sequence, maxlen=max_len_sequences, padding='post', truncating='post')
    
    # Get prediction
    if padded_sequence.shape[0] == 0:
        print("Warning: Text could not be tokenized effectively.")
        return []
        
    prediction_probs = model.predict(padded_sequence)[0]
    
    # Get the top 3 emotions
    top_3_indices = prediction_probs.argsort()[-3:][::-1]
    
    # Define emotion labels (these should match your training data)
    emotion_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
        'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
        'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    top_3_emotions_with_scores = []
    for idx in top_3_indices:
        if idx < len(emotion_labels):
            top_3_emotions_with_scores.append((emotion_labels[idx], prediction_probs[idx]))
        else:
            print(f"Warning: Predicted index {idx} is out of bounds for emotion labels.")

    return top_3_emotions_with_scores

def main():
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Load the Whisper model
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("tiny")
    
    # Print selected audio device
    device_info = sd.query_devices(MICROPHONE_DEVICE_INDEX)
    print(f"\nUsing audio device: {device_info['name']}")
    print(f"Input channels: {device_info['max_input_channels']}")
    
    print("\nStarting continuous recording and analysis...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            # Record audio
            recording = record_audio(duration=10)
            
            # Save and transcribe
            audio_file = save_audio(recording)
            transcription = transcribe_audio(whisper_model, audio_file)
            
            if transcription.strip():
                print(f"\nTranscription: {transcription}")
                
                # Keras sentiment model prediction
                sentiment_keras = prediction_pipeline(transcription, sentiment_model, tokenizer, max_len)
                
                # Emotion model prediction
                emotions = predict_emotion(transcription, emotion_model, tokenizer, max_len)
                
                # LLM prediction
                sentiment_llm, emotion_llm = get_sentiment_and_emotion(transcription)
                
                print(f"\nKeras Model Sentiment: {sentiment_keras}")
                print("Top 3 Emotions:")
                for emotion, score in emotions:
                    print(f"- {emotion}: {score:.4f}")
                print(f"LLM Sentiment: {sentiment_llm}, LLM Emotion: {emotion_llm}\n")
            else:
                print("\nNo speech detected in this recording.")
            
            # Small pause before next recording
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nStopping the recording process...")
        sys.exit(0)

if __name__ == "__main__":
    main()

