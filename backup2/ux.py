import gradio as gr
import whisper
import keras
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re, emoji, contractions
from datetime import datetime
import os
import logging
import soundfile as sf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models and tokenizer
try:
    logger.info("Loading sentiment model...")
    sentiment_model = keras.models.load_model('imdb_gru.keras')
    logger.info("Loading emotion model...")
    emotion_model = keras.models.load_model('emotion_model_transformer.keras')
    logger.info("Loading tokenizer...")
    with open("tokenizer1.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    max_len = 100
    logger.info("Loading Whisper model...")
    whisper_model = whisper.load_model("tiny")
    logger.info("All models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

# Define emotion labels
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = contractions.fix(text)
    text = re.sub(r'\b(Cuz|coz)\b', 'because', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(Ikr)\b', 'I know right', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(Faux pas)\b', 'mistake', text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_emotion(text):
    try:
        logger.info(f"Predicting emotion for text: {text[:50]}...")
        processed = preprocess_text(text)
        sequence = tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
        probs = emotion_model.predict(padded, verbose=0)[0]
        top3 = probs.argsort()[-3:][::-1]
        result = [(emotion_labels[i], float(probs[i])) for i in top3]
        logger.info(f"Emotion prediction result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in emotion prediction: {str(e)}")
        return [("error", 1.0)]

def prediction_pipeline(text):
    try:
        logger.info(f"Predicting sentiment for text: {text[:50]}...")
        processed = preprocess_text(text)
        sequence = tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
        pred = sentiment_model.predict(padded, verbose=0)
        result = "positive" if pred[0][0] > 0.5 else "negative"
        logger.info(f"Sentiment prediction result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in sentiment prediction: {str(e)}")
        return "error"

def get_sentiment_and_emotion(text):
    # Fake stub — replace with real LLM logic or LangChain prompt
    return "neutral", "curiosity"

def analyze_audio(audio):
    try:
        logger.info(f"Received audio input: {audio}")
        logger.info(f"Audio input type: {type(audio)}")
        
        if audio is None:
            logger.warning("No audio provided")
            return "No audio provided", "", [], "", ""
        
        # Handle the audio input which could be a tuple of (sample_rate, audio_data)
        if isinstance(audio, tuple):
            sample_rate, audio_data = audio
            # Save the audio data to a temporary file
            temp_file = "temp_audio.wav"
            sf.write(temp_file, audio_data, sample_rate)
            audio = temp_file
        elif not isinstance(audio, str):
            logger.error(f"Unexpected audio input type: {type(audio)}")
            return "Invalid audio input", "", [], "", ""
            
        # Check if audio file exists and is readable
        if not os.path.exists(audio):
            logger.error(f"Audio file not found: {audio}")
            return "Audio file not found", "", [], "", ""
            
        # Get audio file info
        try:
            audio_info = sf.info(audio)
            logger.info(f"Audio file info: {audio_info}")
            logger.info(f"Audio file duration: {audio_info.duration} seconds")
            logger.info(f"Audio file sample rate: {audio_info.samplerate} Hz")
            logger.info(f"Audio file channels: {audio_info.channels}")
        except Exception as e:
            logger.error(f"Error reading audio file info: {str(e)}")
            
        logger.info("Starting audio transcription...")
        result = whisper_model.transcribe(audio)
        transcription = result["text"]
        logger.info(f"Transcription result: {transcription}")

        if not transcription.strip():
            logger.warning("No speech detected in audio")
            return "No speech detected", "", [], "", ""

        logger.info("Starting sentiment analysis...")
        sentiment = prediction_pipeline(transcription)
        logger.info("Starting emotion analysis...")
        emotions = predict_emotion(transcription)
        sentiment_llm, emotion_llm = get_sentiment_and_emotion(transcription)

        # Clean up temporary file if it was created
        if isinstance(audio, tuple) and os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")

        return transcription, sentiment, emotions, sentiment_llm, emotion_llm
    except Exception as e:
        logger.error(f"Error in audio analysis: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", "", [], "", ""

demo = gr.Interface(
    fn=analyze_audio,
    inputs=gr.Audio(type="filepath"),
    outputs=[
        gr.Text(label="Transcription"),
        gr.Text(label="Sentiment (Keras)"),
        gr.Label(label="Top 3 Emotions (Keras)"),
        gr.Text(label="Sentiment (LLM)"),
        gr.Text(label="Emotion (LLM)")
    ],
    live=False,
    title="🎤 Emotion & Sentiment Analyzer",
    description="Speak into the mic. Whisper will transcribe and two models will analyze your speech."
)

if __name__ == "__main__":
    demo.launch(share=False)
