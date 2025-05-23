from llm_api import get_sentiment_and_emotion
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from prediction_pipeline import prediction_pipeline
import sys
import re
import emoji
import contractions

# Load Keras models and tokenizer
sentiment_model = keras.models.load_model('imdb_gru.keras')
emotion_model = keras.models.load_model('emotion_model_transformer.keras')

with open("tokenizer1.pkl", "rb") as f:
    tokenizer = pickle.load(f)
max_len = 100  # Should match what was used in training

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
    print("Enter a text to analyze (or type 'exit' to quit):")
    while True:
        text = input("> ").strip()
        if text.lower() == "exit":
            break
            
        # Keras sentiment model prediction
        sentiment_keras = prediction_pipeline(text, sentiment_model, tokenizer, max_len)
        
        # Emotion model prediction
        emotions = predict_emotion(text, emotion_model, tokenizer, max_len)
        
        # LLM prediction
        sentiment_llm, emotion_llm = get_sentiment_and_emotion(text)
        
        print(f"\nKeras Model Sentiment: {sentiment_keras}")
        print("Top 3 Emotions:")
        for emotion, score in emotions:
            print(f"- {emotion}: {score:.4f}")
        print(f"LLM Sentiment: {sentiment_llm}, LLM Emotion: {emotion_llm}\n")

if __name__ == "__main__":
    main()

