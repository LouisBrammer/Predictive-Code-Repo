from llm_api import get_sentiment_and_emotion
import pickle
from tensorflow import keras
from prediction_pipeline import prediction_pipeline
import sys

# Load Keras model and tokenizer
model = keras.models.load_model('imdb_conv2.keras')

with open("tokenizer1.pkl", "rb") as f:
    tokenizer = pickle.load(f)
max_len = 100  # Should match what was used in training

def main():
    print("Enter a text to analyze (or type 'exit' to quit):")
    while True:
        text = input("> ").strip()
        if text.lower() == "exit":
            break
        # Keras model prediction
        sentiment_keras = prediction_pipeline(text, model, tokenizer, max_len)
        # LLM prediction
        sentiment_llm, emotion_llm = get_sentiment_and_emotion(text)
        print(f"\nKeras Model Sentiment: {sentiment_keras}")
        print(f"LLM Sentiment: {sentiment_llm}, LLM Emotion: {emotion_llm}\n")

if __name__ == "__main__":
    main()

