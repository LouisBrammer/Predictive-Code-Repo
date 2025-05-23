import pandas as pd
from tensorflow import keras
import pickle
from prediction_pipeline import prediction_pipeline

# Load the sentiment model and tokenizer
sentiment_model = keras.models.load_model('imdb_gru.keras')
with open("tokenizer1.pkl", "rb") as f:
    tokenizer = pickle.load(f)
max_len = 100  # Should match what was used in training

def predict_sentiment(text):
    """Predict sentiment for a given text using the sentiment model."""
    try:
        return prediction_pipeline(text, sentiment_model, tokenizer, max_len)
    except Exception as e:
        print(f"Error predicting sentiment: {str(e)}")
        return None

def read_excel_file():
    try:
        # Read the Excel file
        df = pd.read_excel('whisper_testing.xlsx')
        
        # Print column names to debug
        print("\nAvailable columns in the Excel file:")
        print(df.columns.tolist())
        
        # Add sentiment prediction column
        print("\nPredicting sentiments...")
        df['sentiment predicted'] = df['text'].apply(predict_sentiment)
        
        # Display basic information about the DataFrame
        print("\nDataFrame Info:")
        print(df.info())
        
        print("\nFirst few rows of the data:")
        print(df.head())
        
        # Save the DataFrame as a CSV file
        df.to_csv('full_pipeline_testing.csv', index=False)
        print("\nSaved DataFrame to 'full_pipeline_testing.csv'.")
        
        return df
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return None

if __name__ == "__main__":
    read_excel_file() 