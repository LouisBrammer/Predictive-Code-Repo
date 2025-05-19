import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.metrics import AUC, Precision, Recall
from llm_api import get_sentiment_and_emotion
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, roc_auc_score


# Load test data
df = pd.read_csv("goemotions_test_data.csv")



# Get the one-hot encoded labels from the original DataFrame
original_columns = df.columns.tolist()
num_label_columns = 27
label_column_names = original_columns[-num_label_columns:]

# Create test DataFrame with texts and labels
goemotions_valid_df = pd.DataFrame({
    'texts': df['texts'].tolist()
})

# Add label columns
for i, col in enumerate(label_column_names):
    goemotions_valid_df[col] = df[col].values

# Shuffle the DataFrame and take only 100 rows for testing
goemotions_valid_df = goemotions_valid_df.sample(frac=1, random_state=42).reset_index(drop=True).head(100)

# Reorder columns 
goemotions_valid_df = goemotions_valid_df[['texts','processed_text', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
       'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
       'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
       'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
       'remorse', 'sadness', 'surprise', 'neutral']]


# Function to find the first non-zero emotion label for each row
def get_true_emotion(row):
    # Start from index 2 (third column) which is 'anger'
    for col in row.index[2:]:
        if row[col] == 1:
            return col.lower()
    return None

# Apply function to get true emotion labels
goemotions_valid_df['true_emotion'] = goemotions_valid_df.apply(get_true_emotion, axis=1)
