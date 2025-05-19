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
