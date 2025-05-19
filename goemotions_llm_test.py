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

goemotions_valid_df = pd.read_csv('goemotions_valid_df.csv')

# Align predicted and true emotions
y_true = goemotions_valid_df['true_emotion']
y_pred = goemotions_valid_df['predicted_emotion']

# Print unique values in true and predicted emotions
print("\nUnique true emotions:", y_true.unique())
print("Unique predicted emotions:", y_pred.unique())
print("\nSample of true vs predicted:")
comparison_df = pd.DataFrame({'true': y_true, 'predicted': y_pred})
print(comparison_df.head(10))

# Filter to only those predictions that are in the label set
labels = ['anger', 'annoyance', 'approval', 'caring', 'confusion',
        'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
        'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
        'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
        'remorse', 'sadness', 'surprise', 'neutral']

# Ensure both y_true and y_pred are strings and handle NaN values
y_true = y_true.fillna('unknown').astype(str)
y_pred = y_pred.fillna('unknown').astype(str)

print("\nFirst 10 true values:", y_true[:10].tolist())
print("First 10 predicted values:", y_pred[:10].tolist())

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)
print("\nConfusion matrix shape:", cm.shape)
print("Number of non-zero elements in confusion matrix:", np.count_nonzero(cm))

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

print("\nComputed metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Create metrics DataFrame and save to CSV
metrics_df = pd.DataFrame({
    'accuracy': [accuracy],
    'precision': [precision],
    'recall': [recall]
})
metrics_df.to_csv('llm_goemotions_metrics.csv', index=False)
print("\nMetrics saved to 'llm_goemotions_metrics.csv'")

# Save the processed DataFrame to CSV
goemotions_valid_df.to_csv('goemotions_llm_results.csv', index=False)
print("\nResults saved to 'goemotions_llm_results.csv'")



