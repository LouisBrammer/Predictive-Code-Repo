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

# Convert label columns to numeric
goemotions_valid_df[label_column_names] = goemotions_valid_df[label_column_names].apply(pd.to_numeric, errors='coerce')

# Get the index of the maximum value for each row (the true emotion)
true_emotion_indices = goemotions_valid_df[label_column_names].values.argmax(axis=1)
# Convert indices to emotion names
goemotions_valid_df['true_emotion'] = [label_column_names[idx].lower() for idx in true_emotion_indices]

# Map LLM function over texts to get sentiment and emotion
sentiment_emotion = goemotions_valid_df['texts'].apply(get_sentiment_and_emotion)
goemotions_valid_df['sentiment'] = sentiment_emotion.apply(lambda x: x[0])
goemotions_valid_df['predicted_emotion'] = sentiment_emotion.apply(lambda x: x[1])

# Clean sentiment and emotion columns to lowercase
goemotions_valid_df['sentiment'] = goemotions_valid_df['sentiment'].str.lower()
goemotions_valid_df['predicted_emotion'] = goemotions_valid_df['predicted_emotion'].str.lower()

# Show the first few rows with new columns
print("\nFirst few rows of processed data:")
print(goemotions_valid_df.head())

# Align predicted and true emotions
y_true = goemotions_valid_df['true_emotion']
y_pred = goemotions_valid_df['predicted_emotion']

# Filter to only those predictions that are in the label set
valid_labels = [label.lower() for label in label_column_names]
mask = y_pred.isin(valid_labels)
y_true = y_true[mask]
y_pred = y_pred[mask]

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=valid_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=valid_labels)
fig, ax = plt.subplots(figsize=(12, 12))
disp.plot(ax=ax, xticks_rotation=90)
plt.title('Confusion Matrix for Emotion Classification')
plt.savefig('goemotions_llm_confusion_matrix.png')
plt.tight_layout()
plt.show()

# Save the processed DataFrame to CSV
goemotions_valid_df.to_csv('goemotions_llm_results.csv', index=False)
print("\nResults saved to 'goemotions_llm_results.csv'")

