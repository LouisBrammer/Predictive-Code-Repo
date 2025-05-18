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
test_dataset = keras.utils.text_dataset_from_directory(
    os.path.expanduser('data/aclImdb/test'),
    batch_size=64
)

# Prepare test data
test_texts = []
test_labels = []

for text_batch, label_batch in test_dataset:
    for text, label in zip(text_batch.numpy(), label_batch.numpy()):
        test_texts.append(text.decode('utf-8'))
        test_labels.append(label)

# Create DataFrame with texts and labels
imdb_valid_df = pd.DataFrame({
    'texts': test_texts,
    'labels': test_labels
})

# Show unique values in labels column
print("\nUnique values in labels:")
print(imdb_valid_df['labels'].unique())

# Shuffle the DataFrame and take only 1000 rows
imdb_valid_df = imdb_valid_df.sample(frac=1, random_state=42).reset_index(drop=True).head(100)


'''
GET CHAT GPT RESPONSE
'''


# Map LLM function over texts to get sentiment and emotion
sentiment_emotion = imdb_valid_df['texts'].apply(get_sentiment_and_emotion)
imdb_valid_df['sentiment'] = sentiment_emotion.apply(lambda x: x[0])
imdb_valid_df['emotion'] = sentiment_emotion.apply(lambda x: x[1])

# Show the first few rows with new columns
print(imdb_valid_df.head())


# Clean sentiment and emotion columns to lowercase
imdb_valid_df['sentiment'] = imdb_valid_df['sentiment'].str.lower()
imdb_valid_df['emotion'] = imdb_valid_df['emotion'].str.lower()

# Map binary labels to positive/negative in a new column 'label'
imdb_valid_df['label'] = imdb_valid_df['labels'].map({1: 'positive', 0: 'negative'})

imdb_valid_df.to_csv("imdb_valid_llm_results.csv", index=False)


# Remove rows where sentiment is None
imdb_valid_df = imdb_valid_df[imdb_valid_df['sentiment'].notnull()]

# Create confusion matrix
cm = confusion_matrix(imdb_valid_df['label'], imdb_valid_df['sentiment'], labels=['positive', 'negative'])

# Create and display confusion matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['positive', 'negative'])
disp.plot()
plt.title('Confusion Matrix: LLM Sentiment vs True Labels')
plt.savefig('confusion_matrix_llm_imdb.png')
plt.show()

# Print classification metrics
print("\nClassification Report:")
print(classification_report(imdb_valid_df['label'], imdb_valid_df['sentiment']))

# Compute and store accuracy metrics
metrics_dict = {}
metrics_dict['accuracy'] = accuracy_score(imdb_valid_df['label'], imdb_valid_df['sentiment'])
metrics_dict['precision'] = precision_score(imdb_valid_df['label'], imdb_valid_df['sentiment'], pos_label='positive')
metrics_dict['recall'] = recall_score(imdb_valid_df['label'], imdb_valid_df['sentiment'], pos_label='positive')
# For AUC, need to convert to binary
label_map = {'positive': 1, 'negative': 0}
y_true = imdb_valid_df['label'].map(label_map)
y_pred = imdb_valid_df['sentiment'].map(label_map)
try:
    metrics_dict['auc'] = roc_auc_score(y_true, y_pred)
except Exception:
    metrics_dict['auc'] = None

print("\nLLM IMDB Metrics:")
for k, v in metrics_dict.items():
    print(f"{k}: {v}")

# Save metrics to CSV
metrics_df = pd.DataFrame([metrics_dict])
metrics_df.to_csv("llm_imdb_metrics.csv", index=False)


