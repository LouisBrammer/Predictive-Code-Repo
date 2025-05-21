import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, roc_auc_score
from llm_api import get_sentiment_and_emotion
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import emoji
import contractions


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, roc_auc_score
from llm_api import get_sentiment_and_emotion

# Load test data
df = pd.read_csv("goemotions_test_data.csv")

# Take 100 random rows from the DataFrame
df = df.sample(n=100, random_state=42).reset_index(drop=True)


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

# Keep only the relevant columns and drop the rest
goemotions_valid_df = goemotions_valid_df[['texts', 'processed_text', 'true_emotion']]




# Define emotion clusters
emotion_clusters = {
    'Joy & Excitement': [
        'amusement', 'excitement', 'joy', 'optimism', 'relief'
    ],
    'Love & Caring': [
        'love', 'caring', 'admiration', 'gratitude', 'approval', 'pride', 'desire'
    ],
    'Surprise, Cognition & Curiosity': [
        'surprise', 'realization', 'confusion', 'curiosity'
    ],
    'Fear & Anxiety': [
        'fear', 'nervousness'
    ],
    'Sadness & Shame': [
        'sadness', 'grief', 'disappointment', 'remorse', 'embarrassment'
    ],
    'Anger & Disgust': [
        'anger', 'annoyance', 'disgust', 'disapproval'
    ],
    'Neutral': [
        'neutral'
    ]
}

# Function to map emotion to cluster
def map_emotion_to_cluster(emotion):
    if emotion is None:
        return 'Neutral'  # Default to Neutral if no emotion is found
    for cluster, emotions in emotion_clusters.items():
        if emotion.lower() in [e.lower() for e in emotions]:  # Case-insensitive comparison
            return cluster
    return 'Neutral'  # Default to Neutral if emotion not found in any cluster

# Create true_cluster column
goemotions_valid_df['true_cluster'] = goemotions_valid_df['true_emotion'].apply(map_emotion_to_cluster)

# Create llm_prediction column using get_sentiment_and_emotion (extract only the emotion)
goemotions_valid_df['llm_prediction'] = goemotions_valid_df['texts'].apply(lambda x: get_sentiment_and_emotion(x)[1])

# Create llm_cluster column by mapping llm_prediction to its cluster
goemotions_valid_df['llm_cluster'] = goemotions_valid_df['llm_prediction'].apply(map_emotion_to_cluster)

# Print some statistics to verify
print("\nCluster distribution:")
print(goemotions_valid_df['true_cluster'].value_counts())
print("\nSample rows with their emotions, clusters, and LLM predictions:")
print(goemotions_valid_df[['texts', 'true_emotion', 'true_cluster', 'llm_prediction', 'llm_cluster']].head(10))

# Store the DataFrame as a CSV
goemotions_valid_df.to_csv('emotion_testing_results.csv', index=False)
print("\nResults saved to 'emotion_testing_results.csv'")





import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, roc_auc_score
from llm_api import get_sentiment_and_emotion
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import emoji
import contractions



# --- Pre-processing Function based on the Paper ---
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
        return "" # Return empty string for non-string inputs

    # 1. Convert Emojis to text
    text = emoji.demojize(text, delimiters=(" ", " ")) # e.g., 👍 -> thumbs_up

    # 2. Expand Contractions
    text = contractions.fix(text) # e.g., "I'll" -> "I will"

    # 3. Fix specific Acronyms and Misspellings (examples from paper)
    text = re.sub(r'\b(Cuz|coz)\b', 'because', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(Ikr)\b', 'I know right', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(Faux pas)\b', 'mistake', text, flags=re.IGNORECASE)

    # 4. Lowercase text
    text = text.lower()

    # 5. Normalize repeated characters (e.g., "coooool" -> "cool")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Remove extra spaces that might have been introduced
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Define emotion clusters
emotion_clusters = {
    'Joy & Excitement': [
        'amusement', 'excitement', 'joy', 'optimism', 'relief'
    ],
    'Love & Caring': [
        'love', 'caring', 'admiration', 'gratitude', 'approval', 'pride', 'desire'
    ],
    'Surprise, Cognition & Curiosity': [
        'surprise', 'realization', 'confusion', 'curiosity'
    ],
    'Fear & Anxiety': [
        'fear', 'nervousness'
    ],
    'Sadness & Shame': [
        'sadness', 'grief', 'disappointment', 'remorse', 'embarrassment'
    ],
    'Anger & Disgust': [
        'anger', 'annoyance', 'disgust', 'disapproval'
    ],
    'Neutral': [
        'neutral'
    ]
}




# Function to map emotion to cluster
def map_emotion_to_cluster(emotion):
    if emotion is None:
        return 'Neutral'  # Default to Neutral if no emotion is found
    for cluster, emotions in emotion_clusters.items():
        if emotion.lower() in [e.lower() for e in emotions]:  # Case-insensitive comparison
            return cluster
    return 'Neutral'  # Default to Neutral if emotion not found in any cluster

# Define sentiment mapping for clusters
cluster_to_sentiment = {
    'Joy & Excitement': 'positive',
    'Love & Caring': 'positive',
    'Surprise, Cognition & Curiosity': 'neutral',
    'Neutral': 'neutral',
    'Anger & Disgust': 'negative', 
    'Fear & Anxiety': 'negative',
    'Sadness & Shame': 'negative'
}

# Function to map cluster to sentiment
def map_cluster_to_sentiment(cluster):
    return cluster_to_sentiment.get(cluster, 'neutral')  # Default to neutral if cluster not found



# Load the Keras model
model = keras.models.load_model('emotion_model_conv_advanced.keras', compile=False)

# Load the DataFrame
goemotions_valid_df = pd.read_csv("emotion_testing_results.csv")

# Initialize tokenizer and fit on texts
max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words, oov_token="<unk>")
tokenizer.fit_on_texts(goemotions_valid_df['texts'])

# Create cnn_prediction column using the model's predictions
def predict_emotion(text):
    # Preprocess the text
    processed_text = preprocess_text(text)
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    # Get prediction
    prediction = model.predict(padded_sequence, verbose=0)[0]
    # Get the index of the highest probability
    predicted_index = np.argmax(prediction)
    # Map index to emotion label
    emotion_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
                     'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
                     'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
                     'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
    return emotion_labels[predicted_index]

# Apply prediction function to create cnn_prediction column
goemotions_valid_df['cnn_prediction'] = goemotions_valid_df['texts'].apply(predict_emotion)

# Create cnn_cluster column by mapping cnn_prediction to its cluster
goemotions_valid_df['cnn_cluster'] = goemotions_valid_df['cnn_prediction'].apply(map_emotion_to_cluster)

# Print some statistics to verify
print("\nSample rows with their emotions, clusters, and predictions:")
print(goemotions_valid_df[['texts', 'true_emotion', 'true_cluster', 'llm_prediction', 'llm_cluster', 'cnn_prediction', 'cnn_cluster']].head(10))

# Store the updated DataFrame as a CSV
goemotions_valid_df.to_csv('emotion_testing_results_updated.csv', index=False)
print("\nUpdated results saved to 'emotion_testing_results_updated.csv'")



# Create sentiment columns by mapping clusters to sentiments
goemotions_valid_df['sentiment_true'] = goemotions_valid_df['true_cluster'].apply(map_cluster_to_sentiment)
goemotions_valid_df['cnn_sentiment'] = goemotions_valid_df['cnn_cluster'].apply(map_cluster_to_sentiment)
goemotions_valid_df['llm_sentiment'] = goemotions_valid_df['llm_cluster'].apply(map_cluster_to_sentiment)

# Print sample rows to verify sentiment mappings
print("\nSample rows with clusters and their mapped sentiments:")
print(goemotions_valid_df[['true_cluster', 'sentiment_true', 
                          'cnn_cluster', 'cnn_sentiment',
                          'llm_cluster', 'llm_sentiment']].head())

# Update the CSV with new sentiment columns
goemotions_valid_df.to_csv('emotion_testing_results_updated.csv', index=False)
print("\nUpdated results with sentiment columns saved to 'emotion_testing_results_updated.csv'")






goemotions_valid_df = pd.read_csv('emotion_testing_results_updated.csv')

print(goemotions_valid_df.head())
print(goemotions_valid_df.info())

# Calculate metrics for LLM predictions (cluster level)
llm_accuracy = accuracy_score(goemotions_valid_df['true_cluster'], goemotions_valid_df['llm_cluster'])
llm_precision = precision_score(goemotions_valid_df['true_cluster'], goemotions_valid_df['llm_cluster'], average='weighted')
llm_recall = recall_score(goemotions_valid_df['true_cluster'], goemotions_valid_df['llm_cluster'], average='weighted')

# Calculate metrics for CNN predictions (cluster level)
cnn_accuracy = accuracy_score(goemotions_valid_df['true_cluster'], goemotions_valid_df['cnn_cluster'])
cnn_precision = precision_score(goemotions_valid_df['true_cluster'], goemotions_valid_df['cnn_cluster'], average='weighted')
cnn_recall = recall_score(goemotions_valid_df['true_cluster'], goemotions_valid_df['cnn_cluster'], average='weighted')

# Calculate metrics for sentiment level
llm_sent_accuracy = accuracy_score(goemotions_valid_df['sentiment_true'], goemotions_valid_df['llm_sentiment'])
llm_sent_precision = precision_score(goemotions_valid_df['sentiment_true'], goemotions_valid_df['llm_sentiment'], average='weighted')
llm_sent_recall = recall_score(goemotions_valid_df['sentiment_true'], goemotions_valid_df['llm_sentiment'], average='weighted')

cnn_sent_accuracy = accuracy_score(goemotions_valid_df['sentiment_true'], goemotions_valid_df['cnn_sentiment'])
cnn_sent_precision = precision_score(goemotions_valid_df['sentiment_true'], goemotions_valid_df['cnn_sentiment'], average='weighted')
cnn_sent_recall = recall_score(goemotions_valid_df['sentiment_true'], goemotions_valid_df['cnn_sentiment'], average='weighted')

print("\nLLM Model Metrics (Cluster Level):")
print(f"Accuracy: {llm_accuracy:.4f}")
print(f"Precision: {llm_precision:.4f}") 
print(f"Recall: {llm_recall:.4f}")

print("\nCNN Model Metrics (Cluster Level):")
print(f"Accuracy: {cnn_accuracy:.4f}")
print(f"Precision: {cnn_precision:.4f}")
print(f"Recall: {cnn_recall:.4f}")

print("\nLLM Model Metrics (Sentiment Level):")
print(f"Accuracy: {llm_sent_accuracy:.4f}")
print(f"Precision: {llm_sent_precision:.4f}") 
print(f"Recall: {llm_sent_recall:.4f}")

print("\nCNN Model Metrics (Sentiment Level):")
print(f"Accuracy: {cnn_sent_accuracy:.4f}")
print(f"Precision: {cnn_sent_precision:.4f}")
print(f"Recall: {cnn_sent_recall:.4f}")

# Generate classification reports
print("\nLLM Classification Report (Cluster Level):")
print(classification_report(goemotions_valid_df['true_cluster'], goemotions_valid_df['llm_cluster']))

print("\nCNN Classification Report (Cluster Level):") 
print(classification_report(goemotions_valid_df['true_cluster'], goemotions_valid_df['cnn_cluster']))

print("\nLLM Classification Report (Sentiment Level):")
print(classification_report(goemotions_valid_df['sentiment_true'], goemotions_valid_df['llm_sentiment']))

print("\nCNN Classification Report (Sentiment Level):") 
print(classification_report(goemotions_valid_df['sentiment_true'], goemotions_valid_df['cnn_sentiment']))

# Create a function to generate comparison plots
def create_comparison_plot(metrics, llm_scores, cnn_scores, title, filename):
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, llm_scores, width, label='LLM')
    rects2 = ax.bar(x + width/2, cnn_scores, width, label='CNN')

    ax.set_ylabel('Scores')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Create plots for cluster level metrics
metrics = ['Accuracy', 'Precision', 'Recall']
llm_scores = [llm_accuracy, llm_precision, llm_recall]
cnn_scores = [cnn_accuracy, cnn_precision, cnn_recall]
create_comparison_plot(metrics, llm_scores, cnn_scores, 
                      'Model Performance Comparison (Cluster Level)', 
                      'model_comparison_cluster.png')

# Create plots for sentiment level metrics
llm_sent_scores = [llm_sent_accuracy, llm_sent_precision, llm_sent_recall]
cnn_sent_scores = [cnn_sent_accuracy, cnn_sent_precision, cnn_sent_recall]
create_comparison_plot(metrics, llm_sent_scores, cnn_sent_scores, 
                      'Model Performance Comparison (Sentiment Level)', 
                      'model_comparison_sentiment.png')
