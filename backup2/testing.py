import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd

# Base path for the dataset
dataset_path = 'data/aclImdb'

valid_dataset = keras.utils.text_dataset_from_directory(os.path.expanduser(dataset_path), batch_size=32)    #batch size needs to be changed here

# 1. Prepare text data from dataset
texts = []
labels = []

for text_batch, label_batch in valid_dataset:
    for text, label in zip(text_batch.numpy(), label_batch.numpy()):
        texts.append(text.decode('utf-8'))
        labels.append(label)

# Create DataFrame
validation_imdb = pd.DataFrame({
    'text': texts,
    'label': labels
})

print(validation_imdb.head())
print(validation_imdb.shape)


import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Initialize parameters
max_words = 10000
max_len = 100

# Initialize tokenizer
tokenizer = Tokenizer(num_words=max_words)

# Load the saved model
model = tf.keras.models.load_model('model.keras')

#7. Prediction Pipeline
def prediction_pipeline(text, model, tokenizer, max_len):
    """
    Pipeline function that handles all preprocessing steps and returns the sentiment.
    
    Args:
        text (str): Input text to predict
        model: Trained model
        tokenizer: Tokenizer instance
        max_len: Maximum sequence length
    Returns:
        str: Either "positive" or "negative" sentiment
    """
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded, verbose=0)[0][0]
    return "positive" if prediction > 0.5 else "negative"



# Apply prediction pipeline to text column in batches for better performance
batch_size = 32
predictions = []

for i in range(0, len(validation_imdb), batch_size):
    batch_texts = validation_imdb['text'].iloc[i:i+batch_size].tolist()
    # Tokenize and pad the entire batch at once
    sequences = tokenizer.texts_to_sequences(batch_texts)
    padded = pad_sequences(sequences, maxlen=max_len)
    # Get predictions for the batch
    batch_predictions = model.predict(padded, verbose=0)
    # Convert probabilities to labels
    batch_labels = ['positive' if pred > 0.5 else 'negative' for pred in batch_predictions]
    predictions.extend(batch_labels)

# Assign predictions to new column
validation_imdb['pred_model_1'] = predictions

