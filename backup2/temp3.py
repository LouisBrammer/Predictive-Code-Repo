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