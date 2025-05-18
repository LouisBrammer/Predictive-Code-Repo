import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Hides the GPU from TensorFlow
tf.config.set_visible_devices([], 'GPU') 

# Base path for the dataset
dataset_path = 'data/aclImdb'

train_dataset = keras.utils.text_dataset_from_directory(
    os.path.expanduser(os.path.join(dataset_path, 'train')),
    class_names=['neg', 'pos'],
    batch_size=64
)

valid_dataset = keras.utils.text_dataset_from_directory(
    os.path.expanduser(os.path.join(dataset_path, 'test')),
    batch_size=64
)


# 1. Prepare text data from dataset
texts = []
labels = []

for text_batch, label_batch in train_dataset:
    for text, label in zip(text_batch.numpy(), label_batch.numpy()):
        texts.append(text.decode('utf-8'))
        labels.append([label]) # Convert to list for consistency

print(f"Number of training examples: {len(texts)}")
print(f"Example text: {texts[0][:100]}...")
print(f"Example label: {labels[0]}")

# 2. Tokenize and pad

max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_len)
y = np.array(labels)



# After fitting the tokenizer
# Save it
with open("tokenizer1.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# 3. Load GloVe embeddings
embedding_dim = 50
embeddings_index = {}

glove_path = 'glove.6B.50d.txt'
with open(glove_path, encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
 
# 4. Prepare embedding matrix
word_index = tokenizer.word_index
num_words = min(max_words, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i >= max_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

## CHANGE FROM HERE ONWARDS

"""
# 5. Build a simple model
model = models.Sequential([
    layers.InputLayer(input_shape=(max_len,)),
    layers.Embedding(
        input_dim=num_words,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=False
    ),
    layers.Conv1D(64, 3, activation='relu'),  # Add local feature extraction
    layers.GlobalMaxPooling1D(),  # Max pooling captures the most important features
    layers.Dropout(0.2),  # Add regularization to prevent overfitting
    layers.Dense(32, activation='relu'),  # Increase from 16 to 32
    layers.Dense(16, activation='relu'),  # Add another layer
    layers.Dropout(0.2),  # Additional dropout
    layers.Dense(y.shape[1], activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC', 'Precision', 'Recall']  # Added AUC metric for better evaluation
)

model.summary()

# 6. Train
early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=2, restore_best_weights=True)
model.fit(X, y, epochs=30, verbose=1, validation_split=0.2, callbacks=[early_stop])


# 6.B STORE MODEL
model.save('imdb_conv2.keras')

"""

# other model using GRU alternative - according with the paper: Comparative Study of CNN and RNN for Natural Language Processing the ideal batch size is 50
gru_units = 256
gru_model = models.Sequential([
    layers.InputLayer(input_shape=(max_len,)),
    layers.Embedding(
        input_dim=num_words,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=False
    ),
    layers.SpatialDropout1D(0.2),  # Dropout for embeddings
    layers.Bidirectional(layers.GRU(gru_units, return_sequences=False)),
    layers.Dropout(0.5),
    layers.Dense(100, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(y.shape[1], activation='sigmoid') 
])
gru_model.compile(optimizer='adam',
              loss='binary_crossentropy', 
              metrics=['accuracy', 'AUC', 'Precision', 'Recall'])
model = gru_model

early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model.fit(X, y, epochs=10, verbose=1, validation_split=0.2, callbacks=[early_stop])

model.summary()

model.save('imdb_gru.keras')






#7. Prediction Pipeline
def prediction_pipeline(text, model, tokenizer, max_len):
    """
    Pipeline function that handles all preprocessing steps and returns the sentiment.
    
    Args:
        text (str): Input text to predict
        model: Trained model
    Returns:
        str: Either "positive" or "negative" sentiment
    """
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded, verbose=0)[0][0]
    return "positive" if prediction > 0.5 else "negative"



# Example usage                                                                 
text = "This movie was fantastic! I loved it."
sentiment = prediction_pipeline(text, model, tokenizer, max_len)
print(f"Sentiment: {sentiment}")                                                                                                                                        