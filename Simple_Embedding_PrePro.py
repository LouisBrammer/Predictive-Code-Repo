import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
import os

print("Current working directory:", os.getcwd())

# 1. Example data (replace with your own)
texts = [
    "I love machine learning",
    "Deep learning is fun",
    "Natural language processing with embeddings"
]
labels = [
    [1, 0],  # Example multi-label
    [0, 1],
    [1, 1]
]

# 2. Tokenize and pad
max_words = 10000
max_len = 10
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_len)
y = np.array(labels)

# 3. Load GloVe embeddings
embedding_dim = 50
embeddings_index = {}
current_dir = os.path.dirname(os.path.abspath(__file__))
glove_path = os.path.join(current_dir, 'glove.6B.50d.txt')
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

# 5. Build a simple model
model = models.Sequential([
    layers.Embedding(
        input_dim=num_words,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=False
    ),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(y.shape[1], activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 6. Train
model.fit(X, y, epochs=5, batch_size=2, verbose=1)