import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Hides the GPU from TensorFlow
tf.config.set_visible_devices([], 'GPU') 

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/goemotions/goemotions_filtered.csv')

print(f"Dataset shape: {df.shape}")
print("\nFirst few rows of the dataset:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())

# Extract texts and labels
texts = df['text'].tolist()

# Get the one-hot encoded labels
labels = df.iloc[:, -27:].values  # Convert to numpy array

# check dimensions
print(f"\nNumber of training examples: {len(texts)}")
print(f"Example text: {texts[0][:100]}...")
print(f"Example label shape: {labels[0].shape}")

# 2. Tokenize and pad
max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_len)
y = np.array(labels)

# 3. Load GloVe embeddings
embedding_dim = 50
embeddings_index = {}
current_dir = os.path.dirname(os.path.abspath(__file__))
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


    


    # 5. Build a simple model
model = models.Sequential([
    layers.InputLayer(input_shape=(max_len,)),
    layers.Embedding(
        input_dim=num_words,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=False
    ),
    layers.Conv1D(128, 3, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(27, activation='sigmoid')  # 27 emotion labels
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 6. Train
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history = model.fit(
    X, y,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

# 7. Save model
model.save('emotion_model.keras')

# 8. Make predictions
def predict_emotion(text):
    # Tokenize and pad the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len)
    
    # Get prediction
    prediction = model.predict(padded)[0]
    
    # Get the emotion labels (assuming they're the last 27 columns of the original dataframe)
    emotion_labels = df.columns[-27:]
    
    # Get the top 3 emotions
    top_3_idx = prediction.argsort()[-3:][::-1]
    top_3_emotions = [(emotion_labels[idx], prediction[idx]) for idx in top_3_idx]
    
    return top_3_emotions

# Example prediction
test_text = "I am so happy today!"
predictions = predict_emotion(test_text)
print("\nExample prediction:")
for emotion, score in predictions:
    print(f"{emotion}: {score:.3f}")