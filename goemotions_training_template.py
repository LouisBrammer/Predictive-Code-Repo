import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam # Import Adam optimizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import F1Score # Import F1Score
import re
import emoji # For emoji conversion
import contractions # For expanding contractions
from sklearn.model_selection import train_test_split # Import for train-test split
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add, Input, GlobalAveragePooling1D

# Hides the GPU from TensorFlow if not needed or causing issues
# tf.config.set_visible_devices([], 'GPU')

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
    # Add more rules as needed

    # 4. Lowercase text
    text = text.lower()

    # 5. Normalize repeated characters (e.g., "coooool" -> "cool")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text) # Reduces 3 or more repetitions to 2
                                            # For "cool", change to r'\1' if only one char is desired

    # Remove extra spaces that might have been introduced
    text = re.sub(r'\s+', ' ', text).strip()
    return text



# 1. Load the dataset
print("Loading dataset...")
# Ensure the path to your CSV is correct
try:
    df = pd.read_csv('data/goemotions/goemotions_filtered.csv')
except FileNotFoundError:
    print("Error: 'data/goemotions/goemotions_filtered.csv' not found.")
    print("Please ensure the dataset is in the correct path or update the path in the script.")
    exit()


print(f"Dataset shape: {df.shape}")
print("\nFirst few rows of the dataset:")
print(df.head())
original_columns = df.columns.tolist() # Get column names BEFORE any modifications
print("\nOriginal Column names:")
print(original_columns)

# Get the one-hot encoded labels from the original DataFrame
# The paper mentions 28 emotions. This script assumes 27 based on previous context.
# It's assumed the label columns are the last N columns in the *original* CSV.
num_label_columns = 27 # Adjust if necessary based on your CSV structure
if len(original_columns) < num_label_columns:
    print(f"Error: DataFrame has fewer than {num_label_columns} columns. Cannot extract labels as expected.")
    exit()

label_column_names = original_columns[-num_label_columns:]
print(f"\nIdentified label columns: {label_column_names}")

# Extract labels using the identified column names from the original DataFrame
try:
    labels_from_df = df[label_column_names].values
    # Verify that all label columns are numeric
    if not np.issubdtype(labels_from_df.dtype, np.number):
        print(f"Warning: Labels extracted from columns {label_column_names} are not all numeric. Attempting conversion.")
        # This conversion might fail if there's genuine non-numeric text
        labels_from_df = df[label_column_names].astype(float).values
except KeyError as e:
    print(f"Error extracting label columns: {e}. Check column names in your CSV and `num_label_columns`.")
    exit()
except ValueError as e:
    print(f"Error converting label columns to numeric: {e}. One of the identified label columns likely contains non-numeric text.")
    exit()

# Extract texts and apply pre-processing
print("\nApplying pre-processing to texts...")
# Ensure 'text' column exists
if 'text' not in df.columns:
    print("Error: 'text' column not found in DataFrame. Please check your CSV file.")
    exit()
df['processed_text'] = df['text'].apply(preprocess_text) # Now add the processed_text column
texts = df['processed_text'].tolist()



# 2. Tokenize and pad
max_words = 10000  # Max words to keep in the vocabulary
max_len = 100      # Max length of sequences
tokenizer = Tokenizer(num_words=max_words, oov_token="<unk>") # Added oov_token
tokenizer.fit_on_texts(texts) # Fit tokenizer on ALL texts to build comprehensive vocabulary
sequences = tokenizer.texts_to_sequences(texts)
X_padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post') # Padded sequences

# Ensure labels are of type float32 for TensorFlow/Keras
y_labels = np.array(labels_from_df, dtype=np.float32)


print(f"Shape of X_padded (all data before split): {X_padded.shape}")
print(f"Shape of y_labels (all labels before split): {y_labels.shape}, dtype: {y_labels.dtype}")

# 2.b. Split data into Training, Validation, and Test sets
try:
    stratify_target = None
    if y_labels.ndim > 1 and y_labels.shape[1] > 1: # Multi-label one-hot
        stratify_target = np.argmax(y_labels, axis=1) # Simplification for stratification
    elif y_labels.ndim == 1: # Single-label
        stratify_target = y_labels

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_padded, y_labels, test_size=0.2, random_state=42, stratify=stratify_target
    )
except ValueError as e:
    print(f"Stratification failed: {e}. Falling back to non-stratified split.")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_padded, y_labels, test_size=0.2, random_state=42
    )


print(f"Shape of X_train_val (training + validation data): {X_train_val.shape}")
print(f"Shape of y_train_val (training + validation labels): {y_train_val.shape}, dtype: {y_train_val.dtype}")
print(f"Shape of X_test (test data): {X_test.shape}")
print(f"Shape of y_test (test labels): {y_test.shape}, dtype: {y_test.dtype}")


# 3. Load GloVe embeddings
embedding_dim = 50  # GloVe embeddings dimension (e.g., 50, 100, 200, 300)
embeddings_index = {}
# Ensure the path to your GloVe file is correct
glove_path = 'glove.6B.50d.txt' # Using 50d embeddings
try:
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f"Found {len(embeddings_index)} word vectors in GloVe.")
except FileNotFoundError:
    print(f"Error: GloVe file '{glove_path}' not found.")
    print("Please download GloVe embeddings (e.g., glove.6B.50d.txt) and place it in the correct path or update the path.")
    exit()

# 4. Prepare embedding matrix
word_index = tokenizer.word_index # Use the same tokenizer fitted on all texts
num_words = min(max_words, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i >= num_words: # Use num_words which is min(max_words, actual_vocab_size+1)
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print(f"Shape of embedding matrix: {embedding_matrix.shape}")

# 5. Build the Transformer model (replacing CNN)
# Transformer: Embedding -> Positional Encoding -> Stacked Transformer Encoder -> Pooling -> Dense(output)

# Reusable transformer encoder block
def transformer_encoder_block(x, num_heads, key_dim):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = Add()([x, attention_output])
    x = LayerNormalization()(x)
    ffn_output = layers.Dense(key_dim, activation='relu')(x)
    x = Add()([x, ffn_output])
    x = LayerNormalization()(x)
    return x

inputs = keras.Input(shape=(max_len,))
embedding_layer = layers.Embedding(
    input_dim=num_words,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    input_length=max_len,
    trainable=True
)
x = embedding_layer(inputs)

# Add positional encoding (learnable)
positional_embedding = layers.Embedding(input_dim=max_len, output_dim=embedding_dim)(tf.range(start=0, limit=max_len, delta=1))
x = x + positional_embedding

# Stack 3 transformer encoder blocks
for _ in range(3):
    x = transformer_encoder_block(x, num_heads=4, key_dim=embedding_dim)

# Pooling and output
x = GlobalAveragePooling1D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(y_labels.shape[1], activation='softmax')(x)

model = keras.Model(inputs, outputs)

model.build(input_shape=(None, max_len))  # Force model build before summary

# Optimizer with learning rate from the paper and gradient clipping
custom_optimizer = Adam(learning_rate=0.0002, clipnorm=1.0)

model.compile(
    optimizer=custom_optimizer,
    loss='binary_crossentropy', # As per paper
    metrics=[F1Score(average='micro', threshold=0.5, name='f1_score'), 'accuracy'] # Use F1Score class, 'micro' is good for multi-label
)

model.summary()

# 6. Train the model
# Paper specifies 12 epochs for the CNN model
epochs_from_paper = 12
batch_size = 64 

# Early stopping is a good practice, though not explicitly mentioned for the CNN in the paper
early_stop = EarlyStopping(monitor='val_f1_score', mode='max', patience=3, restore_best_weights=True, verbose=1) # Monitor val_f1_score

print("\nStarting model training...")
history = model.fit(
    X_train_val, y_train_val, # Train on the training+validation split
    epochs=epochs_from_paper,
    batch_size=batch_size,
    validation_split=0.25, # Create validation set from X_train_val (0.25 of 0.8 original data = 0.2 of total)
    callbacks=[early_stop]
)

# 7. Evaluate on Test Set (Important for unbiased performance measure)
print("\nEvaluating model on the test set...")
test_results = model.evaluate(X_test, y_test, verbose=0)
test_metric_names = model.metrics_names
print("Test Set Evaluation:")
for name, value in zip(test_metric_names, test_results):
    print(f"{name}: {value:.4f}")


# 8. Save model
model_save_path = 'emotion_model_transformer2.keras'
model.save(model_save_path)
print(f"\nModel saved to {model_save_path}")









# 9. Make predictions function
def predict_emotion(text_input, trained_model, tokenizer_instance, max_len_sequences, current_label_column_names):
    """Predicts top 3 emotions for a given text."""
    # Pre-process the input text
    processed_text_input = preprocess_text(text_input)
    
    # Tokenize and pad the input text
    sequence = tokenizer_instance.texts_to_sequences([processed_text_input])
    padded_sequence = pad_sequences(sequence, maxlen=max_len_sequences, padding='post', truncating='post')
    
    # Get prediction
    if padded_sequence.shape[0] == 0: # Handle empty sequence after tokenization
        print("Warning: Text could not be tokenized effectively.")
        return []
        
    prediction_probs = trained_model.predict(padded_sequence)[0]
    
    # Get the top 3 emotions
    top_3_indices = prediction_probs.argsort()[-3:][::-1] # Indices of top 3 scores
    
    top_3_emotions_with_scores = []
    for idx in top_3_indices:
        if idx < len(current_label_column_names):
             top_3_emotions_with_scores.append((current_label_column_names[idx], prediction_probs[idx]))
        else:
            print(f"Warning: Predicted index {idx} is out of bounds for emotion labels.")

    return top_3_emotions_with_scores

# Example prediction
test_text = "I am so incredibly happy and excited today, it feels amazing! 😄🎉"
# Use the 'label_column_names' identified during data loading
predictions = predict_emotion(test_text, model, tokenizer, max_len, label_column_names)
print("\nExample prediction for:", test_text)
if predictions:
    for emotion_label, score in predictions:
        print(f"{emotion_label}: {score:.4f}")
else:
    print("No predictions could be made.")

test_text_2 = "This is so frustrating and annoying, I can't believe this happened... 😠"
predictions_2 = predict_emotion(test_text_2, model, tokenizer, max_len, label_column_names)
print("\nExample prediction for:", test_text_2)
if predictions_2:
    for emotion_label, score in predictions_2:
        print(f"{emotion_label}: {score:.4f}")
else:
    print("No predictions could be made.")

