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



# Create test DataFrame with texts and labels
test_df = pd.DataFrame({
    'texts': df['text'].iloc[X_test.shape[0]*-1:].tolist()  # Get original texts corresponding to test set
})

# Add label columns 
for i, col in enumerate(label_column_names):
    test_df[col] = y_test[:,i]

# Add processed texts
test_df['processed_text'] = df['processed_text'].iloc[X_test.shape[0]*-1:].tolist()

# Save test data to CSV
test_df.to_csv("goemotions_test_data.csv", index=False)

print("\nTest data saved to goemotions_test_data.csv")
print(f"Shape of test data: {test_df.shape}")
