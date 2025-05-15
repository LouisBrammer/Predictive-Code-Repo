import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from backup.text_embedder import TextEmbedder

# Load the dataset
df = pd.read_csv('data/goemotions_filtered.csv')

# Initialize the TextEmbedder
embedder = TextEmbedder(
    glove_path="glove.6B.50d.txt",
    max_length=600,
    max_tokens=20000,
    embedding_dim=50
)

# Fit and transform the text data
text_embeddings = embedder.fit_transform(df['text'])

# Split the data into train, validation, and test sets
# First split: 70% train, 30% temp
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
# Second split: split temp into 50/50 for test and validation
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save the processed datasets
train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

# Save the embeddings
np.save('data/train_embeddings.npy', text_embeddings[:len(train_df)])
np.save('data/val_embeddings.npy', text_embeddings[len(train_df):len(train_df)+len(val_df)])
np.save('data/test_embeddings.npy', text_embeddings[len(train_df)+len(val_df):])

print(f"Train set size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Validation set size: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test set size: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)") 