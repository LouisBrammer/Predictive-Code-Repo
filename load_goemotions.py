import numpy as np
import pandas as pd
from typing import Tuple, Dict

def load_data() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load the processed embeddings and their corresponding labels.
    
    Returns:
        Tuple containing:
        - Dictionary of embeddings (train, val, test)
        - Dictionary of labels (train, val, test)
    """
    # Load embeddings
    embeddings = {
        'train': np.load('data/train_embeddings.npy'),
        'val': np.load('data/val_embeddings.npy'),
        'test': np.load('data/test_embeddings.npy')
    }
    
    # Load labels
    labels = {
        'train': pd.read_csv('data/train.csv'),
        'val': pd.read_csv('data/val.csv'),
        'test': pd.read_csv('data/test.csv')
    }
    
    # Get emotion columns (excluding metadata columns)
    emotion_columns = [col for col in labels['train'].columns 
                      if col not in ['text', 'id', 'author', 'subreddit', 
                                   'link_id', 'parent_id', 'created_utc', 
                                   'rater_id', 'example_very_unclear']]
    
    # Extract only the emotion labels
    for split in labels:
        labels[split] = labels[split][emotion_columns].values
    
    return embeddings, labels


if __name__ == "__main__":
    # Load the data
    embeddings, labels = load_data()
    
    # Print information about the loaded data
    print_data_info(embeddings, labels)
    
    # Example of accessing the data
    print("\nExample of first training sample:")
    print("Embedding shape:", embeddings['train'][0].shape)
    print("Label vector:", labels['train'][0])
    
    # Example of getting the original text
    train_df = pd.read_csv('data/train.csv')
    print("\nOriginal text of first sample:")
    print(train_df.iloc[0]['text']) 