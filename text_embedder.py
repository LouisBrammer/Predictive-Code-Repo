import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Optional, Union
import pandas as pd

class TextEmbedder:
    def __init__(
        self,
        glove_path: str = "glove.6B.50d.txt",
        max_length: int = 600,
        max_tokens: int = 20000,
        embedding_dim: int = 50
    ):
        """
        Initialize the TextEmbedder with GloVe embeddings.
        
        Args:
            glove_path: Path to the GloVe embeddings file
            max_length: Maximum sequence length for text
            max_tokens: Maximum vocabulary size
            embedding_dim: Dimension of the word embeddings
        """
        self.max_length = max_length
        self.max_tokens = max_tokens
        self.embedding_dim = embedding_dim
        
        # Load GloVe embeddings
        self.embeddings_index = {}
        with open(os.path.expanduser(glove_path)) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                self.embeddings_index[word] = coefs
        
        # Initialize tokenizer
        self.tokenizer = keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_sequence_length=max_length,
            output_mode="int"
        )
        
        self.embedding_layer = None
        self.is_fitted = False
    
    def fit(self, texts: Union[List[str], pd.Series]) -> None:
        """
        Fit the tokenizer on the provided texts.
        
        Args:
            texts: List or pandas Series of text data
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        # Create a dummy dataset for adaptation
        dummy_dataset = tf.data.Dataset.from_tensor_slices(texts)
        self.tokenizer.adapt(dummy_dataset)
        
        # Create embedding matrix
        vocabulary = self.tokenizer.get_vocabulary()
        word_index = dict(zip(vocabulary, range(len(vocabulary))))
        
        embedding_matrix = np.zeros((self.max_tokens, self.embedding_dim))
        for word, i in word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        # Create embedding layer
        self.embedding_layer = keras.layers.Embedding(
            self.max_tokens,
            self.embedding_dim,
            embeddings_initializer=keras.initializers.Constant(embedding_matrix),
            trainable=False,
            mask_zero=True,
        )
        
        self.is_fitted = True
    
    def transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Transform texts into embedded sequences.
        
        Args:
            texts: List or pandas Series of text data
            
        Returns:
            numpy array of embedded sequences
        """
        if not self.is_fitted:
            raise ValueError("TextEmbedder must be fitted before transform")
            
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        # Tokenize texts
        tokenized = self.tokenizer(texts)
        
        # Convert to embeddings
        embedded = self.embedding_layer(tokenized)
        
        return embedded.numpy()
    
    def fit_transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Fit the embedder and transform the texts in one step.
        
        Args:
            texts: List or pandas Series of text data
            
        Returns:
            numpy array of embedded sequences
        """
        self.fit(texts)
        return self.transform(texts) 