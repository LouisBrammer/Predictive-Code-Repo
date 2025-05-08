import pandas as pd
import numpy as np
from text_embedder import TextEmbedder
from typing import Union, List, Optional
import os

class TextEmbeddingProcessor:
    def __init__(
        self,
        glove_path: str = "glove.6B.50d.txt",
        max_length: int = 600,
        max_tokens: int = 20000,
        embedding_dim: int = 50,
        output_column_prefix: str = "embedded_"
    ):
        """
        Initialize the text embedding processor.
        
        Args:
            glove_path: Path to GloVe embeddings file
            max_length: Maximum sequence length
            max_tokens: Maximum vocabulary size
            embedding_dim: Dimension of word embeddings
            output_column_prefix: Prefix for the output columns
        """
        self.glove_path = glove_path
        self.max_length = max_length
        self.max_tokens = max_tokens
        self.embedding_dim = embedding_dim
        self.output_column_prefix = output_column_prefix
        self.embedder = None
        
    def fit(self, texts: Union[str, List[str], pd.Series]) -> 'TextEmbeddingProcessor':
        """
        Fit the embedder on the provided texts.
        
        Args:
            texts: Single string, list of strings, or pandas Series
            
        Returns:
            self for method chaining
        """
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        self.embedder = TextEmbedder(
            glove_path=self.glove_path,
            max_length=self.max_length,
            max_tokens=self.max_tokens,
            embedding_dim=self.embedding_dim
        )
        self.embedder.fit(texts)
        return self
    
    def transform(
        self,
        texts: Union[str, List[str], pd.Series, pd.DataFrame],
        text_column: Optional[str] = None
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform texts into embeddings.
        
        Args:
            texts: Single string, list of strings, pandas Series, or DataFrame
            text_column: Name of the text column if texts is a DataFrame
            
        Returns:
            numpy array of embeddings or DataFrame with embedded columns
        """
        if self.embedder is None:
            raise ValueError("TextEmbeddingProcessor must be fitted before transform")
            
        # Handle single string input
        if isinstance(texts, str):
            embeddings = self.embedder.transform([texts])
            return embeddings[0]  # Return single embedding
            
        # Handle list of strings or Series
        if isinstance(texts, (list, pd.Series)):
            return self.embedder.transform(texts)
            
        # Handle DataFrame
        if isinstance(texts, pd.DataFrame):
            if text_column is None:
                raise ValueError("text_column must be specified when input is a DataFrame")
                
            embeddings = self.embedder.transform(texts[text_column])
            
            # Create new columns for each embedding dimension
            for i in range(self.embedding_dim):
                col_name = f"{self.output_column_prefix}{i}"
                texts[col_name] = embeddings[:, :, i].mean(axis=1)
                
            return texts
            
        raise TypeError("Input must be string, list of strings, Series, or DataFrame")
    
    def fit_transform(
        self,
        texts: Union[str, List[str], pd.Series, pd.DataFrame],
        text_column: Optional[str] = None
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Fit the processor and transform the texts in one step.
        
        Args:
            texts: Single string, list of strings, pandas Series, or DataFrame
            text_column: Name of the text column if texts is a DataFrame
            
        Returns:
            numpy array of embeddings or DataFrame with embedded columns
        """
        return self.fit(texts).transform(texts, text_column)

def main():
    # Example usage with different input types
    processor = TextEmbeddingProcessor(glove_path='glove.6B.50d.txt')
    
    # Example 1: Single string
    single_text = "This is a sample text for embedding"
    single_embedding = processor.fit_transform(single_text)
    print("\nSingle text embedding shape:", single_embedding.shape)
    
    # Example 2: List of strings
    text_list = [
        'This is a sample text for embedding',
        'Another example text to demonstrate',
        'More text data to process'
    ]
    list_embeddings = processor.fit_transform(text_list)
    print("\nList embeddings shape:", list_embeddings.shape)
    
    # Example 3: DataFrame
    df = pd.DataFrame({
        'text': text_list,
        'label': [0, 1, 0]
    })
    embedded_df = processor.fit_transform(df, text_column='text')
    print("\nEmbedded DataFrame:")
    print(embedded_df.head())

if __name__ == "__main__":
    main() 