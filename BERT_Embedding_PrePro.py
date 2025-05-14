import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os
from transformers import BertTokenizer, TFBertModel

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

# 2. Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# 3. Tokenize and prepare inputs
def prepare_bert_input(texts):
    # Tokenize texts
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,  # BERT's max length
        return_tensors='tf'
    )
    return encodings

# Prepare inputs
encodings = prepare_bert_input(texts)
y = np.array(labels)

# 4. Build a model using BERT embeddings
def create_bert_model(num_labels):
    # Input layers
    input_ids = layers.Input(shape=(128,), dtype=tf.int32, name='input_ids')
    attention_mask = layers.Input(shape=(128,), dtype=tf.int32, name='attention_mask')
    
    # Get BERT embeddings
    bert_output = bert_model([input_ids, attention_mask])[0]  # Get the last hidden state
    
    # Add classification layers
    x = layers.GlobalAveragePooling1D()(bert_output)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_labels, activation='sigmoid')(x)
    
    # Create model
    model = models.Model(inputs=[input_ids, attention_mask], outputs=outputs)
    return model

# Create and compile model
model = create_bert_model(y.shape[1])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 5. Train the model
history = model.fit(
    [encodings['input_ids'], encodings['attention_mask']],
    y,
    epochs=3,
    batch_size=2,
    verbose=1
)

# 6. Create prediction pipeline
def prediction_pipeline(text, model, tokenizer):
    """
    Pipeline function that handles all preprocessing steps and returns the prediction.
    
    Args:
        text (str): Input text to predict
        model: Trained model
        tokenizer: BERT tokenizer
    
    Returns:
        numpy.ndarray: Model prediction
    """
    # Tokenize input
    encodings = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='tf'
    )
    
    # Show tokenization details
    print("\nTokenization details:")
    print(f"Input text: {text}")
    print(f"Tokenized: {tokenizer.tokenize(text)}")
    print(f"Input IDs: {encodings['input_ids'][0]}")
    print(f"Attention mask: {encodings['attention_mask'][0]}")
    
    # Get prediction
    prediction = model.predict(
        [encodings['input_ids'], encodings['attention_mask']],
        verbose=0
    )
    return prediction

# Test the pipeline
test_text = "I love machine learning and deep learning"
prediction = prediction_pipeline(test_text, model, tokenizer)
print("\nFinal prediction:")
print(prediction) 