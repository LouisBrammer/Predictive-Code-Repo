import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from torch import nn
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

# 2. Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# 3. Tokenize and prepare inputs
def prepare_bert_input(texts):
    # Tokenize texts
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,  # BERT's max length
        return_tensors='pt'
    )
    return encodings

# Prepare inputs
encodings = prepare_bert_input(texts)
y = torch.tensor(labels, dtype=torch.float)

# 4. Build a model using BERT embeddings
class BertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)  # BERT base hidden size is 768

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return torch.sigmoid(logits)

# Create model
model = BertClassifier(y.shape[1])
model.train()

# 5. Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.BCELoss()

# Training loop
for epoch in range(3):
    optimizer.zero_grad()
    outputs = model(encodings['input_ids'], encodings['attention_mask'])
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 6. Create prediction pipeline
def prediction_pipeline(text, model, tokenizer):
    """
    Pipeline function that handles all preprocessing steps and returns the prediction.
    
    Args:
        text (str): Input text to predict
        model: Trained model
        tokenizer: BERT tokenizer
    
    Returns:
        torch.Tensor: Model prediction
    """
    # Tokenize input
    encodings = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # Show tokenization details
    print("\nTokenization details:")
    print(f"Input text: {text}")
    print(f"Tokenized: {tokenizer.tokenize(text)}")
    print(f"Input IDs: {encodings['input_ids'][0]}")
    print(f"Attention mask: {encodings['attention_mask'][0]}")
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        prediction = model(encodings['input_ids'], encodings['attention_mask'])
    return prediction

# Test the pipeline
test_text = "I love machine learning and deep learning"
prediction = prediction_pipeline(test_text, model, tokenizer)
print("\nFinal prediction:")
print(prediction) 