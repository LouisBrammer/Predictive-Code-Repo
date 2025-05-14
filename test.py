from load_goemotions import load_data

embeddings, labels = load_data()

print("Input shape:", embeddings['train'].shape)
print("Labels shape:", labels['train'].shape)

import tensorflow as tf
from tensorflow.keras import layers, models

# Convert numpy arrays to tensorflow format if needed
X_train = embeddings['train']  # Shape: (samples, sequence_length, embedding_dim)
y_train = labels['train']      # Shape: (samples, num_classes)

# Define a model that can handle sequential data
model = models.Sequential([
    # Input shape: (sequence_length, embedding_dim)
    layers.Dense(128, activation='relu', input_shape=(600, 50)),
    layers.Dense(64, activation='relu'),
    layers.GlobalAveragePooling1D(),  # Reduce sequence dimension
    layers.Dense(y_train.shape[1], activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Train the model
history = model.fit(
    X_train, 
    y_train,
    epochs=3,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Print final metrics
print("\nFinal training metrics:")
for metric in history.history:
    print(f"{metric}: {history.history[metric][-1]:.4f}")

