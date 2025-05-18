import os

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.metrics import AUC, Precision, Recall

# Load the tokenizer
with open('tokenizer1.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the models without compiling
model1 = keras.models.load_model('imdb_conv1.keras', compile=False)
model2 = keras.models.load_model('imdb_conv2.keras', compile=False)
model3 = keras.models.load_model('imdb_gru.keras', compile=False)

# Recompile with correct metrics
metrics = ['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

# Load test data
test_dataset = keras.utils.text_dataset_from_directory(
    os.path.expanduser('data/aclImdb/test'),
    batch_size=64
)

# Prepare test data
test_texts = []
test_labels = []

for text_batch, label_batch in test_dataset:
    for text, label in zip(text_batch.numpy(), label_batch.numpy()):
        test_texts.append(text.decode('utf-8'))
        test_labels.append(label)

# Tokenize and pad test data
max_len = 100
test_sequences = tokenizer.texts_to_sequences(test_texts)
X_test = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_len)
y_test = np.array(test_labels)

# Evaluate all models
print("\nEvaluating Conv1...")
results1 = model1.evaluate(X_test, y_test, verbose=1)
print("\nEvaluating Conv2...")
results2 = model2.evaluate(X_test, y_test, verbose=1)
print("\nEvaluating Gru...")
results3 = model3.evaluate(X_test, y_test, verbose=1)

# Unpack compile_metrics if present
metric_names = ['loss', 'accuracy', 'auc', 'precision', 'recall']
def unpack_results(results):
    if isinstance(results, list) and len(results) == 2 and isinstance(results[1], (list, np.ndarray)):
        # [loss, [accuracy, auc, precision, recall]]
        return [results[0]] + list(results[1])
    return results

results1 = unpack_results(results1)
results2 = unpack_results(results2)
results3 = unpack_results(results3)

results_dict = {
    'Conv1': dict(zip(metric_names, results1)),
    'Conv2': dict(zip(metric_names, results2)),
    'Gru': dict(zip(metric_names, results3))
}

# Read LLM metrics from CSV
llm_metrics_df = pd.read_csv('llm_imdb_metrics.csv')
llm_metrics = llm_metrics_df.iloc[0].to_dict()

# Print raw results for debugging
print("\nRaw results for Conv1:", results1)
print("Raw results for Conv2:", results2)
print("Raw results for Gru:", results3)
print("LLM metrics:", llm_metrics)
print("Metric names:", metric_names)

# Create subplots for different metrics
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Define metrics to plot
metrics_to_plot = ['accuracy', 'auc', 'precision', 'recall']
titles = ['Accuracy', 'AUC', 'Precision', 'Recall']
axes = [ax1, ax2, ax3, ax4]

# Plot metrics comparison
for metric, title, ax in zip(metrics_to_plot, titles, axes):
    # Get values for all models and LLM
    values = []
    for model_name in ['Conv1', 'Conv2', 'Gru']:
        if metric in results_dict[model_name]:
            values.append(results_dict[model_name][metric])
        else:
            print(f"Warning: {metric} not found in {model_name} results")
            values.append(0)
    # Add LLM metric
    values.append(llm_metrics.get(metric, 0))
    # Create bar plot
    bars = ax.bar(['Conv1', 'Conv2', 'Gru', 'LLM'], values)
    ax.set_title(f'Model {title}')
    ax.set_ylabel(title)
    # Set y-axis limits to show differences better
    if values:
        min_val = min(values)
        max_val = max(values)
        margin = (max_val - min_val) * 0.1
        ax.set_ylim(max(0, min_val - margin), min(1, max_val + margin))
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig('sentiment_detection_metrics.png')
plt.show()

# Print detailed metrics for all models
print("\nDetailed Metrics for Conv1:")
for metric, value in results_dict['Conv1'].items():
    print(f"{metric}: {value:.4f}")

print("\nDetailed Metrics for Conv2:")
for metric, value in results_dict['Conv2'].items():
    print(f"{metric}: {value:.4f}")

print("\nDetailed Metrics for Gru:")
for metric, value in results_dict['Gru'].items():
    print(f"{metric}: {value:.4f}")

# Compare predictions
print("\nComparing predictions on a few examples:")
test_examples = [
    "This movie was fantastic! I loved every minute of it.",
    "The worst film I've ever seen. Terrible acting and plot.",
    "An average movie with some good moments but also many flaws."
]

for example in test_examples:
    # Tokenize and pad the example
    sequence = tokenizer.texts_to_sequences([example])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_len)
    
    # Get predictions from all models
    pred1 = model1.predict(padded, verbose=0)[0][0]
    pred2 = model2.predict(padded, verbose=0)[0][0]
    pred3 = model3.predict(padded, verbose=0)[0][0]
    
    print(f"\nText: {example}")
    print(f"Conv1 prediction: {pred1:.4f} ({'Positive' if pred1 > 0.5 else 'Negative'})")
    print(f"Conv2 prediction: {pred2:.4f} ({'Positive' if pred2 > 0.5 else 'Negative'})")
    print(f"Gru prediction: {pred3:.4f} ({'Positive' if pred3 > 0.5 else 'Negative'})")
