import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('full_pipeline_testing.csv')

# Create confusion matrix
cm = confusion_matrix(df['true_sentiment'], df['sentiment predicted'], 
                     labels=['negative', 'neutral', 'positive'])

# Print confusion matrix
print("\nConfusion Matrix:")
print(cm)

# Print classification report
print("\nClassification Report:")
print(classification_report(df['true_sentiment'], df['sentiment predicted']))

# Create a heatmap of the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['negative', 'neutral', 'positive'],
            yticklabels=['negative', 'neutral', 'positive'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.close()

# Calculate additional metrics
total_samples = len(df)
correct_predictions = (df['true_sentiment'] == df['sentiment predicted']).sum()
accuracy = correct_predictions / total_samples

print(f"\nAdditional Metrics:")
print(f"Total Samples: {total_samples}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2%}")

# Calculate metrics per class
for sentiment in ['negative', 'neutral', 'positive']:
    true_positives = ((df['true_sentiment'] == sentiment) & 
                     (df['sentiment predicted'] == sentiment)).sum()
    false_positives = ((df['true_sentiment'] != sentiment) & 
                      (df['sentiment predicted'] == sentiment)).sum()
    false_negatives = ((df['true_sentiment'] == sentiment) & 
                      (df['sentiment predicted'] != sentiment)).sum()
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{sentiment.upper()} Class Metrics:")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")

# Compute general F1 score and accuracy (micro average)
general_f1 = f1_score(df['true_sentiment'], df['sentiment predicted'], average='micro')
general_accuracy = accuracy_score(df['true_sentiment'], df['sentiment predicted'])

print(f"\nGeneral (Micro-Averaged) F1 Score: {general_f1:.2%}")
print(f"General Accuracy: {general_accuracy:.2%}") 