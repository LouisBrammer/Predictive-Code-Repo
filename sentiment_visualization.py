import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Set style
plt.style.use('default')
sns.set_theme()

# Read the data
df = pd.read_csv('full_pipeline_testing.csv')

# Create a figure with subplots
plt.figure(figsize=(20, 15))

# 1. Confusion Matrix
plt.subplot(2, 2, 1)
cm = confusion_matrix(df['true_sentiment'], df['sentiment predicted'], 
                     labels=['negative', 'neutral', 'positive'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['negative', 'neutral', 'positive'],
            yticklabels=['negative', 'neutral', 'positive'])
plt.title('Confusion Matrix', pad=20, fontsize=12)
plt.ylabel('True Label', fontsize=10)
plt.xlabel('Predicted Label', fontsize=10)

# 2. Class Distribution
plt.subplot(2, 2, 2)
class_dist = df['true_sentiment'].value_counts()
sns.barplot(x=class_dist.index, y=class_dist.values)
plt.title('Distribution of True Sentiments', pad=20, fontsize=12)
plt.xlabel('Sentiment', fontsize=10)
plt.ylabel('Count', fontsize=10)

# 3. Prediction Distribution
plt.subplot(2, 2, 3)
pred_dist = df['sentiment predicted'].value_counts()
sns.barplot(x=pred_dist.index, y=pred_dist.values)
plt.title('Distribution of Predicted Sentiments', pad=20, fontsize=12)
plt.xlabel('Sentiment', fontsize=10)
plt.ylabel('Count', fontsize=10)

# 4. Error Analysis
plt.subplot(2, 2, 4)
df['is_correct'] = df['true_sentiment'] == df['sentiment predicted']
error_by_class = df.groupby('true_sentiment')['is_correct'].mean()
sns.barplot(x=error_by_class.index, y=error_by_class.values)
plt.title('Accuracy by True Sentiment Class', pad=20, fontsize=12)
plt.xlabel('True Sentiment', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)

# Adjust layout and save
plt.tight_layout()
plt.savefig('sentiment_analysis_plots.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a separate plot for error cases
error_cases = df[df['true_sentiment'] != df['sentiment predicted']]
plt.figure(figsize=(12, 6))
error_matrix = pd.crosstab(error_cases['true_sentiment'], error_cases['sentiment predicted'])
sns.heatmap(error_matrix, annot=True, fmt='d', cmap='Reds')
plt.title('Error Analysis: True vs Predicted Sentiments', pad=20, fontsize=12)
plt.ylabel('True Sentiment', fontsize=10)
plt.xlabel('Predicted Sentiment', fontsize=10)
plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualizations have been saved as 'sentiment_analysis_plots.png' and 'error_analysis.png'") 