import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Create directory if it doesn't exist
os.makedirs("static/images", exist_ok=True)

# Load the predictions
df = pd.read_csv("data/mri_predictions.csv")

# Get true labels
y_true = df['y_true']

# Get predicted labels (argmax of the probabilities)
probs = df[['prob_normal', 'prob_mild', 'prob_moderate']].values
y_pred = np.argmax(probs, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
labels = ['Normal', 'Mild', 'Moderate']

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('MRI Model - Confusion Matrix (SVM)', fontsize=14, pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()

# Save
plt.savefig("static/images/mri_confusion_matrix.png", dpi=300)
print("✅ Image saved to static/images/mri_confusion_matrix.png")