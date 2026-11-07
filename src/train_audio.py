import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model():
    # 1. Load the dataset
    csv_path = "dataset/processed/audio_features.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run audio_features.py first.")
        return

    df = pd.read_csv(csv_path)

    # 2. Separate features (X) and labels (y)
    # Drop filename and label from features
    X = df.drop(columns=['filename', 'label'])
    y = df['label']

    # 3. Split the data into Training (70%) and Testing (30%)
    # stratify=y ensures the same proportion of AD and CN in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    # 4. Initialize and train the Random Forest classifier
    # We use class_weight='balanced' because we have few samples
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # 5. Make predictions on the test set
    y_pred = model.predict(X_test)

    # 6. Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy (CN)', 'Alzheimer (AD)']))

    # 7. Save the trained model
    os.makedirs("models", exist_ok=True)
    model_path = "models/audio_classifier.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved successfully to {model_path}")

    # 8. Plot and save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['CN', 'AD'], yticklabels=['CN', 'AD'])
    plt.title('Confusion Matrix - Audio Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("models/audio_confusion_matrix.png")
    print("Confusion matrix saved to models/audio_confusion_matrix.png")

if __name__ == "__main__":
    train_model()