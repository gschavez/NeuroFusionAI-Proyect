import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def train_fusion_model():
    # 1. Define file paths
    mri_path = "data/mri_predictions.csv"
    audio_path = "data/audio_predictions.csv"
    
    # 2. Check if files exist
    if not os.path.exists(mri_path) or not os.path.exists(audio_path):
        print("Error: Missing prediction CSV files in the 'data/' folder.")
        print("Make sure both mri_predictions.csv and audio_predictions.csv are present.")
        return

    print("Loading prediction datasets...")
    df_mri = pd.read_csv(mri_path)
    df_audio = pd.read_csv(audio_path)

    # 3. Create Synthetic Dataset (Random Sampling with replacement)
    # Since the datasets have different lengths (e.g., 5000 MRI vs 300 Audio),
    # we sample 3000 rows from each to create a combined synthetic dataset.
    np.random.seed(42)
    samples = 3000
    
    print(f"Generating synthetic dataset with {samples} samples...")
    mri_sampled = df_mri.sample(n=samples, replace=True).reset_index(drop=True)
    audio_sampled = df_audio.sample(n=samples, replace=True).reset_index(drop=True)

    # 4. Combine Features (6 columns total: 3 MRI probs + 3 Audio probs)
    # add_suffix is required to avoid duplicate column names in the concatenated frame
    X = pd.concat([
        mri_sampled[['prob_normal', 'prob_mild', 'prob_moderate']].add_suffix('_mri'),
        audio_sampled[['prob_normal', 'prob_mild', 'prob_moderate']].add_suffix('_audio')
    ], axis=1)
    
    # 5. Determine the final label using the Clinical Severity Rule
    # 0 = Normal, 1 = Mild, 2 = Moderate
    # The final diagnosis is the most severe of the two predictions.
    y = np.maximum(mri_sampled['y_true'], audio_sampled['y_true'])

    # 6. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 7. Define the 3 required models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine (SVM)": SVC(probability=True, random_state=42)
    }

    best_f1 = 0
    best_model_name = ""
    best_model = None
    metrics = []

    # 8. Train and evaluate each model
    print("\n--- Training and Evaluating Fusion Models ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"  -> Accuracy: {acc*100:.2f}% | F1-Score: {f1:.4f}")
        metrics.append({"model": name, "accuracy": acc, "f1_score": f1})
        
        # Save the best model based on F1-Score
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    # 9. Save the best model and metrics
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/fusion_classifier.pkl")
    
    with open("models/fusion_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n" + "="*50)
    print(f"BEST FUSION MODEL SELECTED: {best_model_name}")
    print(f"Saved to models/fusion_classifier.pkl")
    print(f"Metrics saved to models/fusion_metrics.json")
    print("="*50)

if __name__ == "__main__":
    train_fusion_model()