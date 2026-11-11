import pandas as pd
import joblib
import os
import numpy as np

FEATURES_CSV = "dataset/processed/audio_features.csv"
MODEL_PATH = "models/audio_classifier.pkl"
SCALER_PATH = "models/audio_scaler.pkl"
OUTPUT_CSV = "data/audio_predictions.csv"

def generate_predictions():
    print("--- Generating Audio Predictions for Fusion ---")
    
    df = pd.read_csv(FEATURES_CSV)
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    X = df.drop(columns=['filename', 'label'])
    y_true = df['label']
    
    # Since the dataset is binary (0=Normal, 1=Alzheimer), we need to map y_true to 3 classes for fusion
    # Mapping: 0 -> 0 (Normal), 1 -> 1 (Mild) or 2 (Moderate). 
    # For the synthetic dataset, we will split the '1' (Alzheimer) labels evenly into 1 and 2
    # to allow the fusion model to learn the 3rd class.
    np.random.seed(42)
    y_true_fusion = []
    for label in y_true:
        if label == 0:
            y_true_fusion.append(0) # Normal
        else:
            # Randomly assign Mild (1) or Moderate (2) to balance the classes
            y_true_fusion.append(np.random.choice([1, 2], p=[0.7, 0.3]))

    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)
    
    output_df = pd.DataFrame()
    output_df['patient_id'] = range(len(df))
    output_df['y_true'] = y_true_fusion
    
    # REDISTRIBUTE PROBABILITIES:
    # We take the binary probability (prob_alzheimer) and split it into Mild and Moderate
    prob_alzheimer = probs[:, 1]
    
    output_df['prob_normal'] = probs[:, 0]
    output_df['prob_mild'] = prob_alzheimer * 0.7  # 70% of Alzheimer prob goes to Mild
    output_df['prob_moderate'] = prob_alzheimer * 0.3 # 30% of Alzheimer prob goes to Moderate
    
    os.makedirs("data", exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"✅ Successfully generated {OUTPUT_CSV}")
    print(f"Total rows: {len(output_df)}")
    print("\nClass distribution (y_true):")
    print(output_df['y_true'].value_counts().sort_index())
    print("\nFirst 5 rows:")
    print(output_df.head())

if __name__ == "__main__":
    generate_predictions()