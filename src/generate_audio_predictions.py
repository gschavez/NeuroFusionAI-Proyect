import pandas as pd
import joblib
import os
import numpy as np

# Paths
FEATURES_CSV = "dataset/processed/audio_features.csv"
MODEL_PATH = "models/audio_classifier.pkl"
OUTPUT_CSV = "data/audio_predictions.csv"

def generate_predictions():
    # 1. Load features
    if not os.path.exists(FEATURES_CSV):
        print("Error: audio_features.csv not found.")
        return
        
    df = pd.read_csv(FEATURES_CSV)
    
    # 2. Load model
    if not os.path.exists(MODEL_PATH):
        print("Error: audio_classifier.pkl not found.")
        return
        
    model = joblib.load(MODEL_PATH)
    
    # Load scaler if it exists
    scaler_path = "models/audio_scaler.pkl"
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    # 3. Prepare data for prediction
    X = df.drop(columns=['filename', 'label'])
    y_true = df['label']
    
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    # 4. Get probabilities (0: Normal, 1: Alzheimer)
    probs = model.predict_proba(X_scaled)
    
    # 5. Create the output dataframe
    output_df = pd.DataFrame()
    
    # Create a sequential patient_id (0 to N-1)
    output_df['patient_id'] = range(len(df))
    output_df['y_true'] = y_true
    
    # Audio model outputs binary probabilities. We map them to 3 classes.
    output_df['prob_normal'] = probs[:, 0]
    output_df['prob_mild'] = probs[:, 1]
    output_df['prob_moderate'] = 0.0
    
    # Save to data/
    os.makedirs("data", exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully generated {OUTPUT_CSV} with {len(output_df)} rows.")
    print(output_df.head())

if __name__ == "__main__":
    generate_predictions()