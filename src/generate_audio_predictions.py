import pandas as pd
import joblib
import os
import numpy as np

# Paths
FEATURES_CSV = "dataset/processed/audio_features.csv"
MODEL_PATH = "models/audio_classifier.pkl"
SCALER_PATH = "models/audio_scaler.pkl"
OUTPUT_CSV = "data/audio_predictions.csv"

def generate_predictions():
    print("--- Generating Audio Predictions for Fusion ---")
    
    # 1. Load dataset
    if not os.path.exists(FEATURES_CSV):
        print(f"Error: {FEATURES_CSV} not found.")
        return
        
    df = pd.read_csv(FEATURES_CSV)
    
    # 2. Load model and scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Error: Model or Scaler not found in models/ folder.")
        return
        
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 3. Prepare data
    X = df.drop(columns=['filename', 'label'])
    y_true = df['label']
    
    # Scale features
    X_scaled = scaler.transform(X)

    # 4. Get probabilities (0: Normal, 1: Alzheimer)
    # Note: For a binary model, predict_proba returns [[prob_0, prob_1]]
    probs = model.predict_proba(X_scaled)
    
    # 5. Create the output dataframe
    output_df = pd.DataFrame()
    
    # Create a sequential patient_id (0 to N-1)
    output_df['patient_id'] = range(len(df))
    output_df['y_true'] = y_true
    
    # The Fusion model expects 3 classes (0=Normal, 1=Mild, 2=Moderate)
    # Since our audio model is binary (0=Normal, 1=Alzheimer), we map it:
    output_df['prob_normal'] = probs[:, 0]
    output_df['prob_mild'] = probs[:, 1]
    output_df['prob_moderate'] = 0.0 # Audio model doesn't distinguish moderate
    
    # Save to data/ folder
    os.makedirs("data", exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"✅ Successfully generated {OUTPUT_CSV}")
    print(f"Total rows: {len(output_df)}")
    print("\nFirst 5 rows:")
    print(output_df.head())

if __name__ == "__main__":
    generate_predictions()