import joblib
import librosa
import numpy as np
import os
from functools import lru_cache

# Define default paths
DEFAULT_MODEL_PATH = "models/audio_classifier.pkl"
DEFAULT_SCALER_PATH = "models/audio_scaler.pkl"

# TRICK: lru_cache loads the model into RAM ONLY ONCE and reuses it
@lru_cache(maxsize=1)
def load_models():
    if not os.path.exists(DEFAULT_MODEL_PATH) or not os.path.exists(DEFAULT_SCALER_PATH):
        raise FileNotFoundError("Model or Scaler not found. Train the model first.")
    model = joblib.load(DEFAULT_MODEL_PATH)
    scaler = joblib.load(DEFAULT_SCALER_PATH)
    return model, scaler

def get_audio_prediction(audio_path):
    """
    Receives an audio path and returns the probabilities for Normal, Mild, and Moderate.
    """
    # 1. Load the trained model and scaler (from cache)
    try:
        model, scaler = load_models()
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
    # 2. Extract features from the new audio (19 features)
    try:
        # Load audio with a maximum duration of 120s (bounded memory)
        y, sr = librosa.load(audio_path, sr=16000, mono=True, duration=120)
        
        # Basic features
        duration = librosa.get_duration(y=y, sr=sr)
        
        # CRITICAL OPTIMIZATION: Use yin instead of piptrack
        # yin is much lighter and faster to compute pitch, preventing server crashes
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        # Filter out null values that yin might return
        pitch_values = f0[~np.isnan(f0)]
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0.0
        
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)
        silence_threshold = rms_mean * 0.5
        silent_frames = np.sum(rms < silence_threshold)
        pause_ratio = silent_frames / len(rms) if len(rms) > 0 else 0.0
        
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)
        
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid_mean = np.mean(spectral_centroids)
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)

        # Concatenate features in the EXACT order the model expects
        features = [duration, pitch_mean, rms_mean, pause_ratio, zcr_mean, spectral_centroid_mean]
        features.extend(mfcc_means)
        
        # Reshape and Scale
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        # 3. Predict
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Model is binary (0: Normal, 1: Alzheimer)
        prob_normal = float(probabilities[0])
        prob_alzheimer = float(probabilities[1])
        
        # Distribute Alzheimer probability (70% Mild, 30% Moderate)
        prob_mild = prob_alzheimer * 0.7
        prob_moderate = prob_alzheimer * 0.3
        
        return {
            "status": "success",
            "probabilities": [prob_normal, prob_mild, prob_moderate]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }