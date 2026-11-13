import joblib
import librosa
import numpy as np
import os

def get_audio_prediction(audio_path, model_path="models/audio_classifier.pkl", scaler_path="models/audio_scaler.pkl"):
    """
    Receives an audio path and returns the probabilities for Normal, Mild, and Moderate.
    """
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or Scaler not found. Train the model first.")
        
    # 1. Load the trained model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # 2. Extract features from the new audio (19 features)
    try:
        # Training clips were split into chunks of at most 120 s (trim_audios.py),
        # so we cap the load at 120 s: same range as training, bounded memory.
        y, sr = librosa.load(audio_path, sr=16000, mono=True, duration=120)
        
        # Basic features
        duration = librosa.get_duration(y=y, sr=sr)
        
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0.0
        
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)
        silence_threshold = rms_mean * 0.5
        silent_frames = np.sum(rms < silence_threshold)
        pause_ratio = silent_frames / len(rms) if len(rms) > 0 else 0.0
        
        # NEW Features added to match the 19
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
        
        # Model is binary (0: Normal, 1: Alzheimer), so we split the Alzheimer prob
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