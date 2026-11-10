import joblib
import librosa
import numpy as np
import os

def get_audio_prediction(audio_path, model_path="models/audio_classifier.pkl"):
    """
    Receives an audio path and returns the probability of Alzheimer.
    Returns a dictionary with the probability (0.0 to 1.0) and the predicted class.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train it first.")
        
    # 1. Load the trained model
    model = joblib.load(model_path)
    
    # 2. Extract features from the new audio
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        
        duration = librosa.get_duration(y=y, sr=sr)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0.0
        
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)
        silence_threshold = rms_mean * 0.5
        silent_frames = np.sum(rms < silence_threshold)
        pause_ratio = silent_frames / len(rms) if len(rms) > 0 else 0.0
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)

        # Concatenate features in the exact same order as training
        features = [duration, pitch_mean, rms_mean, pause_ratio]
        features.extend(mfcc_means)
        
        # Reshape for the model
        features_array = np.array(features).reshape(1, -1)
        
        # 3. Predict
        # predict_proba returns [[prob_class_0, prob_class_1]]
        # class_0 = CN (Healthy), class_1 = AD (Alzheimer)
        probabilities = model.predict_proba(features_array)[0]
        
        prob_alzheimer = float(probabilities[1])
        prediction_label = "Alzheimer (AD)" if prob_alzheimer > 0.5 else "Healthy Control (CN)"
        
        # 4. Return the result formatted for the Fusion (Person 3)
        return {
            "status": "success",
            "prediction": prediction_label,
            "probability_ad": round(prob_alzheimer, 4) # Example: 0.8523
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Test with one of the raw audios
    test_audio = "dataset/raw/ad_1.wav"  # Change to cn_1.wav to test the other class
    
    print(f"Testing Audio Prediction for: {test_audio}")
    result = get_audio_prediction(test_audio)
    
    print("\n--- RESULT ---")
    print(f"Status: {result.get('status')}")
    if result.get('status') == 'success':
        print(f"Prediction: {result.get('prediction')}")
        print(f"Probability of AD: {result.get('probability_ad') * 100:.2f}%")
    else:
        print(f"Error: {result.get('message')}")