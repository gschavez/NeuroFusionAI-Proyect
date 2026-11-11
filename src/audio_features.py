import os
import librosa
import numpy as np
import pandas as pd

def extract_features(file_path):
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=16000)
        
        # 1. Basic Features
        duration = librosa.get_duration(y=y, sr=sr)
        
        # 2. Pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0.0
        
        # 3. Energy and Pauses
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)
        silence_threshold = rms_mean * 0.5
        silent_frames = np.sum(rms < silence_threshold)
        pause_ratio = silent_frames / len(rms) if len(rms) > 0 else 0.0
        
        # 4. NEW FEATURES: Zero Crossing Rate & Spectral Centroid
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)
        
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid_mean = np.mean(spectral_centroids)
        
        # 5. MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)

        # Pack features
        features = {
            'duration': duration,
            'pitch_mean': pitch_mean,
            'rms_mean': rms_mean,
            'pause_ratio': pause_ratio,
            'zcr_mean': zcr_mean,
            'spectral_centroid': spectral_centroid_mean
        }
        
        # Add the 13 MFCCs
        for i, mfcc_val in enumerate(mfcc_means):
            features[f'mfcc_{i+1}'] = mfcc_val
            
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

if __name__ == "__main__":
    base_data_dir = "dataset/raw"
    output_csv = "dataset/processed/audio_features.csv"
    
    os.makedirs("dataset/processed", exist_ok=True)
    all_features = []
    
    folders = {
        "ad": 1,  # 1 = Alzheimer
        "cn": 0   # 0 = Healthy
    }
    
    for folder_name, label in folders.items():
        folder_path = os.path.join(base_data_dir, folder_name)
        
        if not os.path.exists(folder_path):
            continue
            
        print(f"\n--- Processing folder: {folder_name.upper()} ---")
        files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
        print(f"Found {len(files)} files in {folder_name}.")
        
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            features = extract_features(file_path)
            if features:
                features['label'] = label
                features['filename'] = f"{folder_name}/{filename}"
                all_features.append(features)
    
    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv(output_csv, index=False)
        print(f"\nDone! Features saved to {output_csv}")
        print(f"Total files processed: {len(df)}")
    else:
        print("\nNo audio files were found or processed.")