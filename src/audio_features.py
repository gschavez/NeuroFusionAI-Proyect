import os
import librosa
import numpy as np
import pandas as pd

def extract_features(file_path):
    try:
        # Load audio (16kHz is the standard for speech)
        y, sr = librosa.load(file_path, sr=16000)
        
        # 1. Total duration
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
        
        # 4. MFCCs (Vocal tract characteristics)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)

        # Create feature dictionary
        features = {
            'duration': duration,
            'pitch_mean': pitch_mean,
            'rms_mean': rms_mean,
            'pause_ratio': pause_ratio,
        }
        
        # Append the 13 MFCCs
        for i, mfcc_val in enumerate(mfcc_means):
            features[f'mfcc_{i+1}'] = mfcc_val
            
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

if __name__ == "__main__":
    data_dir = "dataset/raw"
    output_csv = "dataset/processed/audio_features.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs("dataset/processed", exist_ok=True)
    
    all_features = []
    
    # Read all audio files in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(data_dir, filename)
            print(f"Processing: {filename}")
            
            features = extract_features(file_path)
            if features:
                # Label: 1 for Alzheimer (AD), 0 for Healthy Control (CN)
                if filename.startswith("ad"):
                    features['label'] = 1
                elif filename.startswith("cn"):
                    features['label'] = 0
                else:
                    continue # Ignore files that do not start with ad_ or cn_
                
                features['filename'] = filename
                all_features.append(features)
    
    # Save everything to a DataFrame and export to CSV
    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)
    print(f"\nDone! Features saved to {output_csv}")
    print(df.head())