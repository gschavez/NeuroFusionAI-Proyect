import os
import librosa
import soundfile as sf
import numpy as np

DATA_DIR = "dataset/raw"
folders = ["ad", "cn"]

def add_noise(file_path, output_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        noise = np.random.randn(len(y))
        augmented_y = y + 0.005 * noise
        sf.write(output_path, augmented_y, sr)
        return True
    except Exception as e:
        print(f"Error with {file_path}: {e}")
        return False

if __name__ == "__main__":
    for folder in folders:
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.exists(folder_path): continue
            
        print(f"\n--- Augmenting folder: {folder.upper()} ---")
        files = [f for f in os.listdir(folder_path) if f.endswith(".wav") and "aug" not in f]
        
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            new_filename = filename.replace(".wav", "_aug.wav")
            output_path = os.path.join(folder_path, new_filename)
            
            if not os.path.exists(output_path):
                if add_noise(file_path, output_path):
                    print(f"Created: {new_filename}")
    print("\nAugmentation complete.")