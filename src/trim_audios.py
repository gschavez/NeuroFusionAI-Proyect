import os
import librosa
import soundfile as sf

# --- CONFIGURATION ---
DATA_DIR = "dataset/raw"
MAX_DURATION_SEC = 120  # Trim audio to 2 minutes (120 seconds)

def trim_audio(file_path, max_sec=MAX_DURATION_SEC):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # If the audio is longer than the max allowed, trim it
        if duration > max_sec:
            y_trimmed = y[:max_sec * sr]
            sf.write(file_path, y_trimmed, sr)
            print(f"Trimmed: {file_path} (from {duration:.1f}s to {max_sec}s)")
        else:
            print(f"Skipped: {file_path} (Duration: {duration:.1f}s)")
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    folders = ["ad", "cn"]
    
    for folder in folders:
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found.")
            continue
            
        print(f"\n--- Processing folder: {folder.upper()} ---")
        
        for filename in os.listdir(folder_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(folder_path, filename)
                trim_audio(file_path)
                
    print("\nTrimming complete. You can now run audio_features.py again.")