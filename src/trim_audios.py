import os
import librosa
import soundfile as sf
import math

# --- CONFIGURATION ---
DATA_DIR = "dataset/raw"
CHUNK_LENGTH_SEC = 120  # Split every 2 minutes (120 seconds)

def split_audio(file_path, output_dir, base_filename, max_sec=CHUNK_LENGTH_SEC):
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # If audio is shorter than 2 minutes, just move it (or keep it)
        if duration <= max_sec:
            print(f"Skipped (too short): {file_path} ({duration:.1f}s)")
            return
            
        # Calculate how many chunks we can create
        num_chunks = math.ceil(duration / max_sec)
        print(f"Splitting {file_path} ({duration:.1f}s) into {num_chunks} chunks.")
        
        for i in range(num_chunks):
            start_sample = i * max_sec * sr
            end_sample = min((i + 1) * max_sec * sr, len(y))
            
            # Extract the chunk
            chunk = y[start_sample:end_sample]
            
            # Skip chunks that are too short (e.g., less than 10 seconds of noise at the end)
            if len(chunk) < 10 * sr:
                continue
                
            # Define new filename (e.g., ad_1_part1.wav, ad_1_part2.wav)
            new_filename = f"{base_filename}_part{i+1}.wav"
            new_file_path = os.path.join(output_dir, new_filename)
            
            # Save the chunk
            sf.write(new_file_path, chunk, sr)
            
        # Delete the original file to save space
        os.remove(file_path)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    folders = ["ad", "cn"]
    
    for folder in folders:
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.exists(folder_path):
            continue
            
        print(f"\n--- Processing folder: {folder.upper()} ---")
        
        # Get a list of all wav files
        files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
        
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            # Get filename without extension (e.g., "ad_1")
            base_filename = os.path.splitext(filename)[0] 
            
            split_audio(file_path, folder_path, base_filename)
            
    print("\nSplitting complete! Now run audio_features.py again.")