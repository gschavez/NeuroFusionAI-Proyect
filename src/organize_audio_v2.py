import os
import shutil

SOURCE_BASE = r"C:\Users\xcosm\Desktop\mi_dataset_manual"
DEST_AD = r"C:\Users\xcosm\NeuroFusionAI-Proyect\dataset\raw\ad"
DEST_CN = r"C:\Users\xcosm\NeuroFusionAI-Proyect\dataset\raw\cn"

def process_folder(category, destination):
    source_path = os.path.join(SOURCE_BASE, category, category)
    
    if not os.path.exists(source_path):
        print(f"Warning: Source folder {source_path} does not exist.")
        return

    # Find the highest existing number to avoid overwriting
    existing_files = [f for f in os.listdir(destination) if f.endswith('.wav')]
    max_num = 0
    for f in existing_files:
        try:
            # Extract number from "ad_24.wav" -> 24
            num = int(f.split('_')[1].split('.')[0])
            if num > max_num:
                max_num = num
        except ValueError:
            pass # Ignore files that don't have the standard format
    
    count = max_num + 1
    print(f"Found {len(existing_files)} existing files in {category.upper()}.")
    print(f"Highest number found: {max_num}. New files will start at: {count}.")

    copied_count = 0
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if file.endswith(".wav"):
                new_filename = f"{category.lower()}_{count}.wav"
                new_file_path = os.path.join(destination, new_filename)
                
                # Safety check: if it still exists, keep incrementing
                while os.path.exists(new_file_path):
                    count += 1
                    new_filename = f"{category.lower()}_{count}.wav"
                    new_file_path = os.path.join(destination, new_filename)
                
                old_file_path = os.path.join(root, file)
                shutil.copy2(old_file_path, new_file_path)
                print(f"Copied: {file} -> {new_filename}")
                count += 1
                copied_count += 1
                
    print(f"Total new files copied for {category.upper()}: {copied_count}")

if __name__ == "__main__":
    print("--- Moving NEW AD audios ---")
    process_folder("AD", DEST_AD)
    
    print("\n--- Moving NEW CN audios ---")
    process_folder("CN", DEST_CN)
    
    print("\nDone! All new audio files have been organized safely.")