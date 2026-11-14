import cv2
import joblib
import numpy as np
from skimage.feature import hog
from pathlib import Path

# Paths to your test images (change these to point to real MRI images)
IMG_1 = "normal.jpg"
IMG_2 = "alzheimer.jpg"
MODEL_PATH = Path("models/mri_classifier.pkl")

def debug_image(image_path, model_data):
    print(f"\n--- Debugging: {image_path} ---")
    
    # 1. Load image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: Could not read {image_path}. Check the path.")
        return
        
    # 2. Resize
    img_size = tuple(model_data["img_size"])
    img_resized = cv2.resize(img, img_size)
    
    # Verify image isn't completely black or white
    print(f"Image Max Pixel: {np.max(img_resized)}, Min Pixel: {np.min(img_resized)}, Mean: {np.mean(img_resized):.2f}")
    
    # 3. Extract HOG
    hog_params = model_data["feature_params"]
    feat = hog(img_resized, **hog_params).reshape(1, -1)
    
    # Print feature stats to see if they change between images
    print(f"HOG Feature Shape: {feat.shape}")
    print(f"HOG Feature Sum: {np.sum(feat):.2f}")
    print(f"HOG Feature Mean: {np.mean(feat):.4f}")
    
    # 4. Predict
    model = model_data["model"]
    proba = model.predict_proba(feat)[0]
    print(f"Probabilities: Normal={proba[0]:.4f}, Mild={proba[1]:.4f}, Moderate={proba[2]:.4f}")
    
    return feat

if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print("Model not found!")
        exit()
        
    model_data = joblib.load(MODEL_PATH)
    
    feat1 = debug_image(IMG_1, model_data)
    feat2 = debug_image(IMG_2, model_data)
    
    if feat1 is not None and feat2 is not None:
        if np.array_equal(feat1, feat2):
            print("\n❌ CRITICAL: The HOG features are EXACTLY the same. The model is reading the images as identical.")
        else:
            print("\n✅ GOOD: The HOG features are different. The model's dataset is the problem (underfitting/imbalance).")