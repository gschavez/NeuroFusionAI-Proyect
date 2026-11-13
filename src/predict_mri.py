"""
Inference helper for /predict_mri. Returns 3 probabilities aligned with
CLASSES = ["Normal", "Leve", "Moderado"] used by app.py / train_fusion.py.
"""

from pathlib import Path
import pickle
import numpy as np
import cv2
from skimage.feature import hog

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "mri_classifier.pkl"

_CACHE = None


def _load():
    global _CACHE
    if _CACHE is None:
        with open(MODEL_PATH, "rb") as f:
            _CACHE = pickle.load(f)
    return _CACHE


def predict_mri(image_path):
    bundle = _load()
    model = bundle["model"]
    hog_params = bundle["feature_params"]
    img_size = tuple(bundle["img_size"])
    classes = bundle["classes_3"]  # ["Normal", "Leve", "Moderado"]

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    if img.shape[:2] != img_size:
        img = cv2.resize(img, img_size)

    feat = hog(img, **hog_params).reshape(1, -1)
    proba = model.predict_proba(feat)[0]  # (3,)
    idx = int(np.argmax(proba))

    return {
        "class_name": classes[idx],
        "class_index": idx,
        "confidence": float(proba[idx]),
        "probabilities": {classes[i]: float(proba[i]) for i in range(3)},
        "probabilities_list": [float(proba[i]) for i in range(3)],  # <- what app.py expects
    }
