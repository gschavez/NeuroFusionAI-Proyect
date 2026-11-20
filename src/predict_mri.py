"""
Inference helper for /predict_mri. Returns 3 probabilities aligned with
CLASSES = ["Normal", "Mild", "Moderate"] used by app.py / train_fusion.py.

It also checks whether the image resembles the training data. The model is an
RBF-kernel SVM: when an image is very different from every training example,
all kernel values are ~0 and the SVM falls back to the SAME "default" output
(about 75 % Normal / 25 % Mild) whatever the image shows. In that case the MRI
result is flagged as not informative instead of being shown as a real prediction.
"""

from pathlib import Path
import pickle
import numpy as np
import cv2
from skimage.feature import hog

try:  # works both as "src.predict_mri" (Flask) and from inside src/
    from src.mri_features import extract_features
except ImportError:  # pragma: no cover
    from mri_features import extract_features

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "mri_classifier.pkl"

# Below this kernel similarity to the closest support vector the SVM output is
# essentially its default. Calibrate it with scripts/check_mri_inputs.py on the dataset.
MIN_SIMILARITY = 0.01
# If the probabilities are this close (L1 distance) to the default output, flag them.
DEFAULT_OUTPUT_TOLERANCE = 0.05
# Almost uniform images (blank, black, white) are not valid scans.
MIN_INTENSITY_STD = 10.0

_CACHE = None


def _load():
    global _CACHE
    if _CACHE is None:
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        pipe = bundle["model"]
        svc = pipe.named_steps["clf"]
        # Default output: what the SVM returns for a point infinitely far from all
        # support vectors (every RBF kernel value = 0).
        far = np.full((1, svc.support_vectors_.shape[1]), 1e3)
        bundle["_default_proba"] = svc.predict_proba(far)[0]
        # New models (src/train_mri_robust.py) store their own calibrated threshold.
        bundle["_min_similarity"] = float(bundle.get("similarity_threshold", MIN_SIMILARITY))
        _CACHE = bundle
    return _CACHE


def _similarity(pipe, feat):
    """RBF kernel value to the closest support vector (1 = identical, 0 = unrelated)."""
    svc = pipe.named_steps["clf"]
    z = pipe[:-1].transform(feat)  # every step before the SVM (scaler/PCA)
    d2 = ((svc.support_vectors_ - z) ** 2).sum(axis=1).min()
    return float(np.exp(-svc._gamma * d2))


def predict_mri(image_path):
    bundle = _load()
    model = bundle["model"]
    hog_params = bundle["feature_params"]
    img_size = tuple(bundle["img_size"])
    classes = bundle["classes_3"]  # ["Normal", "Mild", "Moderate"]

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    original_size = [int(img.shape[1]), int(img.shape[0])]
    intensity_std = float(img.std())

    if bundle.get("preprocess") == "brain_crop_v1":
        # Robust model: same cleanup/crop/contrast as in training.
        feat = extract_features(img).reshape(1, -1)
    else:
        # Old model (train_mri_sklearn.py): raw resize + HOG.
        if img.shape[:2] != img_size:
            img = cv2.resize(img, img_size)
        feat = hog(img, **hog_params).reshape(1, -1)
    proba = model.predict_proba(feat)[0]  # (3,)
    idx = int(np.argmax(proba))

    similarity = _similarity(model, feat)
    distance_to_default = float(np.abs(proba - bundle["_default_proba"]).sum())
    reasons = []
    if intensity_std < MIN_INTENSITY_STD:
        reasons.append("The image is almost uniform (blank, black or white).")
    if similarity < bundle["_min_similarity"] or distance_to_default < DEFAULT_OUTPUT_TOLERANCE:
        reasons.append("The image does not resemble the MRI slices used for training, so the model "
                       "returned its default output instead of a real prediction.")

    return {
        "class_name": classes[idx],
        "class_index": idx,
        "confidence": float(proba[idx]),
        "probabilities": {classes[i]: float(proba[i]) for i in range(3)},
        "probabilities_list": [float(proba[i]) for i in range(3)],  # <- what app.py expects
        "quality": {
            "similarity": round(similarity, 6),
            "distance_to_default": round(distance_to_default, 4),
            "original_size": original_size,
            "uncertain_input": bool(reasons),
            "reasons": reasons,
        },
    }