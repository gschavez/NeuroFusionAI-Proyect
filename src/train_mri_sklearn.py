"""
Train and compare 3 classic ML models (RandomForest, SVM, KNN) for MRI.

Uses GroupShuffleSplit with perceptual hash (pHash) to avoid data leakage:
images of the same patient or near-duplicate scans stay in the SAME split.

Fusion contract (confirmed with Persona 3):
    - CSV: patient_id, y_true, prob_Normal, prob_Leve, prob_Moderado
    - y_true: INTEGER 0=Normal, 1=Leve, 2=Moderado
    - patient_id: consecutive 0..N-1

Outputs:
    - models/mri_classifier.pkl
    - models/mri_metrics.json
    - models/mri_predictions.csv
"""

from pathlib import Path
from collections import Counter
import csv
import json
import pickle
import time
import warnings

import numpy as np
import cv2
from skimage.feature import hog

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")


# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "models" / "mri_classifier.pkl"
METRICS_PATH = BASE_DIR / "models" / "mri_metrics.json"
CSV_PATH = BASE_DIR / "models" / "mri_predictions.csv"

CLASSES_4 = {
    1: "Mild Demented",
    2: "Moderate Demented",
    3: "Non Demented",
    4: "Very Mild Demented",
}
SHORT_NAMES_4 = {
    1: "mild_demented",
    2: "moderate_demented",
    3: "non_demented",
    4: "very_mild_demented",
}

CLASSES_3 = ["Normal", "Leve", "Moderado"]
MAP_4_TO_3 = {1: 1, 2: 2, 3: 0, 4: 1}

IMG_SIZE = (128, 128)
RANDOM_SEED = 42
TEST_SIZE = 0.15
PHASH_THRESHOLD = 8  # Hamming distance for near-duplicates

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png"}


# =====================================================
# PERCEPTUAL HASH (for leakage-free grouping)
# =====================================================

def phash(img_bgr, hash_size=8, highfreq_factor=4):
    """DCT-based perceptual hash -> 64-bit int."""
    size = hash_size * highfreq_factor
    img = cv2.resize(img_bgr, (size, size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    low = dct[:hash_size, :hash_size]
    med = np.median(low)
    bits = (low > med).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_hog(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if img.shape[:2] != IMG_SIZE:
        img = cv2.resize(img, IMG_SIZE)
    return hog(img, **HOG_PARAMS)


def load_and_group():
    """Load images + group near-duplicates via pHash."""
    X, y4, y3, paths, hashes = [], [], [], [], []
    print("=" * 55)
    print("LOADING + pHASH GROUPING")
    print("=" * 55)

    for folder, name in CLASSES_4.items():
        folder_path = DATASET_PATH / str(folder)
        count = 0
        t0 = time.time()
        for p in folder_path.rglob("*"):
            if p.suffix.lower() not in SUPPORTED_EXT:
                continue
            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                continue
            feat = extract_hog(p)
            if feat is None:
                continue
            X.append(feat)
            y4.append(folder - 1)
            y3.append(MAP_4_TO_3[folder])
            paths.append(str(p))
            hashes.append(phash(img_bgr))
            count += 1
        print(f"  [{name}] {count} images in {time.time()-t0:.1f}s")

    # Group near-duplicates
    print(f"\nGrouping by pHash (threshold={PHASH_THRESHOLD})...")
    n = len(hashes)
    group_id = np.full(n, -1, dtype=int)
    current = 0
    for i in range(n):
        if group_id[i] != -1:
            continue
        group_id[i] = current
        for j in range(i + 1, n):
            if group_id[j] == -1 and hamming(hashes[i], hashes[j]) <= PHASH_THRESHOLD:
                group_id[j] = current
        current += 1
    print(f"  Total groups: {current} (from {n} images)")
    print(f"  Avg group size: {n/current:.2f}")

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y4, dtype=np.int64),
        np.asarray(y3, dtype=np.int64),
        paths, group_id,
    )


# =====================================================
# MODELS
# =====================================================

def build_models():
    return {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                random_state=RANDOM_SEED, n_jobs=-1,
            )),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=100, random_state=RANDOM_SEED)),
            ("clf", SVC(
                kernel="rbf", C=1.0, gamma="scale",
                class_weight="balanced", probability=True,
                random_state=RANDOM_SEED,
            )),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=50, random_state=RANDOM_SEED)),
            # k=25, uniform: honest KNN without memorization
            ("clf", KNeighborsClassifier(
                n_neighbors=25, weights="uniform", n_jobs=-1,
            )),
        ]),
    }


# =====================================================
# MAIN
# =====================================================

def main():
    np.random.seed(RANDOM_SEED)
    X, y4, y3, paths, groups = load_and_group()
    print(f"\nFeature matrix: {X.shape}")

    # Split by GROUP, not by image -> no patient overlap
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    idx_train, idx_test = next(gss.split(X, y3, groups=groups))
    print(f"Train: {len(idx_train)} | Test: {len(idx_test)}")
    print("Class distribution in test:", Counter(y3[idx_test].tolist()))

    results, fitted, probas = {}, {}, {}
    for name, pipe in build_models().items():
        print(f"\n--- {name} ---")
        t0 = time.time()
        pipe.fit(X[idx_train], y3[idx_train])
        tt = time.time() - t0
        y_pred = pipe.predict(X[idx_test])
        pr = pipe.predict_proba(X[idx_test])

        acc = accuracy_score(y3[idx_test], y_pred)
        mf1 = f1_score(y3[idx_test], y_pred, average="macro", zero_division=0)
        wf1 = f1_score(y3[idx_test], y_pred, average="weighted", zero_division=0)
        mp = precision_score(y3[idx_test], y_pred, average="macro", zero_division=0)
        mr = recall_score(y3[idx_test], y_pred, average="macro", zero_division=0)

        print(f"  Accuracy={acc:.4f}  MacroF1={mf1:.4f}  WeightedF1={wf1:.4f}  Time={tt:.1f}s")
        print(classification_report(y3[idx_test], y_pred, labels=[0, 1, 2],
                                    target_names=CLASSES_3, zero_division=0))

        results[name] = {
            "accuracy": round(float(acc), 4), "f1_score": round(float(mf1), 4),
            "weighted_f1": round(float(wf1), 4), "precision_macro": round(float(mp), 4),
            "recall_macro": round(float(mr), 4), "train_time_sec": round(float(tt), 1),
        }
        fitted[name] = pipe
        probas[name] = pr

    best = max(results, key=lambda k: results[k]["f1_score"])
    best_model = fitted[best]
    print(f"\nBest model (grouped split): {best} (Macro F1 = {results[best]['f1_score']})")

    # Save bundle
    bundle = {
        "model": best_model, "model_name": best,
        "classes_3": CLASSES_3, "classes_4": CLASSES_4,
        "short_names_4": SHORT_NAMES_4, "map_4_to_3": MAP_4_TO_3,
        "feature_type": "HOG", "feature_params": HOG_PARAMS,
        "img_size": IMG_SIZE, "split_strategy": "GroupShuffleSplit_by_pHash",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nModel saved: {MODEL_PATH}")

    # Save metrics
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "best_model": best, "models": results, "classes": CLASSES_3,
            "split_strategy": "GroupShuffleSplit_by_pHash",
            "phash_threshold": PHASH_THRESHOLD,
            "note": "Split by perceptual hash to prevent data leakage from near-duplicate MRI scans.",
        }, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved: {METRICS_PATH}")

    # Save CSV
    pb = probas[best]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "y_true", "prob_Normal", "prob_Leve", "prob_Moderado"])
        for i in range(len(idx_test)):
            w.writerow([
                i, int(y3[idx_test[i]]),
                round(float(pb[i, 0]), 6),
                round(float(pb[i, 1]), 6),
                round(float(pb[i, 2]), 6),
            ])
    print(f"Predictions saved: {CSV_PATH}")

    cm = confusion_matrix(y3[idx_test], best_model.predict(X[idx_test]), labels=[0, 1, 2])
    print("\nConfusion Matrix (grouped split):")
    print(cm)
    print("\nDone.")


if __name__ == "__main__":
    main()