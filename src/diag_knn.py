"""
Quick diagnostic: compare KNN at different k values on the SAME split.
If accuracy drops hard when k increases -> memorization / leakage.
"""
from pathlib import Path
import time
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset"
CLASSES_4 = {1: "Mild", 2: "Moderate", 3: "Non", 4: "VeryMild"}
MAP_4_TO_3 = {1: 1, 2: 2, 3: 0, 4: 1}
IMG_SIZE = (128, 128)
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(16,16),
                  cells_per_block=(2,2), block_norm="L2-Hys")
EXT = {".jpg", ".jpeg", ".png"}

def hog_feat(p):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    if img.shape[:2] != IMG_SIZE: img = cv2.resize(img, IMG_SIZE)
    return hog(img, **HOG_PARAMS)

X, y4, y3 = [], [], []
for folder in CLASSES_4:
    for p in (DATASET_PATH / str(folder)).rglob("*"):
        if p.suffix.lower() not in EXT: continue
        f = hog_feat(p)
        if f is None: continue
        X.append(f); y4.append(folder-1); y3.append(MAP_4_TO_3[folder])

X = np.asarray(X, np.float32); y4 = np.asarray(y4); y3 = np.asarray(y3)
print(f"X={X.shape}")

idx_tr, idx_te = train_test_split(np.arange(len(X)), test_size=0.15,
                                  random_state=42, stratify=y4)

print("\n=== KNN diagnosis: k vs accuracy (same split) ===")
for k, weights in [(7,"distance"), (15,"uniform"), (25,"uniform"), (50,"uniform")]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=50, random_state=42)),
        ("clf", KNeighborsClassifier(n_neighbors=k, weights=weights, n_jobs=-1)),
    ])
    t0 = time.time()
    pipe.fit(X[idx_tr], y3[idx_tr])
    pred = pipe.predict(X[idx_te])
    proba = pipe.predict_proba(X[idx_te])
    n_certain = int((proba.max(axis=1) == 1.0).sum())
    acc = accuracy_score(y3[idx_te], pred)
    f1 = f1_score(y3[idx_te], pred, average="macro", zero_division=0)
    print(f"k={k:3d} w={weights:9s} -> acc={acc:.4f} f1={f1:.4f} "
          f"certain={n_certain}/{len(idx_te)} ({n_certain/len(idx_te)*100:.1f}%) "
          f"({time.time()-t0:.1f}s)")