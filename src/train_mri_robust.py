"""
Robust MRI model (SVM on HOG) - replaces the brittle models/mri_classifier.pkl.

    python src/train_mri_robust.py            # uses ./dataset/1..4
    python src/train_mri_robust.py D:/dataset  # another folder

What changes compared with train_mri_sklearn.py:
  * Preprocessing from src/mri_features.py (background cleanup, brain crop,
    centering, contrast). The SAME function is used by src/predict_mri.py.
  * No StandardScaler on HOG (it was what made the old model collapse to
    75 % Normal / 25 % Mild on any slightly different image).
  * Data augmentation (rotation, shift, zoom, brightness, JPEG) on the TRAIN
    split only.
  * Same leakage-free split as before (GroupShuffleSplit by perceptual hash).
  * A robustness test: the test images are also evaluated after a random
    perturbation, and the script prints how many answers are the "default"
    75/25 output. The old model fails this test; the new one should not.

Outputs (same names as before, so app.py and train_fusion.py keep working):
  models/mri_classifier.pkl, models/mri_metrics.json, data/mri_predictions.csv
The old files are copied to *.old first. After training, run:
  python src/train_fusion.py
because the fusion model is trained on data/mri_predictions.csv.
"""

import csv
import json
import pickle
import shutil
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mri_features import (HOG_PARAMS, IMG_SIZE, PREPROCESS_VERSION,  # noqa: E402
                          augment, extract_features, to_gray)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "models" / "mri_classifier.pkl"
METRICS_PATH = BASE_DIR / "models" / "mri_metrics.json"
CSV_PATH = BASE_DIR / "data" / "mri_predictions.csv"

CLASSES_4 = {1: "Mild Demented", 2: "Moderate Demented", 3: "Non Demented", 4: "Very Mild Demented"}
MAP_4_TO_3 = {1: 1, 2: 2, 3: 0, 4: 1}
CLASSES_3 = ["Normal", "Mild", "Moderate"]

RANDOM_SEED = 42
TEST_SIZE = 0.15
PHASH_THRESHOLD = 8
AUG_COPIES = 2            # augmented copies per training image
AUG_COPIES_MODERATE = 8   # Moderate has very few images: more copies
C_VALUES = [1.0, 5.0, 10.0]
SUPPORTED_EXT = {".jpg", ".jpeg", ".png"}


def phash(gray, hash_size=8, highfreq_factor=4):
    size = hash_size * highfreq_factor
    dct = cv2.dct(np.float32(cv2.resize(gray, (size, size))))
    low = dct[:hash_size, :hash_size]
    bits = (low > np.median(low)).flatten()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def popcount64(x):
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def load_images():
    imgs, y3, paths = [], [], []
    print("=" * 60)
    print(f"LOADING  {DATASET_PATH}")
    print("=" * 60)
    for folder, name in CLASSES_4.items():
        n = 0
        for p in sorted((DATASET_PATH / str(folder)).rglob("*")):
            if p.suffix.lower() not in SUPPORTED_EXT:
                continue
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            imgs.append(img)
            y3.append(MAP_4_TO_3[folder])
            paths.append(str(p))
            n += 1
        print(f"  [{folder}] {name:20s} {n} images")
    if not imgs:
        raise SystemExit(f"No images found under {DATASET_PATH}/1..4")
    return imgs, np.asarray(y3, dtype=np.int64), paths


def group_by_phash(imgs):
    hashes = np.array([phash(i) for i in imgs], dtype=np.uint64)
    n = len(hashes)
    group = np.full(n, -1, dtype=int)
    g = 0
    for i in range(n):
        if group[i] != -1:
            continue
        group[i] = g
        rest = np.nonzero(group[i + 1:] == -1)[0] + i + 1
        if len(rest):
            d = popcount64(np.bitwise_xor(hashes[rest], hashes[i]))
            group[rest[d <= PHASH_THRESHOLD]] = g
        g += 1
    print(f"  pHash groups: {g} (from {n} images)")
    return group


def features(imgs):
    return np.stack([extract_features(i) for i in imgs])


def build(c):
    return Pipeline([
        ("pca", PCA(n_components=100, random_state=RANDOM_SEED)),
        ("clf", SVC(kernel="rbf", C=c, gamma="scale", class_weight="balanced",
                    probability=True, random_state=RANDOM_SEED)),
    ])


def similarity(pipe, X):
    """RBF kernel value to the nearest support vector, per row."""
    z = pipe[:-1].transform(X)
    svc = pipe.named_steps["clf"]
    sv = svc.support_vectors_
    out = []
    for i in range(0, len(z), 256):
        d2 = ((z[i:i + 256, None, :] - sv[None]) ** 2).sum(-1).min(1)
        out.append(np.exp(-svc._gamma * d2))
    return np.concatenate(out)


def default_proba(pipe):
    svc = pipe.named_steps["clf"]
    return svc.predict_proba(np.full((1, svc.support_vectors_.shape[1]), 1e3))[0]


def scores(y, pred):
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "f1_score": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y, pred, average="weighted", zero_division=0)), 4),
        "precision_macro": round(float(precision_score(y, pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y, pred, average="macro", zero_division=0)), 4),
    }


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    imgs, y3, paths = load_images()
    groups = group_by_phash(imgs)

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    tr, te = next(gss.split(np.zeros(len(y3)), y3, groups=groups))
    print(f"  Train: {len(tr)} | Test: {len(te)} | test classes: {dict(Counter(y3[te].tolist()))}")

    # Validation split (inside train, also by group) to choose C.
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_SEED + 1)
    a, b = next(gss2.split(np.zeros(len(tr)), y3[tr], groups=groups[tr]))
    tr_fit, tr_val = tr[a], tr[b]

    def augmented(idx):
        X, y = [imgs[i] for i in idx], list(y3[idx])
        for i in idx:
            k = AUG_COPIES_MODERATE if y3[i] == 2 else AUG_COPIES
            for _ in range(k):
                X.append(augment(imgs[i], rng))
                y.append(y3[i])
        return X, np.asarray(y)

    print("\nExtracting features (with augmentation)...")
    t0 = time.time()
    Xa, ya = augmented(tr_fit)
    Xa = features(Xa)
    Xval = features([imgs[i] for i in tr_val])
    print(f"  train samples after augmentation: {len(ya)}  ({time.time()-t0:.0f}s)")

    best_c, best_f1 = None, -1
    for c in C_VALUES:
        t0 = time.time()
        m = build(c).fit(Xa, ya)
        f1 = f1_score(y3[tr_val], m.predict(Xval), average="macro", zero_division=0)
        print(f"  C={c:<5} validation macro-F1={f1:.4f}  ({time.time()-t0:.0f}s)")
        if f1 > best_f1:
            best_c, best_f1 = c, f1
    print(f"  -> C={best_c}")

    print("\nTraining final model on the whole train split...")
    Xall, yall = augmented(tr)
    Xall = features(Xall)
    t0 = time.time()
    model = build(best_c).fit(Xall, yall)
    train_time = time.time() - t0
    print(f"  done in {train_time:.0f}s, support vectors: {model.named_steps['clf'].support_vectors_.shape[0]}")

    # ---- Evaluation: clean test images
    Xte = features([imgs[i] for i in te])
    yte = y3[te]
    pr = model.predict_proba(Xte)
    pred = pr.argmax(1)
    clean = scores(yte, pred)
    print("\nTEST (clean):", clean)
    print(classification_report(yte, pred, labels=[0, 1, 2], target_names=CLASSES_3, zero_division=0))
    cm = confusion_matrix(yte, pred, labels=[0, 1, 2])
    print("Confusion matrix:\n", cm)

    # ---- Robustness: perturbed test images (shift / rotate / JPEG)
    test_rng = np.random.default_rng(7)
    Xpert = features([augment(imgs[i], test_rng) for i in te])
    pr_p = model.predict_proba(Xpert)
    robust = scores(yte, pr_p.argmax(1))
    dflt = default_proba(model)
    def_rate_clean = float((np.abs(pr - dflt).sum(1) < 0.05).mean())
    def_rate_pert = float((np.abs(pr_p - dflt).sum(1) < 0.05).mean())
    print("TEST (perturbed):", robust)
    print(f"Default-output rate  clean={def_rate_clean:.1%}  perturbed={def_rate_pert:.1%}")
    print(f"Prediction spread (std of P(Normal)) clean={pr[:, 0].std():.3f}")

    # Similarity threshold for 'not informative' warnings in the app:
    # half of the 1st percentile seen on real test images.
    sim = similarity(model, np.vstack([Xte, Xpert]))
    sim_threshold = float(np.percentile(sim, 1) * 0.5)
    print(f"Similarity p1={np.percentile(sim, 1):.4f}  -> threshold {sim_threshold:.4f}")

    # ---- Save (keep a copy of the old files)
    for p in (MODEL_PATH, METRICS_PATH, CSV_PATH):
        if p.exists():
            shutil.copy2(p, p.with_suffix(p.suffix + ".old"))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model, "model_name": "SVM",
        "classes_3": CLASSES_3, "classes_4": CLASSES_4, "map_4_to_3": MAP_4_TO_3,
        "feature_type": "HOG", "feature_params": HOG_PARAMS, "img_size": IMG_SIZE,
        "preprocess": PREPROCESS_VERSION,
        "similarity_threshold": sim_threshold,
        "split_strategy": "GroupShuffleSplit_by_pHash",
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    metrics = {
        "best_model": "SVM",
        "models": {"SVM": {**clean, "train_time_sec": round(train_time, 1), "C": best_c}},
        "robustness": {"perturbed": robust, "default_output_rate_clean": round(def_rate_clean, 4),
                       "default_output_rate_perturbed": round(def_rate_pert, 4)},
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES_3, "preprocess": PREPROCESS_VERSION,
        "augmentation": {"copies": AUG_COPIES, "copies_moderate": AUG_COPIES_MODERATE},
        "split_strategy": "GroupShuffleSplit_by_pHash", "phash_threshold": PHASH_THRESHOLD,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "y_true", "prob_normal", "prob_mild", "prob_moderate"])
        for i in range(len(te)):
            w.writerow([i, int(yte[i])] + [round(float(v), 6) for v in pr[i]])

    print(f"\nSaved {MODEL_PATH.name}, {METRICS_PATH.name}, {CSV_PATH.name} (old ones kept as *.old)")
    print("Next: python src/train_fusion.py")


if __name__ == "__main__":
    main()