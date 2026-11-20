"""
Shared MRI preprocessing + HOG features.

Used by BOTH src/train_mri_robust.py (training) and src/predict_mri.py
(inference), so the model always sees images prepared the same way.

Why this exists: the first model computed HOG on the raw 128x128 image and then
applied a StandardScaler. In the training images the black background is
exactly 0, so ~170 HOG features had almost zero variance. Any tiny change in an
uploaded image (a 1-pixel shift, JPEG noise in the background, a re-saved file)
made those features explode (z-scores of 300+), the image looked "unknown" to
the RBF SVM and it always answered its default ~75 % Normal / 25 % Mild.

The fix:
  1. Clean the background to exactly 0 (removes JPEG / scanner noise).
  2. Crop to the brain and center it on a square canvas (removes shifts and
     different margins / image sizes).
  3. Normalize contrast inside the brain (removes brightness differences).
  4. HOG on the result. No StandardScaler afterwards (HOG is already normalized).
"""

import cv2
import numpy as np
from skimage.feature import hog

IMG_SIZE = (128, 128)
BG_THRESHOLD = 20          # pixels darker than this (after blur) are background
MARGIN = 4                 # pixels of black border kept around the brain
PREPROCESS_VERSION = "brain_crop_v1"

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)


def to_gray(img):
    if img is None:
        raise ValueError("Empty image")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.uint8)


def preprocess(img):
    """Gray image (any size) -> clean, centered, contrast-normalized 128x128 uint8."""
    g = to_gray(img)
    blur = cv2.GaussianBlur(g, (5, 5), 0)
    mask = (blur > BG_THRESHOLD).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Keep the largest connected region (the brain); drops text, rulers, specks.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[biggest, cv2.CC_STAT_AREA] >= 0.02 * mask.size:
            mask = (labels == biggest).astype(np.uint8)
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))

    ys, xs = np.nonzero(mask)
    if len(xs) < 0.02 * mask.size:          # nothing that looks like a brain: keep whole image
        crop, cmask = g, np.ones_like(g)
    else:
        y0, y1 = max(ys.min() - MARGIN, 0), min(ys.max() + MARGIN + 1, g.shape[0])
        x0, x1 = max(xs.min() - MARGIN, 0), min(xs.max() + MARGIN + 1, g.shape[1])
        crop, cmask = g[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    crop = np.where(cmask > 0, crop, 0).astype(np.uint8)   # background -> exactly 0

    # Pad to a square so the aspect ratio is kept, then resize.
    h, w = crop.shape
    side = max(h, w)
    canvas = np.zeros((side, side), np.uint8)
    canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
    out = cv2.resize(canvas, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # Contrast: map the 1st..99th percentile of brain pixels to 0..255.
    brain = out[out > BG_THRESHOLD]
    if brain.size > 50:
        lo, hi = np.percentile(brain, [1, 99])
        if hi - lo > 5:
            scaled = np.clip((out.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
            out = np.where(out > BG_THRESHOLD, scaled, 0).astype(np.uint8)
    return out


def extract_features(img):
    """Image array -> HOG vector (1764,) as float32."""
    return hog(preprocess(img), **HOG_PARAMS).astype(np.float32)


def augment(img, rng):
    """Random realistic change of a raw image (simulates other uploads / scanners)."""
    g = to_gray(img)
    h, w = g.shape
    angle = rng.uniform(-10, 10)
    scale = rng.uniform(0.9, 1.1)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[:, 2] += rng.uniform(-6, 6, size=2)
    g = cv2.warpAffine(g, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    gamma = rng.uniform(0.8, 1.25)
    g = np.clip(255.0 * (g / 255.0) ** gamma * rng.uniform(0.85, 1.15), 0, 255).astype(np.uint8)
    if rng.random() < 0.5:   # JPEG re-encode, like a downloaded / shared image
        q = int(rng.integers(55, 95))
        ok, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            g = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return g