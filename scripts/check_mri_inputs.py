"""Check how the MRI model sees a folder of images.

Usage (from the project root):
    python scripts/check_mri_inputs.py dataset/3            # e.g. Non Demented training folder
    python scripts/check_mri_inputs.py C:\\Users\\me\\Downloads\\test_images

For every image it prints the predicted probabilities, the similarity to the
training data and whether it would be flagged. Dataset images should give varied,
confident probabilities and high similarity; images flagged as "UNCERTAIN" are the
ones that produce the constant ~75 % Normal / 25 % Mild output.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from src.predict_mri import predict_mri, _load

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset")
files = [p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50
sims, flagged = [], 0
print(f"{'file':40s} {'Normal':>7s} {'Mild':>7s} {'Moder.':>7s} {'similar.':>9s}  status")
for p in files[:limit]:
    r = predict_mri(p)
    q = r["quality"]
    pr = r["probabilities_list"]
    sims.append(q["similarity"])
    flagged += q["uncertain_input"]
    print(f"{p.name[:40]:40s} {pr[0]:7.2%} {pr[1]:7.2%} {pr[2]:7.2%} {q['similarity']:9.2e}  "
          f"{'UNCERTAIN' if q['uncertain_input'] else 'ok'}")
if sims:
    threshold = _load()["_min_similarity"]
    print(f"\n{len(sims)} images · flagged {flagged} · similarity percentiles "
          f"p5={np.percentile(sims, 5):.2e} p50={np.median(sims):.2e} · current threshold {threshold:.4g}")