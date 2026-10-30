from pathlib import Path
from PIL import Image
import random

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset"

CLASSES = {
    "1": "Mild Demented",
    "2": "Moderate Demented",
    "3": "Non Demented",
    "4": "Very Mild Demented"
}

for folder, class_name in CLASSES.items():
    images = list((DATASET_PATH / folder).rglob("*.jpg"))
    samples = random.sample(images, min(5, len(images)))

    print(f"\n{class_name}")

    for path in samples:
        with Image.open(path) as image:
            print(
                f"Archivo: {path.name} | "
                f"Tamaño: {image.size} | "
                f"Modo: {image.mode}"
            )
            