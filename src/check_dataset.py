
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset"

CLASSES = {
    "1": "Mild Demented",
    "2": "Moderate Demented",
    "3": "Non Demented",
    "4": "Very Mild Demented"
}

print("VERIFICACIÓN DEL DATASET")
print("-" * 40)

total_images = 0

for folder, class_name in CLASSES.items():
    folder_path = DATASET_PATH / folder

    images = list(folder_path.rglob("*.jpg"))
    count = len(images)

    total_images += count

    print(f"{class_name}: {count} imágenes")

print("-" * 40)
print(f"TOTAL DE IMÁGENES: {total_images}")
