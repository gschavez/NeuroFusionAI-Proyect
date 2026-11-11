
from pathlib import Path
from collections import Counter
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset"

CLASSES = {
    "1": "Mild Demented",
    "2": "Moderate Demented",
    "3": "Non Demented",
    "4": "Very Mild Demented"
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

EXPECTED_SIZE = (128, 128)


def audit_dataset():
    print("=" * 55)
    print("MRI DATASET AUDIT")
    print("=" * 55)

    total_images = 0
    total_corrupted = 0
    total_invalid_dimensions = 0
    total_missing_folders = 0

    for folder, class_name in CLASSES.items():

        folder_path = DATASET_PATH / folder

        if not folder_path.exists():
            print(f"\n[ERROR] Missing folder: {folder_path}")
            total_missing_folders += 1
            continue

        image_paths = [
            path for path in folder_path.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        class_corrupted = 0
        class_invalid_dimensions = 0
        valid_images = 0

        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    image.verify()

                with Image.open(image_path) as image:
                    if image.size != EXPECTED_SIZE:
                        class_invalid_dimensions += 1
                    else:
                        valid_images += 1

            except (UnidentifiedImageError, OSError):
                class_corrupted += 1

        total_images += len(image_paths)
        total_corrupted += class_corrupted
        total_invalid_dimensions += class_invalid_dimensions

        print(f"\nClass: {class_name}")
        print(f"Total images: {len(image_paths)}")
        print(f"Valid images: {valid_images}")
        print(f"Corrupted images: {class_corrupted}")
        print(f"Invalid dimensions: {class_invalid_dimensions}")

    print("\n" + "=" * 55)
    print("AUDIT SUMMARY")
    print("=" * 55)

    print(f"Total images: {total_images}")
    print(f"Corrupted images: {total_corrupted}")
    print(f"Invalid dimensions: {total_invalid_dimensions}")
    print(f"Missing folders: {total_missing_folders}")

    print("=" * 55)

    if (
        total_corrupted == 0
        and total_invalid_dimensions == 0
        and total_missing_folders == 0
    ):
        print("Dataset integrity check passed.")
    else:
        print("Dataset requires further inspection.")


if __name__ == "__main__":
    audit_dataset()
