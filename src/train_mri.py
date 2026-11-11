
from pathlib import Path
from collections import Counter
import csv
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score
)


# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "models" / "mri_classifier.pkl"
CSV_PATH = BASE_DIR / "models" / "mri_predictions.csv"

CLASSES = {
    1: "Mild Demented",
    2: "Moderate Demented",
    3: "Non Demented",
    4: "Very Mild Demented"
}

INPUT_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 6
RANDOM_SEED = 42

DEVICE = torch.device("cpu")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# =====================================================
# 2. LOAD IMAGE PATHS AND LABELS
# =====================================================

image_paths = []
labels = []

for folder, class_name in CLASSES.items():

    folder_path = DATASET_PATH / str(folder)

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder_path}"
        )

    for image_path in folder_path.rglob("*"):

        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            image_paths.append(str(image_path))
            labels.append(folder - 1)


if not image_paths:
    raise RuntimeError("No images were found in the dataset.")

print("=" * 55)
print("MRI CLASSIFIER TRAINING")
print("=" * 55)

print(f"Total images: {len(image_paths)}")

print("\nOriginal class distribution:")

for index, class_name in enumerate(CLASSES.values()):
    print(f"{class_name}: {labels.count(index)}")


# =====================================================
# 3. STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# =====================================================

X_train_val, X_test, y_train_val, y_test = train_test_split(
    image_paths,
    labels,
    test_size=0.15,
    random_state=RANDOM_SEED,
    stratify=labels
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.15 / 0.85,
    random_state=RANDOM_SEED,
    stratify=y_train_val
)

print("\nDataset split:")
print(f"Training: {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Testing: {len(X_test)}")


# =====================================================
# 4. MRI DATASET
# =====================================================

class MRIDataset(Dataset):

    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        image_path = self.paths[index]
        label = self.labels[index]

        with Image.open(image_path) as image:

            image = image.convert("L")
            image = image.resize(INPUT_SIZE)

            image_array = np.array(image, dtype=np.float32) / 255.0

            image_tensor = torch.from_numpy(
                image_array
            ).unsqueeze(0)

        return image_tensor, label


train_dataset = MRIDataset(X_train, y_train)
val_dataset = MRIDataset(X_val, y_val)
test_dataset = MRIDataset(X_test, y_test)


# =====================================================
# 5. BALANCED TRAINING SAMPLER
# =====================================================

class_counts = Counter(y_train)

sample_weights = [
    1.0 / class_counts[label]
    for label in y_train
]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =====================================================
# 6. CNN ARCHITECTURE
# =====================================================

class MRIClassifier(nn.Module):

    def __init__(self):
        super(MRIClassifier, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((2, 2))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * 2 * 2, 4)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = MRIClassifier().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =====================================================
# 7. VALIDATION FUNCTION
# =====================================================

def evaluate_model(model, data_loader):

    model.eval()

    all_predictions = []
    all_targets = []

    total_loss = 0.0

    with torch.no_grad():

        for images, targets in data_loader:

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, targets)

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item()

            all_predictions.extend(predictions.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    average_loss = total_loss / len(data_loader)

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    return average_loss, accuracy, macro_f1


# =====================================================
# 8. TRAINING WITH EARLY STOPPING
# =====================================================

print("\nStarting training...")

best_val_f1 = -1.0
epochs_without_improvement = 0

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in train_loader:

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    train_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    val_loss, val_accuracy, val_f1 = evaluate_model(
        model,
        val_loader
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy * 100:.2f}% | "
        f"Val Macro F1: {val_f1:.4f}"
    )

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": CLASSES,
                "input_size": INPUT_SIZE,
                "best_val_f1": best_val_f1
            },
            MODEL_PATH
        )

        print("Best model checkpoint saved.")

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# =====================================================
# 9. LOAD BEST MODEL
# =====================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])

print("\nBest model loaded.")
print(f"Best validation Macro F1: {best_val_f1:.4f}")


# =====================================================
# 10. FINAL TEST EVALUATION
# =====================================================

model.eval()

all_predictions = []
all_targets = []
all_probabilities = []

with torch.no_grad():

    for images, targets in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_targets.extend(targets.tolist())
        all_probabilities.extend(probabilities.cpu().tolist())


accuracy = accuracy_score(all_targets, all_predictions)

macro_precision = precision_score(
    all_targets,
    all_predictions,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    all_targets,
    all_predictions,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    all_targets,
    all_predictions,
    average="macro",
    zero_division=0
)

matrix = confusion_matrix(
    all_targets,
    all_predictions,
    labels=[0, 1, 2, 3]
)

print("\n" + "=" * 55)
print("FINAL TEST RESULTS")
print("=" * 55)

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")

print("\nClass order:")

for index, name in enumerate(CLASSES.values()):
    print(f"{index}: {name}")

print("\nConfusion Matrix:")
print(matrix)

print("\nClassification Report:")

print(
    classification_report(
        all_targets,
        all_predictions,
        labels=[0, 1, 2, 3],
        target_names=list(CLASSES.values()),
        zero_division=0
    )
)


# =====================================================
# 11. EXPORT TEST PROBABILITIES
# =====================================================

with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    writer.writerow([
        "image",
        "true_label",
        "predicted_label",
        *[
            f"prob_{name.replace(' ', '_')}"
            for name in CLASSES.values()
        ]
    ])

    for index in range(len(X_test)):

        true_label = all_targets[index] + 1
        predicted_label = all_predictions[index] + 1

        writer.writerow([
            X_test[index],
            true_label,
            predicted_label,
            *[
                round(float(probability), 6)
                for probability in all_probabilities[index]
            ]
        ])


print(f"\nModel saved to: {MODEL_PATH}")
print(f"Predictions CSV saved to: {CSV_PATH}")

print("\nMRI training process completed.")
