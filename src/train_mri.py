
from pathlib import Path
from collections import Counter
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report
)


# =====================================================
# 1. CONFIGURACIÓN
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

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")

torch.manual_seed(42)


# =====================================================
# 2. CARGA DE RUTAS Y ETIQUETAS
# =====================================================

image_paths = []
labels = []

for folder, class_name in CLASSES.items():

    folder_path = DATASET_PATH / str(folder)

    for image_path in folder_path.rglob("*.jpg"):
        image_paths.append(str(image_path))
        labels.append(folder - 1)

print(f"Total de imágenes: {len(image_paths)}")


# =====================================================
# 3. DIVISIÓN 70/30
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    random_state=42,
    stratify=labels
)

print(f"Entrenamiento: {len(X_train)}")
print(f"Prueba: {len(X_test)}")


# =====================================================
# 4. PREPARACIÓN DE LAS IMÁGENES
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

            image_tensor = torch.frombuffer(
                bytearray(image.tobytes()),
                dtype=torch.uint8
            )

            image_tensor = image_tensor.reshape(1, 128, 128)
            image_tensor = image_tensor.float() / 255.0

        return image_tensor, label


train_dataset = MRIDataset(X_train, y_train)
test_dataset = MRIDataset(X_test, y_test)


# =====================================================
# 5. BALANCEO DE CLASES
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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =====================================================
# 6. CONSTRUCCIÓN DE LA CNN
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
# 7. ENTRENAMIENTO
# =====================================================

print("\nINICIANDO ENTRENAMIENTO")

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

    accuracy = 100 * correct / total
    average_loss = total_loss / len(train_loader)

    print(
        f"Época [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )


# =====================================================
# 8. EVALUACIÓN
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

        all_predictions.extend(predictions.tolist())
        all_targets.extend(targets.tolist())
        all_probabilities.extend(probabilities.tolist())


# F1-score macro
f1 = f1_score(
    all_targets,
    all_predictions,
    average="macro"
)

# Matriz de confusión
matrix = confusion_matrix(
    all_targets,
    all_predictions,
    labels=[0, 1, 2, 3]
)

print("\nRESULTADOS DE EVALUACIÓN")

print(f"F1-score macro: {f1:.4f}")

print("\nOrden de clases:")
for index, name in enumerate(CLASSES.values()):
    print(f"{index}: {name}")

print("\nMatriz de confusión:")
print(matrix)

print("\nRESULTADOS DE EVALUACIÓN")

print(f"F1-score macro: {f1:.4f}")

print("\nOrden de clases:")
for index, name in enumerate(CLASSES.values()):
    print(f"{index}: {name}")

print("\nMatriz de confusión:")
print(matrix)

print("\nREPORTE POR CLASE")
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
# 9. GUARDAR MODELO
# =====================================================

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": CLASSES,
        "input_size": (128, 128)
    },
    MODEL_PATH
)

print(f"\nModelo guardado en: {MODEL_PATH}")


# =====================================================
# 10. GENERAR CSV DE PROBABILIDADES
# =====================================================

with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    writer.writerow([
        "image",
        "true_label",
        "predicted_label",
        *[f"prob_{name.replace(' ', '_')}" for name in CLASSES.values()]
    ])

    for i in range(len(X_test)):

        true_label = all_targets[i] + 1
        predicted_label = all_predictions[i] + 1

        writer.writerow([
            X_test[i],
            true_label,
            predicted_label,
            *[round(float(p), 6) for p in all_probabilities[i]]
        ])

print(f"CSV generado en: {CSV_PATH}")

print("\nPROCESO FINALIZADO")
