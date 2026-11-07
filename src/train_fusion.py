
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix


# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

MRI_FILE = DATA_DIR / "mri_predictions.csv"
AUDIO_FILE = DATA_DIR / "audio_predictions.csv"

CLASSES = ["Normal", "Leve", "Moderado"]

PROBABILITY_COLUMNS = [
    "prob_Normal",
    "prob_Leve",
    "prob_Moderado"
]


def load_predictions():
    """Carga y combina las predicciones de MRI y audio."""

    mri = pd.read_csv(MRI_FILE)
    audio = pd.read_csv(AUDIO_FILE)

    required_columns = (
        ["patient_id", "y_true"] + PROBABILITY_COLUMNS
    )

    for name, dataframe in [("MRI", mri), ("Audio", audio)]:
        missing = set(required_columns) - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"Faltan columnas en el archivo {name}: {missing}"
            )

        if dataframe["patient_id"].duplicated().any():
            raise ValueError(
                f"Hay pacientes duplicados en el archivo {name}."
            )

    # Combinar por identificador de paciente
    data = pd.merge(
        mri,
        audio,
        on="patient_id",
        suffixes=("_mri", "_audio"),
        validate="one_to_one"
    )

    if len(data) != len(mri) or len(data) != len(audio):
        raise ValueError(
            "Los archivos MRI y audio no contienen los mismos pacientes."
        )

    if not (
        data["y_true_mri"] == data["y_true_audio"]
    ).all():
        raise ValueError(
            "Las etiquetas reales no coinciden entre MRI y audio."
        )

    # Variables de entrada: probabilidades de ambos modelos
    feature_columns = (
        [f"{col}_mri" for col in PROBABILITY_COLUMNS]
        + [f"{col}_audio" for col in PROBABILITY_COLUMNS]
    )

    X = data[feature_columns]
    y = data["y_true_mri"]

    if not set(y.unique()).issubset(CLASSES):
        raise ValueError("Se encontraron etiquetas no reconocidas.")

    if X.isnull().any().any():
        raise ValueError("Hay valores vacíos en las probabilidades.")

    if not X.apply(lambda col: col.between(0, 1).all()).all():
        raise ValueError("Las probabilidades deben estar entre 0 y 1.")

    return X, y


def train_fusion():
    """Entrena y evalúa el clasificador de fusión."""

    print("Cargando predicciones de MRI y audio...")

    X, y = load_predictions()

    smallest_class = y.value_counts().min()

    if smallest_class < 2:
        raise ValueError(
            "Se necesitan al menos 2 pacientes por clase para K-Fold."
        )

    folds = min(5, smallest_class)

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42
    )

    model = LogisticRegression(max_iter=1000)

    print(f"Entrenando con validación cruzada de {folds} particiones...")

    # Predicciones de validación cruzada del meta-clasificador
    y_pred = cross_val_predict(model, X, y, cv=cv)

    print("\nReporte de clasificación:")
    print(classification_report(y, y_pred, zero_division=0))

    print("\nMatriz de confusión:")
    print(confusion_matrix(y, y_pred, labels=CLASSES))

    # Entrenar el modelo final con todos los datos disponibles
    model.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "fusion_classifier.pkl"
    joblib.dump(model, model_path)

    print(f"\nModelo de fusión guardado en: {model_path}")


if __name__ == "__main__":
    train_fusion()
