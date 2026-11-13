from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# SVC(probability=True) is deprecated in recent scikit-learn versions;
# it still works, so the warnings are hidden to keep the output readable.
warnings.filterwarnings("ignore", category=FutureWarning)


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 5
TARGET_THRESHOLD = 0.80

FEATURES = [
    "prob_normal_mri",
    "prob_mild_mri",
    "prob_moderate_mri",
    "prob_normal_audio",
    "prob_mild_audio",
    "prob_moderate_audio"
]

CLASS_NAMES = ["Normal", "Mild", "Moderate"]


# ==================================================
# 2. LOAD AND VALIDATE DATA
# ==================================================

print("\nNEUROFUSION AI - FUSION MODEL TRAINING")

mri = pd.read_csv(DATA_DIR / "mri_predictions.csv")
audio = pd.read_csv(DATA_DIR / "audio_predictions.csv")

required_columns = [
    "prob_normal",
    "prob_mild",
    "prob_moderate",
    "y_true"
]

for column in required_columns:
    if column not in mri.columns:
        raise ValueError(f"Missing MRI column: {column}")

    if column not in audio.columns:
        raise ValueError(f"Missing Audio column: {column}")

if mri[required_columns].isnull().any().any():
    raise ValueError("MRI dataset contains missing values.")

if audio[required_columns].isnull().any().any():
    raise ValueError("Audio dataset contains missing values.")

mri_probabilities = mri[
    ["prob_normal", "prob_mild", "prob_moderate"]
].to_numpy()

audio_probabilities = audio[
    ["prob_normal", "prob_mild", "prob_moderate"]
].to_numpy()

for name, probabilities in [
    ("MRI", mri_probabilities),
    ("Audio", audio_probabilities)
]:
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{name} contains non-finite probabilities.")

    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError(f"{name} contains probabilities outside [0, 1].")

    if not np.allclose(
        probabilities.sum(axis=1),
        1,
        atol=0.01
    ):
        raise ValueError(
            f"{name} probabilities do not sum approximately to 1."
        )

mri_labels = mri["y_true"].to_numpy()
audio_labels = audio["y_true"].to_numpy()

if not np.isin(mri_labels, [0, 1, 2]).all():
    raise ValueError("MRI dataset contains invalid labels.")

if not np.isin(audio_labels, [0, 1, 2]).all():
    raise ValueError("Audio dataset contains invalid labels.")


# ==================================================
# 3. CREATE SYNTHETIC FUSION DATASET
# ==================================================

sample_size = min(len(mri), len(audio))

mri_sample = mri.sample(
    n=sample_size,
    replace=False,
    random_state=RANDOM_STATE
).reset_index(drop=True)

audio_sample = audio.sample(
    n=sample_size,
    replace=False,
    random_state=RANDOM_STATE + 1
).reset_index(drop=True)

X = pd.DataFrame({
    "prob_normal_mri": mri_sample["prob_normal"],
    "prob_mild_mri": mri_sample["prob_mild"],
    "prob_moderate_mri": mri_sample["prob_moderate"],
    "prob_normal_audio": audio_sample["prob_normal"],
    "prob_mild_audio": audio_sample["prob_mild"],
    "prob_moderate_audio": audio_sample["prob_moderate"]
})

# Experimental severity rule.
# The audio model is binary (Healthy vs Alzheimer), so its labels are
# collapsed: 0 = Healthy, 1 = Alzheimer. Its Mild/Moderate split is artificial
# and cannot be predicted from the audio probabilities.
# The final label is the maximum severity between MRI and audio.
# This does not represent real patient ground-truth labels.
audio_binary = np.minimum(audio_sample["y_true"].to_numpy(), 1)

y = np.maximum(
    mri_sample["y_true"].to_numpy(),
    audio_binary
)

print(f"\nMRI records: {len(mri)}")
print(f"Audio records: {len(audio)}")
print(f"Synthetic pairs created: {len(X)}")

print("\nFusion label distribution:")
print(pd.Series(y).value_counts().sort_index())


# ==================================================
# 4. SPLIT TRAINING AND TEST DATA
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")


# ==================================================
# 5. CROSS-VALIDATION CONFIGURATION
# ==================================================

cross_validation = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring_metrics = {
    "accuracy": "accuracy",
    "f1_weighted": "f1_weighted",
    "f1_macro": "f1_macro"
}


# ==================================================
# 6. DEFINE THE THREE MACHINE LEARNING MODELS
# ==================================================

models = {

    "Logistic Regression": {
        "estimator": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),
        "parameters": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__max_iter": [1000, 2000, 3000, 5000],
            "classifier__solver": ["lbfgs"]
        }
    },

    "Random Forest": {
        "estimator": RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "parameters": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },

    "Support Vector Machine": {
        "estimator": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),
        "parameters": {
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__gamma": ["scale", "auto"]
        }
    }
}


# ==================================================
# 7. TRAIN, TUNE AND EVALUATE EACH MODEL
# ==================================================

results = []

best_model = None
best_model_name = None
best_cv_score = -1

for model_name, configuration in models.items():

    print("\n" + "=" * 55)
    print(f"TRAINING AND OPTIMIZING: {model_name}")
    print("=" * 55)

    search = GridSearchCV(
        estimator=configuration["estimator"],
        param_grid=configuration["parameters"],
        scoring=scoring_metrics,
        refit="f1_weighted",
        cv=cross_validation,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )

    search.fit(X_train, y_train)

    tuned_model = search.best_estimator_
    best_index = search.best_index_

    cv_accuracy = search.cv_results_[
        "mean_test_accuracy"
    ][best_index]

    cv_weighted_f1 = search.cv_results_[
        "mean_test_f1_weighted"
    ][best_index]

    cv_macro_f1 = search.cv_results_[
        "mean_test_f1_macro"
    ][best_index]

    # Evaluate the tuned model on the held-out test set.
    predictions = tuned_model.predict(X_test)

    test_accuracy = accuracy_score(y_test, predictions)

    test_weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    test_macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    model_confusion_matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2]
    )

    model_report = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    print("\nBest parameters:")
    print(search.best_params_)

    print("\nCross-validation results:")
    print(f"Accuracy: {cv_accuracy:.4f}")
    print(f"Weighted F1-score: {cv_weighted_f1:.4f}")
    print(f"Macro F1-score: {cv_macro_f1:.4f}")

    print("\nTest set results:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Weighted F1-score: {test_weighted_f1:.4f}")
    print(f"Macro F1-score: {test_macro_f1:.4f}")

    print("\nConfusion Matrix:")
    print(model_confusion_matrix)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    results.append({
        "model": model_name,
        "best_parameters": search.best_params_,
        "cross_validation": {
            "accuracy": float(cv_accuracy),
            "f1_weighted": float(cv_weighted_f1),
            "f1_macro": float(cv_macro_f1)
        },
        "test": {
            "accuracy": float(test_accuracy),
            "f1_weighted": float(test_weighted_f1),
            "f1_macro": float(test_macro_f1)
        },
        "confusion_matrix": model_confusion_matrix.tolist(),
        "classification_report": model_report
    })

    # Select the best model using cross-validation only.
    if cv_weighted_f1 > best_cv_score:
        best_cv_score = cv_weighted_f1
        best_model = tuned_model
        best_model_name = model_name


# ==================================================
# 8. SELECT THE BEST MODEL
# ==================================================

selected_result = next(
    result for result in results
    if result["model"] == best_model_name
)

threshold_reached = bool(
    selected_result["cross_validation"]["accuracy"] >= TARGET_THRESHOLD
    and
    selected_result["cross_validation"]["f1_weighted"] >= TARGET_THRESHOLD
)

print("\n" + "=" * 55)
print("FINAL MODEL SELECTION")
print("=" * 55)

print(f"Selected model: {best_model_name}")

print(
    "Cross-validation Accuracy: "
    f"{selected_result['cross_validation']['accuracy']:.4f}"
)

print(
    "Cross-validation Weighted F1-score: "
    f"{selected_result['cross_validation']['f1_weighted']:.4f}"
)

print(
    "Test Accuracy: "
    f"{selected_result['test']['accuracy']:.4f}"
)

print(
    "Test Weighted F1-score: "
    f"{selected_result['test']['f1_weighted']:.4f}"
)

print(
    "Target status: "
    f"{'PASSED' if threshold_reached else 'NOT PASSED'}"
)


# ==================================================
# 9. SAVE THE SELECTED MODEL AND METRICS
# ==================================================

model_path = MODEL_DIR / "fusion_classifier.pkl"
metrics_path = MODEL_DIR / "fusion_metrics.json"

joblib.dump(best_model, model_path)

metrics = {
    "selected_model": best_model_name,
    "selection_metric": "Cross-validation weighted F1-score",
    "target_threshold": TARGET_THRESHOLD,
    "threshold_reached": threshold_reached,
    "selected_model_cross_validation": selected_result["cross_validation"],
    "selected_model_test": selected_result["test"],
    "class_order": CLASS_NAMES,
    "features": FEATURES,
    "results": results,
    "data_limitations": [
        "MRI and audio records were paired independently.",
        "Patient IDs are not shared identifiers.",
        "Labels were generated using a synthetic maximum-severity rule.",
        "Audio Mild and Moderate probabilities were artificially divided.",
        "Audio labels were collapsed to binary (Healthy/Alzheimer) because the audio model is binary.",
        "Results are experimental and do not represent clinical validation."
    ]
}

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(
        metrics,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\nSaved files:")
print(model_path)
print(metrics_path)

if not threshold_reached:
    print(
        "\nWARNING: The selected model did not reach "
        "the 0.80 Accuracy and Weighted F1-score targets "
        "during cross-validation."
    )

print("\nFUSION MODEL TRAINING COMPLETED.")