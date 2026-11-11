import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

CSV_PATH = "dataset/processed/audio_features.csv"
MODELS_DIR = "models"
SEED = 42
CLASS_NAMES = ['Healthy', 'Alzheimer']


def make_pipe(clf):
    """Scaler + model in a Pipeline: the scaler is fitted only on the training data
    of each fold, which avoids data leakage during cross-validation."""
    return Pipeline([('scaler', StandardScaler()), ('clf', clf)])


def train_audio_models():
    print("--- Starting Audio Model Training & Optimization ---")

    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Please run audio_features.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    X = df.drop(columns=['filename', 'label'])
    y = df['label']

    # 1. Split: the test set is used ONLY ONCE, for the final report
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    print(f"Training set: {len(X_train)} samples | Testing set: {len(X_test)} samples")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # 2. Tune the SVM (inside a Pipeline, so the scaler does not leak into the folds)
    print("\n--- Tuning Support Vector Machine (SVM) ---")
    param_grid = {
        'clf__C': [0.1, 1, 10, 100],
        'clf__gamma': ['scale', 0.1, 0.01, 0.001],
        'clf__kernel': ['rbf', 'linear'],
    }
    svm_pipe = make_pipe(SVC(probability=True, class_weight='balanced', random_state=SEED))
    grid_search = GridSearchCV(svm_pipe, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best SVM parameters found: {grid_search.best_params_}")

    best_svm_params = {k.replace('clf__', ''): v for k, v in grid_search.best_params_.items()}

    # 3. Define the models (including the ensemble)
    def new_rf():
        return RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED)

    def new_svm():
        return SVC(probability=True, class_weight='balanced', random_state=SEED, **best_svm_params)

    def new_lr():
        return LogisticRegression(class_weight='balanced', random_state=SEED, max_iter=1000)

    models = {
        "Random Forest": new_rf(),
        "Support Vector Machine (Optimized)": new_svm(),
        "Logistic Regression": new_lr(),
        "Voting Classifier (Ensemble)": VotingClassifier(
            estimators=[('rf', new_rf()), ('svm', new_svm()), ('lr', new_lr())],
            voting='soft'
        ),
    }

    # 4. Select the best model with cross-validation ON THE TRAINING SET (not the test set)
    print("\n--- Cross-validation on training set (model selection) ---")
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(make_pipe(model), X_train, y_train, cv=cv, scoring='f1_weighted')
        cv_results[name] = (scores.mean(), scores.std())
        print(f"{name}: F1 = {scores.mean():.3f} ± {scores.std():.3f}")

    best_model_name = max(cv_results, key=lambda n: cv_results[n][0])

    # 5. Train all models and evaluate them on the test set (report only, does not pick the winner)
    metrics_list = []
    fitted = {}
    for name, model in models.items():
        print(f"\n{'='*40}\nEvaluating on test set: {name}\n{'='*40}")
        pipe = make_pipe(model)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe

        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Alzheimer (1)']))

        metrics_list.append({
            "model_name": name,
            "cv_f1_mean": round(cv_results[name][0], 4),
            "cv_f1_std": round(cv_results[name][1], 4),
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
        })

    # 6. Save the winner (scaler and classifier saved separately, same format as before)
    best_pipe = fitted[best_model_name]
    best_f1_test = next(m["f1_score"] for m in metrics_list if m["model_name"] == best_model_name)

    print("\n" + "=" * 50)
    print(f"🏆 BEST MODEL (by CV): {best_model_name}")
    print(f"   CV F1: {cv_results[best_model_name][0]:.4f} | Test F1: {best_f1_test:.4f}")
    print("=" * 50)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_pipe.named_steps['clf'], f"{MODELS_DIR}/audio_classifier.pkl")
    joblib.dump(best_pipe.named_steps['scaler'], f"{MODELS_DIR}/audio_scaler.pkl")

    with open(f"{MODELS_DIR}/audio_metrics.json", "w") as f:
        json.dump(metrics_list, f, indent=4)

    # 7. Confusion matrix for the winning model
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_test, best_pipe.predict(X_test))

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{MODELS_DIR}/audio_confusion_matrix.png")
    plt.close()

    print(f"\nSaved '{best_model_name}' to {MODELS_DIR}/audio_classifier.pkl")
    print(f"Saved scaler to {MODELS_DIR}/audio_scaler.pkl")
    print(f"Saved metrics to {MODELS_DIR}/audio_metrics.json")
    print(f"Saved Confusion Matrix to {MODELS_DIR}/audio_confusion_matrix.png")
    print("Process complete!")


if __name__ == "__main__":
    train_audio_models()