import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from supabase import create_client

# Person 2's audio prediction function
from src.predict_audio import get_audio_prediction

# Load environment variables for Supabase
load_dotenv(override=True)

app = Flask(__name__)

# --- SUPABASE CONNECTION ---
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase = None
if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)
else:
    print("Warning: Supabase credentials not found. DB saving will be disabled.")


def save_prediction(record):
    """Insert one row in the Supabase 'predictions' table.
    Never breaks the endpoint if the database fails."""
    if supabase is None:
        print("Supabase not configured: prediction was not saved.")
        return
    try:
        supabase.table("predictions").insert(record).execute()
        print("Prediction saved to Supabase.")
    except Exception as db_error:
        print(f"Error saving prediction to Supabase: {db_error}")


BASE_DIR = Path(__file__).resolve().parent
FUSION_MODEL_PATH = BASE_DIR / "models" / "fusion_classifier.pkl"

CLASSES = ["Normal", "Mild", "Moderate"]

# Must match the feature names/order used in src/train_fusion.py
FEATURES = [
    "prob_normal_mri",
    "prob_mild_mri",
    "prob_moderate_mri",
    "prob_normal_audio",
    "prob_mild_audio",
    "prob_moderate_audio",
]

# The audio model is binary (healthy vs AD). data/audio_predictions.csv splits the
# AD probability into Mild/Moderate with a fixed ratio. The endpoint MUST use the
# same ratio, otherwise the fusion model sees different data than it was trained on.
# NOTE: 0.7 was inferred from the CSV (mild = 0.7 * p_ad, moderate = 0.3 * p_ad).
# Confirm it against src/generate_audio_predictions.py.
AUDIO_MILD_SHARE = 0.7


def audio_to_three_classes(prob_ad):
    prob_ad = float(prob_ad)
    return [
        1.0 - prob_ad,                       # Normal
        AUDIO_MILD_SHARE * prob_ad,          # Mild
        (1.0 - AUDIO_MILD_SHARE) * prob_ad,  # Moderate
    ]


# Load fusion model (if it exists yet)
fusion_model = None
if FUSION_MODEL_PATH.exists():
    fusion_model = joblib.load(FUSION_MODEL_PATH)
else:
    print("Warning: fusion_classifier.pkl not found. /predict_fusion will fail until it is trained.")


def combine_probabilities(mri_probabilities, audio_probabilities):
    values = [float(v) for v in list(mri_probabilities) + list(audio_probabilities)]
    if len(values) != 6:
        raise ValueError("Exactly 6 probabilities are required.")
    if not np.isfinite(values).all():
        raise ValueError("Probabilities must be valid numbers.")
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("Probabilities must be between 0 and 1.")
    if not np.isclose(sum(values[:3]), 1.0, atol=0.01):
        raise ValueError("MRI probabilities must sum to 1.")
    if not np.isclose(sum(values[3:]), 1.0, atol=0.01):
        raise ValueError("Audio probabilities must sum to 1.")
    # DataFrame with the same column names used during training
    return pd.DataFrame([values], columns=FEATURES)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "NeuroFusion AI API is running", "status": "OK"})


@app.route("/predict_fusion", methods=["POST"])
def predict_fusion():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "A valid JSON is required."}), 400

        mri = data.get("mri_probabilities")
        audio = data.get("audio_probabilities")

        if not isinstance(mri, list) or not isinstance(audio, list):
            return jsonify({"error": "You must provide lists of MRI and audio probabilities."}), 400
        if len(mri) != 3 or len(audio) != 3:
            return jsonify({"error": "Each modality must contain exactly 3 probabilities."}), 400

        if fusion_model is None:
            return jsonify({"error": "The fusion model is not trained yet."}), 503

        X = combine_probabilities(mri, audio)

        # The model was trained with labels 0/1/2 -> Normal/Mild/Moderate
        prediction_index = int(fusion_model.predict(X)[0])
        prediction_name = CLASSES[prediction_index]
        probabilities = fusion_model.predict_proba(X)[0]

        result = {
            "prediction": prediction_name,
            "probabilities": {
                CLASSES[int(label)]: round(float(probability), 4)
                for label, probability in zip(fusion_model.classes_, probabilities)
            },
            "notice": "Educational prototype. Not a medical diagnosis."
        }

        # --- SAVE TO SUPABASE (final fusion prediction) ---
        save_prediction({
            "user_id": 1,  # Hardcoded for now
            "patient_id": None,
            "mri_image_path": "N/A",
            "audio_path": "N/A",
            "mri_probability": float(max(mri)),
            "audio_probability": float(max(audio)),
            "final_result": prediction_name,
        })

        return jsonify(result), 200

    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/predict_mri", methods=["POST"])
def predict_mri():
    if "file" not in request.files:
        return jsonify({"error": "No file received. Use the 'file' field."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    # Imported here so the API still starts if the MRI model has a problem
    from src.predict_mri import predict_mri as run_inference

    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    try:
        file.save(tmp.name)
        result = run_inference(tmp.name)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    save_prediction({
        "user_id": 1,
        "patient_id": None,
        "mri_image_path": file.filename,
        "audio_path": "N/A",
        "mri_probability": float(result["confidence"]),
        "audio_probability": 0.0,
        "final_result": result["class_name"],
    })

    return jsonify({
        "message": "MRI image analyzed successfully.",
        "filename": file.filename,
        "class_name": result["class_name"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities_list"],  # [Normal, Mild, Moderate]
    }), 200


@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file received. Use the 'file' field."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    allowed_extensions = {".wav", ".mp3", ".flac", ".ogg"}
    extension = Path(file.filename).suffix.lower()
    if extension not in allowed_extensions:
        return jsonify({"error": "Format not allowed. Use WAV, MP3, FLAC or OGG."}), 400

    # Save file temporarily (safe unique name) so the audio model can read it
    tmp = tempfile.NamedTemporaryFile(suffix=extension, delete=False)
    tmp.close()
    temp_path = Path(tmp.name)
    file.save(temp_path)

    try:
        # --- CALL PERSON 2'S MODEL ---
        result = get_audio_prediction(str(temp_path))

        if result["status"] == "error":
            return jsonify({"error": result["message"]}), 500

        prob_ad = result["probability_ad"]

        # 3 probabilities [Normal, Mild, Moderate], same rule as the training CSV
        audio_probs = [round(p, 6) for p in audio_to_three_classes(prob_ad)]

        save_prediction({
            "user_id": 1,
            "patient_id": None,
            "mri_image_path": "N/A",
            "audio_path": file.filename,
            "mri_probability": 0.0,
            "audio_probability": float(prob_ad),
            "final_result": result["prediction"],
        })

        return jsonify({
            "message": "Audio processed successfully.",
            "filename": file.filename,
            "probabilities": audio_probs  # [Normal, Mild, Moderate]
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Delete temporary file
        try:
            temp_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)