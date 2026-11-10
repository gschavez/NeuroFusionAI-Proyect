import os
from pathlib import Path
import joblib
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from supabase import create_client, Client

# Import Person 2's audio prediction function
from src.predict_audio import get_audio_prediction

# Load environment variables for Supabase
load_dotenv(override=True)

app = Flask(__name__)

# --- SUPABASE CONNECTION ---
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
if supabase_url and supabase_key:
    supabase: Client = create_client(supabase_url, supabase_key)
else:
    print("Warning: Supabase credentials not found. DB saving will be disabled.")

BASE_DIR = Path(__file__).resolve().parent
FUSION_MODEL_PATH = BASE_DIR / "models" / "fusion_classifier.pkl"

CLASSES = ["Normal", "Mild", "Moderate"]

# Load fusion model (if it exists yet)
fusion_model = None
if FUSION_MODEL_PATH.exists():
    fusion_model = joblib.load(FUSION_MODEL_PATH)
else:
    print("Warning: fusion_classifier.pkl not found. /predict_fusion will fail until it is trained.")

def combine_probabilities(mri_probabilities, audio_probabilities):
    values = list(mri_probabilities) + list(audio_probabilities)
    if len(values) != 6:
        raise ValueError("Exactly 6 probabilities are required.")
    if not np.isfinite(values).all():
        raise ValueError("Probabilities must be valid numbers.")
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("Probabilities must be between 0 and 1.")
    if not np.isclose(sum(mri_probabilities), 1.0, atol=0.01):
        raise ValueError("MRI probabilities must sum to 1.")
    if not np.isclose(sum(audio_probabilities), 1.0, atol=0.01):
        raise ValueError("Audio probabilities must sum to 1.")
    return np.array([values])

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
        prediction = fusion_model.predict(X)[0]
        probabilities = fusion_model.predict_proba(X)[0]

        result = {
            "prediction": str(prediction),
            "probabilities": {
                str(label): round(float(probability), 4)
                for label, probability in zip(fusion_model.classes_, probabilities)
            },
            "notice": "Educational prototype. Not a medical diagnosis."
        }

        # --- SAVE TO SUPABASE ---
        try:
            supabase.table("predictions").insert({
                "user_id": 1, # Hardcoded for now
                "patient_id": None,
                "mri_image_path": "N/A",
                "audio_path": "N/A",
                "mri_probability": float(max(mri)),
                "audio_probability": float(max(audio)),
                "final_result": str(prediction)
            }).execute()
            print("Fusion prediction saved to Supabase.")
        except Exception as db_error:
            print(f"Error saving fusion prediction to Supabase: {db_error}")

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
    
    allowed_extensions = {".png", ".jpg", ".jpeg", ".dcm"}
    extension = Path(file.filename).suffix.lower()
    if extension not in allowed_extensions:
        return jsonify({"error": "Format not allowed. Use PNG, JPG, JPEG or DCM."}), 400

    # TODO: Person 1 will integrate their model here later
    return jsonify({
        "message": "MRI image received successfully.",
        "filename": file.filename,
        "probabilities": [0.33, 0.33, 0.34] # Fake probs for frontend testing
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

    # Save file temporarily so the audio model can read it
    temp_path = BASE_DIR / f"temp_{file.filename}"
    file.save(temp_path)

    try:
        # --- CALL PERSON 2'S MODEL ---
        result = get_audio_prediction(str(temp_path))
        
        if result["status"] == "error":
            return jsonify({"error": result["message"]}), 500
            
        prob_ad = result["probability_ad"]
        
        # Adapt binary probability to 3 classes for the Fusion model
        audio_probs = [1.0 - prob_ad, prob_ad, 0.0] 

        # --- SAVE TO SUPABASE ---
        try:
            supabase.table("predictions").insert({
                "user_id": 1, 
                "patient_id": None,
                "mri_image_path": "N/A", 
                "audio_path": file.filename,
                "mri_probability": 0.0,
                "audio_probability": float(prob_ad),
                "final_result": result["prediction"]
            }).execute()
            print("Audio prediction saved to Supabase.")
        except Exception as db_error:
            print(f"Error saving audio prediction to Supabase: {db_error}")

        return jsonify({
            "message": "Audio processed successfully.",
            "filename": file.filename,
            "probabilities": audio_probs
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Delete temporary file
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)