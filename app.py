import os
import gc
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables for Supabase
load_dotenv(override=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-key-super-secret-123")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # max 25 MB per upload

# --- SUPABASE CONNECTION ---
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase = None
if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)
else:
    print("Warning: Supabase credentials not found. DB saving will be disabled.")


def save_prediction(record):
    """Insert one row in the Supabase 'predictions' table."""
    if supabase is None:
        print("Supabase not configured: prediction was not saved.")
        return
    try:
        record["user_id"] = session.get("user_id", 1)
        supabase.table("predictions").insert(record).execute()
        print("Prediction saved to Supabase.")
    except Exception as db_error:
        print(f"Error saving prediction to Supabase: {db_error}")


BASE_DIR = Path(__file__).resolve().parent
FUSION_MODEL_PATH = BASE_DIR / "models" / "fusion_classifier.pkl"
CLASSES = ["Normal", "Mild", "Moderate"]

FEATURES = [
    "prob_normal_mri", "prob_mild_mri", "prob_moderate_mri",
    "prob_normal_audio", "prob_mild_audio", "prob_moderate_audio",
]

fusion_model = None
if FUSION_MODEL_PATH.exists():
    fusion_model = joblib.load(FUSION_MODEL_PATH)
else:
    print("Warning: fusion_classifier.pkl not found.")


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
    return pd.DataFrame([values], columns=FEATURES)


# ==========================================
# API ROUTES
# ==========================================

@app.route("/predict_fusion", methods=["POST"])
def predict_fusion():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "A valid JSON is required."}), 400

        mri = data.get("mri_probabilities")
        audio = data.get("audio_probabilities")
        
        # Get filenames from the frontend request
        mri_filename = data.get("mri_filename", "N/A")   
        audio_filename = data.get("audio_filename", "N/A") 

        if not isinstance(mri, list) or not isinstance(audio, list):
            return jsonify({"error": "You must provide lists of MRI and audio probabilities."}), 400
        if len(mri) != 3 or len(audio) != 3:
            return jsonify({"error": "Each modality must contain exactly 3 probabilities."}), 400

        if fusion_model is None:
            return jsonify({"error": "The fusion model is not trained yet."}), 503

        X = combine_probabilities(mri, audio)
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

        # SAVE EVERYTHING IN A SINGLE ROW
        save_prediction({
            "patient_id": None,
            "mri_image_path": mri_filename,
            "audio_path": audio_filename,
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
        gc.collect()  # <--- AÑADIDO: Limpieza de memoria (MRI)

    # Note: DB saving removed from here to avoid duplicate entries
    return jsonify({
        "message": "MRI image analyzed successfully.",
        "filename": file.filename,
        "class_name": result["class_name"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities_list"],
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

    # Lazy import: librosa/numba are heavy, so they are loaded only when audio is used
    from src.predict_audio import get_audio_prediction

    tmp = tempfile.NamedTemporaryFile(suffix=extension, delete=False)
    tmp.close()
    temp_path = Path(tmp.name)
    file.save(temp_path)

    try:
        result = get_audio_prediction(str(temp_path))

        if result["status"] == "error":
            return jsonify({"error": result["message"]}), 500

        audio_probs = result["probabilities"]

        # Note: DB saving removed from here to avoid duplicate entries
        return jsonify({
            "message": "Audio processed successfully.",
            "filename": file.filename,
            "probabilities": [round(p, 6) for p in audio_probs]
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass
        gc.collect()  # <--- AÑADIDO: Limpieza de memoria (Audio/Librosa)


# ==========================================
# FRONTEND ROUTES (HTML TEMPLATES)
# ==========================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/models", methods=["GET"])
def models_page():
    return render_template("models.html")

@app.route("/model-mri", methods=["GET"])
def model_mri():
    return render_template("model-mri.html")

@app.route("/model-audio", methods=["GET"])
def model_audio():
    return render_template("model-audio.html")

@app.route("/model-fusion", methods=["GET"])
def model_fusion():
    return render_template("model-fusion.html")
    

@app.route("/project", methods=["GET"])
def project_definition():
    return render_template("project-definition.html")


@app.route("/methodology", methods=["GET"])
def methodology():
    return render_template("methodology.html")

@app.route("/team", methods=["GET"])
def team():
    return render_template("team.html")

@app.route("/privacy", methods=["GET"])
def privacy():
    return render_template("privacy.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        session['user_id'] = 1  # Simulated user ID
        session['user_email'] = email
        return redirect(url_for('dashboard'))
    
    return render_template("login.html")


@app.route("/logout", methods=["GET"])
def logout():
    session.pop('user_id', None)
    session.pop('user_email', None)
    return redirect(url_for('home'))


@app.route("/dashboard", methods=["GET"])
def dashboard():
    if not session.get('user_id'):
        flash("Please log in to access the dashboard.", "warning")
        return redirect(url_for('login'))
    return render_template("dashboard.html")


@app.route("/history", methods=["GET"])
def history():
    if not session.get('user_id'):
        flash("Please log in to view the history.", "warning")
        return redirect(url_for('login'))
    
    predictions = []
    if supabase is not None:
        try:
            user_id = session.get('user_id')
            response = supabase.table("predictions").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
            predictions = response.data
        except Exception as e:
            print(f"Error fetching history: {e}")
            
    return render_template("history.html", predictions=predictions)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)