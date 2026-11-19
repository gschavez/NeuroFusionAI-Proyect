import os
import gc
import re
import tempfile
from functools import wraps
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from dotenv import load_dotenv
from supabase import create_client
from werkzeug.security import generate_password_hash, check_password_hash

# Load environment variables for Supabase
load_dotenv(override=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")
if not app.secret_key:
    print("Warning: SECRET_KEY not set; using an insecure development key.")
    app.secret_key = "dev-key-only-for-local-testing"
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
        record["user_id"] = session.get("user_id")
        # patient_id is passed in the 'record' dictionary from the frontend
        supabase.table("predictions").insert(record).execute()
        print("Prediction saved to Supabase.")
    except Exception as db_error:
        print(f"Error saving prediction to Supabase: {db_error}")


def login_required(view):
    """Protect a route: only logged-in users can open it.
    Pages redirect to the login form; JSON/API routes answer 401."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            if request.path.startswith("/predict_") or request.is_json:
                return jsonify({"error": "Authentication required. Please log in."}), 401
            flash("Please log in to access this section.", "warning")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


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


def build_partial_comment(prediction, probabilities, mri_probs, audio_probs):
    """Short, cautious interpretation of the fusion output.
    It is a simulation from an academic prototype, never a diagnosis."""
    confidence = max(probabilities.values())
    base = {
        "Normal": ("The combined MRI and voice patterns are closest to the cognitively normal "
                   "profile seen in the training data. In a real clinical setting, a result like "
                   "this would point to routine follow-up rather than an urgent referral."),
        "Mild": ("The combined patterns are closest to mild cognitive decline (very mild / mild "
                 "dementia in the MRI data and Alzheimer-like speech markers in the voice). In a "
                 "real clinical setting, a result like this would justify a complete "
                 "neuropsychological evaluation."),
        "Moderate": ("The combined patterns are closest to moderate dementia in the training data. "
                     "In a real clinical setting, a result like this would justify a prompt "
                     "referral to a neurologist."),
    }[prediction]

    notes = []
    if confidence < 0.60:
        notes.append(f"The model is not confident ({confidence:.0%}), so this result should be "
                     "treated as inconclusive.")
    mri_label = CLASSES[int(np.argmax(mri_probs))]
    mri_impaired = mri_label != "Normal"
    audio_impaired = float(audio_probs[0]) < 0.5  # the voice model is binary (healthy / AD)
    if mri_impaired != audio_impaired:
        notes.append(f"The MRI ({mri_label}) and the voice "
                     f"({'impairment markers' if audio_impaired else 'no impairment markers'}) "
                     "point in different directions, which lowers the reliability of the "
                     "combined result.")
    return {
        "level": prediction,
        "confidence": round(confidence, 4),
        "summary": base,
        "notes": notes,
        "disclaimer": ("Simulation only: this prototype was trained on public, synthetically paired "
                       "data and does not provide a diagnosis. Only a qualified health "
                       "professional can diagnose."),
    }


# ==========================================
# API ROUTES
# ==========================================

@app.route("/predict_fusion", methods=["POST"])
@login_required
def predict_fusion():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "A valid JSON is required."}), 400

        mri = data.get("mri_probabilities")
        audio = data.get("audio_probabilities")
        
        # Get patient ID and filenames from the frontend request
        patient_id = data.get("patient_id")
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
        result["comment"] = build_partial_comment(prediction_name, result["probabilities"], mri, audio)

        # SAVE EVERYTHING IN A SINGLE ROW (Now including patient_id)
        save_prediction({
            "patient_id": patient_id if patient_id else None,
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
@login_required
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
        gc.collect()  # Memory cleanup (MRI)

    return jsonify({
        "message": "MRI image analyzed successfully.",
        "filename": file.filename,
        "class_name": result["class_name"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities_list"],
    }), 200


@app.route("/predict_audio", methods=["POST"])
@login_required
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
        gc.collect()  # Memory cleanup (Audio/Librosa)


# ==========================================
# FRONTEND & PATIENT ROUTES
# ==========================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/models", methods=["GET"])
@login_required
def models_page():
    return render_template("models.html")

@app.route("/model-mri", methods=["GET"])
@login_required
def model_mri():
    return render_template("model-mri.html")

@app.route("/model-audio", methods=["GET"])
@login_required
def model_audio():
    return render_template("model-audio.html")

@app.route("/model-fusion", methods=["GET"])
@login_required
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


# ==========================================
# AUTHENTICATION (table "users": id, email, password_hash, role, created_at)
# ==========================================

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ALLOWED_GENDERS = {"Male", "Female", "Other"}


def safe_next(default_endpoint="dashboard"):
    """Only follow ?next= when it is a path inside this site."""
    next_url = request.args.get("next", "")
    if next_url.startswith("/") and not next_url.startswith("//") and not urlparse(next_url).netloc:
        return next_url
    return url_for(default_endpoint)


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm") or ""

        if supabase is None:
            flash("Database not configured: add SUPABASE_URL and SUPABASE_KEY to the .env file.", "danger")
            return render_template("register.html", email=email)
        if not EMAIL_RE.match(email):
            flash("Please enter a valid email address.", "danger")
            return render_template("register.html", email=email)
        if len(password) < 8 or not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
            flash("The password must have at least 8 characters, including letters and numbers.", "danger")
            return render_template("register.html", email=email)
        if password != confirm:
            flash("The passwords do not match.", "danger")
            return render_template("register.html", email=email)
        if not request.form.get("habeas_data"):
            flash("You must authorize the processing of your personal data.", "danger")
            return render_template("register.html", email=email)

        try:
            exists = supabase.table("users").select("id").eq("email", email).execute()
            if exists.data:
                flash("An account with that email already exists.", "danger")
                return render_template("register.html", email=email)
            # The role is always "user": nobody can make themselves admin from the form.
            supabase.table("users").insert({
                "email": email,
                "password_hash": generate_password_hash(password),
                "role": "user",
            }).execute()
        except Exception as e:
            print(f"Error registering user: {e}")
            flash("The account could not be created. Please try again later.", "danger")
            return render_template("register.html", email=email)

        flash("Account created. You can log in now.", "success")
        return redirect(url_for("login"))

    return render_template("register.html", email="")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        if supabase is None:
            flash("Database not configured: add SUPABASE_URL and SUPABASE_KEY to the .env file.", "danger")
            return render_template("login.html"), 503

        user = None
        try:
            res = supabase.table("users").select("id, email, password_hash, role").eq("email", email).limit(1).execute()
            user = res.data[0] if res.data else None
        except Exception as e:
            print(f"Error during login: {e}")
            flash("Login is not available right now. Please try again later.", "danger")
            return render_template("login.html"), 503

        if not user or not user.get("password_hash") or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.", "danger")
            return render_template("login.html"), 401

        session.clear()
        session["user_id"] = user["id"]
        session["user_email"] = user["email"]
        session["user_role"] = user.get("role") or "user"
        return redirect(safe_next())

    return render_template("login.html")


@app.route("/logout", methods=["GET"])
def logout():
    session.clear()
    flash("You have logged out.", "info")
    return redirect(url_for('home'))


# ==========================================
# PATIENT MANAGEMENT ROUTES
# ==========================================

def db_missing_redirect():
    flash("Database not configured: add SUPABASE_URL and SUPABASE_KEY to the .env file.", "danger")
    return redirect(url_for('patients_page'))


def read_patient_form():
    """Validate the patient form. Returns (data, error)."""
    full_name = (request.form.get("full_name") or "").strip()
    age_raw = (request.form.get("age") or "").strip()
    gender = (request.form.get("gender") or "").strip()
    if len(full_name) < 2:
        return None, "Patient name is required."
    age = None
    if age_raw:
        try:
            age = int(age_raw)
        except ValueError:
            return None, "Age must be a whole number."
        if not 18 <= age <= 120:
            return None, "Age must be between 18 and 120."
    if gender and gender not in ALLOWED_GENDERS:
        return None, "Invalid gender value."
    return {"full_name": full_name, "age": age, "gender": gender or None}, None


@app.route("/patients", methods=["GET"])
@login_required
def patients_page():
    patients_list = []
    if supabase is None:
        flash("Database not configured: patients cannot be loaded. Add SUPABASE_URL and SUPABASE_KEY to the .env file.", "warning")
    else:
        try:
            # Fetch all patients, ordered by newest first
            response = supabase.table("patients").select("*").order("created_at", desc=True).execute()
            patients_list = response.data
        except Exception as e:
            print(f"Error fetching patients: {e}")
            flash("Patients could not be loaded.", "danger")

    return render_template("patients.html", patients=patients_list)


@app.route("/add_patient", methods=["POST"])
@login_required
def add_patient():
    if supabase is None:
        return db_missing_redirect()

    data, error = read_patient_form()
    if error:
        flash(error, "danger")
        return redirect(url_for('patients_page'))
    try:
        data["user_id"] = session["user_id"]
        supabase.table("patients").insert(data).execute()
        flash("Patient added successfully.", "success")
    except Exception as e:
        print(f"Error adding patient: {e}")
        flash("Error adding patient. Please try again.", "danger")

    return redirect(url_for('patients_page'))


@app.route("/delete_patient/<int:patient_id>", methods=["POST"])
@login_required
def delete_patient(patient_id):
    if supabase is None:
        return db_missing_redirect()
    try:
        # Delete the patient's predictions first: this works even if the
        # foreign key predictions.patient_id has no ON DELETE CASCADE.
        supabase.table("predictions").delete().eq("patient_id", patient_id).execute()
        supabase.table("patients").delete().eq("id", patient_id).execute()
        flash("Patient and their evaluation history deleted.", "success")
    except Exception as e:
        print(f"Error deleting patient: {e}")
        flash("Error deleting patient. Please try again.", "danger")

    return redirect(url_for('patients_page'))


@app.route("/edit_patient/<int:patient_id>", methods=["GET", "POST"])
@login_required
def edit_patient(patient_id):
    if supabase is None:
        return db_missing_redirect()

    if request.method == "POST":
        data, error = read_patient_form()
        if error:
            flash(error, "danger")
            return redirect(url_for('edit_patient', patient_id=patient_id))
        try:
            supabase.table("patients").update(data).eq("id", patient_id).execute()
            flash("Patient updated successfully.", "success")
            return redirect(url_for('patients_page'))
        except Exception as e:
            print(f"Error updating patient: {e}")
            flash("Error updating patient. Please try again.", "danger")

    # Fetch patient for the form (GET)
    try:
        response = supabase.table("patients").select("*").eq("id", patient_id).execute()
        if not response.data:
            flash("Patient not found.", "danger")
            return redirect(url_for('patients_page'))
        patient = response.data[0]
    except Exception as e:
        print(f"Error fetching patient: {e}")
        flash("Error loading patient.", "danger")
        return redirect(url_for('patients_page'))

    return render_template("edit_patient.html", patient=patient)


@app.route("/dashboard", methods=["GET"])
@login_required
def dashboard():    
    # Fetch patients to populate the dropdown
    patients_list = []
    if supabase is not None:
        try:
            response = supabase.table("patients").select("id, full_name, age").order("full_name").execute()
            patients_list = response.data
        except Exception as e:
            print(f"Error fetching patients for dashboard: {e}")

    return render_template("dashboard.html", patients=patients_list)


@app.route("/history", methods=["GET"])
@login_required
def history():    
    predictions = []
    if supabase is not None:
        try:
            user_id = session.get('user_id')
            # Get filters from URL
            result_filter = request.args.get("result")
            patient_filter = request.args.get("patient_id")
            search_query = request.args.get("q")

            # Base query
            query = supabase.table("predictions").select("*, patients(full_name)").eq("user_id", user_id)

            # Apply text search (Search by patient name)
            if search_query:
                # First, find patients matching the name
                patient_res = supabase.table("patients").select("id").ilike("full_name", f"%{search_query}%").execute()
                matching_ids = [p["id"] for p in patient_res.data]
                
                if matching_ids:
                    query = query.in_("patient_id", matching_ids)
                else:
                    # If no patients match the name, force empty result
                    query = query.eq("patient_id", -1) 

            # Apply other filters
            if result_filter:
                query = query.eq("final_result", result_filter)
            if patient_filter:
                query = query.eq("patient_id", patient_filter)

            response = query.order("created_at", desc=True).execute()
            predictions = response.data
        except Exception as e:
            print(f"Error fetching history: {e}")
            
    return render_template(
        "history.html", 
        predictions=predictions, 
        active_filter=request.args.get("result"), 
        active_patient=request.args.get("patient_id"),
        search_query=request.args.get("q")
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)