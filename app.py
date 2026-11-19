import os
import gc
import re
import tempfile
import threading
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


# One inference at a time: the MRI and audio requests arrive in parallel from the
# dashboard, and loading the models from two threads at once can deadlock Python's
# import system ("deadlock detected by _ModuleLock"). It also limits peak RAM.
MODEL_LOCK = threading.Lock()


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


# ------------------------------------------------------------------
# Screening report: explanation of the voice result + structured report
# ------------------------------------------------------------------
AUDIO_MARKERS = {
    # feature: (label, what it measures)
    "pause_ratio": ("Pauses and silences", "share of the recording spent in silence or hesitation"),
    "pitch_mean": ("Average pitch", "how high or low the voice is"),
    "rms_mean": ("Voice energy", "loudness / vocal effort"),
    "zcr_mean": ("Zero-crossing rate", "noisiness / breathiness of the sound"),
    "spectral_centroid": ("Spectral brightness", "balance between low and high frequencies"),
    "duration": ("Recording length", "total duration of the speech sample (s)"),
}
_audio_expl = None


def _load_audio_explainer():
    """Coefficients of the Logistic Regression member of the audio ensemble and the
    training mean/std of each feature. Used only to explain, not to predict."""
    global _audio_expl
    if _audio_expl is None:
        scaler = joblib.load(BASE_DIR / "models" / "audio_scaler.pkl")
        model = joblib.load(BASE_DIR / "models" / "audio_classifier.pkl")
        lr = model.named_estimators_["lr"]
        names = list(scaler.feature_names_in_)
        _audio_expl = {
            "names": names,
            "mean": dict(zip(names, scaler.mean_)),
            "scale": dict(zip(names, scaler.scale_)),
            "coef": dict(zip(names, lr.coef_[0])),
        }
    return _audio_expl


def explain_audio(features):
    """Compare each marker with the training population (z-score) and estimate
    whether it pushed the result towards impairment or towards healthy."""
    e = _load_audio_explainer()
    markers, out_of_range = [], []
    mfcc_push = 0.0
    for name in e["names"]:
        if name not in features:
            continue
        z = (features[name] - e["mean"][name]) / e["scale"][name]
        push = e["coef"][name] * z            # > 0 -> towards Alzheimer's, < 0 -> towards healthy
        if abs(z) > 3:
            out_of_range.append(name)
        if name.startswith("mfcc_"):
            mfcc_push += push
            continue
        label, meaning = AUDIO_MARKERS[name]
        markers.append({
            "feature": name, "label": label, "meaning": meaning,
            "value": round(features[name], 4 if abs(features[name]) < 10 else 1),
            "train_mean": round(e["mean"][name], 4 if abs(e["mean"][name]) < 10 else 1),
            "z": round(z, 2),
            "position": ("well above" if z > 1.5 else "above" if z > 0.5 else
                         "well below" if z < -1.5 else "below" if z < -0.5 else "close to"),
            "push": round(push, 3),
            "direction": "impairment" if push > 0.1 else "healthy" if push < -0.1 else "neutral",
        })
    markers.sort(key=lambda m: -abs(m["push"]))
    markers.append({
        "feature": "mfcc", "label": "Vocal timbre (13 MFCCs)",
        "meaning": "shape of the vocal tract / articulation, combined",
        "value": None, "train_mean": None, "z": None, "position": None,
        "push": round(mfcc_push, 3),
        "direction": "impairment" if mfcc_push > 0.1 else "healthy" if mfcc_push < -0.1 else "neutral",
    })
    duration = features.get("duration", 0.0)
    return {
        "markers": markers,
        "quality": {
            "duration_s": round(duration, 1),
            "short_recording": duration < 30,
            "out_of_range": out_of_range,
        },
    }


MRI_FINDINGS = {
    "Normal": ("Pattern closest to non-demented brains in the training set: ventricles and cortical "
               "sulci within the usual range for the dataset."),
    "Mild": ("Pattern closest to very mild / mild dementia in the training set. In this dataset these "
             "images usually show slightly enlarged ventricles and widened sulci compared with normal brains."),
    "Moderate": ("Pattern closest to moderate dementia in the training set, where images typically show "
                 "clearly enlarged ventricles and marked cortical atrophy."),
}

NEXT_STEPS = {
    "Normal": [
        "Routine cognitive re-screening in about 12 months, or earlier if new memory complaints appear.",
        "Keep a record of this result to compare with future evaluations.",
    ],
    "Mild": [
        "Complete cognitive testing by a professional (for example MoCA or MMSE and a neuropsychological battery).",
        "Clinical review of reversible causes (for example vitamin B12, thyroid function, medication, mood).",
        "Follow-up evaluation in 3 to 6 months to see whether the pattern changes.",
    ],
    "Moderate": [
        "Prompt referral to neurology for a full clinical assessment.",
        "Review of the MRI by a radiologist and complete cognitive testing.",
        "Involve the family or caregiver in the follow-up plan.",
    ],
}


def confidence_band(p):
    if p >= 0.80:
        return "High", "The fusion model is confident about this class."
    if p >= 0.60:
        return "Moderate", "The result is plausible but other classes are not excluded."
    return "Low", "The model is not confident: the result should be treated as inconclusive."


def build_report(prediction, probabilities, mri_probs, audio_probs, audio_quality=None):
    """Structured screening report built only from what the models produce."""
    ordered = sorted(probabilities.items(), key=lambda kv: -kv[1])
    top_p = ordered[0][1]
    second, second_p = ordered[1]
    band, band_text = confidence_band(top_p)

    mri_label = CLASSES[int(np.argmax(mri_probs))]
    mri_conf = float(max(mri_probs))
    p_ad = float(audio_probs[1]) + float(audio_probs[2])
    audio_impaired = p_ad >= 0.5
    agree = (mri_label != "Normal") == audio_impaired

    reliability = []
    if not agree:
        reliability.append("MRI and voice disagree: one modality suggests impairment and the other does not.")
    if band == "Low":
        reliability.append("Low fusion confidence (below 60 %).")
    if top_p - second_p < 0.15:
        reliability.append(f"The second most likely class ({second}, {second_p:.0%}) is close to the first.")
    q = audio_quality or {}
    if q.get("short_recording"):
        reliability.append(f"Short voice sample ({q.get('duration_s', 0):.0f} s); the voice model was trained on ~1–2 minute recordings.")
    if q.get("out_of_range"):
        reliability.append("Some acoustic values are far outside the training range ("
                           + ", ".join(q["out_of_range"][:4]) + "), possibly due to noise or the microphone.")
    if not reliability:
        reliability.append("No reliability warnings: both modalities point in the same direction.")

    return {
        "level": prediction,
        "probabilities": probabilities,
        "confidence": round(top_p, 4),
        "confidence_band": band,
        "confidence_text": band_text,
        "second_class": second,
        "second_probability": round(second_p, 4),
        "mri": {"label": mri_label, "confidence": round(mri_conf, 4), "finding": MRI_FINDINGS[mri_label]},
        "voice": {
            "p_alzheimer": round(p_ad, 4),
            "label": "Alzheimer-like speech markers" if audio_impaired else "No Alzheimer-like speech markers",
        },
        "modalities_agree": agree,
        "reliability": reliability,
        "next_steps": NEXT_STEPS[prediction],
        "limitations": [
            "MRI and voice models were trained on different public datasets (Mendeley MRI, ADCeleb speech) and paired synthetically.",
            "The voice model is binary (healthy vs. Alzheimer's); its Mild/Moderate split is a fixed 70/30 rule.",
            "The training data is imbalanced (very few Moderate cases), so Moderate predictions are the least reliable.",
            "The voice explanation comes from the logistic-regression member of the ensemble and is indicative only.",
        ],
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
        audio_quality = None
        raw_q = data.get("audio_quality")
        if isinstance(raw_q, dict):  # sent back by the browser: sanitize before use
            try:
                audio_quality = {
                    "duration_s": float(raw_q.get("duration_s") or 0),
                    "short_recording": bool(raw_q.get("short_recording")),
                    "out_of_range": [str(x)[:30] for x in list(raw_q.get("out_of_range") or [])[:10]],
                }
            except (TypeError, ValueError):
                audio_quality = None
        result["report"] = build_report(prediction_name, result["probabilities"], mri, audio, audio_quality)

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
        with MODEL_LOCK:
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
        with MODEL_LOCK:
            result = get_audio_prediction(str(temp_path))

        if result["status"] == "error":
            return jsonify({"error": result["message"]}), 500

        audio_probs = result["probabilities"]
        response = {
            "message": "Audio processed successfully.",
            "filename": file.filename,
            "probabilities": [round(p, 6) for p in audio_probs],
        }
        try:
            response.update(explain_audio(result.get("features", {})))
        except Exception as expl_error:  # the explanation must never break the prediction
            print(f"Audio explanation unavailable: {expl_error}")
        return jsonify(response), 200

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