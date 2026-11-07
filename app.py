
from pathlib import Path

import joblib
import numpy as np

from flask import Flask, request, jsonify


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fusion_classifier.pkl"

CLASSES = ["Normal", "Leve", "Moderado"]

# Cargar el modelo de fusión
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "No se encontró el modelo fusion_classifier.pkl"
    )

fusion_model = joblib.load(MODEL_PATH)


def combine_probabilities(mri_probabilities, audio_probabilities):
    """Combina las probabilidades de MRI y audio."""

    values = list(mri_probabilities) + list(audio_probabilities)

    if len(values) != 6:
        raise ValueError("Se necesitan exactamente 6 probabilidades.")

    if not np.isfinite(values).all():
        raise ValueError("Las probabilidades deben ser números válidos.")

    if any(value < 0 or value > 1 for value in values):
        raise ValueError("Las probabilidades deben estar entre 0 y 1.")

    if not np.isclose(sum(mri_probabilities), 1.0, atol=0.01):
        raise ValueError("Las probabilidades MRI deben sumar 1.")

    if not np.isclose(sum(audio_probabilities), 1.0, atol=0.01):
        raise ValueError("Las probabilidades de audio deben sumar 1.")

    return np.array([values])


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "NeuroFusion AI API funcionando",
        "status": "OK"
    })


@app.route("/predict_fusion", methods=["POST"])
def predict_fusion():
    """Recibe las probabilidades generadas por MRI y audio."""

    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "Se requiere un JSON válido."}), 400

        mri = data.get("mri_probabilities")
        audio = data.get("audio_probabilities")

        if not isinstance(mri, list) or not isinstance(audio, list):
            return jsonify({
                "error": "Debes enviar listas de probabilidades MRI y audio."
            }), 400

        if len(mri) != 3 or len(audio) != 3:
            return jsonify({
                "error": "Cada modalidad debe contener 3 probabilidades."
            }), 400

        X = combine_probabilities(mri, audio)

        prediction = fusion_model.predict(X)[0]

        probabilities = fusion_model.predict_proba(X)[0]

        result = {
            "prediction": str(prediction),
            "probabilities": {
                str(label): round(float(probability), 4)
                for label, probability in zip(
                    fusion_model.classes_,
                    probabilities
                )
            },
            "notice": "Prototipo educativo. No constituye un diagnóstico médico."
        }

        return jsonify(result), 200

    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        return jsonify({
            "error": "Ocurrió un error interno al realizar la predicción."
        }), 500


@app.route("/predict_mri", methods=["POST"])
def predict_mri():
    """Recibe una imagen MRI para su futura clasificación."""

    if "file" not in request.files:
        return jsonify({
            "error": "No se recibió ningún archivo. Usa el campo 'file'."
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "No seleccionaste ningún archivo."
        }), 400

    allowed_extensions = {".png", ".jpg", ".jpeg", ".dcm"}

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        return jsonify({
            "error": "Formato no permitido. Usa PNG, JPG, JPEG o DCM."
        }), 400

    # El modelo MRI de Persona 1 todavía debe integrarse.
    return jsonify({
        "message": "Imagen MRI recibida correctamente.",
        "filename": file.filename,
        "status": "Pendiente de integrar el clasificador MRI de Persona 1."
    }), 503


@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    """Recibe un archivo de audio para su futura clasificación."""

    if "file" not in request.files:
        return jsonify({
            "error": "No se recibió ningún archivo. Usa el campo 'file'."
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "No seleccionaste ningún archivo."
        }), 400

    allowed_extensions = {".wav", ".mp3", ".flac", ".ogg"}

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        return jsonify({
            "error": "Formato no permitido. Usa WAV, MP3, FLAC u OGG."
        }), 400

    # El modelo de audio de Persona 2 todavía debe integrarse.
    return jsonify({
        "message": "Audio recibido correctamente.",
        "filename": file.filename,
        "status": "Pendiente de integrar el clasificador de audio de Persona 2."
    }), 503


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
