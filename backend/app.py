from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
from utils.feature_extractor import extract_features
from utils.audioconverter import convert_to_wav
import tempfile

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "https://my-city-speaks.vercel.app"}})

# Load trained components
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Notes
language_notes = {
    "kannada": "This language is commonly spoken in Karnataka. (Based on language, not location prediction.)",
    "tulu": "This language is spoken mainly in coastal Karnataka and parts of Kerala. (Based on language, not location prediction.)",
    "hindi": "This language is widely spoken across North and Central India. (Based on language, not location prediction.)",
    "english": "This language is commonly used in urban and formal settings across India. (Based on language, not location prediction.)"
}

@app.route("/")
def index():
    return "Accent classification backend is running."

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files["file"]
    filename = secure_filename(uploaded_file.filename)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            original_path = os.path.join(temp_dir, filename)
            uploaded_file.save(original_path)

            if not filename.lower().endswith(".wav"):
                filepath = convert_to_wav(original_path)
            else:
                filepath = original_path

            features = extract_features(filepath).reshape(1, -1)
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)
            predicted_label = label_encoder.inverse_transform(prediction)[0]
            note = language_notes.get(predicted_label, "")

            return jsonify({
                "language": predicted_label,
                "note": note
            })

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)