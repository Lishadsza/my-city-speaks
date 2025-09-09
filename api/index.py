from flask import Flask, request, jsonify,send_from_directory
import numpy as np
import joblib
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
from utils.feature_extractor import extract_features
from utils.audioconverter import convert_to_wav 

app = Flask(__name__, static_folder="build") # .\venv\Scripts\activate(virtualenviro)
CORS(app)



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

""" @app.route("/")
def index():
    return "Accent classification backend is running." """


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        uploaded_file = request.files["file"]

        # Save original upload
        filename = secure_filename(uploaded_file.filename)
        os.makedirs("uploads", exist_ok=True)
        original_path = os.path.join("uploads", filename)
        uploaded_file.save(original_path)

        # Convert to WAV if needed
        if not filename.lower().endswith(".wav"):
            filepath = convert_to_wav(original_path)
        else:
            filepath = original_path

        # Extract features
        features = extract_features(filepath).reshape(1, -1)

        # Scale & predict
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        note = language_notes.get(predicted_label, "")

        # Cleanups
        try:
            os.remove(original_path)
            if filepath != original_path:
                os.remove(filepath)
        except:
            pass

        # ReturningResponnse
        return jsonify({
            "language": predicted_label,
            "note": note
        })

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
