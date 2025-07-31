from flask import Flask, request, jsonify
import numpy as np
import joblib
app = Flask(__name__)

#loading the trained compnents
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

#notes
language_notes = {
    "kannada": "This language is commonly spoken in Karnataka. (Based on language, not location prediction.)",
    "tulu": "This language is spoken mainly in coastal Karnataka and parts of Kerala.(Based on language, not location prediction.)",
    "hindi": "This language is widely spoken across North and Central India.(Based on language, not location prediction.)",
    "english": "This language is commonly used in urban and formal settings across India.(Based on language, not location prediction.)"
}
@app.route("/")
def index():
    return "Accent classification backend is running."

@app.route("/predict", methods=["POST"])
def predict():#prdict funtion
    try:
        data = request.get_json()
        if "features" not in data:
            return jsonify({"error": "Missing 'features' in request"}), 400
        features = np.array(data["features"]).reshape(1, -1)
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # Optional notes addition
        note = language_notes.get(predicted_label, "")

        return jsonify({
            "prediction": predicted_label,
            "note": note
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
