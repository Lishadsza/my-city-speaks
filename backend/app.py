import os
import uuid 
import tempfile
from flask import Flask, jsonify, request
from supabase import create_client, Client
from dotenv import load_dotenv, find_dotenv
import numpy as np
import joblib
from werkzeug.utils import secure_filename
from flask_cors import CORS

from utils.feature_extractor import extract_features 
from utils.audioconverter import convert_to_wav 
 
# Define the project root directory relative to the current script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=dotenv_path) 

#  Supabase Initialization
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") 
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET_NAME")

# DEBUGGING OUTPUT(for ref)
print("-" * 40)
print(f"DEBUG: Attempted to load .env from: {dotenv_path}")
print(f"DEBUG: SUPABASE_URL is {'SET' if SUPABASE_URL else 'EMPTY'}")
print(f"DEBUG: SERVICE_KEY is {'SET' if SUPABASE_KEY else 'EMPTY'}")
print("-" * 40)

# Initialize the global Supabase client
if not SUPABASE_URL or not SUPABASE_KEY:
    print("WARNING: Supabase credentials not found. Database features will fail.")
    supabase = None
else:
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase client initialized.")
    except Exception as e:
        print(f"WARNING: Failed to initialize Supabase client: {e}")
        print("Database features will be disabled. ML prediction will still work.")
        supabase = None


app = Flask(__name__)
CORS(app) 


# Load trained components
try:
    model = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    print("ML Models loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load ML models: {e}")
    model = None 
# Notes
language_notes = {
    "kannada": "This language is commonly spoken in Karnataka.",
    "tulu": "This language is spoken mainly in coastal Karnataka and parts of Kerala. ",
    "hindi": "This language is widely spoken across North and Central India.)",
    "english": "This language is commonly used in urban and formal settings across India. )"
}
#Ml endpoints
@app.route("/")
def index():
    return "My City Speaks Backend is running."

@app.route("/predict", methods=["POST"])
def predict_language():
    """Endpoint for ML prediction (unchanged from your original code)."""
    if model is None:
        return jsonify({"error": "ML model failed to load on startup."}), 500

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
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

#database and map endpoints
@app.route("/api/cities", methods=["GET"])
def get_cities():
    """API Endpoint 1: Fetches all cities (lat/lng) for the React map."""
    if supabase is None:
        return jsonify({"error": "Supabase client not initialized."}), 500
    try:
        response = supabase.table("cities").select("id, name, lat, lng").execute()
        return jsonify(response.data), 200
    except Exception as e:
        print(f"Error fetching cities: {e}")
        return jsonify({"error": f"Database error fetching cities: {str(e)}"}), 500

@app.route("/api/phrases", methods=["GET"])
def get_phrases():
    """API Endpoint 2: Fetches all English phrases for the dropdown list."""
    if supabase is None:
        return jsonify({"error": "Supabase client not initialized."}), 500
    try:
        response = supabase.table("phrases").select("id, english_text").execute()
        return jsonify(response.data), 200
    except Exception as e:
        print(f"Error fetching phrases: {e}")
        return jsonify({"error": f"Database error fetching phrases: {str(e)}"}), 500

@app.route("/api/city_recordings/<int:city_id>", methods=["GET"])
def get_city_recordings(city_id):
    """API Endpoint 3: Fetches recordings for a city, joining with phrase text."""
    if supabase is None:
        return jsonify({"error": "Supabase client not initialized."}), 500
    try:
        response = supabase.table("recordings").select(
            "id, audio_url, language, created_at, phrases(english_text)"
        ).eq("city_id", city_id).order("created_at", desc=True).execute()
        
        return jsonify(response.data), 200
    except Exception as e:
        print(f"Error fetching recordings: {e}")
        return jsonify({"error": f"Database error fetching recordings: {str(e)}"}), 500

# NEW PHRASE REQUEST ENDPOINT FOR COMMUNITY BOARD DATA
@app.route("/api/city_requests/<int:city_id>", methods=["GET"])
def get_city_requests(city_id):
    """API Endpoint 4: Fetches pending phrase requests for the Community Board."""
    if supabase is None:
        return jsonify({"error": "Supabase client not initialized."}), 500
    try:
        # Fetch requests where city_id matches AND status is 'pending'
        response = supabase.table("phrase_requests").select("id, request_text, language, created_at").eq("city_id", city_id).eq("status", "pending").execute()
        return jsonify(response.data), 200
    except Exception as e:
        print(f"Request Fetch Error: {e}")
        return jsonify({"error": f"Database error fetching requests: {str(e)}"}), 500


@app.route("/api/request_phrase", methods=["POST"])
def request_phrase():
    """API Endpoint 5: Submits a new phrase request for admin review."""
    if supabase is None:
        return jsonify({"error": "Supabase client not initialized."}), 500
    
    data = request.get_json()
    request_text = data.get('request_text')
    city_id = data.get('city_id')
    language = data.get('language')

    if not request_text or not city_id or not language:
        return jsonify({"error": "Missing required fields (request_text , city_id,or language)."}), 400

    try:
    # Explicitly set status to 'pending'
        data_to_insert = {
            "request_text": request_text,
            "city_id": city_id,
            "language": language, 
            "status": "pending" 
        }
        
        insert_res = supabase.table("phrase_requests").insert(data_to_insert).execute()
        
        return jsonify({"message": "Phrase request saved successfully!", "data": insert_res.data[0]}), 201

    except Exception as e:
        print(f"Phrase Request Error: {e}")
        return jsonify({"error": f"Database error on phrase request: {str(e)}"}), 500

# @app.route("/api/fulfill_request/<int:request_id>", methods=["POST"])
# def fulfill_request(request_id):
#     """
#     API Endpoint: Sets the status of a specific phrase request to 'fulfilled'.
#     This endpoint is called after a user successfully contributes a recording 
#     for a phrase requested by the community.
#     """
#     if supabase is None:
#         return jsonify({"error": "Supabase client not initialized."}), 500

#     try:
#         # Update the 'phrase_requests' table using the request_id passed in the URL
#         response = supabase.table("phrase_requests").update({
#             "status": "fulfilled"
#         }).eq("id", request_id).execute()

#         # Check if the update was successful (Supabase returns data if successful)
#         if response.data:
#             return jsonify({"message": f"Request ID {request_id} fulfilled."}), 200
#         else:
#             return jsonify({"error": f"Request ID {request_id} not found or status already fulfilled."}), 404

#     except Exception as e:
#         print(f"Fulfillment Error for ID {request_id}: {e}")
#         return jsonify({"error": f"Database error during fulfillment: {str(e)}"}), 500

# UPLOAD ENDPOINT 
@app.route("/api/upload", methods=["POST"])
def upload_recording_and_metadata():
    """API Endpoint 6: Handles file upload, phrase upsert, and metadata insertion."""
    if supabase is None:
        return jsonify({"error": "Supabase client not initialized."}), 500
    try:
        # 1. Get form data
        audio_file = request.files.get('audio_file')
        phrase_text = request.form.get('phrase')
        language = request.form.get('language')
        city_id = request.form.get('city_id')

        if not all([audio_file, phrase_text, language, city_id]):
            return jsonify({"error": "Missing required data"}), 400

        #  Handle Phrase (Get ID or Insert New Phrase)
        phrase_res = supabase.table("phrases").select("id").eq("english_text", phrase_text).limit(1).execute()
        
        phrase_id = None
        if phrase_res.data:
            phrase_id = phrase_res.data[0]['id']
        else:
            new_phrase_res = supabase.table("phrases").insert({"english_text": phrase_text}).execute()
            if not new_phrase_res.data: raise Exception("Failed to insert new phrase.")
            phrase_id = new_phrase_res.data[0]['id']


        #  Upload Audio to Supabase Storage (Uses service_role key)
        file_name = f"{city_id}_{phrase_id}_{uuid.uuid4()}.webm" # Use webm since MediaRecorder outputs webm
        
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            file=audio_file.read(), 
            path=file_name,
            file_options={"content-type": audio_file.mimetype} 
        )
        audio_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(file_name)


        #  Insert Metadata into 'recordings' table (Uses service_role key)
        data_to_insert = {
            "audio_url": audio_url,
            "language": language,
            "phrase_id": phrase_id,
            "city_id": int(city_id)
        }
        
        insert_res = supabase.table("recordings").insert(data_to_insert).execute()
        
        return jsonify({"message": "Upload successful!", "data": insert_res.data[0]}), 201

    except Exception as e:
        print(f"Upload Error: {e}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
