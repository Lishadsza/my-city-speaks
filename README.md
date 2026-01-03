# 🗺️ My City Speaks

<div align="center">

![My City Speaks](https://img.shields.io/badge/My%20City%20Speaks-Language%20Mapping%20Platform-purple?style=for-the-badge)
![React](https://img.shields.io/badge/React-19.1-61DAFB?style=for-the-badge&logo=react)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)
![Supabase](https://img.shields.io/badge/Supabase-Database-3ECF8E?style=for-the-badge&logo=supabase)

**A comprehensive web platform for mapping and preserving regional language accents across India**



</div>

---

##  Overview

**My City Speaks** is an innovative web application that combines **AI-powered language identification** with **community-driven accent mapping** to preserve and celebrate India's linguistic diversity. The platform enables users to record, share, and explore regional language variations across different cities.

###  Mission
To create a comprehensive digital archive of India's regional accents and dialects, making linguistic diversity accessible and preserving it for future generations.

---

##  Key Features

###  **AI Language Accent Detection**
- **Real-time language identification** using machine learning (SVM model)
- **Multi-language support**: English, Kannada, Tulu, Hindi, and more
- **Audio feature extraction** using librosa for accurate classification
- **Confidence scoring** with detailed prediction notes

###  **Interactive City Mapping**
- **80+ Indian cities** with precise geo-coordinates
- **Interactive Leaflet maps** for city exploration
- **City-specific language options** based on regional demographics
- **Visual accent distribution** across different regions

###  **Community Recording Platform**
- **Browser-based audio recording** using MediaRecorder API
- **Phrase-based contributions** with curated English phrases
- **Multi-language recording** for each city
- **Audio playback and review** before submission

### **Community Request System**
- **Phrase request board** for community-driven content
- **Pending request management** with admin approval
- **Collaborative contribution** workflow
- **Request fulfillment tracking**

###  **Modern Glassmorphism UI**
- **Glass-effect design** with backdrop blur
- **Responsive layout** for all device sizes
- **Smooth animations** and transitions
- **Accessible interface** with proper contrast

###  **Data Management**
- **Supabase integration** for scalable database
- **Real-time data synchronization**
- **Audio file storage** with CDN delivery
- **Metadata tracking** for all recordings

---

##  Architecture

### **Frontend Stack**
- **React 19.1** - Modern UI framework
- **Vite 7.1** - Fast build tool and dev server
- **Tailwind CSS 4.1** - Utility-first styling
- **Leaflet & React Leaflet** - Interactive maps
- **Axios** - HTTP client for API calls
- **Lucide React** - Beautiful icons

### **Backend Stack**
- **Flask 3.1** - Python web framework
- **scikit-learn 1.6** - Machine learning models
- **librosa 0.10** - Audio processing and feature extraction
- **NumPy & Pandas** - Data manipulation
- **Supabase Python Client** - Database operations

### **Machine Learning Pipeline**
- **SVM Classification Model** - Language identification
- **MFCC Feature Extraction** - Audio signal processing
- **Audio Preprocessing** - Format conversion and normalization
- **Model Serialization** - Joblib for model persistence


## 🚀 Setup and Installation

### **Prerequisites**
- Python 3.11+
- Node.js 18+
- Git

### **Backend Setup**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/my-city-speaks.git
   cd my-city-speaks/backend
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment setup:**
   ```bash
   # Create .env file with your Supabase credentials
   SUPABASE_URL="your_supabase_url"
   SUPABASE_SERVICE_KEY="your_service_key"
   SUPABASE_BUCKET_NAME="your_bucket_name"
   ```

5. **Run the Flask server:**
   ```bash
   python app.py
   ```
   Server starts on `http://localhost:5000`

### **Frontend Setup**

1. **Navigate to frontend:**
   ```bash
   cd ../frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start development server:**
   ```bash
   npm run dev
   ```
   Application available at `http://localhost:5173`

### **Database Setup**

1. **Add cities to database:**
   ```bash
   cd backend
   python add_cities_fixed.py
   ```

2. **Manage phrases:**
   ```bash
   python batch_update_phrases.py
   ```

---

## 📡 API Reference

### **Language Prediction**
```http
POST /predict
Content-Type: multipart/form-data

Body: audio file (webm/wav format)
```

**Response:**
```json
{
  "language": "kannada",
  "note": "This language is commonly spoken in Karnataka."
}
```

### **City Management**
```http
GET /api/cities
```
Returns all cities with coordinates.

### **Recording Management**
```http
GET /api/city_recordings/{city_id}
POST /api/upload
```

### **Community Features**
```http
GET /api/city_requests/{city_id}
POST /api/request_phrase
```

---

##  Usage Examples

### **Recording a Phrase**
1. Select a city on the interactive map
2. Choose local language from dropdown
3. Select English phrase to record
4. Click "Start Recording" and speak the phrase
5. Review audio and submit contribution

### **Fulfilling Community Requests**
1. Browse the Community Request Board
2. Click "Record" next to a requested phrase
3. Record the phrase in the requested language
4. Submit to fulfill the community request

### **Exploring Language Data**
1. Navigate between different cities
2. Listen to existing recordings
3. Compare accents across regions
4. Discover linguistic patterns

---



## Project Statistics

- **80+ Cities** mapped across India
- **4+ Languages** supported for ML detection
- **25+ Regional languages** in city mappings
- **Real-time audio processing** capabilities
- **Community-driven** content creation

---

##  Contact & Support

- **Email**: dsouzalisha24@gmail.com
- **Documentation**: [Full API docs](https://docs.mycityspeaks.com)

---

## Screenshots
-HomePage
<img width="1898" height="916" alt="image" src="https://github.com/user-attachments/assets/0400f497-560d-47b2-9f2b-56f7201dd601" />
<img width="1901" height="906" alt="image" src="https://github.com/user-attachments/assets/35b6bf29-9a22-4eed-a241-60b79e083ddd" />

-MapPage
<img width="1915" height="910" alt="image" src="https://github.com/user-attachments/assets/ae24fb89-d2a4-4d6e-bc97-8f8730303dd1" />
<img width="1917" height="892" alt="image" src="https://github.com/user-attachments/assets/7acbbbda-4b24-41cf-a34a-54f6704db378" />
<img width="1888" height="890" alt="image" src="https://github.com/user-attachments/assets/2de65de6-efd6-47de-ac7d-fbb7bf806485" />
<img width="1914" height="908" alt="image" src="https://github.com/user-attachments/assets/67a3d063-9a10-40ab-92bc-3e8684452abe" />

-AnalysisPage
<img width="1889" height="905" alt="image" src="https://github.com/user-attachments/assets/57020078-386c-4985-9c79-258312882b7c" />
<img width="1888" height="900" alt="image" src="https://github.com/user-attachments/assets/e9a5a6a7-bc93-4f44-9456-ec289db98ada" />


-ContactmePage
<img width="1867" height="902" alt="image" src="https://github.com/user-attachments/assets/d7a93dd8-fcb4-4f8f-a852-42230782905a" />



<div align="center">


[⭐ Star this repo](https://github.com/yourusername/my-city-speaks) • [🍴 Fork it](https://github.com/yourusername/my-city-speaks/fork) • [📢 Share it](https://twitter.com/intent/tweet?text=Check%20out%20My%20City%20Speaks%20-%20A%20platform%20for%20mapping%20regional%20language%20accents%20across%20India!)

</div>
