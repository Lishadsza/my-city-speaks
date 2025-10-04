#for checking result i have craetaed this file(optional)
import requests
import librosa
import numpy as np
file_path = "converted_wav/user004_tulu_meal.wav"
# Load and extract MFCCs
y, sr = librosa.load(file_path, sr=None)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_mean = np.mean(mfcc.T, axis=0)


data = {
    "features": mfcc_mean.tolist()#sending backend
}

response = requests.post("http://127.0.0.1:5000/predict", json=data)

try:
    print("Response JSON:", response.json())
except Exception:
    print("Raw Response:", response.text)
