import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    y = librosa.util.fix_length(y, size=16000)
    
    # Extract only the MFCCs to reduce resource consumption
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    features = np.mean(mfcc.T, axis=0)  # shape = (13,)
    
    return features.reshape(1, -1)  # shape = (1, 13)
