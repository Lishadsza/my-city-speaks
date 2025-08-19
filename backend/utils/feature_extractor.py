import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    y = librosa.util.fix_length(y, size=16000)
    
    # Original MFCC + delta + delta2
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    combined = np.vstack([mfcc, delta, delta2])  # shape = (39, n_frames)
    
    # Reduce to 13 features by taking mean **per original MFCC**
    # Option 1: Use only MFCC mean (ignore delta)
    features = np.mean(mfcc.T, axis=0)  # shape = (13,)
    
    # Option 2: Weighted average of MFCC + delta + delta2
    # features = np.mean(combined.reshape(3, 13, -1), axis=(0,2))  # also gives 13 features
    
    return features.reshape(1, -1)  # shape = (1, 13)
