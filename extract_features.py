import librosa
import numpy as np
import pandas as pd
import os

# Load dataset (correct path)
df = pd.read_csv("dataset/dataset.csv")

# Store MFCC features
features = []

for idx, row in df.iterrows():
    filepath = row['filepath']
    try:
        # Load the .wav file
        y, sr = librosa.load(filepath, sr=None)

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1)  # Mean across time axis

        # Build a feature row with metadata + MFCCs
        feature_row = {
            "user_id": row["user_id"],
            "word": row["word"],
            "state": row["state"],
            "city": row["city"],
            "language": row["language"],
            "gender": row["gender"],
            "agegroup": row["agegroup"],
        }

        for i, val in enumerate(mfcc_mean):
            feature_row[f"mfcc_{i+1}"] = val

        features.append(feature_row)
        print(f"Processed: {filepath}")

    except Exception as e:
        print(f"Failed: {filepath} — {e}")

# Save features to CSV
features_df = pd.DataFrame(features)
features_df.to_csv("audio_features.csv", index=False)
print("Saved MFCC features to audio_features.csv")
