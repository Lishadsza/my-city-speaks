import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler
import warnings

# Ignore precision warnings for now
warnings.filterwarnings("ignore")

# Load features
df = pd.read_csv("audio_features.csv")

# Show class distribution before balancing
print("Original Class Distribution:\n", df["language"].value_counts())

# Define features and target
X = df[[col for col in df.columns if col.startswith("mfcc_")]]
y = df["language"]  # Can switch to 'city' or 'gender'

# Step 1: Oversample minority classes
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Show new distribution
print("\nBalanced Class Distribution:\n", pd.Series(y_resampled).value_counts())

# Step 2: Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Step 3: Train Logistic Regression
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, y_train)

# Step 4: Evaluate
y_pred = clf.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
