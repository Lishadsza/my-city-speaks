import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from sklearn.model_selection import cross_val_score
# dataset
df = pd.read_csv("audio_features.csv")

#distribution
print("Class Distribution:\n", df['language'].value_counts())

# Define features
X = df[[col for col in df.columns if col.startswith("mfcc_")]]
y = df["language"]

# Train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify maintains class balance in split
)

#Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
scores = cross_val_score(clf, X, y, cv=5)
print("Cross-validation accuracy:", scores.mean())