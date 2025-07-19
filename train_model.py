import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
# dataset
df = pd.read_csv("audio_features.csv")

#distribution
print("Class Distribution:\n", df['language'].value_counts())

# Define features
X = df[[col for col in df.columns if col.startswith("mfcc_")]]
y = df["language"]

# Train/test split 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)  
# stratify maintains class balance in split


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


#confusion matrix and plotting
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
print("\nConfusion Matrix:\n", cm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

#saving the modell
joblib.dump(clf, 'random_forest_model.pkl')
print("Model saved as random_forest_model.pkl")