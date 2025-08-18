import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.utils import resample

df = pd.read_csv("audio_features.csv")#loading dataset
# print((df.columns).value_counts())  # Checking actual column names
label_column = "language"

# Balancing oversampling dataset 
balanced_dfs = []
for lang in df['language'].unique():
    class_df = df[df['language'] == lang]
    class_df_balanced = resample(class_df,
                                 replace=True,
                                 n_samples=33,
                                 random_state=42)
    balanced_dfs.append(class_df_balanced)
df_balanced = pd.concat(balanced_dfs, ignore_index=True)
print(df_balanced['language'].value_counts())


#extracting mfcc features
""" X = df[[col for col in df.columns if col.startswith("mfcc_")]]
y = df[label_column] """
X = df_balanced[[f"mfcc_{i}" for i in range(1, 14)]]  # use only first 13 MFCCs
y = df_balanced[label_column]


label_encoder = LabelEncoder()#convrtig labels to numbers
y_encoded = label_encoder.fit_transform(y)
scaler = StandardScaler()#feature scaling
X_scaled = scaler.fit_transform(X)

# Split into train/test 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
#using smote to add some extra samples(balance)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
#priting samples per class
unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(label_encoder.inverse_transform(unique), counts)))


# Training
clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
clf.fit(X_train, y_train)
# Predicting
y_pred = clf.predict(X_test)
print(df['language'].value_counts())
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import joblib
joblib.dump(clf, "svm_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
