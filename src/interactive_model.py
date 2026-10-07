import pandas as pd
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# File paths
# -----------------------------
X_train_path = "dataset/processed/X_train.csv"
y_train_path = "dataset/processed/y_train.csv"

X_test_path = "dataset/processed/X_test.csv"
y_test_path = "dataset/processed/y_test.csv"

model_path = "models/interactive_model.pkl"


# -----------------------------
# Load dataset
# -----------------------------
print("Loading dataset...")

X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path).squeeze()

X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).squeeze()

print("Dataset loaded successfully.")
print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)


# -----------------------------
# Select 5 features
# -----------------------------
# We use the first 5 features from the dataset.
# The actual feature names are stored with the model.

selected_features = list(X_train.columns[:5])

print("\nFeatures selected for interactive model:")

for i, feature in enumerate(selected_features, start=1):
    print(f"{i}. {feature}")


# -----------------------------
# Select only these features
# -----------------------------
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]


# -----------------------------
# Create SVM model
# -----------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        C=10,
        gamma=0.001
    ))
])


# -----------------------------
# Train model
# -----------------------------
print("\nTraining interactive SVM model...")

model.fit(X_train_selected, y_train)

print("Training completed.")


# -----------------------------
# Evaluate model
# -----------------------------
y_pred = model.predict(X_test_selected)

accuracy = accuracy_score(y_test, y_pred)

print("\n--------------------------------")
print("INTERACTIVE MODEL RESULTS")
print("--------------------------------")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------
# Save model + feature names
# -----------------------------
model_bundle = {
    "model": model,
    "features": selected_features
}

os.makedirs("models", exist_ok=True)

joblib.dump(model_bundle, model_path)

print("--------------------------------")
print("Model saved successfully!")
print(f"Location: {model_path}")
print("--------------------------------")