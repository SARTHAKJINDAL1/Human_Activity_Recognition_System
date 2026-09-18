"""
Human Activity Recognition (HAR)
Member 2 - Model Training and Hyperparameter Tuning

This script:
1. Loads the processed UCI HAR training data.
2. Trains and tunes 4 ML models:
   - Logistic Regression
   - Random Forest
   - SVM
   - KNN
3. Uses 3-fold cross-validation on TRAINING data only for tuning.
4. Saves the best version of each model to models/.
5. Saves tuning results to results/tuning_comparison.csv.

IMPORTANT:
- X_test and y_test are intentionally NOT used for tuning.
- Final test-set evaluation is handled by Member 3 in evaluate_models.py.

"""

import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# ============================================================
# STEP 1 - PROJECT PATHS
# ============================================================

DATA_PATH = "dataset/processed"
MODEL_PATH = "models"
RESULT_PATH = "results"

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)


# ============================================================
# STEP 2 - LOAD PROCESSED DATA
# ============================================================

print("=" * 70)
print("HUMAN ACTIVITY RECOGNITION - MODEL TRAINING")
print("Member 2: MANAS SAINI (25BCE10183) : ML Model ")
print("=" * 70)

print("\nLoading processed dataset...")

X_train = pd.read_csv(f"{DATA_PATH}/X_train.csv")
y_train = pd.read_csv(f"{DATA_PATH}/y_train.csv").squeeze()

# Loaded only for verification. They are NOT used during tuning.
X_test = pd.read_csv(f"{DATA_PATH}/X_test.csv")
y_test = pd.read_csv(f"{DATA_PATH}/y_test.csv").squeeze()

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"y_test shape : {y_test.shape}")

print("\nTest data will remain untouched during hyperparameter tuning.")


# ============================================================
# STEP 3 - ACTIVITY LABELS
# ============================================================

activity_names = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}


# ============================================================
# STEP 4 - LOGISTIC REGRESSION TUNING
# ============================================================

print("\n" + "=" * 60)
print("1/4 - LOGISTIC REGRESSION TUNING")
print("=" * 60)

logistic_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

logistic_param_grid = {
    "model__C": [0.1, 1, 10]
}

logistic_grid = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid=logistic_param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

logistic_grid.fit(X_train, y_train)

logistic_tuned_model = logistic_grid.best_estimator_

print("\nBest Logistic Regression parameters:")
print(logistic_grid.best_params_)
print(
    f"Best CV accuracy: "
    f"{logistic_grid.best_score_ * 100:.2f}%"
)

joblib.dump(
    logistic_tuned_model,
    f"{MODEL_PATH}/logistic_regression.pkl"
)

print("Saved: models/logistic_regression.pkl")


# ============================================================
# STEP 5 - RANDOM FOREST TUNING
# ============================================================

print("\n" + "=" * 60)
print("2/4 - RANDOM FOREST TUNING")
print("=" * 60)

random_forest_base = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

random_forest_param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [None, 20],
    "min_samples_split": [2, 5]
}

random_forest_grid = GridSearchCV(
    estimator=random_forest_base,
    param_grid=random_forest_param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

random_forest_grid.fit(X_train, y_train)

random_forest_tuned_model = random_forest_grid.best_estimator_

print("\nBest Random Forest parameters:")
print(random_forest_grid.best_params_)
print(
    f"Best CV accuracy: "
    f"{random_forest_grid.best_score_ * 100:.2f}%"
)

joblib.dump(
    random_forest_tuned_model,
    f"{MODEL_PATH}/random_forest.pkl"
)

print("Saved: models/random_forest.pkl")


# ============================================================
# STEP 6 - SVM TUNING
# ============================================================

print("\n" + "=" * 60)
print("3/4 - SVM TUNING")
print("=" * 60)

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ))
])

svm_param_grid = {
    "model__C": [1, 10, 100],
    "model__gamma": ["scale", 0.01, 0.001]
}

svm_grid = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=svm_param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

svm_grid.fit(X_train, y_train)

svm_tuned_model = svm_grid.best_estimator_

print("\nBest SVM parameters:")
print(svm_grid.best_params_)
print(
    f"Best CV accuracy: "
    f"{svm_grid.best_score_ * 100:.2f}%"
)

joblib.dump(
    svm_tuned_model,
    f"{MODEL_PATH}/svm.pkl"
)

print("Saved: models/svm.pkl")


# ============================================================
# STEP 7 - KNN TUNING
# ============================================================

print("\n" + "=" * 60)
print("4/4 - KNN TUNING")
print("=" * 60)

knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(
        n_jobs=-1
    ))
])

knn_param_grid = {
    "model__n_neighbors": [3, 5, 7, 9],
    "model__weights": ["uniform", "distance"]
}

knn_grid = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=knn_param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

knn_grid.fit(X_train, y_train)

knn_tuned_model = knn_grid.best_estimator_

print("\nBest KNN parameters:")
print(knn_grid.best_params_)
print(
    f"Best CV accuracy: "
    f"{knn_grid.best_score_ * 100:.2f}%"
)

joblib.dump(
    knn_tuned_model,
    f"{MODEL_PATH}/knn.pkl"
)

print("Saved: models/knn.pkl")


# ============================================================
# STEP 8 - CREATE TUNING COMPARISON TABLE
# ============================================================

print("\n" + "=" * 60)
print("CREATING TUNING COMPARISON")
print("=" * 60)

tuning_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "SVM",
        "KNN"
    ],
    "Best CV Accuracy (%)": [
        logistic_grid.best_score_ * 100,
        random_forest_grid.best_score_ * 100,
        svm_grid.best_score_ * 100,
        knn_grid.best_score_ * 100
    ]
})

tuning_results = tuning_results.sort_values(
    by="Best CV Accuracy (%)",
    ascending=False
).reset_index(drop=True)

tuning_results.to_csv(
    f"{RESULT_PATH}/tuning_comparison.csv",
    index=False
)

print("\nTuning comparison:")
print(tuning_results.to_string(index=False))

print("\nSaved: results/tuning_comparison.csv")


# ============================================================
# STEP 9 - FINAL MEMBER 2 SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MEMBER 2 MODEL TRAINING COMPLETED")
print("=" * 60)

print("\nSaved models:")
print("1. models/logistic_regression.pkl")
print("2. models/random_forest.pkl")
print("3. models/svm.pkl")
print("4. models/knn.pkl")

print("\nSaved results:")
print("5. results/tuning_comparison.csv")

print("\nIMPORTANT:")
print("- X_test/y_test were not used for tuning.")
print("- Member 3 will perform the final evaluation.")
print("- Member 3 will select the final best model.")
