"""
HUMAN ACTIVITY RECOGNITION
Member 3 - Model Evaluation

This file:
1. Imports evaluation libraries
2. Creates result folders
3. Loads test dataset
4. Verifies activity labels
5. Loads all 4 trained models
6. Generates prediction dictionary
7. Generates probability dictionary
8. Calculates Accuracy
9. Calculates Precision, Recall and F1-Score
10. Calculates AUC-ROC
11. Saves evaluation tables
12. Creates and saves confusion matrices
13. Creates one ROC curve containing all 4 models
14. Creates separate classification reports
15. Creates activity-wise F1 comparison
16. Creates activity winner table
17. Calculates the best model
18. Saves the best model as best_model.pkl

NO MODEL TRAINING OR TUNING IS DONE HERE.
"""


# ============================================================
# STEP 1 - IMPORT EVALUATION LIBRARIES
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize


print("=" * 70)
print("HUMAN ACTIVITY RECOGNITION - MEMBER 3 EVALUATION")
print("=" * 70)


# ============================================================
# STEP 2 - CREATE RESULT FOLDERS
# ============================================================

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(
    os.path.join(RESULTS_DIR, "confusion_matrices"),
    exist_ok=True
)

os.makedirs(
    os.path.join(RESULTS_DIR, "roc_curves"),
    exist_ok=True
)

os.makedirs(
    os.path.join(RESULTS_DIR, "classification_reports"),
    exist_ok=True
)

print("\nResult folders created/verified successfully.")


# ============================================================
# STEP 3 - LOAD TEST DATASET
# ============================================================

X_TEST_PATH = "dataset/processed/X_test.csv"
Y_TEST_PATH = "dataset/processed/y_test.csv"

print("\nLoading test dataset...")

X_test = pd.read_csv(X_TEST_PATH)

y_test = pd.read_csv(Y_TEST_PATH).squeeze()

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("\nTest dataset loaded successfully.")


# ============================================================
# STEP 4 - VERIFY ACTIVITY LABELS
# ============================================================

activity_names = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

classes = list(activity_names.keys())

class_names = list(activity_names.values())


print("\nActivity labels:")

for activity_id, activity_name in activity_names.items():
    print(f"{activity_id} -> {activity_name}")


print("\nLabels found in test dataset:")

print(sorted(y_test.unique().tolist()))


# Verify labels

if not set(y_test.unique()).issubset(set(classes)):

    raise ValueError(
        "Unexpected activity label found in y_test."
    )


print("\nActivity labels verified successfully.")


# ============================================================
# STEP 5 - LOAD ALL 4 TRAINED MODELS
# ============================================================

model_paths = {

    "Logistic Regression":
        "models/logistic_regression.pkl",

    "Random Forest":
        "models/random_forest.pkl",

    "SVM":
        "models/svm.pkl",

    "KNN":
        "models/knn.pkl"
}


models = {}


print("\n" + "=" * 70)
print("LOADING TRAINED MODELS")
print("=" * 70)


for name, path in model_paths.items():

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    models[name] = joblib.load(path)

    print(f"{name} loaded successfully.")


print("\nAll 4 trained models loaded successfully.")


# ============================================================
# STEP 6 - GENERATE PREDICTION DICTIONARY
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)


predictions = {}


for name, model in models.items():

    predictions[name] = model.predict(X_test)

    print(
        f"{name}: "
        f"{predictions[name].shape}"
    )


print("\nPrediction dictionary generated successfully.")


# ============================================================
# STEP 7 - GENERATE PROBABILITY DICTIONARY
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PROBABILITY PREDICTIONS")
print("=" * 70)


probabilities = {}


for name, model in models.items():

    if not hasattr(model, "predict_proba"):

        raise AttributeError(
            f"{name} does not support predict_proba()."
        )

    probabilities[name] = model.predict_proba(X_test)

    print(
        f"{name}: "
        f"{probabilities[name].shape}"
    )


print("\nProbability dictionary generated successfully.")


# ============================================================
# STEP 8 - ACCURACY
# ============================================================

print("\n" + "=" * 70)
print("ACCURACY")
print("=" * 70)


accuracy_results = {}


for name in models:

    accuracy = accuracy_score(
        y_test,
        predictions[name]
    )

    accuracy_results[name] = accuracy

    print(
        f"{name}: "
        f"{accuracy * 100:.2f}%"
    )


# ============================================================
# STEP 9 - PRECISION, RECALL AND F1-SCORE
# ============================================================

print("\n" + "=" * 70)
print("PRECISION, RECALL AND F1-SCORE")
print("=" * 70)


evaluation_results = []


for name in models:

    y_pred = predictions[name]

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    evaluation_results.append({

        "Model": name,

        "Accuracy":
            accuracy_results[name] * 100,

        "Precision":
            precision * 100,

        "Recall":
            recall * 100,

        "F1-Score":
            f1 * 100
    })


    print(f"\n{name}")

    print(
        f"Precision: "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1-Score: "
        f"{f1 * 100:.2f}%"
    )


# ============================================================
# STEP 10 - AUC-ROC SCORE
# ============================================================

print("\n" + "=" * 70)
print("AUC-ROC")
print("=" * 70)


# Convert actual labels to binary format

y_test_binary = label_binarize(
    y_test,
    classes=classes
)


for result in evaluation_results:

    name = result["Model"]

    auc_score = roc_auc_score(
        y_test_binary,
        probabilities[name],
        multi_class="ovr",
        average="macro"
    )

    result["AUC-ROC"] = auc_score * 100

    print(
        f"{name}: "
        f"{auc_score * 100:.2f}%"
    )


# ============================================================
# STEP 11 - SAVE MAIN EVALUATION TABLE
# ============================================================

evaluation_df = pd.DataFrame(
    evaluation_results
)


evaluation_df = evaluation_df.round(2)


print("\n" + "=" * 70)
print("FINAL MODEL EVALUATION TABLE")
print("=" * 70)

print(
    evaluation_df.to_string(index=False)
)


evaluation_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "model_evaluation.csv"
    ),
    index=False
)


print(
    "\nSaved:"
    " results/model_evaluation.csv"
)


# ============================================================
# STEP 12 - MODEL RANKING
# ============================================================

print("\n" + "=" * 70)
print("MODEL RANKING")
print("=" * 70)


ranking_df = evaluation_df.copy()


ranking_df["Average Score"] = ranking_df[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "AUC-ROC"
    ]
].mean(axis=1)


ranking_df = ranking_df.sort_values(
    "Average Score",
    ascending=False
).reset_index(drop=True)


ranking_df.insert(
    0,
    "Rank",
    range(1, len(ranking_df) + 1)
)


ranking_df = ranking_df.round(2)


print(
    ranking_df.to_string(index=False)
)


ranking_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "final_model_comparison.csv"
    ),
    index=False
)


print(
    "\nSaved:"
    " results/final_model_comparison.csv"
)


# ============================================================
# STEP 13 - CONFUSION MATRIX FOR ALL 4 MODELS
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRICES")
print("=" * 70)


for name in models:

    y_pred = predictions[name]


    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=classes
    )


    safe_name = (
        name.lower()
        .replace(" ", "_")
    )


    # -----------------------------
    # Save CSV
    # -----------------------------

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )


    cm_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "confusion_matrices",
            f"{safe_name}_confusion_matrix.csv"
        )
    )


    # -----------------------------
    # Save PNG
    # -----------------------------

    plt.figure(
        figsize=(9, 7)
    )


    plt.imshow(
        cm,
        interpolation="nearest"
    )


    plt.title(
        f"{name} - Confusion Matrix"
    )


    plt.colorbar()


    tick_marks = np.arange(
        len(class_names)
    )


    plt.xticks(
        tick_marks,
        class_names,
        rotation=45,
        ha="right"
    )


    plt.yticks(
        tick_marks,
        class_names
    )


    threshold = cm.max() / 2


    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                )
            )


    plt.ylabel(
        "Actual Activity"
    )

    plt.xlabel(
        "Predicted Activity"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "confusion_matrices",
            f"{safe_name}_confusion_matrix.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"{name}: CSV and PNG saved."
    )


# ============================================================
# STEP 14 - ROC CURVE OF ALL 4 MODELS
# ============================================================

print("\n" + "=" * 70)
print("ROC CURVE COMPARISON")
print("=" * 70)


plt.figure(
    figsize=(10, 8)
)


for name in models:

    proba = probabilities[name]


    fpr = {}
    tpr = {}


    # ROC for each activity

    for i in range(len(classes)):

        fpr[i], tpr[i], _ = roc_curve(
            y_test_binary[:, i],
            proba[:, i]
        )


    # Combine all class curves

    all_fpr = np.unique(
        np.concatenate(
            [
                fpr[i]
                for i in range(len(classes))
            ]
        )
    )


    mean_tpr = np.zeros_like(
        all_fpr
    )


    for i in range(len(classes)):

        mean_tpr += np.interp(
            all_fpr,
            fpr[i],
            tpr[i]
        )


    mean_tpr /= len(classes)


    macro_auc = auc(
        all_fpr,
        mean_tpr
    )


    plt.plot(
        all_fpr,
        mean_tpr,
        label=(
            f"{name} "
            f"(AUC = {macro_auc:.4f})"
        )
    )


# Random classifier

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Multiclass ROC Curve Comparison"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "roc_curves",
        "roc_curve_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.show()

plt.close()


print(
    "Saved:"
    " results/roc_curves/roc_curve_comparison.png"
)


# ============================================================
# STEP 15 - CLASSIFICATION REPORTS FOR ALL 4 MODELS
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORTS")
print("=" * 70)


for name in models:

    report = classification_report(
        y_test,
        predictions[name],
        labels=classes,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )


    report_df = pd.DataFrame(
        report
    ).transpose()


    safe_name = (
        name.lower()
        .replace(" ", "_")
    )


    report_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "classification_reports",
            f"{safe_name}_report.csv"
        )
    )


    print(
        f"{name}: report saved."
    )


# ============================================================
# STEP 16 - ACTIVITY-WISE F1 COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("ACTIVITY-WISE F1-SCORE COMPARISON")
print("=" * 70)


class_f1_comparison = pd.DataFrame(
    index=class_names
)


for name in models:

    report = classification_report(
        y_test,
        predictions[name],
        labels=classes,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )


    class_f1_comparison[name] = [

        report[activity]["f1-score"]

        for activity in class_names

    ]


class_f1_comparison.index.name = "Activity"


# Convert to percentage

class_f1_comparison_percent = (
    class_f1_comparison * 100
).round(2)


print(
    class_f1_comparison_percent
)


class_f1_comparison_percent.to_csv(
    os.path.join(
        RESULTS_DIR,
        "class_f1_comparison.csv"
    )
)


print(
    "\nSaved:"
    " results/class_f1_comparison.csv"
)


# ============================================================
# STEP 17 - ACTIVITY WINNER TABLE
# ============================================================

print("\n" + "=" * 70)
print("ACTIVITY WINNER TABLE")
print("=" * 70)


best_model_per_activity = (
    class_f1_comparison.idxmax(
        axis=1
    )
)


best_f1_per_activity = (
    class_f1_comparison.max(
        axis=1
    ) * 100
).round(2)


activity_winners = pd.DataFrame({

    "Activity":
        class_names,

    "Best Model":
        best_model_per_activity.values,

    "Best F1-Score":
        best_f1_per_activity.values
})


print(
    activity_winners.to_string(
        index=False
    )
)


activity_winners.to_csv(
    os.path.join(
        RESULTS_DIR,
        "activity_winners.csv"
    ),
    index=False
)


print(
    "\nSaved:"
    " results/activity_winners.csv"
)


# ============================================================
# STEP 18 - CALCULATE BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("BEST MODEL CALCULATION")
print("=" * 70)


best_model_name = ranking_df.iloc[0]["Model"]


best_model_score = (
    ranking_df.iloc[0]["Average Score"]
)


print(
    f"Best Model: {best_model_name}"
)


print(
    f"Average Evaluation Score: "
    f"{best_model_score:.2f}%"
)


# Get complete performance

best_row = evaluation_df[
    evaluation_df["Model"]
    == best_model_name
].iloc[0]


print("\nBest model performance:")


print(
    f"Accuracy : "
    f"{best_row['Accuracy']:.2f}%"
)


print(
    f"Precision: "
    f"{best_row['Precision']:.2f}%"
)


print(
    f"Recall   : "
    f"{best_row['Recall']:.2f}%"
)


print(
    f"F1-Score : "
    f"{best_row['F1-Score']:.2f}%"
)


print(
    f"AUC-ROC  : "
    f"{best_row['AUC-ROC']:.2f}%"
)


# ============================================================
# STEP 19 - SAVE BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING BEST MODEL")
print("=" * 70)


best_model = models[
    best_model_name
]


joblib.dump(
    best_model,
    "models/best_model.pkl"
)


print(
    "Best model saved successfully!"
)


print(
    f"Selected model: "
    f"{best_model_name}"
)


print(
    "Location: "
    "models/best_model.pkl"
)


# ============================================================
# STEP 20 - FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MEMBER 3 EVALUATION COMPLETED")
print("=" * 70)


print("\nGenerated:")

print("1. model_evaluation.csv")
print("2. final_model_comparison.csv")
print("3. Confusion matrices - CSV + PNG")
print("4. Combined ROC curve - PNG")
print("5. Classification reports - 4 CSV files")
print("6. class_f1_comparison.csv")
print("7. activity_winners.csv")
print("8. best_model.pkl")


print(
    f"\nFINAL BEST MODEL: "
    f"{best_model_name}"
)

print("=" * 70)