import joblib
import pandas as pd


# -----------------------------
# Load interactive model
# -----------------------------
MODEL_PATH = "models/interactive_model.pkl"

model_bundle = joblib.load(MODEL_PATH)

model = model_bundle["model"]
features = model_bundle["features"]


# -----------------------------
# Activity names
# -----------------------------
ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}


# -----------------------------
# Prediction function
# -----------------------------
def predict_activity(values):

    # Check number of values
    if len(values) != 5:
        raise ValueError("Exactly 5 values are required.")

    # Convert input values into DataFrame
    input_data = pd.DataFrame(
        [values],
        columns=features
    )

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Convert numeric label to activity name
    try:
        prediction = int(prediction)
        activity = ACTIVITY_NAMES.get(
            prediction,
            str(prediction)
        )
    except (ValueError, TypeError):
        activity = str(prediction)

    return activity