import sys
import os

# Add src folder to Python path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    )
)

from predict import predict_activity, features


# -----------------------------
# Display heading
# -----------------------------
print()
print("==========================================")
print("     HUMAN ACTIVITY RECOGNITION")
print("     Interactive Prediction Application")
print("==========================================")
print()

print("This application uses 5 selected")
print("features from the HAR dataset.")
print()

print("Features used:")

for i, feature in enumerate(features, start=1):
    print(f"{i}. {feature}")

print()


# -----------------------------
# Main iterative loop
# -----------------------------
while True:

    print("------------------------------------------")
    print("Enter values for a new prediction")
    print("------------------------------------------")

    values = []

    # Take 5 inputs
    for i, feature in enumerate(features, start=1):

        while True:

            try:
                value = float(
                    input(f"Enter value for Feature {i}: ")
                )

                values.append(value)
                break

            except ValueError:
                print("Invalid input.")
                print("Please enter a numerical value.")


    # -----------------------------
    # Make prediction
    # -----------------------------
    try:

        activity = predict_activity(values)

        print()
        print("==========================================")
        print("         PREDICTION RESULT")
        print("==========================================")
        print(f"Predicted Activity: {activity}")
        print("==========================================")
        print()

    except Exception as e:

        print()
        print("Prediction failed.")
        print("Error:", e)
        print()


    # -----------------------------
    # Ask for another prediction
    # -----------------------------
    again = input(
        "Do you want to make another prediction? (y/n): "
    ).strip().lower()

    if again != "y":
        print()
        print("Application closed.")
        print("Thank you!")
        break

    print()