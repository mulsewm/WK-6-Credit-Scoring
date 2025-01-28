# scripts/api.py

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the best model at startup
MODEL_PATH = "./models/RandomForest.pkl"  # or "LogisticRegression.pkl"
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the API!"

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON input with feature values.
    Example:
    {
      "feature1": 3.2,
      "feature2": 5,
      "feature3": 10.1
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # Convert JSON to a pandas DataFrame (single row)
    # The order of columns must match what the model expects
    # Adjust 'feature_cols' to match your actual training columns
    feature_cols = ["R", "F", "M", "some_other_feature"]  # Example
    input_df = pd.DataFrame([data], columns=feature_cols)

    # Perform prediction
    prediction = model.predict(input_df)[0]  # 0 or 1 for Good vs Bad
    probability = model.predict_proba(input_df)[0, 1]

    response = {
        "prediction": int(prediction),
        "probability_of_good": float(probability)
    }

    return jsonify(response)

if __name__ == "__main__":
    # Run Flask server
    app.run(host="0.0.0.0", port=8000, debug=True)
