
from pathlib import Path

import joblib
import numpy as np


# Define the location of the exported model artifact.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model" / "approved_model.joblib"


# Load and validate the approved Autoencoder model bundle.
def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Approved model file not found: {MODEL_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict):
        raise ValueError("The model artifact must contain a dictionary bundle.")

    required_keys = {"model", "features", "threshold"}

    missing_keys = required_keys - set(bundle.keys())

    if missing_keys:
        raise ValueError(
            f"Model bundle is missing required keys: {missing_keys}"
        )

    model = bundle["model"]
    features = bundle["features"]
    threshold = float(bundle["threshold"])
    scaler = bundle.get("scaler")

    if not features:
        raise ValueError("The model feature list is empty.")

    return {
        "model": model,
        "features": features,
        "threshold": threshold,
        "scaler": scaler
    }


# Calculate the reconstruction error produced by the Autoencoder.
def calculate_anomaly_score(model, input_data):
    reconstructed_data = model.predict(input_data)

    reconstruction_error = np.mean(
        np.square(input_data - reconstructed_data),
        axis=1
    )

    return float(reconstruction_error[0])


# Validate input features, apply preprocessing, and predict an anomaly.
def predict_anomaly(model_bundle, input_features):
    features = model_bundle["features"]

    if not isinstance(input_features, dict):
        raise ValueError("Input features must be provided as a JSON object.")

    missing_features = [
        feature for feature in features
        if feature not in input_features
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    try:
        values = [
            float(input_features[feature])
            for feature in features
        ]
    except (TypeError, ValueError) as error:
        raise ValueError("All feature values must be numeric.") from error

    if not np.all(np.isfinite(values)):
        raise ValueError("Feature values must be finite numbers.")

    input_array = np.asarray(values, dtype=np.float64).reshape(1, -1)

    scaler = model_bundle["scaler"]

    if scaler is not None:
        input_array = scaler.transform(input_array)

    score = calculate_anomaly_score(
        model_bundle["model"],
        input_array
    )

    threshold = model_bundle["threshold"]

    is_anomaly = score > threshold

    return {
        "prediction": "ANOMALY" if is_anomaly else "NORMAL",
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": score,
        "threshold": threshold
    }
