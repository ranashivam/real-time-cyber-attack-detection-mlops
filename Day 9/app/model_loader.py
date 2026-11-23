
"""
Load the approved Autoencoder model and perform anomaly predictions.
"""

from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "model"
    / "approved_model.joblib"
)


# Load the approved model bundle and validate its required components.
def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Approved model file not found: {MODEL_PATH}"
        )

    model_bundle = joblib.load(MODEL_PATH)

    if not isinstance(model_bundle, dict):
        raise ValueError(
            "The approved model artifact must contain a dictionary."
        )

    required_keys = {"model", "features", "threshold"}
    missing_keys = required_keys.difference(model_bundle.keys())

    if missing_keys:
        raise ValueError(
            f"Model bundle is missing required keys: {sorted(missing_keys)}"
        )

    if not callable(getattr(model_bundle["model"], "predict", None)):
        raise ValueError("The loaded model does not provide predict().")

    if not model_bundle["features"]:
        raise ValueError("The model feature list is empty.")

    if not np.isfinite(float(model_bundle["threshold"])):
        raise ValueError("The model threshold must be a finite number.")

    return model_bundle


# Calculate the Autoencoder reconstruction error for an input sample.
def calculate_anomaly_score(model, input_data):
    reconstructed_data = model.predict(input_data)

    reconstruction_error = np.mean(
        np.square(input_data - reconstructed_data),
        axis=1
    )

    return float(reconstruction_error[0])


# Validate input features, apply preprocessing, and generate a prediction.
def predict_anomaly(model_bundle, input_features):
    feature_names = model_bundle["features"]

    missing_features = [
        feature for feature in feature_names
        if feature not in input_features
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    ordered_values = [
        float(input_features[feature])
        for feature in feature_names
    ]

    input_array = np.asarray(
        [ordered_values],
        dtype=np.float64
    )

    if not np.isfinite(input_array).all():
        raise ValueError("Input features must contain finite numbers.")

    scaler = model_bundle.get("scaler")

    if scaler is not None:
        input_array = scaler.transform(input_array)

    anomaly_score = calculate_anomaly_score(
        model_bundle["model"],
        input_array
    )

    threshold = float(model_bundle["threshold"])
    is_anomaly = anomaly_score > threshold

    return {
        "prediction": "ANOMALY" if is_anomaly else "NORMAL",
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": anomaly_score,
        "threshold": threshold
    }
