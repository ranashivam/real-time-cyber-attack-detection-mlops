
from pathlib import Path
import json

import joblib
import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient


# Configure the registry record, test dataset, and existing MLflow server.
PROJECT_ROOT = Path.cwd()

REGISTRY_DIR = PROJECT_ROOT / "artifacts" / "registry"
APPROVED_MODEL_PATH = REGISTRY_DIR / "approved_model.json"

TEST_DATA_PATH = (
    PROJECT_ROOT / "data" / "generated" / "v1.0.0" / "test.csv"
)

TRACKING_URI = "http://localhost:5000"

# Map registered model names to their model types and display names.
MODEL_CONFIG = {
    "CyberAttack_Autoencoder": {
        "display_name": "Autoencoder",
        "kind": "autoencoder",
    },
    "CyberAttack_Isolation_Forest": {
        "display_name": "Isolation Forest",
        "kind": "isolation_forest",
    },
}


# Load the approval record created by promote_model.py.
def load_approval_record():
    if not APPROVED_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Approved model record not found. "
            "Run promote_model.py first."
        )

    with APPROVED_MODEL_PATH.open("r", encoding="utf-8") as file:
        approval = json.load(file)

    if approval.get("approval_status") != "APPROVED":
        raise ValueError("The selected model is not approved.")

    return approval


# Retrieve the exact registered model version from the MLflow Registry.
def get_registered_version(client, registered_name, version_number):
    return client.get_model_version(
        name=registered_name,
        version=str(version_number),
    )


# Download the artifact associated with the registered model version.
def download_model_artifact(model_version):
    artifact_uri = model_version.source

    print(f"Artifact source: {artifact_uri}")

    downloaded_path = mlflow.artifacts.download_artifacts(
        artifact_uri=artifact_uri,
        tracking_uri=TRACKING_URI,
    )

    artifact_path = Path(downloaded_path)

    # Accept either a direct joblib file or a directory containing one.
    if artifact_path.is_file():
        if artifact_path.suffix == ".joblib":
            return artifact_path

        raise ValueError(
            f"Expected a .joblib artifact, received: {artifact_path}"
        )

    joblib_files = list(artifact_path.rglob("*.joblib"))

    if not joblib_files:
        raise FileNotFoundError(
            f"No .joblib model artifact found in: {artifact_path}"
        )

    if len(joblib_files) > 1:
        raise ValueError(
            "Multiple .joblib files found. "
            "The registered artifact must identify one model bundle."
        )

    return joblib_files[0]


# Load and validate the saved model bundle.
def load_model_bundle(artifact_path):
    bundle = joblib.load(artifact_path)

    if not isinstance(bundle, dict):
        raise TypeError(
            "Expected the saved model artifact to be a dictionary."
        )

    required_keys = ["model", "features", "threshold"]

    missing = [
        key for key in required_keys
        if bundle.get(key) is None
    ]

    if missing:
        raise KeyError(
            f"Model bundle is missing required entries: {missing}"
        )

    return bundle


# Load one test record using the features expected by the trained model.
def load_test_sample(features):
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DATA_PATH}"
        )

    test_data = pd.read_csv(TEST_DATA_PATH)

    missing = [
        feature for feature in features
        if feature not in test_data.columns
    ]

    if missing:
        raise ValueError(
            f"Test dataset is missing features: {missing}"
        )

    if test_data.empty:
        raise ValueError("The test dataset contains no records.")

    return test_data[features].head(1).copy()


# Calculate the anomaly score and prediction using the selected model.
def predict_anomaly(bundle, model_kind, sample):
    model = bundle["model"]
    scaler = bundle.get("scaler")
    features = bundle["features"]
    threshold = float(bundle["threshold"])

    input_data = sample[features].copy()

    if scaler is not None:
        transformed_data = scaler.transform(input_data)
    else:
        transformed_data = input_data

    if model_kind == "isolation_forest":
        scores = -model.decision_function(transformed_data)

    elif model_kind == "autoencoder":
        reconstructed = model.predict(transformed_data)
        scores = (
            (transformed_data - reconstructed) ** 2
        ).mean(axis=1)

    else:
        raise ValueError(f"Unsupported model type: {model_kind}")

    anomaly_score = float(scores[0])
    prediction = int(anomaly_score >= threshold)

    return anomaly_score, prediction, threshold


# Load the approved registered version and perform one sample inference.
def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    client = MlflowClient(
        tracking_uri=TRACKING_URI,
        registry_uri=TRACKING_URI,
    )

    approval = load_approval_record()

    registered_name = approval["registered_name"]
    version_number = approval["version"]

    if registered_name not in MODEL_CONFIG:
        raise ValueError(
            f"Unsupported registered model: {registered_name}"
        )

    config = MODEL_CONFIG[registered_name]

    print("\nLoading approved model...")
    print(f"Model: {config['display_name']}")
    print(f"Registry name: {registered_name}")
    print(f"Version: {version_number}")

    model_version = get_registered_version(
        client,
        registered_name,
        version_number,
    )

    if model_version.tags.get("validation_status") != "APPROVED":
        raise ValueError(
            "The registered model version does not have "
            "an APPROVED validation tag."
        )

    artifact_path = download_model_artifact(model_version)
    bundle = load_model_bundle(artifact_path)

    sample = load_test_sample(bundle["features"])

    anomaly_score, prediction, threshold = predict_anomaly(
        bundle,
        config["kind"],
        sample,
    )

    print("\nSample network traffic:")
    print(sample.to_string(index=False))

    print("\nInference result:")
    print(f"Anomaly score: {anomaly_score:.6f}")
    print(f"Decision threshold: {threshold:.6f}")

    if prediction == 1:
        print("Prediction: ANOMALY DETECTED")
    else:
        print("Prediction: NORMAL TRAFFIC")

    print("\nApproved model inference completed successfully.")


if __name__ == "__main__":
    main()
