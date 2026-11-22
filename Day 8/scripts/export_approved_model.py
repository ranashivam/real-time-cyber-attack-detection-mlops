
"""
Export the approved Autoencoder model from its MLflow run
into the Day 8 local artifacts directory.
"""

import json
import shutil
from pathlib import Path

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAY7_ROOT = PROJECT_ROOT.parent / "Day 7"

APPROVAL_FILE = (
    DAY7_ROOT / "artifacts" / "registry" / "approved_model.json"
)

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "model"
OUTPUT_FILE = OUTPUT_DIR / "approved_model.joblib"

TRACKING_URI = "http://localhost:5000"

# This is the run ID confirmed from the MLflow Version 3 details.
APPROVED_RUN_ID = "e6063970034e420cb37385352868d8d9"


# Load the approval record and verify that it represents an approved model.
def load_approval_record():
    if not APPROVAL_FILE.exists():
        raise FileNotFoundError(
            f"Approval file not found: {APPROVAL_FILE}"
        )

    with APPROVAL_FILE.open("r", encoding="utf-8") as file:
        approval = json.load(file)

    if approval.get("approval_status") != "APPROVED":
        raise ValueError("The model is not marked as APPROVED.")

    return approval


# Download the Autoencoder artifact from the confirmed MLflow run.
def download_model_artifact():
    artifact_uri = (
        f"runs:/{APPROVED_RUN_ID}/trained_model/autoencoder.joblib"
    )

    print(f"Downloading from MLflow run: {APPROVED_RUN_ID}")
    print(f"Artifact URI: {artifact_uri}")

    downloaded_path = mlflow.artifacts.download_artifacts(
        artifact_uri=artifact_uri
    )

    downloaded_file = Path(downloaded_path)

    if not downloaded_file.is_file():
        raise FileNotFoundError(
            f"Downloaded model file not found: {downloaded_file}"
        )

    return downloaded_file


# Copy the downloaded model into the Day 8 artifacts directory.
def export_approved_model():
    load_approval_record()

    source_file = download_model_artifact()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_file, OUTPUT_FILE)

    print("\nModel exported successfully.")
    print(f"Source: {source_file}")
    print(f"Destination: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size:,} bytes")


# Configure MLflow and execute the model export.
def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    export_approved_model()


if __name__ == "__main__":
    main()
