
from pathlib import Path
import json

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


# Configure the project paths and existing MLflow tracking server.
PROJECT_ROOT = Path.cwd()
REGISTRY_DIR = PROJECT_ROOT / "artifacts" / "registry"
REGISTRY_PATH = REGISTRY_DIR / "registry.json"

TRACKING_URI = "http://localhost:5000"

# Map local catalogue names to their native MLflow registered model names.
MODEL_NAMES = {
    "CyberAttack_Isolation_Forest": "Isolation Forest",
    "CyberAttack_Autoencoder": "Autoencoder",
}


# Load the local catalogue containing model versions, run IDs, and validation decisions.
def load_registry():
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Registry catalogue not found: {REGISTRY_PATH}"
        )

    with REGISTRY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


# Find an existing registered model or create it when it does not exist.
def get_or_create_registered_model(client, model_name):
    try:
        return client.get_registered_model(model_name)

    except MlflowException:
        print(f"Creating registered model: {model_name}")
        return client.create_registered_model(model_name)


# Find an existing model version linked to a run to avoid duplicate registration.
def find_existing_version(client, model_name, run_id):
    versions = client.search_model_versions(
        filter_string=f"name='{model_name}'"
    )

    for version in versions:
        if version.run_id == run_id:
            return version

    return None


# Apply validation tags and move approved versions into the Staging stage.
def apply_validation_status(client, model_name, version_number, status):
    client.set_model_version_tag(
        name=model_name,
        version=version_number,
        key="validation_status",
        value=status,
    )

    client.set_model_version_tag(
        name=model_name,
        version=version_number,
        key="project",
        value="Real-Time Cyber-Attack Detection",
    )

    if status == "APPROVED":
        client.transition_model_version_stage(
            name=model_name,
            version=version_number,
            stage="Staging",
            archive_existing_versions=False,
        )

        print("Stage: Staging")

    else:
        print("Stage: None (rejected candidate)")


# Register one existing experiment run as a native MLflow model version.
def register_model_version(client, registered_name, version_record):
    run_id = version_record.get("run_id")
    artifact_path = version_record.get("artifact_path")
    status = version_record.get("validation_status")

    if not run_id:
        raise ValueError(
            f"Missing MLflow run ID for {registered_name}."
        )

    if not artifact_path:
        raise ValueError(
            f"Missing artifact path for {registered_name}."
        )

    if status not in ("APPROVED", "REJECTED"):
        raise ValueError(
            f"Invalid validation status for {registered_name}: {status}"
        )

    # Confirm that the original MLflow run still exists.
    client.get_run(run_id)

    # Build the source URI using the model artifact from the original run.
    model_source = f"runs:/{run_id}/{artifact_path}"

    get_or_create_registered_model(
        client,
        registered_name,
    )

    # Reuse an existing version if this run was already registered.
    existing_version = find_existing_version(
        client,
        registered_name,
        run_id,
    )

    if existing_version is not None:
        version = existing_version
        print(
            f"\nExisting version found: "
            f"{registered_name} v{version.version}"
        )

    else:
        print(f"\nRegistering: {registered_name}")

        version = client.create_model_version(
            name=registered_name,
            source=model_source,
            run_id=run_id,
            description=(
                f"Video 7 model version. "
                f"Validation status: {status}"
            ),
        )

        print(f"Created model version: {version.version}")

    # Record the validation decision on the native model version.
    apply_validation_status(
        client,
        registered_name,
        version.version,
        status,
    )

    print(f"Validation: {status}")
    print(f"Run ID: {run_id}")


# Register all validated models from the local catalogue in MLflow Model Registry.
def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    client = MlflowClient(
        tracking_uri=TRACKING_URI,
        registry_uri=TRACKING_URI,
    )

    registry = load_registry()

    for registered_name, model_name in MODEL_NAMES.items():
        versions = registry.get(registered_name, [])

        if not versions:
            raise ValueError(
                f"No local versions found for {registered_name}."
            )

        # Register the latest locally validated version.
        latest_version = max(
            versions,
            key=lambda item: item["version"],
        )

        print(f"\n{'=' * 55}")
        print(f"Model: {model_name}")

        register_model_version(
            client,
            registered_name,
            latest_version,
        )

    print(f"\n{'=' * 55}")
    print("Native MLflow model registration completed.")
    print("Open the MLflow UI and select the Models tab.")


if __name__ == "__main__":
    main()
