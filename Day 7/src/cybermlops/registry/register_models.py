
from datetime import datetime, timezone
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
EXPERIMENT_NAME = "Cyber Attack Anomaly Detection"

# Map each registered model to its MLflow run name and artifact filename.
MODEL_CONFIG = {
    "CyberAttack_Isolation_Forest": {
        "model_name": "Isolation Forest",
        "model_file": "isolation_forest.joblib",
    },
    "CyberAttack_Autoencoder": {
        "model_name": "Autoencoder",
        "model_file": "autoencoder.joblib",
    },
}


# Load the local catalogue containing all previously registered versions.
def load_registry():
    if not REGISTRY_PATH.exists():
        return {}

    with REGISTRY_PATH.open("r", encoding="utf-8") as file:
        registry = json.load(file)

    if not isinstance(registry, dict):
        raise ValueError("The registry catalogue must contain a JSON object.")

    return registry


# Save the complete catalogue while preserving all previous model versions.
def save_registry(registry):
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    with REGISTRY_PATH.open("w", encoding="utf-8") as file:
        json.dump(registry, file, indent=4)


# Find an existing registered model or create it if it does not exist.
def get_or_create_registered_model(client, registered_name):
    try:
        return client.get_registered_model(registered_name)

    except MlflowException:
        print(f"Creating registered model: {registered_name}")
        return client.create_registered_model(registered_name)


# Find an existing registered version associated with the specified MLflow run.
def find_existing_version(client, registered_name, run_id):
    versions = client.search_model_versions(
        filter_string=f"name='{registered_name}'"
    )

    for version in versions:
        if version.run_id == run_id:
            return version

    return None


# Find the newest completed MLflow run for a specific model using its model tag.
def find_latest_model_run(client, experiment_id, model_name):
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.model = '{model_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )

    for run in runs:
        if run.info.status == "FINISHED":
            return run

    raise RuntimeError(
        f"No completed MLflow run was found for {model_name}. "
        "Run the MLflow tracking script first."
    )


# Add a new version record without replacing existing catalogue entries.
def add_catalogue_version(
    registry,
    registered_name,
    model_name,
    model_version,
    run_id,
    artifact_path,
):
    versions = registry.setdefault(registered_name, [])

    existing_record = next(
        (
            item for item in versions
            if str(item.get("version")) == str(model_version)
        ),
        None,
    )

    if existing_record is not None:
        existing_record["run_id"] = run_id
        existing_record["artifact_path"] = artifact_path
        return

    versions.append(
        {
            "version": int(model_version),
            "model_name": model_name,
            "run_id": run_id,
            "artifact_path": artifact_path,
            "status": "Pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "validation_status": "PENDING",
        }
    )

    versions.sort(key=lambda item: int(item["version"]))


# Register the latest experiment run as a new native MLflow model version.
def register_model_version(
    client,
    registry,
    registered_name,
    model_name,
    model_file,
    experiment_id,
):
    run = find_latest_model_run(
        client,
        experiment_id,
        model_name,
    )

    run_id = run.info.run_id

    # The tracking script logs model files under this exact artifact directory.
    artifact_path = f"trained_model/{model_file}"
    model_source = f"runs:/{run_id}/{artifact_path}"

    get_or_create_registered_model(
        client,
        registered_name,
    )

    # Avoid creating duplicate versions when the same run is registered again.
    existing_version = find_existing_version(
        client,
        registered_name,
        run_id,
    )

    if existing_version is not None:
        model_version = existing_version

        print(
            f"\nExisting version found: "
            f"{registered_name} v{model_version.version}"
        )

    else:
        print(f"\nRegistering new version: {registered_name}")
        print(f"MLflow run ID: {run_id}")
        print(f"Artifact source: {model_source}")

        model_version = client.create_model_version(
            name=registered_name,
            source=model_source,
            run_id=run_id,
            description=(
                f"New {model_name} model version "
                f"from MLflow experiment run {run_id}."
            ),
        )

        print(f"Created model version: {model_version.version}")

    # Mark the newly registered candidate as pending validation.
    client.set_model_version_tag(
        name=registered_name,
        version=str(model_version.version),
        key="project",
        value="Real-Time Cyber-Attack Detection",
    )

    client.set_model_version_tag(
        name=registered_name,
        version=str(model_version.version),
        key="validation_status",
        value="PENDING",
    )

    add_catalogue_version(
        registry=registry,
        registered_name=registered_name,
        model_name=model_name,
        model_version=model_version.version,
        run_id=run_id,
        artifact_path=artifact_path,
    )

    print(f"Model: {model_name}")
    print(f"Registered name: {registered_name}")
    print(f"Version: {model_version.version}")
    print(f"Run ID: {run_id}")
    print("Validation status: PENDING")


# Register the latest run for both models and save the complete version catalogue.
def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    client = MlflowClient(
        tracking_uri=TRACKING_URI,
        registry_uri=TRACKING_URI,
    )

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{EXPERIMENT_NAME}' was not found."
        )

    registry = load_registry()

    for registered_name, config in MODEL_CONFIG.items():
        print("\n" + "=" * 60)
        print(f"Model: {config['model_name']}")

        register_model_version(
            client=client,
            registry=registry,
            registered_name=registered_name,
            model_name=config["model_name"],
            model_file=config["model_file"],
            experiment_id=experiment.experiment_id,
        )

    save_registry(registry)

    print("\n" + "=" * 60)
    print("Native MLflow model registration completed.")
    print(f"Version catalogue: {REGISTRY_PATH}")
    print("Run validate_models.py next.")


if __name__ == "__main__":
    main()
