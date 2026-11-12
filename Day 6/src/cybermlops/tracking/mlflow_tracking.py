
"""Track Video 5 anomaly-detection experiments using the existing MLflow server."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import yaml


# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------

# Get the current project directory.
PROJECT_ROOT = Path.cwd()

# Define the locations of trained models and configuration files.
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models" / "v1.0.0"
CONFIG_PATH = PROJECT_ROOT / "configs" / "model_training.yaml"

# Define the locations of evaluation results and predictions.
METRICS_PATH = MODEL_DIR / "metrics.json"
PREDICTIONS_PATH = MODEL_DIR / "predictions.csv"

# Connect to the existing MLflow server from our previous series.
MLFLOW_TRACKING_URI = "http://localhost:5000"

# Define a separate experiment for our cyber-attack detection project.
EXPERIMENT_NAME = "Cyber Attack Anomaly Detection"


# Define the two models and their corresponding output files.
# Each model will have its own independent MLflow run.
MODELS = {
    "Isolation Forest": {
        "model_file": "isolation_forest.joblib",
        "confusion_matrix": "isolation_forest_confusion_matrix.png",
        "metrics_key": "isolation_forest",
    },
    "Autoencoder": {
        "model_file": "autoencoder.joblib",
        "confusion_matrix": "autoencoder_confusion_matrix.png",
        "metrics_key": "autoencoder",
    },
}


# Configure logging to display useful information in the terminal.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Load project files
# ---------------------------------------------------------

# FUNCTION 1:
# Loads the evaluation results from the JSON file generated in Video 5.
# It also checks whether the file exists and contains valid dictionary data.
def load_json(path: Path) -> dict[str, Any]:
    """Read the evaluation metrics from a JSON file."""

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return data


# FUNCTION 2:
# Loads the YAML configuration containing the model training settings.
# These settings will later be recorded as MLflow parameters.
def load_yaml(path: Path) -> dict[str, Any]:
    """Read the model training configuration."""

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    return data


# ---------------------------------------------------------
# Prepare parameters for MLflow
# ---------------------------------------------------------

# FUNCTION 3:
# Converts nested configuration sections into a flat dictionary.
# This makes the configuration easier to record as MLflow parameters.
def flatten_config(
    data: dict[str, Any],
    prefix: str = "",
) -> dict[str, str]:
    """Convert nested configuration values into flat parameters."""

    flattened = {}

    for key, value in data.items():
        parameter_name = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            flattened.update(
                flatten_config(value, parameter_name)
            )

        elif isinstance(value, (list, tuple)):
            flattened[parameter_name] = json.dumps(value)

        elif value is not None:
            flattened[parameter_name] = str(value)

    return flattened


# FUNCTION 4:
# Selects the configuration parameters relevant to the chosen model.
# It then records those parameters in the active MLflow experiment run.
def log_model_parameters(
    config: dict[str, Any],
    model_name: str,
) -> None:
    """Log the relevant configuration parameters for one model."""

    parameters = flatten_config(config)

    # Keep the parameters relevant to the selected model.
    if model_name == "Isolation Forest":
        relevant_sections = ("training.", "isolation_forest.", "features")
    else:
        relevant_sections = ("training.", "autoencoder.", "features")

    selected_parameters = {
        key: value
        for key, value in parameters.items()
        if key.startswith(relevant_sections)
    }

    selected_parameters["model_name"] = model_name

    # MLflow limits parameter names and values.
    selected_parameters = {
        key[:250]: value[:500]
        for key, value in selected_parameters.items()
    }

    mlflow.log_params(selected_parameters)


# ---------------------------------------------------------
# Extract evaluation metrics
# ---------------------------------------------------------

# FUNCTION 5:
# Extracts numerical values from the evaluation metrics dictionary.
# It also handles nested dictionaries so metrics can be recorded individually.
def extract_numeric_metrics(
    data: Any,
    prefix: str = "",
) -> dict[str, float]:
    """Extract numeric values from nested metric dictionaries."""

    results = {}

    if isinstance(data, dict):
        for key, value in data.items():
            metric_name = f"{prefix}_{key}" if prefix else str(key)

            if isinstance(value, dict):
                results.update(
                    extract_numeric_metrics(value, metric_name)
                )

            elif isinstance(value, (int, float)) and not isinstance(
                value, bool
            ):
                results[metric_name] = float(value)

    return results


# FUNCTION 6:
# Retrieves the evaluation metrics belonging to the selected model.
# For example, it extracts Isolation Forest metrics or Autoencoder metrics.
def get_model_metrics(
    all_metrics: dict[str, Any],
    metrics_key: str,
) -> dict[str, float]:
    """Get the evaluation metrics for the selected model."""

    model_metrics = all_metrics.get(metrics_key)

    if not isinstance(model_metrics, dict):
        raise ValueError(
            f"Metrics section '{metrics_key}' was not found "
            "in metrics.json."
        )

    metrics = extract_numeric_metrics(model_metrics)

    if not metrics:
        raise ValueError(
            f"No numeric metrics found for '{metrics_key}'."
        )

    return metrics


# FUNCTION 7:
# Sends the extracted evaluation metrics to the active MLflow run.
# Examples include accuracy, precision, recall, and F1-score.
def log_model_metrics(metrics: dict[str, float]) -> None:
    """Record evaluation metrics in the active MLflow run."""

    mlflow.log_metrics(
        {
            name[:250]: value
            for name, value in metrics.items()
        }
    )


# ---------------------------------------------------------
# Log model and supporting artifacts
# ---------------------------------------------------------

# FUNCTION 8:
# Uploads the trained model and its supporting files to MLflow.
# These artifacts include the confusion matrix, predictions, and configuration.
def log_model_artifacts(
    model_file: Path,
    confusion_matrix: Path,
) -> None:
    """Log the saved model and supporting project files."""

    # Save the original trained model file as an MLflow artifact.
    mlflow.log_artifact(
        str(model_file),
        artifact_path="trained_model",
    )

    # Log the model's confusion matrix when available.
    if confusion_matrix.is_file():
        mlflow.log_artifact(
            str(confusion_matrix),
            artifact_path="evaluation",
        )
    else:
        logger.warning(
            "Confusion matrix not found: %s",
            confusion_matrix,
        )

    # Log the predictions file when available.
    if PREDICTIONS_PATH.is_file():
        mlflow.log_artifact(
            str(PREDICTIONS_PATH),
            artifact_path="predictions",
        )
    else:
        logger.warning(
            "Predictions file not found: %s",
            PREDICTIONS_PATH,
        )

    # Keep a copy of the configuration used for training.
    if CONFIG_PATH.is_file():
        mlflow.log_artifact(
            str(CONFIG_PATH),
            artifact_path="configuration",
        )


# ---------------------------------------------------------
# Track one model
# ---------------------------------------------------------

# FUNCTION 9:
# Creates an independent MLflow run for the selected model.
# It combines the parameters, evaluation metrics, tags, and artifacts
# so that each model's experiment can be inspected separately.
def track_model(
    model_name: str,
    model_details: dict[str, str],
    all_metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Create one MLflow run for a model and log its results."""

    model_file = MODEL_DIR / model_details["model_file"]

    confusion_matrix = MODEL_DIR / model_details["confusion_matrix"]

    if not model_file.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {model_file}"
        )

    metrics = get_model_metrics(
        all_metrics,
        model_details["metrics_key"],
    )

    logger.info("Starting experiment run: %s", model_name)

    with mlflow.start_run(run_name=model_name):

        # Add descriptive labels to identify this experiment.
        mlflow.set_tags(
            {
                "project": "Real-Time Cyber-Attack Detection",
                "episode": "Video 6",
                "model": model_name,
                "tracking_stage": "experiment_tracking",
            }
        )

        # Record the model's training configuration.
        log_model_parameters(config, model_name)

        # Record the model's evaluation results.
        log_model_metrics(metrics)

        # Upload the trained model and supporting files.
        log_model_artifacts(
            model_file,
            confusion_matrix,
        )

        # Retrieve the unique ID assigned to this MLflow run.
        run_id = mlflow.active_run().info.run_id

        logger.info(
            "Completed %s. MLflow Run ID: %s",
            model_name,
            run_id,
        )


# ---------------------------------------------------------
# Main workflow
# ---------------------------------------------------------

# FUNCTION 10:
# Coordinates the complete MLflow tracking workflow.
# It connects to the existing MLflow server, loads the Video 5 outputs,
# and creates separate experiment runs for both anomaly-detection models.
def main() -> None:
    """Track both anomaly-detection models in MLflow."""

    logger.info("Starting MLflow experiment tracking.")

    # Connect to the same MLflow server used in the previous series.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Create or reuse the cyber-attack detection experiment.
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("MLflow URL: %s", MLFLOW_TRACKING_URI)
    logger.info("Experiment: %s", EXPERIMENT_NAME)

    # Load the results and configuration from Video 5.
    all_metrics = load_json(METRICS_PATH)
    config = load_yaml(CONFIG_PATH)

    # Create one independent run for each model.
    for model_name, model_details in MODELS.items():
        track_model(
            model_name=model_name,
            model_details=model_details,
            all_metrics=all_metrics,
            config=config,
        )

    logger.info("All experiments were tracked successfully.")
    logger.info("Open the MLflow UI at %s", MLFLOW_TRACKING_URI)


# Execute the main workflow when this file is run directly.
if __name__ == "__main__":
    main()
