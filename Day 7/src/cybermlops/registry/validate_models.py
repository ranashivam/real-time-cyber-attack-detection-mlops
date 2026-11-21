
from pathlib import Path
import json


# Configure project paths and the model acceptance thresholds.
PROJECT_ROOT = Path.cwd()
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models" / "v1.0.0"
METRICS_PATH = MODEL_DIR / "metrics.json"

REGISTRY_DIR = PROJECT_ROOT / "artifacts" / "registry"
REGISTRY_PATH = REGISTRY_DIR / "registry.json"
VALIDATION_PATH = REGISTRY_DIR / "validation_results.json"

MIN_ANOMALY_RECALL = 0.90
MAX_FALSE_POSITIVE_RATE = 0.05

# Map evaluation metric keys to the corresponding catalogue names.
MODEL_NAMES = {
    "isolation_forest": "CyberAttack_Isolation_Forest",
    "autoencoder": "CyberAttack_Autoencoder",
}


# Load the evaluation metrics generated during model training.
def load_metrics():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_PATH}"
        )

    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


# Load the model-version catalogue created during registration.
def load_registry():
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Model registry catalogue not found: {REGISTRY_PATH}"
        )

    with REGISTRY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


# Save the updated model-version catalogue after validation.
def save_registry(registry):
    with REGISTRY_PATH.open("w", encoding="utf-8") as file:
        json.dump(registry, file, indent=4)


# Extract anomaly recall and false-positive rate from the training metrics.
def extract_model_metrics(model_metrics):
    report = model_metrics.get("classification_report", {})
    anomaly_metrics = report.get("anomaly", {})
    confusion_matrix = model_metrics.get("confusion_matrix")

    if "recall" not in anomaly_metrics:
        raise KeyError(
            "Anomaly recall was not found in classification_report."
        )

    if not confusion_matrix or len(confusion_matrix) != 2:
        raise ValueError(
            "A valid 2x2 confusion matrix was not found."
        )

    if any(len(row) != 2 for row in confusion_matrix):
        raise ValueError(
            "The confusion matrix must have two rows and two columns."
        )

    # The matrix layout is [[true negatives, false positives],
    #                       [false negatives, true positives]].
    true_negatives = confusion_matrix[0][0]
    false_positives = confusion_matrix[0][1]

    normal_count = true_negatives + false_positives

    if normal_count == 0:
        raise ValueError(
            "Cannot calculate false-positive rate without normal records."
        )

    anomaly_recall = float(anomaly_metrics["recall"])
    false_positive_rate = false_positives / normal_count

    return anomaly_recall, false_positive_rate


# Apply the configured acceptance criteria to one model's evaluation results.
def validate_model(model_key, model_metrics):
    anomaly_recall, false_positive_rate = extract_model_metrics(
        model_metrics
    )

    passed = (
        anomaly_recall >= MIN_ANOMALY_RECALL
        and false_positive_rate <= MAX_FALSE_POSITIVE_RATE
    )

    return {
        "model_name": model_key,
        "anomaly_recall": anomaly_recall,
        "false_positive_rate": false_positive_rate,
        "minimum_recall": MIN_ANOMALY_RECALL,
        "maximum_false_positive_rate": MAX_FALSE_POSITIVE_RATE,
        "validation_status": "APPROVED" if passed else "REJECTED",
    }


# Update the latest local catalogue version with its validation decision.
def update_registry_status(registry, registered_name, validation_status):
    versions = registry.get(registered_name)

    if not versions:
        raise KeyError(
            f"No registered versions found for {registered_name}."
        )

    latest_version = max(
        versions,
        key=lambda item: item["version"],
    )

    latest_version["validation_status"] = validation_status
    latest_version["status"] = (
        "Approved" if validation_status == "APPROVED"
        else "Rejected"
    )


# Validate both candidate models, update their catalogue entries, and save results.
def main():
    metrics = load_metrics()
    registry = load_registry()

    results = {}

    for model_key, registered_name in MODEL_NAMES.items():
        if model_key not in metrics:
            raise KeyError(
                f"Metrics for {model_key} were not found."
            )

        result = validate_model(
            model_key,
            metrics[model_key],
        )

        update_registry_status(
            registry,
            registered_name,
            result["validation_status"],
        )

        results[registered_name] = result

        print(f"\nModel: {registered_name}")
        print(
            f"Anomaly recall: "
            f"{result['anomaly_recall']:.2%}"
        )
        print(
            f"False-positive rate: "
            f"{result['false_positive_rate']:.2%}"
        )
        print(
            f"Validation: "
            f"{result['validation_status']}"
        )

    save_registry(registry)

    with VALIDATION_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nModel validation completed.")
    print(f"Updated catalogue: {REGISTRY_PATH}")
    print(f"Validation report: {VALIDATION_PATH}")


if __name__ == "__main__":
    main()
