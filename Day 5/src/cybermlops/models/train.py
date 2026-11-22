"""Train and evaluate anomaly-detection models."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger("cybermlops.models")

LABEL_COLUMN = "is_anomaly"


## "This function loads our training and testing CSV files. It also checks that the required feature columns
## and anomaly labels are available, and verifies that the data is valid."


def load_config(config_path: Path) -> dict:
    """Load model-training settings from YAML."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The configuration file must contain a YAML mapping.")

    return config


## "This function loads our training and testing CSV files. It also checks that the required feature
## columns and anomaly labels are available, and verifies that the data is valid."

def load_datasets(
    train_path: Path,
    test_path: Path,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load datasets and validate the required columns."""
    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Testing dataset not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    required_train = set(features + [LABEL_COLUMN])
    required_test = set(features + [LABEL_COLUMN])

    missing_train = required_train - set(train_df.columns)
    missing_test = required_test - set(test_df.columns)

    if missing_train:
        raise ValueError(
            f"Training dataset is missing columns: {sorted(missing_train)}"
        )

    if missing_test:
        raise ValueError(
            f"Testing dataset is missing columns: {sorted(missing_test)}"
        )

    for name, dataframe in (
        ("training", train_df),
        ("testing", test_df),
    ):
        dataframe[LABEL_COLUMN] = pd.to_numeric(
            dataframe[LABEL_COLUMN],
            errors="raise",
        )

        if not dataframe[LABEL_COLUMN].isin([0, 1]).all():
            raise ValueError(
                f"{name.capitalize()} labels must contain only 0 and 1."
            )

        dataframe[features] = dataframe[features].apply(
            pd.to_numeric,
            errors="raise",
        )

        if not np.isfinite(dataframe[features].to_numpy(dtype=float)).all():
            raise ValueError(
                f"{name.capitalize()} features contain missing or infinite values."
            )

    if not (train_df[LABEL_COLUMN] == 0).all():
        raise ValueError(
            "The training dataset must contain normal events only. "
            "Please check the Video 4 train.csv file."
        )

    if test_df[LABEL_COLUMN].nunique() < 2:
        raise ValueError(
            "The test dataset must contain both normal and anomaly examples."
        )

    return train_df, test_df


## "This function prepares the data before training. It separates normal training observations into
## training and calibration sets, and scales the numerical features so they're on comparable ranges."

def prepare_training_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    calibration_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Split normal training data and scale numerical features."""
    if not 0.05 <= calibration_fraction <= 0.40:
        raise ValueError(
            "calibration_fraction must be between 0.05 and 0.40."
        )

    normal_features = train_df[features].to_numpy(dtype=float)
    test_features = test_df[features].to_numpy(dtype=float)

    fit_data, calibration_data = train_test_split(
        normal_features,
        test_size=calibration_fraction,
        random_state=random_state,
        shuffle=True,
    )

    scaler = StandardScaler()

    fit_scaled = scaler.fit_transform(fit_data)
    calibration_scaled = scaler.transform(calibration_data)
    test_scaled = scaler.transform(test_features)

    return (
        fit_scaled,
        calibration_scaled,
        test_scaled,
        test_df[LABEL_COLUMN].to_numpy(dtype=int),
        scaler,
    )


## "This function calculates the anomaly threshold using the scores from our held-out normal
## observations. We'll use this threshold to decide whether a new observation should be flagged as an anomaly."

def calculate_threshold(
    normal_scores: np.ndarray,
    percentile: float,
) -> float:
    """Calculate an anomaly threshold using held-out normal observations."""
    if not 50 <= percentile <= 100:
        raise ValueError("threshold_percentile must be between 50 and 100.")

    return float(np.percentile(normal_scores, percentile))


## "This function creates and trains our Isolation Forest model using normal activity. It then calculates anomaly scores,
## determines the detection threshold, and generates predictions for the test data."

def train_isolation_forest(
    fit_data: np.ndarray,
    calibration_data: np.ndarray,
    test_data: np.ndarray,
    config: dict,
    percentile: float,
    random_state: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Train Isolation Forest and predict anomalies."""
    model = IsolationForest(
        n_estimators=int(config["n_estimators"]),
        max_samples=config["max_samples"],
        contamination="auto",
        random_state=int(config.get("random_state", random_state)),
        n_jobs=1,
    )

    model.fit(fit_data)

    # Higher values represent more anomalous observations.
    calibration_scores = -model.decision_function(calibration_data)
    test_scores = -model.decision_function(test_data)

    threshold = calculate_threshold(
        calibration_scores,
        percentile,
    )

    predictions = (test_scores > threshold).astype(int)

    result = {
        "model": model,
        "threshold": threshold,
        "calibration_scores": calibration_scores,
    }

    return result, test_scores, predictions


## "This function builds and trains our neural network autoencoder. It learns to reconstruct normal activity,
## calculates reconstruction errors, and flags observations whose errors exceed the detection threshold."

def train_autoencoder(
    fit_data: np.ndarray,
    calibration_data: np.ndarray,
    test_data: np.ndarray,
    config: dict,
    percentile: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Train a neural-network autoencoder and predict anomalies."""
    model = MLPRegressor(
        hidden_layer_sizes=tuple(config["hidden_layer_sizes"]),
        activation="relu",
        solver="adam",
        learning_rate_init=float(config["learning_rate_init"]),
        alpha=float(config["alpha"]),
        max_iter=int(config["max_iter"]),
        early_stopping=False,
        random_state=int(config["random_state"]),
        tol=1e-4,
    )

    # An autoencoder learns to reproduce its own input.
    model.fit(fit_data, fit_data)

    calibration_reconstructed = model.predict(calibration_data)
    test_reconstructed = model.predict(test_data)

    calibration_errors = np.mean(
        np.square(calibration_data - calibration_reconstructed),
        axis=1,
    )

    test_errors = np.mean(
        np.square(test_data - test_reconstructed),
        axis=1,
    )

    threshold = calculate_threshold(
        calibration_errors,
        percentile,
    )

    predictions = (test_errors > threshold).astype(int)

    result = {
        "model": model,
        "threshold": threshold,
        "calibration_errors": calibration_errors,
    }

    return result, test_errors, predictions

## "This function compares the model's predictions with the actual test labels. It calculates accuracy, precision, recall,
## F1-score, and the confusion matrix, helping us understand how well the model performed."

def evaluate_model(
    actual: np.ndarray,
    predictions: np.ndarray,
) -> dict:
    """Calculate classification metrics."""
    return {
        "accuracy": float(accuracy_score(actual, predictions)),
        "precision": float(
            precision_score(actual, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(actual, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(actual, predictions, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            actual,
            predictions,
            labels=[0, 1],
        ).tolist(),
        "classification_report": classification_report(
            actual,
            predictions,
            labels=[0, 1],
            target_names=["normal", "anomaly"],
            output_dict=True,
            zero_division=0,
        ),
    }


## "This function creates and saves a visual confusion matrix. It helps us see how many normal events and anomalies were correctly
## or incorrectly classified."

def save_confusion_matrix(
    matrix: list,
    model_name: str,
    output_path: Path,
) -> None:
    """Save a confusion-matrix image."""
    figure, axis = plt.subplots(figsize=(5, 4))

    image = axis.imshow(matrix, interpolation="nearest")

    figure.colorbar(image, ax=axis)

    axis.set(
        title=f"{model_name} - Confusion Matrix",
        xlabel="Predicted label",
        ylabel="Actual label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )

    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(matrix[row][column]),
                ha="center",
                va="center",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


## "This function saves the actual labels, anomaly scores, and predictions from both models into a CSV
## file. We can use this later to inspect individual predictions."

def save_predictions(
    test_df: pd.DataFrame,
    isolation_scores: np.ndarray,
    isolation_predictions: np.ndarray,
    autoencoder_scores: np.ndarray,
    autoencoder_predictions: np.ndarray,
    output_path: Path,
) -> None:
    """Save actual labels, anomaly scores, and model predictions."""
    results = test_df.copy()

    results["isolation_forest_score"] = isolation_scores
    results["isolation_forest_prediction"] = isolation_predictions

    results["autoencoder_score"] = autoencoder_scores
    results["autoencoder_prediction"] = autoencoder_predictions

    results.to_csv(output_path, index=False)


## "This is the main function that brings everything together. It loads the configuration and datasets, prepares the features,
## trains both models, evaluates their performance, and saves the models and results."

def main() -> None:
    """Run the complete model-training workflow."""
    parser = argparse.ArgumentParser(
        description="Train cyber-attack anomaly-detection models."
    )

    parser.add_argument(
        "--config",
        default="configs/model_training.yaml",
        help="Path to the model-training YAML configuration.",
    )

    args = parser.parse_args()

    config = load_config(Path(args.config))

    data_config = config["data"]
    training_config = config["training"]
    output_config = config["output"]

    features = config["features"]

    output_dir = Path(output_config["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    LOGGER.info("Loading training and testing datasets.")

    train_df, test_df = load_datasets(
        Path(data_config["train_path"]),
        Path(data_config["test_path"]),
        features,
    )

    LOGGER.info("Training rows: %s", len(train_df))
    LOGGER.info("Testing rows: %s", len(test_df))
    LOGGER.info("Features: %s", features)

    random_state = int(training_config["random_state"])
    calibration_fraction = float(
        training_config["calibration_fraction"]
    )
    percentile = float(training_config["threshold_percentile"])

    (
        fit_data,
        calibration_data,
        test_data,
        actual,
        scaler,
    ) = prepare_training_data(
        train_df=train_df,
        test_df=test_df,
        features=features,
        calibration_fraction=calibration_fraction,
        random_state=random_state,
    )

    LOGGER.info("Normal training observations: %s", len(fit_data))
    LOGGER.info("Normal calibration observations: %s", len(calibration_data))

    LOGGER.info("Training Isolation Forest.")

    isolation_result, isolation_scores, isolation_predictions = (
        train_isolation_forest(
            fit_data=fit_data,
            calibration_data=calibration_data,
            test_data=test_data,
            config=config["isolation_forest"],
            percentile=percentile,
            random_state=random_state,
        )
    )

    LOGGER.info("Training neural-network autoencoder.")

    autoencoder_result, autoencoder_scores, autoencoder_predictions = (
        train_autoencoder(
            fit_data=fit_data,
            calibration_data=calibration_data,
            test_data=test_data,
            config=config["autoencoder"],
            percentile=percentile,
        )
    )

    LOGGER.info("Evaluating both models.")

    isolation_metrics = evaluate_model(
        actual,
        isolation_predictions,
    )

    autoencoder_metrics = evaluate_model(
        actual,
        autoencoder_predictions,
    )

    # Save model bundles, including the scaler and feature order.
    joblib.dump(
        {
            "model": isolation_result["model"],
            "scaler": scaler,
            "features": features,
            "threshold": isolation_result["threshold"],
        },
        output_dir / "isolation_forest.joblib",
    )

    joblib.dump(
        {
            "model": autoencoder_result["model"],
            "scaler": scaler,
            "features": features,
            "threshold": autoencoder_result["threshold"],
        },
        output_dir / "autoencoder.joblib",
    )

    save_predictions(
        test_df=test_df,
        isolation_scores=isolation_scores,
        isolation_predictions=isolation_predictions,
        autoencoder_scores=autoencoder_scores,
        autoencoder_predictions=autoencoder_predictions,
        output_path=output_dir / "predictions.csv",
    )

    save_confusion_matrix(
        isolation_metrics["confusion_matrix"],
        "Isolation Forest",
        output_dir / "isolation_forest_confusion_matrix.png",
    )

    save_confusion_matrix(
        autoencoder_metrics["confusion_matrix"],
        "Autoencoder",
        output_dir / "autoencoder_confusion_matrix.png",
    )

    metrics = {
        "dataset": {
            "training_rows": int(len(train_df)),
            "testing_rows": int(len(test_df)),
            "normal_training_rows_used": int(len(fit_data)),
            "normal_calibration_rows": int(len(calibration_data)),
            "test_normal_rows": int((actual == 0).sum()),
            "test_anomaly_rows": int((actual == 1).sum()),
        },
        "configuration": {
            "features": features,
            "threshold_percentile": percentile,
            "calibration_fraction": calibration_fraction,
            "random_state": random_state,
        },
        "isolation_forest": {
            "threshold": isolation_result["threshold"],
            **isolation_metrics,
        },
        "autoencoder": {
            "threshold": autoencoder_result["threshold"],
            **autoencoder_metrics,
        },
    }

    metrics_path = output_dir / "metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n" + "=" * 60)
    print("ANOMALY DETECTION MODEL RESULTS")
    print("=" * 60)

    print("\nIsolation Forest")
    print(f"Accuracy : {isolation_metrics['accuracy']:.4f}")
    print(f"Precision: {isolation_metrics['precision']:.4f}")
    print(f"Recall   : {isolation_metrics['recall']:.4f}")
    print(f"F1-score : {isolation_metrics['f1_score']:.4f}")
    print(f"Threshold: {isolation_result['threshold']:.6f}")

    print("\nNeural Network Autoencoder")
    print(f"Accuracy : {autoencoder_metrics['accuracy']:.4f}")
    print(f"Precision: {autoencoder_metrics['precision']:.4f}")
    print(f"Recall   : {autoencoder_metrics['recall']:.4f}")
    print(f"F1-score : {autoencoder_metrics['f1_score']:.4f}")
    print(f"Threshold: {autoencoder_result['threshold']:.6f}")

    print("\nSaved output files:")
    for filename in (
        "isolation_forest.joblib",
        "autoencoder.joblib",
        "metrics.json",
        "predictions.csv",
        "isolation_forest_confusion_matrix.png",
        "autoencoder_confusion_matrix.png",
    ):
        print(f"  {output_dir / filename}")

    print("\nTraining and evaluation completed successfully.")


if __name__ == "__main__":
    main()