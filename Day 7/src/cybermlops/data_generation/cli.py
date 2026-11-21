
"""Command-line entry point for synthetic telemetry generation."""

import argparse
import json
from pathlib import Path

import yaml

from cybermlops.data_generation.generator import generate_dataset
from cybermlops.data_generation.schema import SCHEMA_VERSION
from cybermlops.data_generation.validation import validate_dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "data_generator.yaml"


# Defines the command-line options, including the configuration file,
# event rate, duration, seed, scenario, anomaly fraction, and output directory.

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic cyber-security telemetry."
    )

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument("--event-rate", type=int)
    parser.add_argument("--duration-seconds", type=int)
    parser.add_argument("--seed", type=int)

    parser.add_argument(
        "--scenario",
        choices=[
            "normal",
            "failed_logins",
            "connection_burst",
            "api_spike",
            "mixed",
        ],
    )

    parser.add_argument("--anomaly-fraction", type=float)
    parser.add_argument("--output-dir")

    return parser.parse_args()


# The main workflow:
# 1. Reads the YAML configuration.
# 2. Applies command-line overrides when supplied.
# 3. Generates the training and test datasets using deterministic seeds.
# 4. Validates both datasets before saving either CSV.
# 5. Prints row counts, scenario counts, anomaly counts, and validation results.
# 6. Saves train.csv, test.csv, and metadata.json.


def main():
    """Generate, validate, and save training and test datasets."""
    args = parse_args()

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = REPOSITORY_ROOT / config_path

    if not config_path.is_file():
        raise SystemExit(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise SystemExit(
            "The YAML configuration must contain a mapping of settings."
        )

    # Command-line arguments override values in the YAML configuration.
    settings = {
        "event_rate": (
            args.event_rate
            if args.event_rate is not None
            else config.get("event_rate", 5)
        ),
        "duration_seconds": (
            args.duration_seconds
            if args.duration_seconds is not None
            else config.get("duration_seconds", 60)
        ),
        "seed": (
            args.seed
            if args.seed is not None
            else config.get("seed", 42)
        ),
        "scenario": (
            args.scenario
            if args.scenario is not None
            else config.get("scenario", "mixed")
        ),
        "anomaly_fraction": (
            args.anomaly_fraction
            if args.anomaly_fraction is not None
            else config.get("anomaly_fraction", 0.10)
        ),
    }

    output_dir_value = (
        args.output_dir
        if args.output_dir is not None
        else config.get("output_dir", "data/generated/v1.0.0")
    )

    output_dir = Path(output_dir_value)

    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate normal-only training data.
    train = generate_dataset(
        **settings,
        split="train",
    )

    # Use a separate deterministic seed for the test dataset.
    test_settings = dict(settings)
    test_settings["seed"] = settings["seed"] + 1

    test = generate_dataset(
        **test_settings,
        split="test",
    )

    # Validate both datasets before saving either one.
    datasets = {
        "train": train,
        "test": test,
    }

    validation_errors = {}

    for name, dataset in datasets.items():
        errors = validate_dataset(dataset)

        if errors:
            validation_errors[name] = errors

    if validation_errors:
        for name, errors in validation_errors.items():
            print(f"\nValidation failed for {name} dataset:")

            for error in errors:
                print(f"  - {error}")

        raise SystemExit(1)

    # Save datasets only after both have passed validation.
    for name, dataset in datasets.items():
        output_path = output_dir / f"{name}.csv"

        dataset.to_csv(
            output_path,
            index=False,
        )

        print(f"\n{name.upper()} dataset")
        print(f"Rows: {len(dataset)}")

        print("\nScenario counts:")
        print(
            dataset["scenario"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print("\nAnomaly counts:")
        print(
            dataset["is_anomaly"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print("Validation: PASS")
        print(f"Saved to: {output_path}")

    # Record the settings used for reproducibility.
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "seed": settings["seed"],
        "train_seed": settings["seed"],
        "test_seed": test_settings["seed"],
        "event_rate": settings["event_rate"],
        "duration_seconds": settings["duration_seconds"],
        "scenario": settings["scenario"],
        "anomaly_fraction": settings["anomaly_fraction"],
        "train_rows": len(train),
        "test_rows": len(test),
        "train_file": "train.csv",
        "test_file": "test.csv",
    }

    metadata_path = output_dir / "metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")

    print(f"\nDatasets saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
