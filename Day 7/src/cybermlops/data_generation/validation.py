"""Data quality checks for synthetic telemetry."""

import pandas as pd

from cybermlops.data_generation.schema import (
    NUMERIC_RANGES,
    REQUIRED_COLUMNS,
)


# This function receives a pandas DataFrame and returns a list of validation errors.
#
# It checks:
# - Whether all required columns exist.
# - Whether the dataset is empty.
# - Whether event IDs are missing or duplicated.
# - Whether required fields contain missing values.
# - Whether timestamps can be parsed.
# - Whether numerical values fall within the configured ranges.
# - Whether anomaly labels and scenario labels are consistent.


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Return a list of validation errors. Empty means valid."""

    errors = []

    missing_columns = sorted(
        set(REQUIRED_COLUMNS) - set(df.columns)
    )

    if missing_columns:
        return [f"Missing required columns: {missing_columns}"]

    if df.empty:
        errors.append("Dataset is empty.")

    if df["event_id"].isna().any():
        errors.append("Missing event IDs.")

    if df["event_id"].duplicated().any():
        errors.append("Duplicate event IDs.")

    if df[REQUIRED_COLUMNS].isna().any().any():
        errors.append("Missing values found in required columns.")

    parsed_timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    if parsed_timestamps.isna().any():
        errors.append("Invalid timestamp values.")

    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        values = pd.to_numeric(df[column], errors="coerce")

        if values.isna().any():
            errors.append(f"{column} contains non-numeric values.")
            continue

        if ((values < minimum) | (values > maximum)).any():
            errors.append(
                f"{column} contains values outside "
                f"{minimum} to {maximum}."
            )

    if not df["is_anomaly"].isin([0, 1]).all():
        errors.append("is_anomaly must contain only 0 or 1.")

    if not df["scenario"].isin(
        ["normal", "failed_logins", "connection_burst", "api_spike"]
    ).all():
        errors.append("Unknown scenario label found.")

    if not df["is_anomaly"].eq(
        df["scenario"].ne("normal").astype(int)
    ).all():
        errors.append(
            "Scenario labels and anomaly labels are inconsistent."
        )

    return errors