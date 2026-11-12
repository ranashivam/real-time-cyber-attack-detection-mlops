"""Reproducible synthetic telemetry dataset generation."""

import numpy as np
import pandas as pd

from cybermlops.data_generation.scenarios import (
    ANOMALY_SCENARIOS,
    apply_anomaly_scenario,
    generate_normal_events,
)


START_TIMESTAMP = pd.Timestamp("2025-01-01T00:00:00Z")


# event_rate — how many events to generate per simulated second.
# duration_seconds — how long the simulated period lasts.
# seed — the random seed used for reproducibility.
# scenario — which anomaly scenario to simulate.
# anomaly_fraction — the proportion of test records selected to be anomalous.
# split — whether to generate the training or test dataset.


def generate_dataset(
    *,
    event_rate: int,
    duration_seconds: int,
    seed: int,
    scenario: str,
    anomaly_fraction: float = 0.10,
    split: str = "test",
) -> pd.DataFrame:
    """
    Generate synthetic events.

    event_rate is the number of records generated per simulated second.
    For the training split, only normal events are returned.
    """

    if event_rate < 1:
        raise ValueError("event_rate must be at least 1.")

    if duration_seconds < 1:
        raise ValueError("duration_seconds must be at least 1.")

    if not 0.0 <= anomaly_fraction <= 1.0:
        raise ValueError(
            "anomaly_fraction must be between 0 and 1."
        )

    if scenario not in {
        "normal",
        "failed_logins",
        "connection_burst",
        "api_spike",
        "mixed",
    }:
        raise ValueError(f"Unsupported scenario: {scenario}")

##  This section controls how we introduce anomalies into our dataset. First, we validate the split and generate the normal events. If we're creating training data, we keep everything normal. For the test dataset, we select a percentage of records based on our anomaly fraction.
##  Next, we either choose a mixture of anomaly scenarios or apply one specific scenario, such as repeated failed logins or a connection burst. Finally, we modify the selected records to simulate those unusual patterns.
##  So, the main idea is that we start with normal data and selectively introduce anomalies to create a controlled test dataset.


    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'.")

    count = event_rate * duration_seconds
    rng = np.random.default_rng(seed)

    events = generate_normal_events(count, rng)

    # Training uses only normal examples.
    if split == "train" or scenario == "normal":
        selected_anomalies = np.array([], dtype=int)

    else:
        anomaly_count = int(round(count * anomaly_fraction))
        anomaly_count = min(max(anomaly_count, 0), count)

        selected_anomalies = rng.choice(
            count,
            size=anomaly_count,
            replace=False,
        )

    if len(selected_anomalies):
        if scenario == "mixed":
            selected_scenarios = rng.choice(
                ANOMALY_SCENARIOS,
                size=len(selected_anomalies),
            )
        else:
            selected_scenarios = np.full(
                len(selected_anomalies),
                scenario,
                dtype=object,
            )

        # Apply each selected scenario to its chosen rows.
        for anomaly_name in ANOMALY_SCENARIOS:
            positions = selected_anomalies[
                selected_scenarios == anomaly_name
            ]

            if len(positions) == 0:
                continue

            anomalous_rows = apply_anomaly_scenario(
                events.iloc[positions],
                anomaly_name,
                rng,
            )

            # Copy generated anomaly features into the original rows.
            events.loc[
                events.index[positions],
                anomalous_rows.columns,
            ] = anomalous_rows.to_numpy()

    # Give every record a stable, unique ID.
    prefix = "train" if split == "train" else "test"
    events.insert(
        0,
        "event_id",
        [f"{prefix}-{i:08d}" for i in range(count)],
    )

    # Timestamps are deterministic and ordered.
    events.insert(
        0,
        "timestamp",
        START_TIMESTAMP
        + pd.to_timedelta(
            np.arange(count) / event_rate,
            unit="s",
        ),
    )

    return events.reset_index(drop=True)