"""Synthetic baseline behavior and anomaly scenarios."""

import numpy as np
import pandas as pd

from cybermlops.data_generation.schema import SOURCE_CATEGORIES


ANOMALY_SCENARIOS = [
    "failed_logins",
    "connection_burst",
    "api_spike",
]




# Creates the requested number of baseline events.
# Generates illustrative values for bytes transferred, connection counts, failed logins, and API request rates.
# Assigns normal scenario labels and sets is_anomaly to 0.
# Uses the supplied random number generator so the output is reproducible.

def generate_normal_events(
    count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate illustrative baseline events."""

    ports = np.array([22, 53, 80, 443, 5432, 8080, 8443])

    return pd.DataFrame(
        {
            "source_category": rng.choice(
                SOURCE_CATEGORIES, size=count
            ),
            "destination_port": rng.choice(ports, size=count),
            "bytes_transferred": rng.integers(
                100, 50_001, size=count
            ),
            "connection_count": rng.integers(
                1, 21, size=count
            ),
            "failed_logins": rng.integers(
                0, 3, size=count
            ),
            "api_requests_per_minute": rng.integers(
                1, 121, size=count
            ),
            "scenario": np.full(count, "normal", dtype=object),
            "is_anomaly": np.zeros(count, dtype=int),
        }
    )




# Takes a set of generated events and modifies selected features to represent a specific synthetic anomaly.
# failed_logins creates unusually high failed-login counts.
# connection_burst creates unusually high connection counts.
# api_spike creates unusually high API request rates.
# Updates the scenario and anomaly labels accordingly.


def apply_anomaly_scenario(
    events: pd.DataFrame,
    scenario: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Apply one synthetic anomaly pattern to selected rows."""

    result = events.copy()
    count = len(result)

    if count == 0:
        return result

    if scenario == "failed_logins":
        result["source_category"] = "authentication"
        result["failed_logins"] = rng.integers(
            10, 51, size=count
        )

    elif scenario == "connection_burst":
        result["connection_count"] = rng.integers(
            100, 1001, size=count
        )

    elif scenario == "api_spike":
        result["source_category"] = "api"
        result["api_requests_per_minute"] = rng.integers(
            500, 10_001, size=count
        )

    else:
        raise ValueError(
            f"Unsupported anomaly scenario: {scenario}"
        )

    result["scenario"] = scenario
    result["is_anomaly"] = 1

    return result