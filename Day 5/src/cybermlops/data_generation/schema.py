"""Schema definitions for synthetic security telemetry."""

## Lists the fields every event must contain, such as timestamp,
## event ID, network features, scenario, and anomaly label.

REQUIRED_COLUMNS = [
    "timestamp",
    "event_id",
    "source_category",
    "destination_port",
    "bytes_transferred",
    "connection_count",
    "failed_logins",
    "api_requests_per_minute",
    "scenario",
    "is_anomaly",
]

## Defines acceptable ranges for numerical fields,
## such as destination ports and connection counts.

NUMERIC_RANGES = {
    "destination_port": (1, 65535),
    "bytes_transferred": (0, 1_000_000),
    "connection_count": (0, 1000),
    "failed_logins": (0, 50),
    "api_requests_per_minute": (0, 10_000),
    "is_anomaly": (0, 1),
}

## Defines the source types the generator can assign, such as web, API, and authentication.

SOURCE_CATEGORIES = [
    "web",
    "api",
    "authentication",
    "database",
    "internal_service",
]

## Lists the supported scenario names.

SCENARIOS = [
    "normal",
    "failed_logins",
    "connection_burst",
    "api_spike",
    "mixed",
]

## Identifies the version of the data format so we can track schema changes later.

SCHEMA_VERSION = "1.0"