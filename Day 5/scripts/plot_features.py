"""Plot one synthetic telemetry feature distribution."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd



# This function reads a generated CSV file, selects a numerical feature, and creates a histogram using Matplotlib.
#
# You can choose from features such as:
# - connection_count
# - bytes_transferred
# - failed_logins
# - api_requests_per_minute
# It saves the plot as a PNG file so we can inspect the generated data visually.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/generated/v1.0.0/test.csv",
    )
    parser.add_argument(
        "--feature",
        default="connection_count",
        choices=[
            "bytes_transferred",
            "connection_count",
            "failed_logins",
            "api_requests_per_minute",
        ],
    )
    parser.add_argument(
        "--output",
        default="data/generated/v1.0.0/feature_distribution.png",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(df[args.feature], bins=30, edgecolor="black")
    plt.title(f"Distribution of {args.feature}")
    plt.xlabel(args.feature)
    plt.ylabel("Number of events")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()

    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()