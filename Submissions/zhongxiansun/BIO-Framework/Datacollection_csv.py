"""
Battery EOL Prediction and Policy Evaluation Pipeline.

This module processes battery cycling test data to compute End-of-Life (EOL) metrics
based on retention curves, associates them with policy indices, and aggregates results
for analysis or visualization.
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import interpolate, optimize


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the EOL prediction pipeline.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Battery EOL prediction from cycling data.")
    parser.add_argument(
        "--round_idx",
        type=int,
        default=3,
        help="Round index for experiment tracking (default: 3)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Dual_salt/data",
        help="Directory containing batch and pred subdirectories (default: 'data')",
    )
    parser.add_argument(
        "--policies_path",
        type=str,
        default="policies.csv",
        help="Path to the policies CSV file (default: 'policies.csv')",
    )
    parser.add_argument(
        "--test_files",
        type=str,
        default="Dual_salt/raw",
        help="Directory containing test data folders like baseline/ and round*/ (default: 'Dual_salt/raw')",
    )
    return parser.parse_args()


class EOLExtractor:
    """Extracts End-of-Life (EOL) metric from a single battery cycling curve.

    Processes raw cycling data, validates its quality, interpolates retention curve,
    and computes a weighted EOL score based on time-to-80% and time-to-60% capacity.

    Attributes:
        file (str): Path to the input CSV file with cycling data.
        data_dir (str): Base directory for saving intermediate outputs.
        round_idx (int): Current round index used in output paths.
        alpha (float): Weight for EOL_80 in final EOL calculation (default 0.75).
        data (tuple): Processed cycle and retention data.
        is_valid (bool): Flag indicating whether the data passed quality checks.
        eol (float): Computed EOL score; NaN if invalid.
    """

    def __init__(self, args: argparse.Namespace, file: str):
        """Initialize EOLExtractor with arguments and file path.

        Args:
            args: Parsed command-line arguments.
            file: Path to the CSV file containing battery cycling data.
        """
        self.file = file
        self.is_valid = False
        self.alpha = 0.75
        self.data_dir = args.data_dir
        self.round_idx = args.round_idx

        self.data = self.process_data()
        self.clean()
        self.eol = self.get_eol()
        self.appender()

    def process_data(self) -> tuple:
        """Load and extract cycle number and retention percentage from CSV.

        Only rows 3 to 49 are used to avoid unstable initial cycles.

        Returns:
            Tuple of (cycle numbers, retention percentages), or (0, 0) if empty.
        """
        df = pd.read_csv(self.file)
        cyc = df["Cyc"].tolist()
        retention = df["Retention(%)"].tolist()

        cyc_data = cyc[3:50]
        retention_data = retention[3:50]

        if not cyc_data or not retention_data:
            return (0, 0)
        return (cyc_data, retention_data)

    def clean(self) -> None:
        """Validate data quality and set `is_valid` flag accordingly.

        Invalid conditions:
            - Less than or equal to 50 data points.
            - Retention exceeds 102%.
            - Average retention between cycles 20–30 or 18–22 >= 90% (indicating anomaly).

        Sets:
            self.is_valid (bool): True only if data passes all checks.
        """
        if len(self.data[0]) <= 50:
            self.data = (0, 0)
        if self.data == (0, 0):
            self.is_valid = False
            return

        _, retention_data = self.data
        if (
            max(retention_data) >= 102
            or np.mean(retention_data[20:30]) >= 90
            or np.mean(retention_data[18:22]) >= 90
        ):
            self.is_valid = False
            return

        self.is_valid = True

    def get_eol(self) -> float:
        """Calculate weighted EOL score using interpolation and root-finding.

        EOL = alpha * EOL@80% + (1-alpha) * EOL@60%

        Uses extrapolated linear interpolation and Newton-Raphson method to find
        when retention drops below 80% and 60%.

        Returns:
            Weighted EOL value in cycles; NaN if data is invalid.
        """

        def call_eol(target: float) -> float:
            if not self.is_valid:
                return np.nan
            try:
                func = interpolate.interp1d(
                    self.data[0], self.data[1], fill_value="extrapolate"
                )

                def residual(x): return func(x) - target
                # Initial guess near expected crossing point
                result = optimize.newton(residual, x0=100 - target)
                return result - 3  # Adjustment heuristic
            except Exception:
                return np.nan

        eol_80 = call_eol(80)
        eol_60 = call_eol(60)
        return self.alpha * eol_80 + (1 - self.alpha) * eol_60

    def appender(self) -> None:
        """Append policy vector and computed EOL to pred CSV file.

        Reads corresponding policy from `batch/<round_idx>.csv`, finds matching row,
        then appends policy values and EOL to `pred/<round_idx>.csv`.
        """
        batch_file = os.path.join(self.data_dir, "batch", f"{self.round_idx}.csv")
        pred_file = os.path.join(self.data_dir, "pred", f"{self.round_idx}.csv")

        # Extract policy index from filename (e.g., Testdata/round3/data_5.csv -> index 5)
        match = re.search(r"(\d+)", os.path.basename(self.file))
        if not match:
            raise ValueError(f"Could not extract index from filename: {self.file}")
        policy_idx = int(match.group(1))

        # Read policy sample
        policies_sample = pd.read_csv(batch_file, header=None, skip_blank_lines=True)
        policy = policies_sample.iloc[policy_idx]

        # Append to pred file
        with open(pred_file, "a") as outfile:
            policy_str = ",".join(map(str, policy))
            outfile.write(f"{policy_str},{self.eol}\n")


class Sum:
    """Aggregates policy indices and EOL results across multiple experiments.

    Matches predicted policies with original policy list to retrieve their global index,
    then collects associated EOL scores for downstream use (e.g., plotting).
    """

    def __init__(self, args: argparse.Namespace):
        """Initialize Sum with command-line arguments.

        Args:
            args: Parsed arguments containing round_idx, policies_path, and data_dir.
        """
        self.round_idx = args.round_idx
        self.policies = pd.read_csv(args.policies_path, header=1).astype(int)
        self.data_dir = args.data_dir

        self.indexs = self.get_policy_indexs()
        self.eols = self.get_eols()

    def get_policy_indexs(self) -> list:
        """Find global policy indices by matching sampled policies.

        Loads policy samples from `batch/<round_idx>.csv`, matches each against
        the full policy list (`policies.csv`), and returns the matched row indices.

        Returns:
            List of matched policy indices.
        """
        indexs = []
        policies_list = []

        pattern = os.path.join(self.data_dir, "batch", f"{self.round_idx}.csv")
        all_files = glob.glob(pattern)

        for file in all_files:
            df = pd.read_csv(file, header=None, skip_blank_lines=True)
            policies_list.append(df)
        policies_sample = pd.concat(policies_list, ignore_index=True).astype(int)

        for _, sample in policies_sample.iterrows():
            match = (self.policies == sample.tolist()).all(axis=1)
            if match.any():
                indexs.append(match.idxmax())
            else:
                indexs.append(np.nan)  # No match found

        return indexs

    def get_eols(self) -> pd.Series:
        """Load EOL scores from prediction file.

        Reads `pred/<round_idx>.csv`, extracts last column as EOL values,
        converts to numeric (coercing errors to NaN).

        Returns:
            Pandas Series of EOL values.
        """
        pred_file = os.path.join(self.data_dir, "pred", f"{self.round_idx}.csv")
        df = pd.read_csv(pred_file, header=None, skip_blank_lines=True)
        return pd.to_numeric(df.iloc[:, -1], errors="coerce")


def baseline_analysis(args: argparse.Namespace) -> None:
    """Perform statistical analysis on baseline group EOL values.

    Computes mean, standard error, and 95% confidence interval for baseline performance.

    Args:
        args: Command-line arguments with test_files path.
    """
    baselines_files = glob.glob(os.path.join(args.test_files, "baseline", "*.csv"))
    baseline_eols = [EOLExtractor(args, file).eol for file in baselines_files]
    baseline_eols = [e for e in baseline_eols if not np.isnan(e)]

    if not baseline_eols:
        print("No valid baseline EOL values found.")
        return

    mean_baseline = np.mean(baseline_eols)
    n = len(baseline_eols)
    std_err = np.std(baseline_eols, ddof=1) / np.sqrt(n)
    t_critical = stats.t.ppf((1 + 0.95) / 2, df=n - 1)
    margin_of_error = t_critical * std_err

    print(
        f"Baseline EOL: {mean_baseline:.2f}, "
        f"95% CI margin: ±{margin_of_error:.2f} cycles"
    )


def collect_results(args: argparse.Namespace) -> None:
    """Collect policy indices and EOL scores into a structured CSV.

    Aggregates results via `Sum` class and saves to `round<idx>.csv` for visualization.

    Args:
        args: Command-line arguments.
    """
    agent = Sum(args)
    data = pd.DataFrame({"index": agent.indexs, "EOL": agent.eols})
    data = data.fillna(np.nan)

    output_dir = os.path.join(args.data_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"round{args.round_idx}.csv")
    data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def main() -> None:
    """Main entry point for EOL extraction and result aggregation."""
    args = parse_args()

    # Create pred directory if it doesn't exist
    pred_dir = os.path.join(args.data_dir, "pred")
    os.makedirs(pred_dir, exist_ok=True)

    # Uncomment next line to run baseline analysis
    # baseline_analysis(args)

    pulse_files = glob.glob(os.path.join(args.test_files, f"round{args.round_idx}", "*.csv"))
    for file in pulse_files:
        EOLExtractor(args, file)

    collect_results(args)


if __name__ == "__main__":
    main()