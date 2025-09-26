"""
Battery EOL Prediction and Policy Evaluation Pipeline.

This module processes battery cycling test data (in .ndax format) to compute End-of-Life (EOL)
metrics based on capacity retention curves, associates them with policy indices from active
learning rounds, and aggregates results for analysis or visualization. It supports baseline
statistical evaluation and detailed result export.
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import interpolate
import NewareNDA


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the EOL prediction pipeline.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Battery EOL prediction from NDAX cycling data.")
    parser.add_argument(
        "--round_idx",
        type=int,
        required=True,
        help="Round index for experiment tracking (e.g., 1, 2, 3)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="LHCE/data",
        help="Root directory containing raw data, batch/, pred/, and Testdata/ (default: 'LHCE/data')",
    )
    parser.add_argument(
        "--policies_path",
        type=str,
        default="policies.csv",
        help="Path to the global policies CSV file (default: 'policies.csv')",
    )
    parser.add_argument(
        "--test_files",
        type=str,
        default="LHCE/raw",
        help="Subdirectory containing test folders like baseline/ and round*/ (default: 'LHCE/raw')",
    )
    return parser.parse_args()


class EOLExtractor:
    """Extracts End-of-Life (EOL) metric from a single battery cycling curve in .ndax format.

    Loads raw cycling data using NewareNDA, computes discharge capacity retention,
    cleans noisy data, and calculates EOL via cubic interpolation at 80% and 60% thresholds.

    Attributes:
        file (str): Path to the input .ndax file with cycling data.
        data_dir (str): Base directory for saving intermediate outputs.
        round_idx (int): Current round index used in output paths.
        alpha (float): Weight for EOL@80% in final score (default 0.75).
        cycles (np.ndarray): Validated cycle numbers after cleaning.
        retentions (np.ndarray): Corresponding retention percentages.
        is_valid (bool): Flag indicating whether the data passed quality checks.
        eol (float): Computed EOL score; NaN if invalid.
    """

    def __init__(self, args: argparse.Namespace, file: str):
        """Initialize EOLExtractor with arguments and file path.

        Args:
            args: Parsed command-line arguments.
            file: Path to the .ndax file containing battery cycling data.
        """
        self.file = file
        self.data_dir = args.data_dir
        self.round_idx = args.round_idx
        self.alpha = 0.75

        self.cycles, self.retentions = self.clean()
        self.is_valid = len(self.cycles) > 1
        self.eol = self.calculate_eol()
        self.append_to_pred()

    def clean(self) -> tuple[np.ndarray, np.ndarray]:
        """Load and clean cycling data by grouping on cycle and computing retention.

        Applies filtering for valid cycles where:
            - Retention is between 0% and 105%
            - Discharge capacity is non-negative
            - Cycle numbers are strictly increasing

        Only uses stable cycles (after initial few).

        Returns:
            Tuple of (valid cycles, corresponding retentions), or ([0], [0]) if invalid.
        """
        try:
            df = NewareNDA.read(self.file)
            grouped = df.groupby('Cycle')['Discharge_Capacity(mAh)'].max().reset_index()
            if len(grouped) < 50:
                return np.array([0]), np.array([0])

            # Compute retention relative to cycle 4
            initial_capacity = grouped['Discharge_Capacity(mAh)'].iloc[3]
            grouped['Retention'] = (grouped['Discharge_Capacity(mAh)'] / initial_capacity) * 100

            cycles = grouped['Cycle'].values
            retentions = grouped['Retention'].values
            dis_capacity = grouped['Discharge_Capacity(mAh)'].values

            # Filter valid entries
            valid_mask = (
                (retentions >= 0) &
                (retentions <= 105) &
                (dis_capacity >= 0) &
                (np.diff(cycles, prepend=cycles[0] - 1) > 0)
            )

            valid_cycles = cycles[valid_mask]
            valid_retentions = retentions[valid_mask]

            if len(valid_cycles) == 0:
                return np.array([0]), np.array([0])

            return valid_cycles, valid_retentions

        except Exception as e:
            print(f"Error processing {self.file}: {e}")
            return np.array([0]), np.array([0])

    def calculate_eol(self) -> float:
        """Calculate weighted EOL score using cubic interpolation.

        EOL = alpha * EOL@80% + (1-alpha) * EOL@60%

        Uses cubic spline interpolation over dense cycle grid to find when retention
        drops below threshold. Adjusts result by -3 cycles as heuristic.

        Returns:
            Weighted EOL value in cycles; NaN if invalid or no crossing.
        """

        def find_crossing(target: float) -> float:
            if not self.is_valid:
                return np.nan
            try:
                f = interpolate.interp1d(
                    self.cycles, self.retentions,
                    kind='cubic', fill_value='extrapolate'
                )
                cycles_dense = np.arange(self.cycles.min() + 3, self.cycles.max() + 0.1, 0.1)
                retention_interp = f(cycles_dense)
                idx = np.where(retention_interp <= target)[0]
                if len(idx) == 0:
                    return np.nan
                return float(np.round(cycles_dense[idx[0]] - 3, 3))
            except Exception:
                return np.nan

        eol_80 = find_crossing(80)
        eol_60 = find_crossing(60)
        return self.alpha * eol_80 + (1 - self.alpha) * eol_60

    def append_to_pred(self) -> None:
        """Append policy vector and computed EOL to pred CSV file.

        Reads corresponding policy from `batch/<round_idx>.csv`, finds matching row
        by filename index, then appends policy values and EOL to `pred/<round_idx>.csv`.
        """
        batch_file = os.path.join(self.data_dir, "batch", f"{self.round_idx}.csv")
        pred_file = os.path.join(self.data_dir, "pred", f"{self.round_idx}.csv")

        # Extract policy index from filename (e.g., Testdata/round2/data_5.ndax -> index 5)
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

        self.indices = self.get_policy_indices()
        self.eols = self.get_eols()

    def get_policy_indices(self) -> list:
        """Find global policy indices by matching sampled policies.

        Loads policy samples from `batch/<round_idx>.csv`, matches each against
        the full policy list (`policies.csv`), and returns the matched row indices.

        Returns:
            List of matched policy indices; np.nan for unmatched rows.
        """
        indices = []
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
                indices.append(match.idxmax())
            else:
                indices.append(np.nan)

        return indices

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

    Computes mean, standard deviation, and prints summary for baseline performance.

    Args:
        args: Command-line arguments with test_files path.
    """
    baseline_files = glob.glob(os.path.join(args.data_dir, args.test_files, "baseline", "*.ndax"))
    baseline_eols = [EOLExtractor(args, file).eol for file in baseline_files]
    baseline_eols = [e for e in baseline_eols if not np.isnan(e)]

    if not baseline_eols:
        print("No valid baseline EOL values found.")
        return

    mean_baseline = np.mean(baseline_eols)
    std_baseline = np.std(baseline_eols, ddof=1)

    print(f"Baseline EOL: {mean_baseline:.2f} ± {std_baseline:.2f} cycles")


def collect_results(args: argparse.Namespace) -> None:
    """Collect policy indices and EOL scores into a structured CSV.

    Aggregates results via `Sum` class and saves to `results/round<idx>.csv`.

    Args:
        args: Command-line arguments.
    """
    agent = Sum(args)
    data = pd.DataFrame({"index": agent.indices, "EOL": agent.eols})
    data = data.dropna(subset=["index"])  # Remove unmatched policies
    data["index"] = data["index"].astype(int)

    output_dir = os.path.join(args.data_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"round{args.round_idx}.csv")
    data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def main() -> None:
    """Main entry point for EOL extraction and result aggregation."""
    args = parse_args()

    # Ensure pred directory exists
    pred_dir = os.path.join(args.data_dir, "pred")
    os.makedirs(pred_dir, exist_ok=True)

    # Run baseline analysis if requested (uncomment next line)
    # baseline_analysis(args)

    # Process all files in current round
    pulse_files = glob.glob(os.path.join(args.data_dir, args.test_files, f"round{args.round_idx}", "*.ndax"))
    for file in pulse_files:
        print(f"Processing {os.path.basename(file)}...")
        EOLExtractor(args, file)

    # Aggregate and save results
    collect_results(args)


if __name__ == "__main__":
    main()