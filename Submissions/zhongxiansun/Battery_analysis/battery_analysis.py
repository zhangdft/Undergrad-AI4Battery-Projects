"""
battery_analysis.py - Battery Cycle Data Analysis from Neware NDAX Files

This script reads battery cycling data from a Neware .ndax file, processes charge/discharge
capacity and voltage, computes dQ/dV, and visualizes the results across cycles.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Tuple, Dict, Any
import NewareNDA

# Global plot settings
plt.rcParams.update(
    {
        "font.size": 12,
        "figure.figsize": (8, 10),
        "figure.dpi": 300,
        "font.family": "sans-serif",
    }
)


def smooth(
    data: np.ndarray, window_length: int = 191, polyorder: int = 3
) -> np.ndarray:
    """Apply Savitzky-Golay filter to smooth data.
    Args:
        data (np.ndarray): Input 1D array to smooth.
        window_length (int): The length of the filter window (must be odd). Default is 191.
        polyorder (int): The order of the polynomial used to fit the samples. Default is 3.
    Returns:
        np.ndarray: Smoothed array of same length as input.
    """
    return savgol_filter(
        data, window_length=window_length, polyorder=polyorder, mode="nearest"
    )


class BatteryData:
    """
    A class to handle battery cycling data from Neware .ndax files.

    This class loads data, extracts cycle information, computes smoothed capacity/voltage,
    and generates plots for voltage vs capacity, dQ/dV analysis, and capacity fade over cycles.

    Attributes:
        file_path (str): Path to the .ndax file.
        data (pd.DataFrame): Loaded raw battery data.
        time (pd.Series): Time in seconds since start.
        cycles (np.ndarray): Unique cycle numbers.
        groups (pd.core.groupby.DataFrameGroupBy): Grouped data by cycle.
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.data = self.load_data()
        # self.time = self.timestamp()
        self.cycles = self.get_unique_cycles()
        self.groups = self.data.groupby("Cycle")

    def load_data(self) -> pd.DataFrame:
        """Load battery data from .ndax file.

        Returns:
            pd.DataFrame: Raw battery data.

        Raises:
            FileNotFoundError: If file is not found.
            Exception: For any other loading errors.
        """
        try:
            data = NewareNDA.read(self.file_path)
            if data.empty:
                raise ValueError(f"No data found in file: {self.file_path}")
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.file_path}")
        except Exception as e:
            raise Exception(f"Error reading file {self.file_path}: {e}")

    def timestamp(self) -> pd.Series:
        """
        Convert timestamps to elapsed seconds from the first measurement.

        Returns:
            pd.Series: Series of time in seconds (float).
        """
        time = pd.to_datetime(self.data["Timestamp"], format="%Y-%m-%d %H:%M:%S")
        return (time - time.iloc[0]).dt.total_seconds()

    def get_unique_cycles(self) -> np.ndarray:
        """Get unique cycle numbers in the dataset.

        Returns:
            np.ndarray: Sorted array of unique cycle indices.
        """
        return self.data["Cycle"].unique()

    def get_cycle_data(
        self, cycle_number: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract and smooth discharge/charge capacity and voltage for a given cycle.

        Args:
            cycle_number (int): The cycle number to extract.

        Returns:
            Tuple[np.ndarray, ...]: discharge capacity, charge capacity,
                                    discharge voltage, charge voltage.
        """
        try:
            group = self.groups.get_group(cycle_number)
        except KeyError:
            raise KeyError(f"Cycle number {cycle_number} not found in data.")

        dis_mask = group["Current(mA)"] < 0
        ch_mask = group["Current(mA)"] > 0

        dis_cap = group["Discharge_Capacity(mAh)"][dis_mask].values
        ch_cap = group["Charge_Capacity(mAh)"][ch_mask].values
        dis_vol = group["Voltage"][dis_mask].values
        ch_vol = group["Voltage"][ch_mask].values

        # Sort by voltage before returning
        dis_sort_idx = np.argsort(dis_vol)
        ch_sort_idx = np.argsort(ch_vol)

        return (
            dis_cap[dis_sort_idx],
            ch_cap[ch_sort_idx],
            dis_vol[dis_sort_idx],
            ch_vol[ch_sort_idx],
        )

    def plot_cycle(self, cycle_number: int, cmap) -> None:
        """Plot voltage-capacity and dQ/dV curves for a single cycle.

        Args:
            cycle_number (int): Cycle number to plot.
            cmap: Matplotlib colormap instance.
        """
        dis_cap, ch_cap, dis_vol, ch_vol = self.get_cycle_data(cycle_number)
        color = cmap(cycle_number / 50)

        # dQ/dV calculation via gradient
        dq_dv_dis = np.gradient(dis_cap, dis_vol)
        dq_dv_ch = np.gradient(ch_cap, ch_vol)

        ax1.plot(ch_cap, ch_vol, color=color, label=f"Cycle {cycle_number}")
        ax1.plot(dis_cap, dis_vol, color=color)
        ax2.plot(dis_vol, smooth(dq_dv_dis, window_length=81, polyorder=3), color=color)
        ax2.plot(ch_vol, smooth(dq_dv_ch, window_length=81, polyorder=3), color=color)

    def plot_capacity_fade(self, cmap) -> None:
        """Plot maximum discharge capacity vs cycle number.

        Args:
            cmap: Matplotlib colormap instance.
        """
        cap_per_cycle = self.groups["Discharge_Capacity(mAh)"].max()
        ax3.plot(
            self.cycles,
            cap_per_cycle,
            color=cmap(0.5),  # single color for capacity fade
            marker="o",
            linestyle="-",
            markersize=4,
        )

    def plot_all(self, start_cycle: int = 1, end_cycle: int = 100) -> None:
        """
        Generate all three subplots: V vs Q, dQ/dV, and capacity fade.

        Args:
            start_cycle (int): First cycle to plot in the first two axes. Default is 1.
            end_cycle (int): Last cycle to plot. Default is 100.
        """
        cmap = plt.cm.viridis

        for cycle in range(start_cycle, min(end_cycle + 1, self.cycles.max() + 1)):
            if cycle in self.cycles:
                self.plot_cycle(cycle, cmap)  # ax1, ax2: V-Q and dQ/dV plots

        self.plot_capacity_fade(cmap)  # ax3: capacity fade plot

        ax1.set_xlabel("Capacity (mAh)", fontsize=12)
        ax1.set_ylabel("Voltage (V)", fontsize=12)
        ax1.set_title("Voltage vs Capacity", fontsize=14)

        ax2.set_xlabel("Voltage (V)", fontsize=12)
        ax2.set_ylabel("dQ/dV (mAh/V)", fontsize=12)
        ax2.set_title("dQ/dV vs Voltage", fontsize=14)

        ax3.set_xlabel("Cycle Number", fontsize=12)
        ax3.set_ylabel("Discharge Capacity (mAh)", fontsize=12)
        ax3.set_title("Discharge Capacity over Cycles", fontsize=14)

        plt.tight_layout()
        # save as png file
        plt.savefig("Battery_Cycles.png", dpi=1200, bbox_inches="tight")
        # save as svg file
        plt.savefig("Battery_Cycles.svg", dpi=1200, bbox_inches="tight")
        plt.close()


def main() -> None:
    """Main execution function."""
    file_path = "FullCell.ndax"
    battery = BatteryData(file_path)
    battery.plot_all(start_cycle=7, end_cycle=41)
    print(
        f"Analysis complete. Plot saved as 'Battery_Cycles.png' and 'Battery_Cycles.svg'."
    )


if __name__ == "__main__":
    # Create subplots once at module level (only for plotting context)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
    main()
