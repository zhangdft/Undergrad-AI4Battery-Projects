
import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Tuple, Dict, Any
import NewareNDA

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

class BatteryBatch:
    def __init__(self, file_path: str) -> None:
        self.file_path = self.cook()

    def cook(self):
        