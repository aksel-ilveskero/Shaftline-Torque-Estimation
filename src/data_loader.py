"""
Data loading utilities for state estimation.

This module provides functions to load measurement data from various formats
(CSV, Feather) and return standardized data structures compatible with the
estimation framework.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence
import numpy as np
import pandas as pd


def plot_series(
    data_series: Sequence[np.ndarray],
    titles: Sequence[str],
    xlabel: str = "Sample index",
    ylabel: str = "",
    figsize: Optional[tuple[float, float]] = None,
) -> None:
    """
    Plot a collection of time-series on individual subplots.

    Parameters
    ----------
    data_series : sequence of np.ndarray
        Iterable of one-dimensional arrays to plot.
    titles : sequence of str
        Titles for each subplot. Must match the length of `data_series`.
    xlabel : str, optional
        Label for the x-axis. Defaults to "Sample index".
    ylabel : str, optional
        Shared label applied to all subplots' y-axes. Defaults to empty string.
    figsize : tuple, optional
        Figure size passed to matplotlib.
    """
    if not data_series:
        return

    if len(data_series) != len(titles):
        raise ValueError("`data_series` and `titles` must have the same length.")

    import matplotlib.pyplot as plt

    n_plots = len(data_series)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize or (8, 2.5 * n_plots), squeeze=False, sharex=True)

    axes = axes.flatten()

    for ax, series, title in zip(axes, data_series, titles):
        ax.plot(series)
        ax.set_title(title)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.grid(True)

    fig.tight_layout()
    plt.show()


def load_csv(csv_path: str | Path, start_idx: int = 14500, end_idx: int = 16000) -> Dict:
    """
    Load step measurement CSV and return standardized data structure.
    
    Reads CSV file with encoder and torque measurements, computes angular
    velocities from encoder angles, and returns processed data arrays.
    
    Parameters:
    -----------
    csv_path : str | Path
        Path to CSV file
    start_idx : int, optional
        Start index for data slice (default: 14500)
    end_idx : int, optional
        End index for data slice (default: 16000)
        
    Returns:
    --------
    dict
        Dictionary with keys:
        - 'time': time vector (seconds)
        - 'measurements': dict with 'torque' and 'velocity' arrays
        - 'inputs': dict with 'motor' and 'load' arrays
        - 'reference': dict with reference measurements
        
    Expected CSV header:
      time,en1time,en1angle,en2time,en2angle,torque1,torque2,MotorTorque,MotorVelocity,PropellerTorque,PropellerVelocity
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)

    time = data["time"][start_idx:end_idx]
    enc1_time = data["en1time"][start_idx:end_idx]
    enc1_angle = data["en1angle"][start_idx:end_idx]
    enc2_time = data["en2time"][start_idx:end_idx]
    enc2_angle = data["en2angle"][start_idx:end_idx]
    torque1 = data["torque1"][start_idx:end_idx]
    torque2 = data["torque2"][start_idx:end_idx]
    motor_torque = data["MotorTorque"][start_idx:end_idx]
    motor_velocity = data["MotorVelocity"][start_idx:end_idx]
    prop_torque = data["PropellerTorque"][start_idx:end_idx]

    # Compute encoder angular velocities (rad/s) with unwrap to handle 2π discontinuities
    enc1_velocity = np.gradient(np.unwrap(enc1_angle), enc1_time)
    enc2_velocity = np.gradient(np.unwrap(enc2_angle), enc2_time)

    plot_series(
        [motor_torque, torque1, prop_torque],
        ["Motor Torque", "Shaft Torque 1", "Propeller Torque"],
        ylabel="Magnitude",
    )
    
    return {
        'time': time,
        'measurements': {
            'torque': np.column_stack([torque1, torque2]) if len(torque1) > 0 else np.array([]).reshape(0, 0),
            'velocity': np.column_stack([enc1_velocity, enc2_velocity]) if len(enc1_velocity) > 0 else np.array([]).reshape(0, 0)
        },
        'inputs': {
            'motor': motor_torque,
            'load': prop_torque
        },
        "reference": {
            "xout_rows": [8,18,26,27],
            "xout": np.column_stack([torque1, torque2, enc1_velocity, enc2_velocity]),
            "u2": prop_torque,
        },
    }


def load_feather(feather_path: str | Path = 'data/ice_aligned.feather', 
                 start_idx: int = 6000, end_idx: int = 8000,
                 sample_id: Optional[int] = None) -> Dict:
    """
    Load measurement data from Feather file with ice excitation data.
    
    Reads a Feather file containing aligned ice excitation measurements,
    groups data by sample_id, and extracts time-series data for a specific
    sample slice.
    
    Parameters:
    -----------
    feather_path : str | Path, optional
        Path to Feather file (default: 'data/ice_aligned.feather')
    start_idx : int, optional
        Start index for data slice (default: 6000)
    end_idx : int, optional
        End index for data slice (default: 8000)
    sample_id : int, optional
        Specific sample_id to load. If None, loads first sample.
        
    Returns:
    --------
    dict
        Dictionary with keys:
        - 'time': time vector (seconds)
        - 'measurements': dict with 'torque' and 'velocity' arrays
        - 'inputs': dict with 'motor' and 'load' arrays
        - 'reference': dict with reference measurements
    """
    feather_path = Path(feather_path)
    if not feather_path.exists():
        raise FileNotFoundError(f"Feather file not found: {feather_path}")
    
    measurements = pd.read_feather(str(feather_path))

    # Group by sample_id
    sample_groups = measurements.groupby('sample_id')
    
    # Select sample
    if sample_id is None:
        sample_id = measurements['sample_id'].iloc[0]
    
    if sample_id not in sample_groups.groups:
        raise ValueError(f"Sample ID {sample_id} not found in data")
    
    group = sample_groups.get_group(sample_id)
    group = group.sort_values('time')

    # Select slice
    sample_slice = group.iloc[start_idx:end_idx]

    # Extract the columns
    time = sample_slice["time"].to_numpy()
    e1 = sample_slice["E1"].to_numpy()  # velocity measurement
    e2 = sample_slice["E2"].to_numpy()  # velocity measurement
    t1 = sample_slice["T1"].to_numpy()  # torque measurement
    t2 = sample_slice["T2"].to_numpy()  # torque measurement
    u_p = sample_slice["u_p"].to_numpy()  # load torque input
    u_m = sample_slice["u_m"].to_numpy()  # motor torque input

    plot_series(
        [u_m, e1, t1],
        [
            "Motor Torque Input",
            "Encoder 1 Velocity",
            "Torque Sensor 1",
        ],
        ylabel="Magnitude",
    )

    return {
        'time': time,
        'measurements': {
            'torque': np.column_stack([t1, t2]) if len(t1) > 0 else np.array([]).reshape(0, 0),
            'velocity': np.column_stack([e1, e2]) if len(e1) > 0 else np.array([]).reshape(0, 0)
        },
        'inputs': {
            'motor': u_m,
            'load': u_p
        },
        "reference": {
            "xout_rows": [8,18,26,27],
            "xout": np.column_stack([t1, t2, e1, e2]),
            "u2": u_p,
        },
    }

