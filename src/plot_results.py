"""
Standalone plotting functions for estimation results.

This module provides functions to plot estimation results from saved .npz files
without requiring an estimator instance. Useful for post-processing and visualization
of saved estimation results.
"""

from pathlib import Path
from typing import Optional, Dict, Sequence
import numpy as np
import matplotlib.pyplot as plt


def plot_results(
    results_path: str,
    state_indices: Optional[Sequence[int]] = None,
):
    """
    Load and plot estimation results from saved file.
    
    This function can be used independently of the estimator object to plot
    results from saved .npz files. Plots in a 2x2 grid:
    - Top left: Driving Motor Torque (u1)
    - Top right: Load Motor Torque (u2, estimated and optionally true)
    - Bottom left: Driving Motor Velocity (first velocity state)
    - Bottom right: Load Motor Velocity (last velocity state)
    
    Parameters:
    -----------
    results_path : str
        Path to saved results file (.npz). If relative, loads from data/
    state_indices : sequence of int, optional
        Deprecated parameter, kept for backward compatibility. Not used.
    """
    # Resolve input path from data/ by default if only filename provided
    input_path = Path(results_path)
    if input_path.parent == Path('.'):
        input_path = Path('data') / input_path.name

    # Load saved arrays. allow_pickle=True is required because some entries
    # (like reference_data) are stored as Python dictionaries.
    data = np.load(str(input_path), allow_pickle=True)
    # Estimated states and inputs produced by the estimator
    xhat = data["xhat"]
    uhat = data["uhat"]
    
    # Optional ground-truth data for comparison
    ref = data["reference_data"].item() if "reference_data" in data else None
    if ref is not None:
        ref_data_rows = ref['xout_rows']
        ref_data = ref['xout']
        ref_u2 = ref['u2']
    else:
        ref_data_rows = None
        ref_data = None
        ref_u2 = None

    if ref_data_rows is not None and ref_data is not None:
        ref_data_rows = np.asarray(ref_data_rows).astype(int)
        ref_row_to_idx = {int(row): pos for pos, row in enumerate(ref_data_rows)}
    else:
        ref_row_to_idx = {}

    t = data["t"]
    y = data["y"]
    u1 = data["u1"]
    
    # Skip first 250 and last 50 data points for all plots
    skip_start = 250
    skip_end = 50
    if len(t) > skip_start + skip_end:
        t = t[skip_start:-skip_end]
        y = y[skip_start:-skip_end, :]
        if u1.ndim == 1:
            u1 = u1[skip_start:-skip_end]
        else:
            u1 = u1[:, skip_start:-skip_end]
        xhat = xhat[:, skip_start:-skip_end]
        uhat = uhat[:, skip_start:-skip_end]
        if ref_data is not None:
            ref_data = ref_data[skip_start:-skip_end, :]
        if ref_u2 is not None:
            ref_u2 = ref_u2[skip_start:-skip_end]
    
    # Determine state categories from minimal-form ordering
    n_states = xhat.shape[0]
    if n_states % 2 == 0:
        torque_state_boundary = (n_states // 2)
    else:
        torque_state_boundary = (n_states - 1) // 2

    # Determine first and last velocity state indices
    first_velocity_idx = torque_state_boundary
    last_velocity_idx = n_states - 1

    # Create 2x2 grid as in simulate_data
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))

    # Upper row: Input torques
    # Top left: u1 (Driving Motor Torque)
    if u1.ndim == 1:
        axes[0, 0].plot(t, u1, "r", linewidth=1.5, label='simulated')
    else:
        axes[0, 0].plot(t, u1[0, :], "r", linewidth=1.5, label='simulated')
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Torque (Nm)")
    axes[0, 0].set_title("Driving Motor Torque")
    axes[0, 0].text(0.02, 1.06, "a)", transform=axes[0, 0].transAxes, fontsize=12, verticalalignment='top')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Top right: u2 (Load Motor Torque)
    if ref_u2 is not None:
        axes[0, 1].plot(t, ref_u2, "r", alpha=0.7, linewidth=1.5, label='simulated')
    axes[0, 1].plot(t, uhat[0, :], "C0", alpha=0.8, linewidth=1.2, label='estimated')
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Torque (Nm)")
    axes[0, 1].set_title("Load Motor Torque")
    axes[0, 1].text(0.02, 1.06, "b)", transform=axes[0, 1].transAxes, fontsize=12, verticalalignment='top')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Lower row: First and last velocity states
    # Bottom left: First velocity state (Driving Motor Velocity)
    #axes[1, 0].plot(t, xhat[first_velocity_idx, :], "r", linewidth=1.5, label='estimated')
    
    # Plot true state if available
    ref_pos = ref_row_to_idx.get(int(first_velocity_idx))
    if ref_pos is not None and ref_data is not None:
        true_trajectory = ref_data[:, ref_pos]
        if true_trajectory.ndim == 1 and true_trajectory.size == t.size:
            axes[1, 0].plot(t, true_trajectory, color='r', linewidth=1.5, label='simulated')
    
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Velocity (rad/s)")
    axes[1, 0].set_title("Driving Motor Velocity")
    axes[1, 0].text(0.02, 1.06, "c)", transform=axes[1, 0].transAxes, fontsize=12, verticalalignment='top')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Bottom right: Last velocity state (Load Motor Velocity)
    axes[1, 1].plot(t, xhat[last_velocity_idx, :], "r", linewidth=1.5, label='estimated')
    
    # Plot true state if available
    ref_pos = ref_row_to_idx.get(int(last_velocity_idx))
    if ref_pos is not None and ref_data is not None:
        true_trajectory = ref_data[:, ref_pos]
        if true_trajectory.ndim == 1 and true_trajectory.size == t.size:
            axes[1, 1].plot(t, true_trajectory, color='C0', alpha=0.85, linewidth=1.2, label='simulated')
    
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Velocity (rad/s)")
    axes[1, 1].set_title("Load Motor Velocity")
    axes[1, 1].text(0.02, 1.06, "d)", transform=axes[1, 1].transAxes, fontsize=12, verticalalignment='top')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

    # Second plot of intermediate disk states
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(t, xhat[11, :], "r", linewidth=1.5, label='estimated')
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Torque (Nm)")
    axes[0].set_title("Disk 10 Torque")
    axes[0].text(0.02, 1.06, "a)", transform=axes[0].transAxes, fontsize=12, verticalalignment='top')
    axes[0].grid(True, alpha=0.3)
    

    axes[1].plot(t, xhat[30, :], "r", linewidth=1.5, label='estimated')
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Velocity (rad/s)")
    axes[1].set_title("Disk 10 Velocity")
    axes[1].text(0.02, 1.06, "b)", transform=axes[1].transAxes, fontsize=12, verticalalignment='top')
    axes[1].grid(True, alpha=0.3)

    # Add true states
    ref_pos = ref_row_to_idx.get(int(11))
    if ref_pos is not None and ref_data is not None:
        true_trajectory = ref_data[:, ref_pos]
        if true_trajectory.ndim == 1 and true_trajectory.size == t.size:
            axes[0].plot(t, true_trajectory, color='C0', alpha=0.85, linewidth=1.2, label='simulated')
    ref_pos = ref_row_to_idx.get(int(31))
    if ref_pos is not None and ref_data is not None:
        true_trajectory = ref_data[:, ref_pos]
        if true_trajectory.ndim == 1 and true_trajectory.size == t.size:
            axes[1].plot(t, true_trajectory, color='C0', alpha=0.85, linewidth=1.2, label='simulated')

    axes[0].legend()
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    


if __name__ == "__main__":
    plot_results('data/mhe_results.npz')

