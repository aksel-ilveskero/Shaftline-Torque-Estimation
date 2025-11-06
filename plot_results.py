"""
Standalone plotting functions for estimation results.

This module provides functions to plot estimation results from saved .npz files
without requiring an estimator instance. Useful for post-processing and visualization
of saved estimation results.
"""

from pathlib import Path
from typing import Optional, Dict
import numpy as np
import matplotlib.pyplot as plt


def plot_results(results_path: str, reference_data: Optional[Dict] = None):
    """
    Load and plot estimation results from saved file.
    
    This function can be used independently of the estimator object to plot
    results from saved .npz files.
    
    Parameters:
    -----------
    results_path : str
        Path to saved results file (.npz). If relative, loads from data/
    reference_data : dict, optional
        Reference data for comparison with keys:
        - 'torque': measured torque data (N x n_torque_sensors)
        - 'velocity': measured velocity data (N x n_velocity_sensors)
        - 'prop_torque': propeller/load torque (N,)
        - 'u1_noisy': noisy motor torque input (N,) or (1 x N)
    """
    # Resolve input path from data/ by default if only filename provided
    input_path = Path(results_path)
    if input_path.parent == Path('.'):
        input_path = Path('data') / input_path.name

    data = np.load(str(input_path), allow_pickle=True)
    xhat = data["xhat"]
    uhat = data["uhat"]
    t = data["t"]
    y = data["y"]
    u1 = data["u1"]
    
    # Load sensor metadata
    torque_sensors = data.get("torque_sensors", [])  # Disk numbers
    velocity_sensors = data.get("velocity_sensors", [])  # Disk numbers
    torque_sensor_state_indices = data.get("torque_sensor_state_indices", [])
    velocity_sensor_state_indices = data.get("velocity_sensor_state_indices", [])
    sensor_types = data.get("sensor_types", [])
    sensor_indices = data.get("sensor_indices", [])

    # Use converted state indices if available, otherwise fall back to disk numbers (for backward compatibility)
    if len(torque_sensor_state_indices) > 0:
        torque_state_indices = torque_sensor_state_indices
    else:
        torque_state_indices = torque_sensors  # Fallback for old files
    
    if len(velocity_sensor_state_indices) > 0:
        velocity_state_indices = velocity_sensor_state_indices
    else:
        velocity_state_indices = velocity_sensors  # Fallback for old files

    true_x = data.get("true_x")
    true_u2 = data.get("true_u2")
    u1_noisy = data.get("u1_noisy")  # Check if u1_noisy is saved in file
    if u1_noisy is None and reference_data and 'u1_noisy' in reference_data:
        u1_noisy = reference_data['u1_noisy']

    # Determine number of plots needed
    n_plots = 1  # Inputs plot
    if len(torque_state_indices) > 0:
        n_plots += len(torque_state_indices)
    if len(velocity_state_indices) > 0:
        n_plots += len(velocity_state_indices)
    
    fig, axs = plt.subplots(n_plots, figsize=(10, 3*n_plots))
    if n_plots == 1:
        axs = [axs]
    
    plot_idx = 0
    
    # Plot inputs
    if u1_noisy is not None:
        if isinstance(u1_noisy, np.ndarray) and u1_noisy.ndim == 2:
            axs[plot_idx].plot(t, u1_noisy[0, :], 'k', alpha=0.5, linewidth=2, label='measured input u1')
        else:
            axs[plot_idx].plot(t, u1_noisy, 'k', alpha=0.5, linewidth=2, label='measured input u1')
    else:
        if u1.ndim == 1:
            axs[plot_idx].plot(t, u1, 'k', alpha=0.5, linewidth=2, label='input u1')
        else:
            axs[plot_idx].plot(t, u1[0, :], 'k', alpha=0.5, linewidth=2, label='input u1')
    
    if true_u2 is not None and true_u2.size > 0:
        axs[plot_idx].plot(t, true_u2, 'b', alpha=0.5, linewidth=2, label='true input u2')
    axs[plot_idx].scatter(t, uhat[0, :], alpha=0.25, color='r', edgecolor=None, label='estimated input')
    
    if reference_data and 'prop_torque' in reference_data:
        axs[plot_idx].plot(t, reference_data['prop_torque'], alpha=0.25, color='g', label='known propeller input')
    
    axs[plot_idx].set_ylabel('Torque (Nm)')
    axs[plot_idx].legend()
    axs[plot_idx].set_title('Inputs')
    plot_idx += 1
    
    # Plot torque sensors
    if len(torque_state_indices) > 0:
        if reference_data and 'torque' in reference_data:
            ref_torque = reference_data['torque']
            if ref_torque.ndim == 1:
                ref_torque = ref_torque.reshape(-1, 1)
        
        for i, sensor_idx in enumerate(torque_state_indices):
            disk_num = torque_sensors[i] if i < len(torque_sensors) else sensor_idx
            axs[plot_idx].scatter(t, xhat[sensor_idx, :], alpha=0.25, color='r', edgecolor=None, label='estimated torque')
            
            if reference_data and 'torque' in reference_data:
                if ref_torque.shape[1] > i:
                    axs[plot_idx].plot(t, ref_torque[:, i], label='measured torque')
            
            if true_x is not None and true_x.size > 0:
                axs[plot_idx].plot(t, true_x[:, sensor_idx], color='k', alpha=0.5, label='true torque')
            
            axs[plot_idx].set_ylabel('Shaft Torque (Nm)')
            axs[plot_idx].legend()
            axs[plot_idx].set_title(f'Torque Sensor at Disk {disk_num} (State {sensor_idx})')
            plot_idx += 1
    
    # Plot velocity sensors
    if len(velocity_state_indices) > 0:
        if reference_data and 'velocity' in reference_data:
            ref_velocity = reference_data['velocity']
            if ref_velocity.ndim == 1:
                ref_velocity = ref_velocity.reshape(-1, 1)
        
        for i, sensor_idx in enumerate(velocity_state_indices):
            disk_num = velocity_sensors[i] if i < len(velocity_sensors) else sensor_idx
            axs[plot_idx].scatter(t, xhat[sensor_idx, :], alpha=0.25, color='r', edgecolor=None, label='estimated velocity')
            
            if reference_data and 'velocity' in reference_data:
                if ref_velocity.shape[1] > i:
                    axs[plot_idx].plot(t, ref_velocity[:, i], label='measured velocity')
            
            if true_x is not None and true_x.size > 0:
                axs[plot_idx].plot(t, true_x[:, sensor_idx], color='k', alpha=0.5, label='true velocity')
            
            axs[plot_idx].set_ylabel('Velocity (rad/s)')
            axs[plot_idx].legend()
            axs[plot_idx].set_title(f'Velocity Sensor at Disk {disk_num} (State {sensor_idx})')
            plot_idx += 1
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <results_file.npz>")
        print("Example: python plot_results.py data/mhe_results.npz")
        sys.exit(1)
    
    results_file = sys.argv[1]
    plot_results(results_file)

