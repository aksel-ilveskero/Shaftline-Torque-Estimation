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
    results from saved .npz files.
    
    Parameters:
    -----------
    results_path : str
        Path to saved results file (.npz). If relative, loads from data/
    state_indices : sequence of int, optional
        State indices to plot. Defaults to indices corresponding to measured sensors
        (data['sensor_indices']) when available, otherwise plots all states.
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
    
    # Load sensor metadata
    # Sensor metadata is optional because logged data can come from different configurations
    torque_sensors = data.get("torque_sensors")
    velocity_sensors = data.get("velocity_sensors")
    torque_sensor_state_indices = data.get("torque_sensor_state_indices")
    velocity_sensor_state_indices = data.get("velocity_sensor_state_indices")
    sensor_types = data.get("sensor_types", [])
    sensor_indices = data.get("sensor_indices", [])

    # Normalize to lists for consistent handling
    def _to_int_list(value):
        if value is None:
            return []
        arr = np.asarray(value).astype(int)
        return arr.tolist()

    torque_sensors = _to_int_list(torque_sensors)
    velocity_sensors = _to_int_list(velocity_sensors)
    torque_sensor_state_indices = _to_int_list(torque_sensor_state_indices)
    velocity_sensor_state_indices = _to_int_list(velocity_sensor_state_indices)
    sensor_types = list(np.asarray(sensor_types).astype(str)) if len(sensor_types) > 0 else []
    sensor_indices = _to_int_list(sensor_indices)
    
    # Determine which states to plot
    if state_indices is None:
        if len(sensor_indices) > 0:
            # Use order provided by sensor indices (may have duplicates, keep first occurrence)
            seen = set()
            state_indices_to_plot = []
            for idx in sensor_indices:
                if idx not in seen:
                    state_indices_to_plot.append(int(idx))
                    seen.add(int(idx))
        else:
            state_indices_to_plot = list(range(xhat.shape[0]))
    else:
        state_indices_to_plot = [int(idx) for idx in state_indices]

    if len(state_indices_to_plot) == 0:
        raise ValueError("No state indices provided to plot.")

    # Maps for quick lookup of sensor indices by state index
    sensor_index_map = {}
    for meas_idx, state_idx in enumerate(sensor_indices):
        sensor_index_map.setdefault(int(state_idx), []).append(meas_idx)

    torque_index_map = {int(idx): pos for pos, idx in enumerate(torque_sensor_state_indices)}
    velocity_index_map = {int(idx): pos for pos, idx in enumerate(velocity_sensor_state_indices)}

    # Determine state categories from minimal-form ordering
    n_states = xhat.shape[0]
    if n_states % 2 == 0:
        torque_state_boundary = (n_states // 2)
    else:
        torque_state_boundary = (n_states - 1) // 2

    # Determine number of plots needed
    n_plots = 1 + len(state_indices_to_plot)
    
    fig, axs = plt.subplots(n_plots, figsize=(10, 3*n_plots))
    if n_plots == 1:
        axs = [axs]
    
    plot_idx = 0
    
    # Plot inputs
    if u1.ndim == 1:
        axs[plot_idx].plot(t, u1, 'k', alpha=0.5, linewidth=2, label='input u1')
    else:
        axs[plot_idx].plot(t, u1[0, :], 'k', alpha=0.5, linewidth=2, label='input u1')
    
    if ref_u2 is not None:
        axs[plot_idx].plot(t, ref_u2, alpha=0.5, linewidth=2, label='true input u2')

    axs[plot_idx].plot(t, uhat[0, :], alpha=0.5, color='r', label='estimated input u2')
    
    axs[plot_idx].set_ylabel('Torque (Nm)')
    axs[plot_idx].legend()
    axs[plot_idx].set_title('Inputs')
    plot_idx += 1

    # Plot states
    for state_idx in state_indices_to_plot:
        state_type = None
        label_suffix = f"State {state_idx}"
        ylabel = 'State Value'
        sensor_pos = None

        # Determine state type and sensor position
        if state_idx < torque_state_boundary:
            state_type = 'torque'
            label_suffix = f"Torque State {state_idx}"
            ylabel = 'Shaft Torque (Nm)'
            sensor_pos = torque_index_map.get(state_idx)
        elif state_idx < n_states:
            state_type = 'velocity'
            label_suffix = f"Velocity State {state_idx}"
            ylabel = 'Velocity (rad/s)'
            sensor_pos = velocity_index_map.get(state_idx)

        # Plot estimated state
        axs[plot_idx].plot(t, xhat[state_idx, :], color='r', alpha=0.7, linewidth=1.5, label=f'estimated {label_suffix.lower()}')

        # Plot true state if available
        ref_pos = ref_row_to_idx.get(int(state_idx))
        if ref_pos is not None and ref_data is not None:
            true_trajectory = ref_data[:,ref_pos]
            if true_trajectory.shape == t.shape or true_trajectory.shape == t.shape[::-1]:
                axs[plot_idx].plot(t, true_trajectory, color='C0', alpha=0.85, linewidth=1.2, label=f'true {label_suffix.lower()}')
            elif true_trajectory.ndim == 1 and true_trajectory.size == t.size:
                axs[plot_idx].plot(t, true_trajectory, color='C0', alpha=0.85, linewidth=1.2, label=f'true {label_suffix.lower()}')

        # Plot measurements
        measured_indices = sensor_index_map.get(state_idx, [])
        if measured_indices:
            for meas_idx in measured_indices:
                meas_label = 'measurement'
                if meas_idx < len(sensor_types):
                    meas_label = f"measurement ({sensor_types[meas_idx]})"
                axs[plot_idx].plot(t, y[:, meas_idx], color='k', alpha=0.5, linewidth=1, label=meas_label)

        # Add disk label to title if available
        disk_label = ""
        if state_type == 'torque' and torque_sensors is not None and len(torque_sensors) > 0:
            if sensor_pos is not None and sensor_pos < len(torque_sensors):
                disk_label = f"Disk {torque_sensors[sensor_pos]}"
        elif state_type == 'velocity' and velocity_sensors is not None and len(velocity_sensors) > 0:
            if sensor_pos is not None and sensor_pos < len(velocity_sensors):
                disk_label = f"Disk {velocity_sensors[sensor_pos]}"
        
        title_parts = [label_suffix]
        if disk_label:
            title_parts.append(f"({disk_label})")
        axs[plot_idx].set_title(' '.join(title_parts))
        axs[plot_idx].set_ylabel(ylabel)
        axs[plot_idx].legend()
        plot_idx += 1
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_results('data/mhe_results.npz', state_indices=[8,18,26,40])

