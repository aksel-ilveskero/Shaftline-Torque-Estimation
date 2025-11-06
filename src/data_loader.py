"""
Data loading and simulation module for state estimation.

This module provides functions to load measurement data from various formats
(CSV, Feather) and simulate data using OpenTorsion assemblies. All functions
return standardized data structures compatible with the estimation framework.
"""

from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        'reference': {
            'torque': np.column_stack([torque1, torque2]),
            'velocity': np.column_stack([enc1_velocity, enc2_velocity]),
            'prop_torque': prop_torque
        }
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
        'reference': {
            'torque': np.column_stack([t1, t2]),
            'velocity': np.column_stack([e1, e2]),
            'prop_torque': u_p
        }
    }


def simulate_data(assembly, config: Dict) -> Dict:
    """
    Simulate data using OpenTorsion assembly.
    
    Parameters:
    -----------
    assembly : ot.Assembly
        OpenTorsion assembly object
    config : dict
        Simulation configuration with keys:
        - 'time': time vector or dict with 'start', 'end', 'n_points'
        - 'u1': motor torque input (N,) or function
        - 'u2': load torque input (N,) or function
        - 'process_noise_std': float (default 0.01)
        - 'measurement_noise_std': float (default 0.1)
        - 'initial_state': ndarray (optional)
        - 'measurement_config': dict with sensor locations
        
    Returns:
    --------
    dict
        Dictionary with keys:
        - 'time': time vector
        - 'measurements': dict with 'torque' and 'velocity' arrays
        - 'inputs': dict with 'motor' and 'load' arrays
        - 'reference': dict with true states and inputs
    """
    from ot_assembly import minimal_state_space
    from utils import c2d
    
    # Get state-space matrices
    A_c, B_c, C_c, D_c = minimal_state_space(assembly)
    
    # Setup measurement matrix if provided
    if 'measurement_config' in config:
        from ot_assembly import get_state_structure, create_measurement_matrix
        state_struct = get_state_structure(assembly)
        C_meas, _ = create_measurement_matrix(
            n_states=state_struct['n_states'],
            torque_sensor_locations=config['measurement_config'].get('torque_sensors', []),
            velocity_sensor_locations=config['measurement_config'].get('velocity_sensors', [])
        )
    else:
        C_meas = C_c
    
    # Setup time vector
    if isinstance(config['time'], dict):
        t = np.linspace(config['time']['start'], config['time']['end'], config['time']['n_points'])
    else:
        t = np.array(config['time'])
    
    N = len(t)
    dt = t[1] - t[0] if N > 1 else 0.01
    
    # Discretize
    A, B = c2d(A_c, B_c, dt)
    
    # Setup inputs
    u1 = config.get('u1', np.zeros(N))
    u2 = config.get('u2', np.zeros(N))
    
    if callable(u1):
        u1 = np.array([u1(ti) for ti in t])
    if callable(u2):
        u2 = np.array([u2(ti) for ti in t])
    
    u1 = np.asarray(u1).flatten()
    u2 = np.asarray(u2).flatten()
    
    if len(u1) != N:
        raise ValueError(f"u1 length {len(u1)} doesn't match time length {N}")
    if len(u2) != N:
        raise ValueError(f"u2 length {len(u2)} doesn't match time length {N}")
    
    # Initialize state
    n_states = A.shape[0]
    if 'initial_state' in config:
        x = np.asarray(config['initial_state']).flatten()
        if len(x) != n_states:
            raise ValueError(f"initial_state length {len(x)} doesn't match system states {n_states}")
    else:
        x = np.zeros(n_states)
    
    # Noise parameters
    process_noise_std = config.get('process_noise_std', 0.01)
    measurement_noise_std = config.get('measurement_noise_std', 0.1)
    
    # Simulation arrays
    xout = np.zeros((N, n_states))
    y = np.zeros((N, C_meas.shape[0]))
    
    # Simulation loop
    for i in range(N):
        xout[i] = x.copy()
        
        # Calculate outputs
        y_clean = C_meas @ x
        y[i] = y_clean + np.random.normal(0, measurement_noise_std, C_meas.shape[0])
        
        # State update with process noise
        if i < N - 1:
            process_noise = np.random.normal(0, process_noise_std, n_states)
            u_input = np.array([u1[i], u2[i]])
            x = A @ x + B @ u_input + process_noise
    
    # Extract torque and velocity measurements if measurement_config provided
    if 'measurement_config' in config:
        torque_sensors = config['measurement_config'].get('torque_sensors', [])
        velocity_sensors = config['measurement_config'].get('velocity_sensors', [])
        
        # Map measurements to torque/velocity based on sensor metadata
        from ot_assembly import create_measurement_matrix
        _, sensor_metadata = create_measurement_matrix(
            n_states=n_states,
            torque_sensor_locations=torque_sensors,
            velocity_sensor_locations=velocity_sensors
        )
        
        torque_measurements = []
        velocity_measurements = []
        
        for i, sensor_type in enumerate(sensor_metadata['sensor_types']):
            if sensor_type == 'torque':
                torque_measurements.append(y[:, i])
            elif sensor_type == 'velocity':
                velocity_measurements.append(y[:, i])
        
        torque_meas = np.column_stack(torque_measurements) if torque_measurements else np.array([]).reshape(N, 0)
        velocity_meas = np.column_stack(velocity_measurements) if velocity_measurements else np.array([]).reshape(N, 0)
    else:
        # Use all outputs as measurements
        torque_meas = np.array([]).reshape(N, 0)
        velocity_meas = y
    
    return {
        'time': t,
        'measurements': {
            'torque': torque_meas,
            'velocity': velocity_meas
        },
        'inputs': {
            'motor': u1,
            'load': u2
        },
        'reference': {
            'x': xout,
            'u2': u2,
            'u1_noisy': u1  # Can add noise here if needed
        }
    }

