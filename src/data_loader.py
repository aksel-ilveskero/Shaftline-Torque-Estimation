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

class PIDController:
    """PID controller for speed control with anti-windup."""
    
    def __init__(self, kp, ki, kd, dt, output_limits=(-np.inf, np.inf), anti_windup=True):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_limits = output_limits
        self.anti_windup = anti_windup
        
        # Initialize controller state
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_output = 0.0
        
    def update(self, setpoint, current_value):
        """Update PID controller and return control output."""
        error = setpoint - current_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * self.dt
        if self.anti_windup and self.previous_output is not None:
            # Anti-windup: limit integral if output is saturated
            if self.previous_output >= self.output_limits[1]:
                self.integral = min(self.integral, 0)
            elif self.previous_output <= self.output_limits[0]:
                self.integral = max(self.integral, 0)
        
        i_term = self.ki * self.integral
        
        # Derivative term
        d_term = self.kd * (error - self.previous_error) / self.dt
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply output limits
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # Update state
        self.previous_error = error
        self.previous_output = output
        
        return output


def get_load_case(load_case: str, N: int) -> np.ndarray:
    """
    Generate load torque sequence based on load case type.
    
    Parameters:
    -----------
    load_case : str
        Type of load case: 'step' or 'impulse'
    N : int
        Number of time points in the simulation
        
    Returns:
    --------
    ndarray
        Load torque sequence (N,) with values in Nm
        
    Raises:
    -------
    ValueError
        If load_case is not recognized
    """
    u2 = np.zeros(N)
    
    if load_case == 'step':
        # Step loads: multiple steps throughout simulation
        step_points = [int(N/4), int(N/2), int(3*N/4)]
        vals = [0.1, 1.3, 0.1, 1.3]
        prev = 0
        for sp, v in zip(step_points + [N], vals):
            u2[prev:sp] = v
            prev = sp
    elif load_case == 'impulse':
        # Impulse loads: short duration impulses
        impulse_times = [int(N/4), int(N/2), int(3*N/4)]
        impulse_duration = int(N * 0.01)  # 1% of simulation time
        impulse_magnitude = 2.0
        for imp_time in impulse_times:
            end_idx = min(imp_time + impulse_duration, N)
            u2[imp_time:end_idx] = impulse_magnitude
    else:
        raise ValueError(f"Unknown load_case '{load_case}'. Supported cases: 'step', 'impulse'")
    
    return u2


def simulate_data(assembly, config: Dict) -> Dict:
    """
    Simulate data using OpenTorsion assembly with PID control.
    
    Parameters:
    -----------
    assembly : ot.Assembly
        OpenTorsion assembly object
    config : dict
        Simulation configuration with keys:
        - 'time': time vector or dict with 'start', 'end', 'n_points'
        - 'load_case': str, required - 'step' or 'impulse' for load torque pattern
        - 'process_noise_std': float (default 0.01)
        - 'measurement_noise_std': float (default 0.1)
        - 'actuator_noise_std': float (default 0.05)
        - 'initial_state': ndarray (optional)
        - 'measurement_config': dict with sensor locations (required)
        - 'pid_params': dict with 'kp', 'ki', 'kd' (optional, defaults provided)
        - 'speed_target': float (default 200.0 rad/s)
        
    Returns:
    --------
    dict
        Dictionary with keys:
        - 'time': time vector
        - 'measurements': dict with 'torque' and 'velocity' arrays
        - 'inputs': dict with 'motor' and 'load' arrays
        - 'reference': dict with true states and inputs
    """
    from utils import create_measurement_matrix
    from utils import c2d, minimize
    
    # Check required parameters
    if 'load_case' not in config:
        raise ValueError("'load_case' must be specified in config. Use 'step' or 'impulse'.")
    if 'measurement_config' not in config:
        raise ValueError("measurement_config must be provided for simulation")
    
    load_case = config['load_case']
    if load_case not in ['step', 'impulse']:
        raise ValueError(f"load_case must be 'step' or 'impulse', got '{load_case}'")
    
    # Get state-space matrices
    A_full, B_full, _, _ = assembly.state_space()
    
    # Transform to minimal form using minimize function
    A_c = minimize(A_full, assembly, 'state')  # X @ A @ X_inv
    B_c = minimize(B_full, assembly, 'input')  # X @ B

    # Only retain the first and last columns of B_c
    if B_c.shape[1] >= 2:
        B_c = B_c[:, [0, -1]]
    else:
        raise ValueError("B_c does not have enough columns to retain first and last.")
    
    # Setup time vector
    if isinstance(config['time'], dict):
        t = np.linspace(config['time']['start'], config['time']['end'], config['time']['n_points'])
    else:
        t = np.array(config['time'])
    
    N = len(t)
    dt = t[1] - t[0] if N > 1 else 0.01
    
    # Discretize
    A, B = c2d(A_c, B_c, dt)
    n_states = A.shape[0]
    
    # Setup measurement matrix
    torque_sensors = config['measurement_config'].get('torque_sensors', [])
    velocity_sensors = config['measurement_config'].get('velocity_sensors', [])
    C_meas, sensor_metadata = create_measurement_matrix(
        n_states=n_states,
        torque_sensor_locations=torque_sensors,
        velocity_sensor_locations=velocity_sensors
    )
    
    # Initialize state
    if 'initial_state' in config:
        x = np.asarray(config['initial_state']).flatten()
        if len(x) != n_states:
            raise ValueError(f"initial_state length {len(x)} doesn't match system states {n_states}")
    else:
        x = np.zeros(n_states)
    
    # Noise parameters
    process_noise_std = config.get('process_noise_std', 0.01)
    measurement_noise_std = config.get('measurement_noise_std', 0.1)
    actuator_noise_std = config.get('actuator_noise_std', 0.05)
    
    # PID parameters
    speed_target = config.get('speed_target', 200.0)
    pid_params = config.get('pid_params', {'kp': 0.4, 'ki': 0.15, 'kd': 0.001})
    
    # Initialize PID controller
    pid = PIDController(
        kp=pid_params.get('kp', 0.4),
        ki=pid_params.get('ki', 0.15),
        kd=pid_params.get('kd', 0.001),
        dt=dt,
        output_limits=(-10, 10)
    )
    
    # Determine velocity state index for PID feedback
    # Velocity states are typically in the second half of the state vector
    # Use the first velocity state (typically at index (n_states+1)//2)
    velocity_state_idx = (n_states + 1) // 2
    
    # Generate load torque (u2) based on load_case
    u2 = get_load_case(load_case, N)
    
    # Simulation arrays
    xout = np.zeros((N, n_states))
    y = np.zeros((N, C_meas.shape[0]))
    u1 = np.zeros(N)
    u1_clean = np.zeros(N)  
    
    # Convert speed_target to array if scalar
    if np.isscalar(speed_target):
        speed_target_array = np.full(N, speed_target)
    else:
        speed_target_array = np.asarray(speed_target)
        if len(speed_target_array) != N:
            raise ValueError(f"speed_target length {len(speed_target_array)} doesn't match time length {N}")
    
    # Simulation loop with PID control
    for i in range(N):
        xout[i] = x.copy()
        
        # Calculate outputs
        y_clean = C_meas @ x
        y[i] = y_clean + np.random.normal(0, measurement_noise_std, C_meas.shape[0])
        
        # PID control for u1 (motor torque) based on speed feedback
        current_speed = x[velocity_state_idx]
        u1_clean[i] = pid.update(speed_target_array[i], current_speed)
        
        # Apply limits to clean u1
        u1_clean[i] = np.clip(u1_clean[i], -10, 10)
        
        # Add actuator noise to u1
        actuator_noise = np.random.normal(0, actuator_noise_std)
        u1[i] = u1_clean[i] + actuator_noise
        
        # State update with process noise
        if i < N - 1:
            process_noise = np.random.normal(0, process_noise_std, n_states)
            u_input = np.array([u1[i], u2[i]])
            x = A @ x + B @ u_input + process_noise
    
    # Extract torque and velocity measurements based on sensor metadata
    torque_measurements = []
    velocity_measurements = []
    
    for i, sensor_type in enumerate(sensor_metadata['sensor_types']):
        if sensor_type == 'torque':
            torque_measurements.append(y[:, i])
        elif sensor_type == 'velocity':
            velocity_measurements.append(y[:, i])
    
    torque_meas = np.column_stack(torque_measurements) if torque_measurements else np.array([]).reshape(N, 0)
    velocity_meas = np.column_stack(velocity_measurements) if velocity_measurements else np.array([]).reshape(N, 0)
    
    # Plot simulation results for verification
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    # Plot inputs (motor torque and load)
    axes[0].plot(t, u1, 'b-', linewidth=1.5, label='Motor Torque (u1)')
    axes[0].plot(t, u2, 'orange', linewidth=1.5, label='Load Torque (u2)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Torque (Nm)')
    axes[0].set_title('Inputs: Motor and Load Torque')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot one velocity state from xout
    axes[1].plot(t, xout[:, velocity_state_idx], 'g-', linewidth=1.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Velocity (rad/s)')
    axes[1].set_title('Velocity State (from xout)')
    axes[1].grid(True, alpha=0.3)
    
    # Plot one angle/torque state from xout (first state, typically angle)
    axes[2].plot(t, xout[:, 7], 'r-', linewidth=1.5)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Shaft Torque (Nm)')
    axes[2].set_title('Shaft Torque (from xout)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
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
            'u1_noisy': u1
        }
    }

