"""
Simulation utilities for generating synthetic data using OpenTorsion assemblies.

This module contains PID control logic and helper functions that were
previously embedded in `data_loader.py`. The functionality is separated out
to keep loading utilities distinct from simulation routines.
"""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


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

    Parameters
    ----------
    load_case : str
        Type of load case: 'step' or 'impulse'
    N : int
        Number of time points in the simulation

    Returns
    -------
    ndarray
        Load torque sequence (N,) with values in Nm

    Raises
    ------
    ValueError
        If load_case is not recognized
    """
    u2 = np.zeros(N)

    if load_case == "step":
        # Step loads: multiple steps throughout simulation
        step_points = [int(N / 4), int(N / 2), int(3 * N / 4)]
        vals = [1, 5, 1, 5]
        prev = 0
        for sp, v in zip(step_points + [N], vals):
            u2[prev:sp] = v
            prev = sp
    elif load_case == "impulse":
        # Impulse loads: short duration impulses
        impulse_times = [int(N / 4), int(N / 2), int(3 * N / 4)]
        impulse_duration = int(N * 0.03)
        impulse_magnitude = 6.0
        for imp_time in impulse_times:
            end_idx = min(imp_time + impulse_duration, N)
            u2[imp_time:end_idx] = impulse_magnitude
    else:
        raise ValueError(f"Unknown load_case '{load_case}'. Supported cases: 'step', 'impulse'")

    return u2


def simulate_data(assembly, config: Dict) -> Dict:
    """
    Simulate data using OpenTorsion assembly with PID control.

    Parameters
    ----------
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

    Returns
    -------
    dict
        Dictionary with keys:
        - 'time': time vector
        - 'measurements': dict with 'torque' and 'velocity' arrays
        - 'inputs': dict with 'motor' and 'load' arrays
        - 'reference': dict with true states and inputs
    """
    from utils import c2d, create_measurement_matrix, minimize

    # Check required parameters
    if "load_case" not in config:
        raise ValueError("'load_case' must be specified in config. Use 'step' or 'impulse'.")
    if "measurement_config" not in config:
        raise ValueError("measurement_config must be provided for simulation")

    load_case = config["load_case"]
    if load_case not in ["step", "impulse"]:
        raise ValueError(f"load_case must be 'step' or 'impulse', got '{load_case}'")

    # Get state-space matrices
    A_full, B_full, _, _ = assembly.state_space()

    # Transform to minimal form using minimize function
    A_c = minimize(A_full, assembly, "state")  # X @ A @ X_inv
    B_c = minimize(B_full, assembly, "input")  # X @ B

    # Only retain the first and last columns of B_c
    if B_c.shape[1] >= 2:
        B_c = B_c[:, [0, -1]]
    else:
        raise ValueError("B_c does not have enough columns to retain first and last.")

    # Setup time vector
    if isinstance(config["time"], dict):
        t = np.linspace(config["time"]["start"], config["time"]["end"], config["time"]["n_points"])
    else:
        t = np.array(config["time"])

    N = len(t)
    dt = t[1] - t[0] if N > 1 else 0.01

    # Discretize
    A, B = c2d(A_c, B_c, dt)
    n_states = A.shape[0]

    # Setup measurement matrix
    torque_sensors = config["measurement_config"].get("torque_sensors", [])
    velocity_sensors = config["measurement_config"].get("velocity_sensors", [])
    C_meas, sensor_metadata = create_measurement_matrix(
        n_states=n_states,
        torque_sensor_locations=torque_sensors,
        velocity_sensor_locations=velocity_sensors,
    )

    # Initialize state
    if "initial_state" in config:
        x = np.asarray(config["initial_state"]).flatten()
        if len(x) != n_states:
            raise ValueError(f"initial_state length {len(x)} doesn't match system states {n_states}")
    else:
        x = np.zeros(n_states)

    # Noise parameters
    process_noise_std = config.get("process_noise_std", 0.01)
    measurement_noise_std = config.get("measurement_noise_std", 0.1)
    actuator_noise_std = config.get("actuator_noise_std", 0.05)

    # PID parameters
    speed_target = config.get("speed_target", 200.0)
    pid_params = config.get("pid_params", {"kp": 0.4, "ki": 0.15, "kd": 0.001})

    # Initialize PID controller
    pid = PIDController(
        kp=pid_params.get("kp", 0.4),
        ki=pid_params.get("ki", 0.15),
        kd=pid_params.get("kd", 0.001),
        dt=dt,
        output_limits=(-10, 10),
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

    for i, sensor_type in enumerate(sensor_metadata["sensor_types"]):
        if sensor_type == "torque":
            torque_measurements.append(y[:, i])
        elif sensor_type == "velocity":
            velocity_measurements.append(y[:, i])

    torque_meas = np.column_stack(torque_measurements) if torque_measurements else np.array([]).reshape(N, 0)
    velocity_meas = np.column_stack(velocity_measurements) if velocity_measurements else np.array([]).reshape(N, 0)

    # Plot simulation results for verification
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # Plot inputs (motor torque and load)
    axes[0].plot(t, u1, "b-", linewidth=1.5, label="Motor Torque (u1)")
    axes[0].plot(t, u2, "orange", linewidth=1.5, label="Load Torque (u2)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Torque (Nm)")
    axes[0].set_title("Inputs: Motor and Load Torque")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot one velocity state from xout
    axes[1].plot(t, xout[:, velocity_state_idx], "g-", linewidth=1.5)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Velocity (rad/s)")
    axes[1].set_title("Velocity State (from xout)")
    axes[1].grid(True, alpha=0.3)

    # Plot one angle/torque state from xout (first state, typically angle)
    axes[2].plot(t, xout[:, 7], "r-", linewidth=1.5)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Shaft Torque (Nm)")
    axes[2].set_title("Shaft Torque (from xout)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        "time": t,
        "measurements": {
            "torque": torque_meas,
            "velocity": velocity_meas,
        },
        "inputs": {
            "motor": u1,
            "load": u2,
        },
        "reference": {
            "xout_rows": np.linspace(0, n_states-1, n_states).astype(int),
            "xout": xout,
            "u2": u2,
        },
    }


