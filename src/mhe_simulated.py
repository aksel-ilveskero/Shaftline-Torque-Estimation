"""
Moving Horizon Estimation (MHE) simulation module with PID control.

This module implements a complete simulation and estimation pipeline that:
1. Creates a linear state-space model from an OpenTorsion test bench assembly
2. Simulates the system dynamics with PID-controlled motor torque
3. Applies process, measurement, and actuator noise
4. Uses Moving Horizon Estimation to estimate states and unknown inputs
5. Visualizes the estimation results

The simulation includes a PID controller for speed regulation and generates
synthetic measurement data with configurable noise characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt

from ot_assembly import test_bench, minimal_state_space
from estimator import MHEEstimator
from utils import c2d


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


def custom_linear_simulation(A, B, C, D, t, u2, speed_target, process_noise_std=0.01, 
                            measurement_noise_std=0.1, actuator_noise_std=0.1, pid_params=None, 
                            initial_state=None, output_indices=None):
    """
    Custom linear simulation with process error, measurement noise, and PID control.
    
    Parameters:
    -----------
    A, B, C, D : ndarray
        State space matrices
    t : ndarray
        Time vector
    u2 : ndarray
        Load torque input (step loads)
    speed_target : float or ndarray
        Target speed for PID control (rad/s)
    process_noise_std : float
        Standard deviation of process noise
    measurement_noise_std : float
        Standard deviation of measurement noise
    actuator_noise_std : float
        Standard deviation of actuator noise (u1)
    pid_params : dict, optional
        PID parameters {'kp': float, 'ki': float, 'kd': float}
    initial_state : ndarray, optional
        Initial state vector
    output_indices : list, optional
        Indices of outputs to use for feedback (default: velocity states)
    
    Returns:
    --------
    tout : ndarray
        Time vector
    y : ndarray
        Output measurements (with noise)
    xout : ndarray
        State trajectory
    u1 : ndarray
        PID controlled motor torque
    """
    
    N = len(t)
    dt = t[1] - t[0] if N > 1 else 0.01
    
    # Initialize arrays
    n_states = A.shape[0]
    n_outputs = C.shape[0]
    
    if initial_state is None:
        x = np.zeros(n_states)
    else:
        x = initial_state.copy()
    
    xout = np.zeros((N, n_states))
    y = np.zeros((N, n_outputs))
    
    # Determine which output to use for speed feedback
    if output_indices is None:
        # Assume velocity states are typically at indices 30+ (adjust based on your system)
        # Look for velocity-related outputs in C matrix
        speed_output_idx = 1 if n_outputs > 1 else 0  # Default to second output
    else:
        speed_output_idx = output_indices[0] if len(output_indices) > 0 else 0
    
    # Initialize PID controller
    if pid_params is None:
        pid_params = {'kp': 1.0, 'ki': 0.1, 'kd': 0.01}
    
    pid = PIDController(
        kp=pid_params['kp'],
        ki=pid_params['ki'], 
        kd=pid_params['kd'],
        dt=dt,
        output_limits=(-10, 10)  # Reasonable torque limits
    )
    
    # Convert speed_target to array if scalar
    if np.isscalar(speed_target):
        speed_target = np.full(N, speed_target)
    
    u1 = np.zeros(N)
    u1_clean = np.zeros(N)  # Clean u1 without actuator noise
    
    # Simulation loop
    for i in range(N):
        # Store current state
        xout[i] = x.copy()
        
        # Calculate outputs
        y_clean = C @ x
        y[i] = y_clean + np.random.normal(0, measurement_noise_std, n_outputs)
        
        # PID control for u1 (motor torque) based on speed feedback
        current_speed = x[speed_output_idx]
        print(current_speed)
        u1_clean[i] = pid.update(speed_target[i], current_speed)

        # Apply limits to clean u1
        if u1_clean[i] > 10:
            u1_clean[i] = 10
        if u1_clean[i] < -10:
            u1_clean[i] = -10
        
        # Add actuator noise to u1
        actuator_noise = np.random.normal(0, actuator_noise_std)
        u1[i] = u1_clean[i] + actuator_noise
        
        # Apply limits to noisy u1 as well
        if u1[i] > 10:
            u1[i] = 10
        if u1[i] < -10:
            u1[i] = -10
        
        # State update with process noise
        if i < N - 1:
            # Process noise
            process_noise = np.random.normal(0, process_noise_std, n_states)
            
            # State equation: x[k+1] = A*x[k] + B*u[k] + process_noise
            u_input = np.array([u1[i], u2[i]])  # [u1, u2]
            x = A @ x + B @ u_input + process_noise

    return t, y, xout, u1

# Simulation
ot_asm = test_bench()

A,B,C,D = minimal_state_space(ot_asm)

# Measurement configuration: torque sensor at disk 8, velocity sensor at disk 8
# (corresponds to torque state 8 and velocity state 30 in a 43-state system)
measurement_config = {
    'torque_sensors': [8],  # Disk number (maps directly to torque state 8)
    'velocity_sensors': [8],  # Disk number (maps to velocity state 22+8=30)
    'inputs': ['motor', 'load']
}

# Parameters
N = 20000
t = np.linspace(0,30,N)

A,B = c2d(A,B,np.mean(np.diff(t)))

# Create step loads for u2 (load torque)
u2 = np.ones(N) * 2
step_points = [int(N/4), int(N/2), int(3*N/4)]
vals = [0.1, 1.3, 0.1, 1.3]
prev = 0
for sp, v in zip(step_points + [N], vals):
    u2[prev:sp] = v
    prev = sp

# Speed target for PID control (rad/s)
speed_target = 210.0  # Constant speed target

# PID parameters for speed control
pid_params = {
    'kp': 0.4,    # Proportional gain
    'ki': 0.15,    # Integral gain  
    'kd': 0.001     # Derivative gain
}

# Custom simulation with PID control and noise
tout, y_noisy, xout, u1 = custom_linear_simulation(
    A=A, B=B, C=C, D=D,
    t=t,
    u2=u2,  # Load torque (step loads)
    speed_target=speed_target,
    process_noise_std=0.01,      # Process noise standard deviation
    measurement_noise_std=0.5,   # Measurement noise standard deviation
    actuator_noise_std=0.05,     # Actuator noise standard deviation
    pid_params=pid_params,
    output_indices=[24]  # Use second output (velocity) for feedback
)

# Ensure u1 is shape (1, N) for consistency
if u1.ndim == 1:
    u1 = u1.reshape(1, -1)

u1_noisy = u1 + np.random.normal(0, 0.05)

# Downsample data to every Nth point for estimation
downsample_factor = 15
indices = np.arange(0, N, downsample_factor)

# Downsample all data
t_downsampled = t[indices]
y_downsampled = y_noisy[indices]
u1_downsampled = u1[indices] if u1.ndim == 1 else u1[:, indices]
u2_downsampled = u2[indices]
xout_downsampled = xout[indices]
u1_noisy_downsampled = u1_noisy[indices] if u1_noisy.ndim == 1 else u1_noisy[:, indices]

# Estimator settings
estimator_settings = {
    "horizon_length": 10,
    "Ts": t_downsampled[1] - t_downsampled[0],  # Use downsampled time step
    "Q_v_scale": 0.1,
    "Q_w_scale": 0.1,
    "lambda_": 1000.0,
}

# Initial state
x_init = np.zeros(43)

# Torque values
x_init[0:21] = 2.3*np.ones(21)
x_init[11:16] = x_init[11:16]*3
x_init[16:] = x_init[16:]*4

x_init[21:] = 222*np.ones(22)
x_init[33:39] = x_init[33:39]/3
x_init[39:] = x_init[39:]/4

# Create measurement matrix C for simulation
from ot_assembly import create_measurement_matrix
C_meas, _ = create_measurement_matrix(
    n_states=A.shape[0],
    torque_sensor_locations=measurement_config['torque_sensors'],
    velocity_sensor_locations=measurement_config['velocity_sensors']
)

# Update C for simulation
C = C_meas

est = MHEEstimator(ot_asm, measurement_config, estimator_settings)
est.load_data(y=y_downsampled, u1=u1_noisy_downsampled, t=t_downsampled, 
              truth={"x": xout_downsampled, "u2": u2_downsampled, "u1_noisy": u1_noisy_downsampled})
results = est.estimate(x_init=x_init)
est.save_results("mhe_results.npz", results)
est.plot_results("mhe_results.npz")