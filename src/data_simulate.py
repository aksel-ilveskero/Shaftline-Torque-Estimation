"""
Data simulation module for torque estimation using Moving Horizon Estimation (MHE).

This module simulates torque dynamics using a linear state-space model derived from
an OpenTorsion assembly. It generates synthetic measurement data with process and
measurement noise, and compares simulated results with measured data from impulse
aligned feather files.

The simulation uses a discretized state-space model and allows for custom noiseå
characteristics and initial conditions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from ot_assembly import test_bench, minimal_state_space
from utils import c2d

def custom_linear_simulation(A, B, C, D, t, u1, u2, process_noise_std=0.01, 
                            measurement_noise_std=0.1, 
                            initial_state=None):
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
    
    # Simulation loop
    for i in range(N):
        # Store current state
        xout[i] = x.copy()
        
        # Calculate outputs
        y_clean = C @ x
        y[i] = y_clean + np.random.normal(0, measurement_noise_std, n_outputs)
        
        # PID control for u1 (motor torque) based on speed feedback
        current_speed = y_clean[0]
        print(current_speed)
        
        # State update with process noise
        if i < N - 1:
            # Process noise
            process_noise = np.random.normal(0, process_noise_std, n_states)
            
            # State equation: x[k+1] = A*x[k] + B*u[k] + process_noise
            u_input = np.array([u1[i], u2[i]])  # [u1, u2]
            x = A @ x + B @ u_input + process_noise

    return t, y, xout, u1

measurements = pd.read_feather('impulse_aligned.feather')

# Group by sample_id
sample_groups = measurements.groupby('sample_id')
print(measurements['sample_id'].unique)

for sample_id, group in sample_groups:
    # print(sample_id)
    # Ensure the group is sorted by time if needed
    group = group.sort_values('time')

    # Select rows [1000:6000] for the current sample
    sample_slice = group.iloc[0:9000]

    # Extract the columns in order: T1, T2, u_m, u_p
    time = sample_slice["time"].to_numpy()
    e1 = sample_slice["E1"].to_numpy()  # input
    e2 = sample_slice["E2"].to_numpy()  # input
    t1 = sample_slice["T1"].to_numpy()  # input
    t2 = sample_slice["T2"].to_numpy()  # evaluation
    # u_m = np.roll(sample_slice["u_m"].to_numpy(), 650)
    # u_p = np.roll(sample_slice["u_p"].to_numpy(), 650)
    u_p = sample_slice["u_p"].to_numpy()
    u_m = sample_slice["u_m"].to_numpy()
    # print(group["sample_id"].unique)
    # Append as a tuple (or list) into our master list

    plt.figure(figsize=(12, 8))

    #plt.plot(time, e1, label="e1")
    #plt.plot(time, e2, label="e2")

    # plt.plot(time, t2, label="t2")
    plt.plot(time, u_m, label="u_m")
    plt.plot(time, u_p, label="u_p")
    # plt.plot(time, t1, label="t1")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Torque (Nm)")
    
    plt.tight_layout()
    plt.legend()
    plt.show()

# Simulation
ot_asm = test_bench()

A,B,C,D = minimal_state_space(ot_asm)

C_rows = [30, 31]
C = np.eye(A.shape[0])[ C_rows, : ]

A,B = c2d(A,B,np.mean(np.diff(time)))

# Initial state
x_init = np.zeros(A.shape[0])

# Torque values
x_init[0:21] = 1.5*np.ones(21)
x_init[11:16] = x_init[11:16]*3
x_init[16:] = x_init[16:]*3*4

x_init[21:] = 222*np.ones(22)
x_init[33:39] = -x_init[33:39]/3
x_init[39:] = x_init[39:]/3/4

# Custom simulation with PID control and noise
tout, y, xout, u1 = custom_linear_simulation(
    A=A, B=B, C=C, D=D, t=time,
    u1=u_m,
    u2=-1*u_p,
    process_noise_std=0.0001,      # Process noise standard deviation
    measurement_noise_std=0.0001,   # Measurement noise standard deviation
    initial_state = x_init
)

fig, axs = plt.subplots(2)

axs[0].plot(time, t1, label='measured torque')
axs[0].scatter(time, xout[:, 8], alpha=0.25, color='r', edgecolor=None, label='estimated torque')
axs[0].scatter(time, xout[:, 19], alpha=0.25, color='b', edgecolor=None, label='estimated torque')
axs[0].set_ylabel('Shaft Torque (Nm)')
axs[0].legend()

axs[1].plot(time, e1, label='measured velocity')
axs[1].scatter(time, xout[:, 30], alpha=0.25, color='r', edgecolor=None, label='estimated velocity')
axs[1].scatter(time, xout[:, 41], alpha=0.25, color='b', edgecolor=None, label='estimated velocity')
axs[1].set_ylabel('Shaft Velocity (rad/s)')
axs[1].legend()

plt.tight_layout()
plt.show()