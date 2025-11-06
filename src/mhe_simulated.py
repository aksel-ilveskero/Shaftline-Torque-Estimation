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

from ot_assembly import test_bench
from run_estimation import run_estimation

if __name__ == "__main__":
    # Measurement configuration: velocity sensors at disk numbers 5, 6
    measurement_config = {
        'torque_sensors': [],  # Torque sensors
        'velocity_sensors': [6, 7],  # Disk numbers (will be converted to state indices)
        'inputs': ['motor']
    }
    
    # Estimator settings
    estimator_settings = {
        'horizon_length': 10,
        'Q_v_scale': 0.25,
        'Q_w_scale': 0.1,
        'lambda_': 1.0
    }
    
    # Initial state
    x_init = np.zeros(43)

    # Torque values
    x_init[0:21] = 2.3 * np.ones(21)
    x_init[11:16] = x_init[11:16] * 3
    x_init[16:] = x_init[16:] * 4

    x_init[21:] = 222 * np.ones(22)
    x_init[33:39] = x_init[33:39] / 3
    x_init[39:] = x_init[39:] / 4


    # Data configuration
    data_config = {
        'initial_state': x_init,
        'load_case': 'step',
        'measurement_config': measurement_config,
        'pid_params': {'kp': 0.4, 'ki': 0.15, 'kd': 0.001},
        'speed_target': 200.0,
        'actuator_noise_std': 0.05,
        'process_noise_std': 0.01,
        'measurement_noise_std': 0.1,
        'time': {'start': 0, 'end': 5, 'n_points': 1500}
    }
    
    # Run estimation using unified workflow
    run_estimation(
        assembly_fn=test_bench,
        data_source='simulate',
        data_config=data_config,
        measurement_config=measurement_config,
        estimator_settings=estimator_settings,
        output_path='mhe_results.npz',
        plot=True
    )