"""
Moving Horizon Estimation (MHE) for testbench data analysis.

This module processes real testbench measurement data (from CSV or Feather files)
and applies Moving Horizon Estimation to estimate torque and velocity states.
Uses the unified estimation framework with sensor-based configuration.

The module supports both CSV files with encoder and torque measurements, as well
as Feather files with aligned ice excitation data.
"""

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from ot_assembly import test_bench
from run_estimation import run_estimation

if __name__ == "__main__":
    # Measurement configuration: velocity sensors at disk numbers 5, 6
    measurement_config = {
        'torque_sensors': [],  # Torque sensors
        'velocity_sensors': [26, 27],  # Velocity sensors
        'inputs': ['motor'],  # Inputs
    }
    
    # Estimator settings
    estimator_settings = {
        'horizon_length': 15,
        'Q_v_scale': 0.1,
        'Q_w_scale': 0.5,
        'lambda_': 1
    }
    
    # Initial state
    x_init = np.zeros(43)

    # Torque values
    x_init[0:21] = 1.5 * np.ones(21)
    x_init[11:16] = x_init[11:16] * 3
    x_init[16:] = x_init[16:] * 3 * 4

    x_init[21:] = 210 * np.ones(22)
    x_init[33:39] = x_init[33:39] / 3
    x_init[39:] = x_init[39:] / 3 / 4


    # Data configuration
    data_config = {
        'initial_state': x_init,
        'path': 'data/impulse_aligned.feather',
        'start_idx': 6500,
        'end_idx': 8500
    }
    
    # Run estimation using unified workflow
    run_estimation(
        assembly_fn=test_bench,
        data_source='feather',
        data_config=data_config,
        measurement_config=measurement_config,
        estimator_settings=estimator_settings,
        output_path='mhe_results.npz',
    )


