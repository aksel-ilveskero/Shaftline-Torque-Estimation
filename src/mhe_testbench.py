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
from data_loader import load_csv, load_feather
from run_estimation import run_estimation

if __name__ == "__main__":
    # Example: Load feather data and run estimation
    data = load_feather()
    
    # Measurement configuration: velocity sensors at disk numbers 5, 6
    measurement_config = {
        'torque_sensors': [],  # No torque sensors
        'velocity_sensors': [6, 7],  # Disk numbers (will be converted to state indices)
        'inputs': ['motor', 'load']
    }
    
    # Estimator settings
    estimator_settings = {
        'horizon_length': 10,
        'Q_v_scale': 0.25,
        'Q_w_scale': 0.1,
        'lambda_': 1.0
    }
    
    # Data configuration
    data_config = {
        'path': 'data/ice_aligned.feather',
        'start_idx': 6000,
        'end_idx': 8000
    }
    
    # Run estimation using unified workflow
    run_estimation(
        assembly_fn=test_bench,
        data_source='feather',
        data_config=data_config,
        measurement_config=measurement_config,
        estimator_settings=estimator_settings,
        output_path='mhe_results.npz',
        plot=True
    )


