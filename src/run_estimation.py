"""
Main workflow script for state estimation.

This script implements the unified workflow for state estimation:
1. Create OpenTorsion system
2. Load or simulate data
3. Define measurement points (sensor locations)
4. Create system matrices (automatic from assembly + config)
5. Perform estimation
6. Save and plot results
"""

from pathlib import Path
from typing import Dict, Optional, Callable
import numpy as np

from ot_assembly import test_bench
from estimator import MHEEstimator
from data_loader import load_csv, load_feather, simulate_data
from utils import minimize
from plot_results import plot_results as plot_estimation_results


def run_estimation(
    assembly_fn: Callable = test_bench,
    data_source: str = 'simulate',
    data_config: Optional[Dict] = None,
    measurement_config: Optional[Dict] = None,
    estimator_settings: Optional[Dict] = None,
    output_path: str = 'estimation_results.npz',
    plot: bool = True
) -> Dict:
    """
    Run complete state estimation workflow.
    
    Parameters:
    -----------
    assembly_fn : callable, optional
        Function that returns an OpenTorsion assembly (default: test_bench)
    data_source : str, optional
        Data source: 'simulate', 'csv', or 'feather' (default: 'simulate')
    data_config : dict, optional
        Configuration for data loading/simulation
        measurement_config : dict, optional
        Measurement configuration with keys:
        - 'torque_sensors': list of disk numbers (can be empty)
        - 'velocity_sensors': list of disk numbers (can be empty)
        - 'inputs': list of input names, e.g., ['motor'], ['load'], or ['motor', 'load']
        Inputs specified in this list are measured (go to B1), others are unmeasured (go to B2).
        Note: Disk numbers are automatically converted to state indices with appropriate offsets.
    estimator_settings : dict, optional
        Estimator-specific settings (MHE settings, etc.)
    output_path : str, optional
        Path to save results (default: 'estimation_results.npz')
    plot : bool, optional
        Whether to plot results (default: True)
        
    Returns:
    --------
    dict
        Dictionary with estimation results and metadata
    """
    
    # Step 1: Create OpenTorsion system
    print("Step 1: Creating OpenTorsion system...")
    assembly = assembly_fn()
    print(f"  Created assembly with {assembly.dofs} degrees of freedom")
    
    # Step 2: Load or simulate data
    print("\nStep 2: Loading/simulating data...")
    
    if data_source == 'simulate':
        if data_config is None:
            raise ValueError("data_config must be provided for simulation")
        data = simulate_data(assembly, data_config)
        print(f"  Simulated {len(data['time'])} time points")
    
    elif data_source == 'csv':
        csv_path = data_config.get('path') if data_config else None
        if csv_path is None:
            raise ValueError("data_config['path'] must be provided for CSV loading")
        start_idx = data_config.get('start_idx', 14500)
        end_idx = data_config.get('end_idx', 16000)
        data = load_csv(csv_path, start_idx=start_idx, end_idx=end_idx)
        print(f"  Loaded {len(data['time'])} time points from CSV")
    
    elif data_source == 'feather':
        feather_path = data_config.get('path', 'data/ice_aligned.feather') if data_config else 'data/ice_aligned.feather'
        start_idx = data_config.get('start_idx', 6000) if data_config else 6000
        end_idx = data_config.get('end_idx', 8000) if data_config else 8000
        sample_id = data_config.get('sample_id') if data_config else None
        data = load_feather(feather_path, start_idx=start_idx, end_idx=end_idx, sample_id=sample_id)
        print(f"  Loaded {len(data['time'])} time points from Feather file")
    
    else:
        raise ValueError(f"Unknown data_source: {data_source}")
    
    # Step 3: Define measurement points
    print("\nStep 3: Defining measurement points...")
    if measurement_config is None:
        # Default: use all available measurements
        torque_sensors = []
        velocity_sensors = []
        
        # Try to infer from data
        if data['measurements']['torque'].shape[1] > 0:
            # Assume torque sensors at common locations (need to be specified)
            print("  Warning: torque sensors detected but locations not specified")
        if data['measurements']['velocity'].shape[1] > 0:
            # Assume velocity sensors at common locations (need to be specified)
            print("  Warning: velocity sensors detected but locations not specified")
        
        measurement_config = {
            'torque_sensors': torque_sensors,
            'velocity_sensors': velocity_sensors,
            'inputs': ['motor', 'load']
        }
    
    torque_sensors = measurement_config.get('torque_sensors', [])
    velocity_sensors = measurement_config.get('velocity_sensors', [])
    print(f"  Torque sensors at disk numbers: {torque_sensors}")
    print(f"  Velocity sensors at disk numbers: {velocity_sensors}")
    
    # Prepare measurement vector y
    y_list = []
    if len(torque_sensors) > 0:
        torque_data = data['measurements']['torque']
        if torque_data.shape[1] != len(torque_sensors):
            raise ValueError(f"Number of torque measurements ({torque_data.shape[1]}) doesn't match number of torque sensors ({len(torque_sensors)})")
        y_list.append(torque_data)
    
    if len(velocity_sensors) > 0:
        velocity_data = data['measurements']['velocity']
        if velocity_data.shape[1] != len(velocity_sensors):
            raise ValueError(f"Number of velocity measurements ({velocity_data.shape[1]}) doesn't match number of velocity sensors ({len(velocity_sensors)})")
        y_list.append(velocity_data)
    
    if len(y_list) == 0:
        raise ValueError("No sensors specified in measurement_config")
    
    y = np.hstack(y_list) if len(y_list) > 1 else y_list[0]
    
    # Prepare input vector u1 (measured inputs)
    inputs_config = measurement_config.get('inputs', ['motor', 'load'])
    u1_list = []
    for inp_name in inputs_config:
        if inp_name not in data['inputs']:
            raise ValueError(f"Input '{inp_name}' not found in data. Available inputs: {list(data['inputs'].keys())}")
        inp_data = data['inputs'][inp_name]
        if inp_data.ndim == 1:
            inp_data = inp_data.reshape(1, -1)
        u1_list.append(inp_data)
    
    # Stack inputs horizontally: each input becomes a column
    if len(u1_list) > 0:
        u1 = np.vstack(u1_list)  # Shape: (n_inputs, N)
    else:
        raise ValueError("No inputs specified in measurement_config['inputs']")
    
    # Step 4: Create system matrices in minimal form
    print("\nStep 4: Creating system matrices in minimal form...")
    if estimator_settings is None:
        estimator_settings = {}
    
    # Add sample time if not provided
    if 'Ts' not in estimator_settings:
        dt = np.mean(np.diff(data['time']))
        estimator_settings['Ts'] = dt
        print(f"  Using sample time: {dt:.6f} s")
    
    # Ensure assembly has X matrix (needed for minimal form transformation)
    # This is done automatically when state_space() is called, but we ensure it's available
    if not hasattr(assembly, 'X'):
        _ = assembly.state_space()  # This populates assembly.X

    
    # Initialize estimator (uses minimal form internally via minimize function)
    estimator = MHEEstimator(assembly, measurement_config, estimator_settings)
    print(f"  System has {estimator.n_states} states (minimal form)")
    print(f"  Measurement matrix C: {estimator.C.shape}")
    
    # Load data into estimator
    estimator.load_data(y=y, u1=u1, t=data['time'])
    
    # Step 5: Perform estimation
    print("\nStep 5: Performing estimation...")
    
    # Prepare initial state if provided in data_config
    x_init = None
    if data_config and 'initial_state' in data_config:
        x_init = np.asarray(data_config['initial_state'])
        if len(x_init) != estimator.n_states:
            raise ValueError(f"initial_state length {len(x_init)} doesn't match system states {estimator.n_states}")
    
    results = estimator.estimate(x_init=x_init)
    
    # Step 6: Save and plot results
    print("\nStep 6: Saving and plotting results...")
    estimator.save_results(output_path, results)
    print(f"  Results saved to: {output_path}")
    
    if plot:
        # Prepare reference data for plotting
        reference_data = None
        if 'reference' in data:
            ref = data['reference']
            reference_data = {}
            if 'torque' in ref:
                reference_data['torque'] = ref['torque']
            if 'velocity' in ref:
                reference_data['velocity'] = ref['velocity']
            if 'prop_torque' in ref:
                reference_data['prop_torque'] = ref['prop_torque']
        
        # Use standalone plotting function (can also use estimator.plot_results())
        plot_estimation_results(output_path, reference_data=reference_data)
    
    return {
        'estimator': estimator,
        'results': results,
        'data': data,
        'measurement_config': measurement_config
    }


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("STATE ESTIMATION WORKFLOW")
    print("=" * 60)
    
    # Example 1: Simulated data
    print("\nExample: Simulated data with MHE")
    
    # Measurement configuration
    # Disk numbers are automatically converted to state indices:
    # - Torque sensors: disk_number -> state_index = disk_number
    # - Velocity sensors: disk_number -> state_index = (n_states+1)//2 + disk_number
    measurement_config = {
        'torque_sensors': [8, 18],      # Disk numbers for torque sensors
        'velocity_sensors': [5, 6],     # Disk numbers for velocity sensors
        'inputs': ['motor', 'load']
    }
    
    # Estimator settings
    estimator_settings = {
        'horizon_length': 10,
        'Q_v_scale': 0.25,
        'Q_w_scale': 0.1,
        'lambda_': 1.0
    }
    
    # Simulation configuration
    data_config = {
        'time': {'start': 0, 'end': 30, 'n_points': 2000},
        'u1': lambda t: 2.0 * np.ones_like(t),
        'u2': lambda t: 2.0 + 0.5 * np.sin(0.5 * t),
        'process_noise_std': 0.01,
        'measurement_noise_std': 0.5,
        'measurement_config': measurement_config
    }
    
    run_estimation(
        assembly_fn=test_bench,
        data_source='simulate',
        data_config=data_config,
        measurement_config=measurement_config,
        estimator_settings=estimator_settings,
        output_path='example_results.npz',
        plot=True
    )

