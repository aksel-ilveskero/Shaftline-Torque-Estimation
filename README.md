# Torque MHE - State Estimation Framework

A generalized state estimation framework for OpenTorsion systems supporting multiple estimation methods with flexible sensor configuration.

## Features

- **Generalized Framework**: Works with any OpenTorsion system (N states, odd number)
- **Flexible Sensor Configuration**: Specify sensors using disk numbers (automatically converted to state indices)
- **Multiple Data Sources**: Support for CSV, Feather files, and simulation
- **Modular Design**: Separate modules for estimation, data loading, plotting, and utilities
- **Standalone Plotting**: Plot results from saved files without running estimation

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from run_estimation import run_estimation
from ot_assembly import test_bench

# Define measurement configuration
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

# Run estimation
run_estimation(
    assembly_fn=test_bench,
    data_source='simulate',
    measurement_config=measurement_config,
    estimator_settings=estimator_settings,
    output_path='results.npz',
    plot=True
)
```

### Plotting Results

```python
from plot_results import plot_results

# Plot from saved file
plot_results('data/mhe_results.npz', reference_data={
    'torque': measured_torque,
    'velocity': measured_velocity,
    'prop_torque': propeller_torque
})
```

Or from command line:
```bash
python plot_results.py data/mhe_results.npz
```

## Project Structure

```
.
├── estimator.py          # Base estimator class and MHE implementation
├── data_loader.py        # Data loading and simulation functions
├── plot_results.py       # Standalone plotting functions
├── run_estimation.py     # Main workflow script
├── ot_assembly.py        # OpenTorsion assembly definitions
├── utils.py              # Utility functions (discretization, minimization)
├── requirements.txt      # Python dependencies
└── data/                 # Data directory (not tracked in git)
```

## Workflow

The framework implements a 6-step workflow:

1. **Create OpenTorsion system** - Define the physical system
2. **Load or simulate data** - Get measurement data
3. **Define measurement points** - Specify sensor locations as disk numbers
4. **Create system matrices** - Automatic minimal form transformation
5. **Perform estimation** - Run state estimation algorithm
6. **Save and plot results** - Visualize and store results

## Sensor Configuration

Sensors are specified using disk numbers, which are automatically converted to state indices:

- **Torque sensors**: Disk number maps directly to torque state index (0 to (N-1)//2)
- **Velocity sensors**: Disk number maps to velocity state index with offset: (N+1)//2 + disk_number

For a 43-state system:
- Torque disk 8 → State 8
- Velocity disk 5 → State 27 (22 + 5)
- Velocity disk 6 → State 28 (22 + 6)

## State Structure

For any OpenTorsion system with N states (odd number):
- **Torque states**: Indices 0 to (N-1)//2 (first half, odd amount)
- **Velocity states**: Indices (N+1)//2 to N-1 (second half, even amount)

## License

[Add your license here]

## Authors

[Add author information here]

