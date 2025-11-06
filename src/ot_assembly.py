"""
OpenTorsion assembly definition and state-space conversion.

This module defines the physical test bench assembly using OpenTorsion library,
including shaft elements, disk inertias, and gear ratios. It provides functions
to create the assembly and convert it to a minimal state-space representation
suitable for estimation and control.

The test bench model includes:
- 24 disk elements with various inertias
- 21 shaft elements with stiffness and damping
- 4 gear elements representing gearbox ratios
- State-space transformation to minimal representation
"""

import opentorsion as ot

from scipy.linalg import pinv
from utils import minimize
import numpy as np

# PARAMETERS
I1 = 7.94e-4
k1 = 1.90e5
c1 = 8.08
d1 = 0.0030

I2 = 3.79e-6
k2 = 6.95e3
c2 = 0.29

I3 = 3.00e-6
k3 = 90.00
c3 = 0.24

I4 = 2.00e-6
k4 = 90.00
c4 = 0.24

I5 = 7.81e-3
k5 = 90.00
c5 = 0.24

I6 = 2.00e-6
k6 = 90.00
c6 = 0.24

I7 = 3.29e-6
k7 = 94.06
c7 = 0.00

I8 = 5.01e-5
k8 = 4.19e4  
c8 = 1.78

I9 = 6.50e-6
k9 = 5.40e3     
c9 = 0.23

I10 = 5.65e-5
k10 = 4.19e4
c10 = 1.78

I11 = 4.27e-6
k11 = 1.22e3
c11 = 0.52

I12 = 3.25e-4
k12 = 4.33e4
c12 = 1.84
d12 = 0.0042

I13 = 1.20e-4
k13 = 3.10e4
c13 = 1.32

I14 = 1.15e-5
k14 = 1.14e3
c14 = 0.05

I15 = 1.32e-4
k15 = 3.10e4
c15 = 1.32

I16 = 4.27e-6
k16 = 1.22e4
c16 = 0.52

I17 = 2.69e-4
k17 = 4.43e4
c17 = 1.88
d17 = 0.0042

I18 = 1.80e-4
k18 = 1.38e5
c18 = 5.86

I19 = 2.00e-5
k19 = 2.00e4
c19 = 0.85

I20 = 2.00e-4
k20 = 1.38e5
c20 = 5.86

I21 = 4.27e-6
k21 = 1.22e4
c21 = 0.52

I22 = 4.95e-2
d22 = 0.24

def test_bench():
    """
    Create and return the OpenTorsion assembly for the test bench.
    
    Constructs a complete assembly model with shaft elements (representing
    torsional stiffness and damping), disk elements (representing rotational
    inertias), and gear elements (representing gearbox ratios).
    
    Returns:
    --------
    ot.Assembly
        Configured OpenTorsion assembly object ready for state-space
        conversion or simulation.
    """
    # Shaft elements
    shafts = []
    shafts.append(ot.Shaft(0, 1, k=k1, c=c1))
    shafts.append(ot.Shaft(1, 2, k=k2, c=c2))
    shafts.append(ot.Shaft(2, 3, k=k3, c=c3))
    shafts.append(ot.Shaft(3, 4, k=k4, c=c4))
    shafts.append(ot.Shaft(4, 5, k=k5, c=c5))
    shafts.append(ot.Shaft(5, 6, k=k6, c=c6))
    shafts.append(ot.Shaft(6, 7, k=k7, c=c7))
    shafts.append(ot.Shaft(7, 8, k=k8, c=c8))
    shafts.append(ot.Shaft(8, 9, k=k9, c=c9))
    shafts.append(ot.Shaft(9, 10, k=k10, c=c10))
    shafts.append(ot.Shaft(10, 11, k=k11, c=c11))

    shafts.append(ot.Shaft(12, 13, k=k12, c=c12))
    shafts.append(ot.Shaft(13, 14, k=k13, c=c13))
    shafts.append(ot.Shaft(14, 15, k=k14, c=c14))
    shafts.append(ot.Shaft(15, 16, k=k15, c=c15))
    shafts.append(ot.Shaft(16, 17, k=k16, c=c16))

    shafts.append(ot.Shaft(18, 19, k=k17, c=c17))
    shafts.append(ot.Shaft(19, 20, k=k18, c=c18))
    shafts.append(ot.Shaft(20, 21, k=k19, c=c19))
    shafts.append(ot.Shaft(21, 22, k=k20, c=c20))
    shafts.append(ot.Shaft(22, 23, k=k21, c=c21))

    """ Disk elements """
    disks = []
    disks.append(ot.Disk(0, I=I1, c=d1))
    disks.append(ot.Disk(1, I=I2))
    disks.append(ot.Disk(2, I=I3))
    disks.append(ot.Disk(3, I=I4))
    disks.append(ot.Disk(4, I=I5))
    disks.append(ot.Disk(5, I=I6))
    disks.append(ot.Disk(6, I=I7))
    disks.append(ot.Disk(7, I=I8))
    disks.append(ot.Disk(8, I=I9))
    disks.append(ot.Disk(9, I=I10))
    disks.append(ot.Disk(10, I=I11))
    disks.append(ot.Disk(11, I=I12/2, c=d12))
    disks.append(ot.Disk(12, I=I12/2, c=d12))
    disks.append(ot.Disk(13, I=I13))
    disks.append(ot.Disk(14, I=I14))
    disks.append(ot.Disk(15, I=I15))
    disks.append(ot.Disk(16, I=I16))
    disks.append(ot.Disk(17, I=I17/2, c=d17))
    disks.append(ot.Disk(18, I=I17/2, c=d17))
    disks.append(ot.Disk(19, I=I18))
    disks.append(ot.Disk(20, I=I19))
    disks.append(ot.Disk(21, I=I20))
    disks.append(ot.Disk(22, I=I21))
    disks.append(ot.Disk(23, I=I22, c=d22))

    """ Gear elements """
    gear1 = ot.Gear(11, 0, 10)
    gear2 = ot.Gear(12, 0, 30, parent=gear1)

    gear3 = ot.Gear(17, 0, 10)
    gear4 = ot.Gear(18, 0, 40, parent=gear3)

    gears = [gear1, gear2, gear3, gear4]

    """ Assembly """
    return ot.Assembly(shafts, disks, gear_elements=gears)

def get_state_structure(assembly):
    """
    Determine the state structure of an OpenTorsion assembly.
    
    For any OpenTorsion system with N states (odd number):
    - First half (odd amount): torque states, indices 0 to (N-1)//2
    - Second half (even amount): velocity states, indices (N+1)//2 to N-1
    
    Parameters:
    -----------
    assembly : ot.Assembly
        OpenTorsion assembly object
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'n_states': total number of states (odd)
        - 'n_torques': number of torque states ((N+1)/2)
        - 'n_velocities': number of velocity states ((N-1)/2)
        - 'torque_indices': list of torque state indices [0, ..., (N-1)//2]
        - 'velocity_indices': list of velocity state indices [(N+1)//2, ..., N-1]
    """
    A, B, C, D = assembly.state_space()

    n_states = A.shape[0]
    
    if n_states % 2 == 0:
        raise ValueError(f"OpenTorsion system must have odd number of states, got {n_states}")
    
    n_torques = (n_states + 1) // 2
    n_velocities = (n_states - 1) // 2
    
    torque_indices = list(range(0, (n_states - 1) // 2 + 1))
    velocity_indices = list(range((n_states + 1) // 2, n_states))
    
    return {
        'n_states': n_states,
        'n_torques': n_torques,
        'n_velocities': n_velocities,
        'torque_indices': torque_indices,
        'velocity_indices': velocity_indices
    }


def create_measurement_matrix(n_states, torque_sensor_locations=None, velocity_sensor_locations=None):
    """
    Create measurement matrix C from sensor location specifications.
    
    Parameters:
    -----------
    n_states : int
        Total number of states (must be odd)
    torque_sensor_locations : list of int, optional
        State indices for torque sensors (can be empty list)
    velocity_sensor_locations : list of int, optional
        State indices for velocity sensors (can be empty list)
        
    Returns:
    --------
    C : ndarray
        Measurement matrix (n_outputs x n_states)
    sensor_metadata : dict
        Dictionary containing:
        - 'torque_sensors': list of torque sensor state indices
        - 'velocity_sensors': list of velocity sensor state indices
        - 'sensor_types': list indicating type of each output row ('torque' or 'velocity')
        - 'sensor_indices': list indicating which state index each output row measures
    """
    if n_states % 2 == 0:
        raise ValueError(f"System must have odd number of states, got {n_states}")
    
    # Default to empty lists if not provided
    if torque_sensor_locations is None:
        torque_sensor_locations = []
    if velocity_sensor_locations is None:
        velocity_sensor_locations = []
    
    # Ensure inputs are lists
    torque_sensor_locations = list(torque_sensor_locations)
    velocity_sensor_locations = list(velocity_sensor_locations)
    
    # Validate at least one sensor type is provided
    if len(torque_sensor_locations) == 0 and len(velocity_sensor_locations) == 0:
        raise ValueError("At least one sensor type (torque or velocity) must be provided")
    
    # Combine all sensor locations
    all_sensor_indices = torque_sensor_locations + velocity_sensor_locations
    sensor_types = ['torque'] * len(torque_sensor_locations) + ['velocity'] * len(velocity_sensor_locations)
    
    # Create C matrix by selecting rows from identity matrix
    n_outputs = len(all_sensor_indices)
    C = np.eye(n_states)[all_sensor_indices, :]
    
    sensor_metadata = {
        'torque_sensors': torque_sensor_locations,
        'velocity_sensors': velocity_sensor_locations,
        'torque_sensor_state_indices': torque_sensor_locations,
        'velocity_sensor_state_indices': velocity_sensor_locations,
        'sensor_types': sensor_types,
        'sensor_indices': all_sensor_indices
    }
    
    return C, sensor_metadata


def minimal_state_space(assembly):
    """
    Convert OpenTorsion assembly to minimal state-space representation.
    
    Transforms the state-space matrices from the assembly's natural coordinates
    to a minimal representation using the transformation matrix X. Also extracts
    only the first and last columns of the B matrix to represent motor torque
    (u1) and propeller/load torque (u2) inputs.
    
    Parameters:
    -----------
    assembly : ot.Assembly
        OpenTorsion assembly object
        
    Returns:
    --------
    A : ndarray
        Minimal state matrix (n_states x n_states)
    B : ndarray
        Minimal input matrix (n_states x 2), where columns are [u1, u2]
    C : ndarray
        Output matrix (unchanged from assembly)
    D : ndarray
        Feedthrough matrix (unchanged from assembly)
    """
    A, B, C, D = assembly.state_space()

    X = assembly.X
    X_inv = pinv(X)
    A = X @ A @ X_inv
    BcX_full = X @ B
    B = np.vstack((BcX_full[:,0], BcX_full[:,-1])).T

    return A,B,C,D