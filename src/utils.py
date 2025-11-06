"""
Utility functions for system discretization and transformation.

This module provides helper functions for converting continuous-time state-space
models to discrete-time representations using zero-order hold (ZOH) discretization,
and for transforming state-space matrices to minimal form using OpenTorsion's
transformation matrix X.
"""

import numpy as np
from scipy.linalg import expm, pinv


def c2d(A: np.ndarray, B: np.ndarray, Ts: float):
    """Discretize continuous-time (A, B) with zero-order hold over sample time Ts.

    Returns (Ad, Bd).
    """
    m, n = A.shape
    mB, nb = B.shape
    s = np.concatenate([A, B], axis=1)
    s = np.concatenate([s, np.zeros((nb, n + nb))], axis=0)
    S = expm(s * Ts)
    Ad = S[0:n, 0:n]
    Bd = S[0:n, n:n + nb + 1]
    return Ad, Bd


def minimize(M: np.ndarray, assembly, matrix_type: str = 'auto'):
    """
    Transform a state-space matrix to minimal form using OpenTorsion's transformation matrix X.
    
    The transformation matrix X from OpenTorsion converts state-space matrices from
    the assembly's natural coordinates to a minimal representation.
    
    Parameters:
    -----------
    M : ndarray
        Matrix to transform. Can be:
        - State matrix (n_states x n_states): transformed as X @ M @ X_inv
        - Input matrix (n_states x n_inputs): transformed as X @ M
        - Output matrix (n_outputs x n_states): transformed as M @ X_inv
    assembly : ot.Assembly
        OpenTorsion assembly object (must have X attribute)
    matrix_type : str, optional
        Type of matrix: 'state', 'input', 'output', or 'auto' (default: 'auto')
        If 'auto', determines type based on dimensions:
        - Square matrix -> 'state'
        - Rows match assembly states -> 'input'
        - Columns match assembly states -> 'output'
        
    Returns:
    --------
    M_minimal : ndarray
        Transformed matrix in minimal form
        
    Examples:
    --------
    >>> A_minimal = minimize(A, assembly, 'state')  # X @ A @ X_inv
    >>> B_minimal = minimize(B, assembly, 'input')  # X @ B
    >>> C_minimal = minimize(C, assembly, 'output')  # C @ X_inv
    """
    if not hasattr(assembly, 'X'):
        raise AttributeError("Assembly must have X attribute (call assembly.state_space() first)")
    
    X = assembly.X
    X_inv = pinv(X)
    
    n_states_full = M.shape[0]  # Number of states in full representation
    n_states_minimal = X.shape[0]  # Number of states in minimal representation
    
    M_shape = M.shape
    
    # Auto-detect matrix type if not specified
    if matrix_type == 'auto':
        if M_shape[0] == M_shape[1] == n_states_full:
            matrix_type = 'state'
        elif M_shape[0] == n_states_full:
            matrix_type = 'input'
        elif M_shape[1] == n_states_full:
            matrix_type = 'output'
        else:
            raise ValueError(f"Cannot auto-detect matrix type. Matrix shape {M_shape} doesn't match "
                           f"full state dimension {n_states_full}. Specify matrix_type explicitly.")
    
    # Apply appropriate transformation based on matrix type
    if matrix_type == 'state':
        # State matrix: X @ M @ X_inv
        if M_shape[0] != n_states_full or M_shape[1] != n_states_full:
            raise ValueError(f"State matrix must be square ({n_states_full}x{n_states_full}), got {M_shape}")
        M_minimal = X @ M @ X_inv
    elif matrix_type == 'input':
        # Input matrix: X @ M
        if M_shape[0] != n_states_full:
            raise ValueError(f"Input matrix must have {n_states_full} rows, got {M_shape[0]}")
        M_minimal = X @ M
    elif matrix_type == 'output':
        # Output matrix: M @ X_inv
        if M_shape[1] != n_states_full:
            raise ValueError(f"Output matrix must have {n_states_full} columns, got {M_shape[1]}")
        M_minimal = M @ X_inv
    else:
        raise ValueError(f"Invalid matrix_type: {matrix_type}. Must be 'state', 'input', 'output', or 'auto'")
    
    return M_minimal


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
