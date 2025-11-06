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

