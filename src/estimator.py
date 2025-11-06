"""
Unified state estimation interface for OpenTorsion systems.

This module provides a base StateEstimator class and MHEEstimator implementation
for state and input estimation in rotating machinery systems. The framework
supports flexible sensor configuration using sensor location lists.
"""

import time
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from abc import ABC, abstractmethod

import numpy as np
import cvxpy as cp
from utils import c2d, minimize, create_measurement_matrix


class StateEstimator(ABC):
    """
    Base class for state estimators in OpenTorsion systems.
    
    Provides a common interface for different estimation methods with
    sensor-based measurement configuration.
    """
    
    def __init__(self, assembly, measurement_config: dict, settings: dict):
        """
        Initialize estimator with assembly and configuration.
        
        Parameters:
        -----------
        assembly : ot.Assembly
            OpenTorsion assembly object
        measurement_config : dict
            Dictionary with keys:
            - 'torque_sensors': list of disk numbers for torque sensors (can be empty)
            - 'velocity_sensors': list of disk numbers for velocity sensors (can be empty)
            - 'inputs': list of input names, e.g., ['motor'], ['load'], or ['motor', 'load']
            Inputs specified in this list are measured (go to B1), others are unmeasured (go to B2).
            Note: Disk numbers are automatically converted to state indices with appropriate offsets.
        settings : dict
            Estimator-specific settings dictionary
        """
        self.assembly = assembly
        self.measurement_config = measurement_config.copy()
        self.settings = settings.copy()
        
        # First: Get continuous-time state-space matrices from assembly and minimize them
        # This also populates assembly.X needed for minimal form transformation
        A_full, B_full, _, _ = assembly.state_space()
        
        # Transform to minimal form using minimize function
        A_c = minimize(A_full, assembly, 'state')  # X @ A @ X_inv
        B_c = minimize(B_full, assembly, 'input')  # X @ B
        
        # Second: Get state structure from minimized state matrix
        # Determine structure from the minimized A matrix dimensions
        n_states = A_c.shape[0]
        if n_states % 2 == 0:
            raise ValueError(f"Minimal state-space must have odd number of states, got {n_states}")
        
        n_torques = (n_states - 1) // 2
        n_velocities = (n_states + 1) // 2
        
        self.state_structure = {
            'n_states': n_states,
            'n_torques': n_torques,
            'n_velocities': n_velocities,
            'torque_indices': list(range(0, n_torques)),
            'velocity_indices': list(range(n_torques, n_states))
        }
        self.n_states = n_states
        
        # Third: Create measurement matrix
        torque_sensors = self.measurement_config.get('torque_sensors', [])
        velocity_sensors = self.measurement_config.get('velocity_sensors', [])
        
        self.C, self.sensor_metadata = create_measurement_matrix(
            n_states=self.n_states,
            torque_sensor_locations=torque_sensors,
            velocity_sensor_locations=velocity_sensors
        )
        
        # Discretize if sample time is provided
        Ts = self.settings.get('Ts')
        if Ts is None:
            raise ValueError("settings['Ts'] must be provided for discretization")
        
        self.A, self.B = c2d(A_c, B_c, Ts)
        
        # Split B matrix based on inputs configuration
        self.inputs_config = self.measurement_config.get('inputs', ['motor', 'load'])
        
        # Map input names to column indices: motor -> 0, load -> 1
        input_name_to_col = {'motor': 0, 'load': 1}
        
        # Validate input names
        valid_inputs = set(input_name_to_col.keys())
        for inp in self.inputs_config:
            if inp not in valid_inputs:
                raise ValueError(f"Invalid input name '{inp}'. Must be one of {valid_inputs}")
        
        # Determine which columns are measured (B1) and unmeasured (B2)
        measured_indices = [input_name_to_col[inp] for inp in self.inputs_config]
        all_indices = list(input_name_to_col.values())
        unmeasured_indices = [idx for idx in all_indices if idx not in measured_indices]
        
        # B1 contains all measured inputs (columns specified in inputs list)
        # B2 contains all unmeasured inputs (columns not in inputs list)
        if len(measured_indices) > 0:
            self.B1 = self.B[:, measured_indices]
        else:
            raise ValueError("At least one input must be specified in measurement_config['inputs']")
        
        if len(unmeasured_indices) > 0:
            self.B2 = self.B[:, unmeasured_indices]
        else:
            # If all inputs are measured, B2 should be empty (no unmeasured inputs to estimate)
            # Create empty matrix with correct number of rows
            self.B2 = np.zeros((self.B.shape[0], 0))
        
        # Placeholders for data
        self.y = None
        self.u1 = None
        self.t = None
    
    def setup_system_matrices(self):
        """
        Set up system matrices from assembly and measurement config.
        
        This is called automatically in __init__, but can be called again
        if measurement config changes.
        """
        # Recreate measurement matrix
        torque_sensors = self.measurement_config.get('torque_sensors', [])
        velocity_sensors = self.measurement_config.get('velocity_sensors', [])
        
        self.C, self.sensor_metadata = create_measurement_matrix(
            n_states=self.n_states,
            torque_sensor_locations=torque_sensors,
            velocity_sensor_locations=velocity_sensors
        )
    
    def load_data(self, y: np.ndarray, u1: np.ndarray, t: np.ndarray):
        """
        Load measurement data and optional ground truth for estimation.
        
        Parameters:
        -----------
        y : ndarray
            Measurement sequence (N x n_outputs)
            Must match the number of sensors defined in measurement_config
        u1 : ndarray
            Known input sequence (n_inputs1 x N) or (N,) for single input
        t : ndarray
            Time vector (N,)
        """
        # Validate y shape
        expected_outputs = self.C.shape[0]
        if y.shape[1] != expected_outputs:
            raise ValueError(f"y must have {expected_outputs} columns (one per sensor), got {y.shape[1]}")
        
        self.y = y
        self.u1 = u1
        self.t = t
    
    @abstractmethod
    def estimate(self, x_init: Optional[np.ndarray] = None):
        """
        Perform state estimation.
        
        Parameters:
        -----------
        x_init : ndarray, optional
            Initial state estimate
            
        Returns:
        --------
        dict
            Dictionary with estimation results
        """
        pass
    
    @abstractmethod
    def save_results(self, output_path: str):
        """
        Save estimation results to file.
        
        Parameters:
        -----------
        output_path : str
            Path to save results
        """
        pass
    
    @abstractmethod
    def plot_results(self, results_path: str, reference_data: Optional[dict] = None):
        """
        Plot estimation results.
        
        Parameters:
        -----------
        results_path : str
            Path to saved results file
        reference_data : dict, optional
            Reference data for comparison plots
        """
        pass


class MHEEstimator(StateEstimator):
    """
    Moving Horizon Estimation (MHE) implementation for state and input estimation.
    
    Implements a moving horizon estimation algorithm using convex optimization
    (CVXPY) to solve a constrained optimization problem over a sliding window
    of measurements.
    """
    
    def __init__(self, assembly, measurement_config: dict, settings: dict):
        """
        Initialize MHE estimator.
        
        Parameters:
        -----------
        assembly : ot.Assembly
            OpenTorsion assembly object
        measurement_config : dict
            Dictionary with keys:
            - 'torque_sensors': list of disk numbers for torque sensors (can be empty)
            - 'velocity_sensors': list of disk numbers for velocity sensors (can be empty)
            - 'inputs': list of input names, e.g., ['motor'], ['load'], or ['motor', 'load']
            Inputs specified in this list are measured (go to B1), others are unmeasured (go to B2).
            Note: Disk numbers are automatically converted to state indices with appropriate offsets.
        settings : dict
            Dictionary with keys:
            - 'horizon_length': int
            - 'Ts': float (discrete sample time)
            - 'Q_v_scale': float (default 0.05)
            - 'Q_w_scale': float (default 0.0001)
            - 'lambda_': float (default 10.0)
        """
        super().__init__(assembly, measurement_config, settings)
        
        self.horizon_length = int(self.settings.get("horizon_length", 10))
        self.lambda_ = float(self.settings.get("lambda_", 10.0))
        self.Q_v_scale = float(self.settings.get("Q_v_scale", 0.05))
        self.Q_w_scale = float(self.settings.get("Q_w_scale", 0.0001))
    
    def _run_window(self,
        y_window: np.ndarray,
        u1_window: np.ndarray,
        Q_v_inv: np.ndarray,
        Q_w_inv: np.ndarray,
        x_prev: np.ndarray,
        u_prev: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve MHE optimization problem for a single time window.
        
        Parameters:
        -----------
        y_window : ndarray
            Measurement sequence for the window (horizon_length x n_outputs)
        u1_window : ndarray
            Known input sequence (n_inputs1 x horizon_length)
        Q_v_inv : ndarray
            Inverse measurement noise covariance matrix
        Q_w_inv : ndarray
            Inverse process noise covariance matrix
        x_prev : ndarray
            Previous state estimate (used for first step in window)
        u_prev : ndarray
            Previous input estimate (used for smoothness constraint)
            
        Returns:
        --------
        x_win : ndarray
            Estimated state sequence (n_states x horizon_length)
        u2_win : ndarray
            Estimated unknown input sequence (n_inputs2 x horizon_length)
        win_time : float
            Time taken to solve the optimization problem (seconds)
        """
        n_states = self.A.shape[0]
        n_inputs = self.B2.shape[1]

        x_var = cp.Variable((n_states, self.horizon_length), complex=False)
        u2_var = cp.Variable((n_inputs, self.horizon_length), complex=False)
        delta_var = cp.Variable((n_inputs, self.horizon_length), complex=False)

        start_time = time.time()

        objective_expr = 0
        for k in range(self.horizon_length):
            if k == 0:
                objective_expr += cp.quad_form(y_window[k] - self.C @ x_var[:, k], Q_v_inv)
                objective_expr += cp.quad_form(self.A @ x_prev + self.B1 @ u1_window[:, k] + self.B2 @ u2_var[:, k] - x_var[:, k], Q_w_inv)
            else:
                objective_expr += cp.quad_form(y_window[k] - self.C @ x_var[:, k], Q_v_inv)
                objective_expr += cp.quad_form(self.A @ x_var[:, k - 1] + self.B1 @ u1_window[:, k] + self.B2 @ u2_var[:, k] - x_var[:, k], Q_w_inv)

            objective_expr += self.lambda_ * cp.sum_squares(delta_var)

        constraints = []
        for k in range(self.horizon_length - 1):
            if k == 0:
                constraints.append(delta_var[:, k] == u_prev - 2 * u2_var[:, k] + u2_var[:, k + 1])
            else:
                constraints.append(delta_var[:, k] == u2_var[:, k - 1] - 2 * u2_var[:, k] + u2_var[:, k + 1])

        problem = cp.Problem(cp.Minimize(objective_expr), constraints)
        problem.solve()

        end_time = time.time()
        return x_var.value, u2_var.value, end_time - start_time

    def _estimate_full_series(self,
        y: np.ndarray,
        u1: np.ndarray,
        Q_v: Optional[np.ndarray] = None,
        Q_w: Optional[np.ndarray] = None,
        x_init: Optional[np.ndarray] = None,
        u_init: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[float], float]:
        """
        Run MHE estimation over the entire measurement sequence.
        
        Parameters:
        -----------
        y : ndarray
            Measurement sequence (N x n_outputs)
        u1 : ndarray
            Known input sequence (n_inputs1 x N)
        Q_v : ndarray, optional
            Measurement noise covariance (default: Q_v_scale * I)
        Q_w : ndarray, optional
            Process noise covariance (default: Q_w_scale * I)
        x_init : ndarray, optional
            Initial state estimate (default: zeros)
        u_init : ndarray, optional
            Initial input estimate (default: zeros)
            
        Returns:
        --------
        xhat : ndarray
            Estimated state sequence (n_states x N)
        uhat : ndarray
            Estimated unknown input sequence (n_inputs2 x N)
        window_times : List[float]
            Computation time for each window (seconds)
        total_time : float
            Total computation time (seconds)
        """
        N = y.shape[0]
        n_states = self.A.shape[0]
        n_inputs = self.B2.shape[1]

        if Q_v is None:
            Q_v = self.Q_v_scale * np.eye(self.C.shape[0])
        if Q_w is None:
            Q_w = self.Q_w_scale * np.eye(n_states)

        Q_v_inv = np.linalg.inv(Q_v)
        Q_w_inv = np.linalg.inv(Q_w)

        xhat = np.zeros((n_states, N))
        uhat = np.zeros((n_inputs, N))

        if x_init is None:
            x_init = np.zeros(n_states)
        if u_init is None:
            u_init = np.zeros(n_inputs)

        window_times: List[float] = []
        total_start = time.time()

        for i in range(N - self.horizon_length + 1):
            print(i)
            y_window = y[i : i + self.horizon_length]
            u1_window = u1[:, i : i + self.horizon_length]

            if i == 0:
                x_prev = x_init
                u_prev = u_init
            else:
                x_prev = xhat[:, i - 1]
                u_prev = uhat[:, i - 1]

            x_win, u2_win, win_time = self._run_window(
                y_window=y_window,
                u1_window=u1_window,
                Q_v_inv=Q_v_inv,
                Q_w_inv=Q_w_inv,
                x_prev=x_prev,
                u_prev=u_prev,
            )

            window_times.append(win_time)

            if i == N - self.horizon_length:
                xhat[:, -self.horizon_length:] = x_win
                uhat[:, -self.horizon_length:] = u2_win
            else:
                xhat[:, i] = x_win[:, 0]
                uhat[:, i] = u2_win[:, 0]

        total_end = time.time()
        return xhat, uhat, window_times, total_end - total_start

    def estimate(self, x_init: Optional[np.ndarray] = None):
        """
        Run MHE estimation and return results.
        
        Parameters:
        -----------
        x_init : ndarray, optional
            Initial state estimate (default: zeros)
            
        Returns:
        --------
        dict
            Dictionary with keys:
            - 'xhat': estimated states (n_states x N)
            - 'uhat': estimated inputs (n_inputs2 x N)
            - 'times': window computation times
            - 'total_time': total computation time
        """
        if self.y is None or self.u1 is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        n_outputs = self.C.shape[0]
        n_states = self.A.shape[0]
        Q_v = self.Q_v_scale * np.eye(n_outputs)
        Q_w = self.Q_w_scale * np.eye(n_states)

        if x_init is None:
            x_init = np.zeros(n_states)

        xhat, uhat, times, total_time = self._estimate_full_series(
            y=self.y,
            u1=self.u1,
            Q_v=Q_v,
            Q_w=Q_w,
            x_init=x_init,
            u_init=2.3 * np.ones(self.B2.shape[1]),
        )

        # Display timing statistics
        print("=" * 60)
        print("MOVING HORIZON ESTIMATION TIMING RESULTS")
        print("=" * 60)
        print(f"Total simulation length: {self.t[-1]} seconds")
        print(f"Total estimation time: {total_time:.4f} seconds")
        print(f"Number of iterations: {len(times)}")
        if len(times) > 0:
            print(f"Average window time: {np.mean(times):.4f} seconds")
            print(f"Min window time: {np.min(times):.4f} seconds")
            print(f"Max window time: {np.max(times):.4f} seconds")
            print(f"Standard deviation: {np.std(times):.4f} seconds")
        print("=" * 60)

        return {
            'xhat': xhat,
            'uhat': uhat,
            'times': times,
            'total_time': total_time
        }

    def save_results(self, output_path: str, results: Optional[dict] = None):
        """
        Save estimation results to file.
        
        Parameters:
        -----------
        output_path : str
            Path to save results (.npz file). If relative path, saves to data/
        results : dict, optional
            Results dictionary from estimate(). If None, calls estimate() first.
        """
        if results is None:
            results = self.estimate()
        
        if self.y is None or self.u1 is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        # Resolve output path to data/ by default
        target_path = Path(output_path)
        if target_path.parent == Path('.'):
            target_path = Path('data') / target_path.name
        target_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            str(target_path),
            xhat=results['xhat'],
            uhat=results['uhat'],
            times=np.array(results['times']),
            total_time=results['total_time'],
            t=self.t,
            y=self.y,
            u1=self.u1,
            torque_sensors=np.array(self.sensor_metadata['torque_sensors']),  # Disk numbers
            velocity_sensors=np.array(self.sensor_metadata['velocity_sensors']),  # Disk numbers
            torque_sensor_state_indices=np.array(self.sensor_metadata['torque_sensor_state_indices']),  # Converted state indices
            velocity_sensor_state_indices=np.array(self.sensor_metadata['velocity_sensor_state_indices']),  # Converted state indices
            sensor_types=np.array(self.sensor_metadata['sensor_types']),
            sensor_indices=np.array(self.sensor_metadata['sensor_indices']),
        )

    def plot_results(self, results_path: str, reference_data: Optional[dict] = None):
        """
        Plot estimation results from saved file.
        
        This method calls the standalone plotting function. For direct plotting
        without an estimator instance, use plot_results.plot_results() directly.
        
        Parameters:
        -----------
        results_path : str
            Path to saved results file (.npz). If relative, loads from data/
        reference_data : dict, optional
            Reference data for comparison with keys:
            - 'torque': measured torque data (N x n_torque_sensors)
            - 'velocity': measured velocity data (N x n_velocity_sensors)
            - 'prop_torque': propeller/load torque (N,)
            - 'u1_noisy': noisy motor torque input (N,) or (1 x N)
        """
        from plot_results import plot_results as plot_func
        
        # Add u1_noisy from truth if available
        if reference_data is None:
            reference_data = {}
        if "u1_noisy" in self.truth:
            reference_data['u1_noisy'] = self.truth['u1_noisy']
        
        plot_func(results_path, reference_data)

