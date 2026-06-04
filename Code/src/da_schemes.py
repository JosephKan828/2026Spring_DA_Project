# ====================================================
# This script is to pack the module for data assimilation
# ====================================================

# ====================================================
# Import packages
# ====================================================
import numpy as np

from tqdm import tqdm
from scipy.optimize import minimize
from typing import Callable, Tuple

import l96_model as l96   #type: ignore
import integrate as integ #type: ignore

# ====================================================
# function for forecast-analysis cycle
# ====================================================

def forecast_analysis(
        init_state: l96.L96State,
        target_layers: Tuple[str, ...],
        obs_dict: dict[str, np.ndarray],
        params: l96.L96Params,
        dt: float,
        t_eval: float,
        da_scheme: Callable,
):
    # Time configuration
    num_steps = int(t_eval / dt)

    # Pre-allocate trajectory array
    X_traj: np.ndarray = np.zeros((num_steps, params.K))
    Y_traj: np.ndarray = np.zeros((num_steps, params.K, params.J))
    Z_traj: np.ndarray = np.zeros((num_steps, params.K, params.J, params.L))

    # current state
    curr_state: l96.L96State = init_state

    offset_Y: int = params.K
    offset_Z: int = params.K + (params.K * params.J)

    # O(1) Pointers
    ptr_x, ptr_y, ptr_z = 0, 0, 0
    idx_X, idx_Y, idx_Z = obs_dict["X_idx"], obs_dict["Y_idx"], obs_dict["Z_idx"]
    val_X, val_Y, val_Z = obs_dict["X"], obs_dict["Y"], obs_dict["Z"]

    for t in tqdm(range(num_steps), desc="DA Cycle"):
        
        arr_X = (ptr_x < len(idx_X)) and (t == idx_X[ptr_x])
        arr_Y = (ptr_y < len(idx_Y)) and (t == idx_Y[ptr_y])
        arr_Z = (ptr_z < len(idx_Z)) and (t == idx_Z[ptr_z])

        do_X = arr_X and ("X" in target_layers)
        do_Y = arr_Y and ("Y" in target_layers)
        do_Z = arr_Z and ("Z" in target_layers)

        if do_X or do_Y or do_Z:
            # Zero-Innovation Padding Trick
            state_flat = np.concatenate([
                curr_state[0].flatten(),
                curr_state[1].flatten(),
                curr_state[2].flatten()
            ])
            pad_obs = np.copy(state_flat)

            if do_X: pad_obs[:offset_Y] = val_X[ptr_x].flatten()
            if do_Y: pad_obs[offset_Y:offset_Z] = val_Y[ptr_y].flatten()
            if do_Z: pad_obs[offset_Z:] = val_Z[ptr_z].flatten()

            # ========================================================
            # DYNAMIC MATH INJECTION
            # Calls whichever DA method (OI or 3DVar) was passed in
            # ========================================================
            flags = (do_X, do_Y, do_Z)
            curr_state = da_scheme(curr_state, pad_obs, flags)

        # Advance pointers
        if arr_X: ptr_x += 1
        if arr_Y: ptr_y += 1
        if arr_Z: ptr_z += 1

        # Store trajectory and integrate forward
        X_traj[t] = curr_state[0]
        Y_traj[t] = curr_state[1]
        Z_traj[t] = curr_state[2]

        curr_state = integ.rk4_step(
            state=curr_state, params=params, dt=dt, tendency_fn=l96.compute_tendencies
        )

    return X_traj, Y_traj, Z_traj

# ====================================================
# Optimal interpolation
# ====================================================

def optimal_interpolation(
        state: l96.L96State,
        Kalman_gain: np.ndarray,
        pad_obs: np.ndarray,
) -> l96.L96State:
    
    # Flatten the background state
    state_flat: np.ndarray = np.concatenate([
        state[0].flatten(),
        state[1].flatten(),
        state[2].flatten()
    ])

    # Calculate analysis field: x_a = x_b + K(y-x_b)
    state_update: np.ndarray = state_flat + Kalman_gain @ (pad_obs - state_flat)

    # Reshape back to L96 geometry
    X_shape: int = state[0].size
    Y_shape: int = state[1].size
    Z_shape: int = state[2].size

    X_update: np.ndarray = state_update[:X_shape].reshape(state[0].shape)
    Y_update: np.ndarray = state_update[X_shape : X_shape+Y_shape].reshape(state[1].shape)
    Z_update: np.ndarray = state_update[-Z_shape:].reshape(state[2].shape)

    return (X_update, Y_update, Z_update)

# ====================================================
# incremental 3D-Var
# ====================================================
def incremental_3DVar(
        state: l96.L96State,
        B_inv  : np.ndarray,
        R_inv_diag  : np.ndarray,
        observations: np.ndarray,
        outer_times: int = 1
) -> l96.L96State:
    """
    Executes Incremental 3D-Var using the analytical gradient and 
    Conjugate Gradient minimization. Optimized for speed.
    """
    
    state_flat: np.ndarray = np.concatenate([
        state[0].flatten(),
        state[1].flatten(),
        state[2].flatten()
    ])

    x_b: np.ndarray = np.copy(state_flat)

    # ------------------------------------------------------------------
    # Outer Loop
    # ------------------------------------------------------------------
    for _ in range(outer_times):
        
        d: np.ndarray = observations - x_b

        def cost_function(dx: np.ndarray) -> float:
            # Replaced linalg.solve with fast matrix multiplication
            B_inv_dx = B_inv @ dx
            term1 = 0.5 * np.dot(dx, B_inv_dx)
            
            diff = dx - d
            term2 = 0.5 * np.sum(diff * R_inv_diag * diff)
            
            return term1 + term2

        def gradient(dx: np.ndarray) -> np.ndarray:
            # Replaced linalg.solve with fast matrix multiplication
            B_inv_dx = B_inv @ dx
            diff = dx - d
            grad = B_inv_dx + (R_inv_diag * diff)
            return grad

        dx0: np.ndarray = np.zeros_like(x_b)
        
        res = minimize(
            fun=cost_function,
            x0=dx0,
            jac=gradient,
            method='L-BFGS-B', 
            options={'disp': False, 'maxiter': 50} # Reduced maxiter for safety
        )
        
        x_b = x_b + res.x

    # ------------------------------------------------------------------
    # Reshape back to L96 geometry
    # ------------------------------------------------------------------
    X_shape: int = state[0].size
    Y_shape: int = state[1].size
    Z_shape: int = state[2].size

    X_update: np.ndarray = x_b[:X_shape].reshape(state[0].shape)
    Y_update: np.ndarray = x_b[X_shape : X_shape+Y_shape].reshape(state[1].shape)
    Z_update: np.ndarray = x_b[-Z_shape:].reshape(state[2].shape)

    return (X_update, Y_update, Z_update)

# ====================================================
# Placeholder for normal 3D-Var
# ====================================================

def normal_3DVar(
        state: l96.L96State,
        BEC_inv  : np.ndarray,
        R_inv_diag  : np.ndarray,
        observations: np.ndarray,
) -> l96.L96State:
    
    # 1. flatten the background state
    state_flat: np.ndarray = np.concatenate([
        state[0].flatten(),
        state[1].flatten(),
        state[2].flatten()
    ])

    x_b: np.ndarray = np.copy(state_flat)
    
    # ------------------------------------------------------------------
    # Combined Cost & Gradient Function
    # ------------------------------------------------------------------
    # OPTIMIZATION 2: Calculate both at once to halve the matrix multiplications
    def cost_and_gradient(x: np.ndarray) -> tuple[float, np.ndarray]:
        
        # Calculate the heavy B matrix math exactly ONCE
        B_inv_dx = BEC_inv @ (x - x_b)
        
        # Calculate the fast 1D R vector math
        obs_diff = x - observations
        R_inv_dx = R_inv_diag * obs_diff
        
        # 1. Calculate Cost (Scalar)
        J_b = 0.5 * np.dot((x - x_b), B_inv_dx)
        J_o = 0.5 * np.dot(obs_diff, R_inv_dx)
        cost = J_b + J_o
        
        # 2. Calculate Gradient (1D Array)
        grad = B_inv_dx + R_inv_dx
        
        return cost, grad

    # ------------------------------------------------------------------
    # Minimize Cost Function
    # ------------------------------------------------------------------
    res = minimize(
        fun=cost_and_gradient,
        x0=state_flat,
        jac=True,               # <-- Tells SciPy that 'fun' returns (cost, gradient)
        method='L-BFGS-B', 
        options={'disp': False, 'maxiter': 50} 
    )

    x_a: np.ndarray = res.x

    # ------------------------------------------------------------------
    # Reshape back to L96 geometry
    # ------------------------------------------------------------------
    X_shape: int = state[0].size
    Y_shape: int = state[1].size
    Z_shape: int = state[2].size

    X_update: np.ndarray = x_a[:X_shape].reshape(state[0].shape)
    Y_update: np.ndarray = x_a[X_shape : X_shape+Y_shape].reshape(state[1].shape)
    Z_update: np.ndarray = x_a[-Z_shape:].reshape(state[2].shape)

    return (X_update, Y_update, Z_update)