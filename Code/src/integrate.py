"""
Numerical Integration Steppers for Data Assimilation Workflows
"""

from typing import Callable
from l96_model import L96State, L96Params

def rk4_step(
    state: L96State, 
    params: L96Params, 
    dt: float, 
    tendency_fn: Callable[[L96State, L96Params], L96State]
) -> L96State:
    """Advances the 3-scale L96 state by a single time-step dt using 4th-order Runge-Kutta.
    
    Args:
        state: Tuple of arrays (X, Y, Z) at time t
        params: Instantiated L96Params containing scales and constants
        dt: The integration time step size
        tendency_fn: Function that computes model physics (e.g., compute_tendencies)
    """
    X, Y, Z = state
    
    # Stage 1
    k1_X, k1_Y, k1_Z = tendency_fn(state, params)
    
    # Stage 2
    state_k2 = (X + 0.5 * dt * k1_X, Y + 0.5 * dt * k1_Y, Z + 0.5 * dt * k1_Z)
    k2_X, k2_Y, k2_Z = tendency_fn(state_k2, params)
    
    # Stage 3
    state_k3 = (X + 0.5 * dt * k2_X, Y + 0.5 * dt * k2_Y, Z + 0.5 * dt * k2_Z)
    k3_X, k3_Y, k3_Z = tendency_fn(state_k3, params)
    
    # Stage 4
    state_k4 = (X + dt * k3_X, Y + dt * k3_Y, Z + dt * k3_Z)
    k4_X, k4_Y, k4_Z = tendency_fn(state_k4, params)
    
    # Weighted update
    X_next = X + (dt / 6.0) * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
    Y_next = Y + (dt / 6.0) * (k1_Y + 2 * k2_Y + 2 * k3_Y + k4_Y)
    Z_next = Z + (dt / 6.0) * (k1_Z + 2 * k2_Z + 2 * k3_Z + k4_Z)
    
    return X_next, Y_next, Z_next