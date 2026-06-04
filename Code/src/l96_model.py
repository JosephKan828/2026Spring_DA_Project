"""
Generalized Three-Layer Lorenz '96 Model Implementation - Core Model Logic
"""

import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class L96Params:
    """Parameters for the generalized 3-scale L96 model.
    
    Default values match the classic configuration. For OSSE setups, pass 
    different values (e.g., h=1.0 for Nature, h=0.95 for Assimilation).
    """
    K: int = 40       # Number of large-scale variables (X)
    J: int = 8        # Number of intermediate variables (Y) per X
    L: int = 8        # Number of small-scale variables (Z) per Y
    F: float = 10.0   # Forcing term
    c: float = 10.0   # Time-scale ratio for Y
    b: float = 10.0   # Spatial-scale ratio for Y
    z: float = 1.0    # Damping factor for Z
    h: float = 1.0    # Coupling constant

    @property
    def ch(self) -> float:
        return self.c
    
    @property
    def bh(self) -> float:
        return self.b


# Type alias for the 3-layer state tuple (X, Y, Z)
L96State = tuple[np.ndarray, np.ndarray, np.ndarray]


def compute_tendencies(state: L96State, params: L96Params) -> L96State:
    """Calculates the ODE tendencies (dXdt, dYdt, dZdt) for the 3-scale L96 model."""
    X, Y, Z = state
    
    # --- 1. Large-Scale (X) Tendencies ---
    X_n1 = np.roll(X, 1)
    X_n2 = np.roll(X, 2)
    X_p1 = np.roll(X, -1)
    
    Y_sum = np.sum(Y, axis=1)  # Spatial coupling: average Y over its subgrid J
    
    dXdt = X_n1 * (X_p1 - X_n2) - X + params.F - (params.h * params.c / params.b) * Y_sum
    
    # --- 2. Intermediate-Scale (Y) Tendencies ---
    # Flattening ensures proper global periodic boundaries across the subgrid blocks
    Y_flat = Y.flatten()
    Y_n1 = np.roll(Y_flat, 1).reshape(params.K, params.J)
    Y_p1 = np.roll(Y_flat, -1).reshape(params.K, params.J)
    Y_p2 = np.roll(Y_flat, -2).reshape(params.K, params.J)
    
    Z_sum = np.sum(Z, axis=2)  # Spatial coupling: average Z over its subgrid L
    
    dYdt = -params.c * params.b * Y_p1 * (Y_p2 - Y_n1) - params.c * Y + \
           (params.h * params.c / params.b) * X[:, None] - \
           (params.h * params.ch / params.bh) * Z_sum
           
    # --- 3. Small-Scale (Z) Tendencies ---
    Z_flat = Z.flatten()
    Z_n1 = np.roll(Z_flat, 1).reshape(params.K, params.J, params.L)
    Z_n2 = np.roll(Z_flat, 2).reshape(params.K, params.J, params.L)
    Z_p1 = np.roll(Z_flat, -1).reshape(params.K, params.J, params.L)
    Z_p2 = np.roll(Z_flat, -2).reshape(params.K, params.J, params.L)
    
    dZdt = params.ch * params.bh * Z_n1 * (Z_p1 - Z_n2) - params.z * params.ch * Z + \
           (params.h * params.ch / params.bh) * Y[:, :, None]
    
    return dXdt, dYdt, dZdt