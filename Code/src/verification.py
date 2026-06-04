# ====================================================
# This script is to implement verification mertics
# ====================================================

# ====================================================
# Import packages
# ====================================================

import numpy as np
from typing import Tuple

import l96_model as l96   # type: ignore


# ====================================================
# RMSE Calculation
# ====================================================
def calc_RMSE(
        truth: l96.L96State,
        estim: l96.L96State
) -> Tuple[float, float, float]:
    rmse_X = np.sqrt(np.mean((truth[0] - estim[0]) ** 2))
    rmse_Y = np.sqrt(np.mean((truth[1] - estim[1]) ** 2))
    rmse_Z = np.sqrt(np.mean((truth[2] - estim[2]) ** 2))
    return rmse_X, rmse_Y, rmse_Z

# ====================================================
# anomaly correlation coefficient (ACC)
# ====================================================

def calc_ACC(
        truth : l96.L96State,
        estim : l96.L96State,
        nature: l96.L96State
) -> Tuple[np.ndarray, ...]:
    
    # Calculate climatology
    clim_step: int = 3000

    clim_X: np.ndarray = np.mean(nature[0][clim_step:], axis=0, keepdims=True)
    clim_Y: np.ndarray = np.mean(nature[1][clim_step:], axis=0, keepdims=True)
    clim_Z: np.ndarray = np.mean(nature[2][clim_step:], axis=0, keepdims=True)

    # Calculate anomalies
    truth_X_anom: np.ndarray = truth[0] - clim_X
    truth_Y_anom: np.ndarray = truth[1] - clim_Y
    truth_Z_anom: np.ndarray = truth[2] - clim_Z

    estim_X_anom: np.ndarray = estim[0] - clim_X
    estim_Y_anom: np.ndarray = estim[1] - clim_Y
    estim_Z_anom: np.ndarray = estim[2] - clim_Z
    
    # Calculate ACC
    acc_X: np.ndarray = np.sum(truth_X_anom * estim_X_anom) / np.sqrt(np.sum(truth_X_anom**2) * np.sum(estim_X_anom**2))
    acc_Y: np.ndarray = np.sum(truth_Y_anom * estim_Y_anom) / np.sqrt(np.sum(truth_Y_anom**2) * np.sum(estim_Y_anom**2))
    acc_Z: np.ndarray = np.sum(truth_Z_anom * estim_Z_anom) / np.sqrt(np.sum(truth_Z_anom**2) * np.sum(estim_Z_anom**2))

    return (acc_X, acc_Y, acc_Z)