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
        truth: Tuple[np.ndarray, ...],
        estim: Tuple[np.ndarray, ...]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    rmse_X = np.sqrt(np.mean((truth[0] - estim[0]) ** 2, axis=(1)))
    rmse_Y = np.sqrt(np.mean((truth[1] - estim[1]) ** 2, axis=(1, 2)))
    rmse_Z = np.sqrt(np.mean((truth[2] - estim[2]) ** 2, axis=(1, 2, 3)))
    return rmse_X, rmse_Y, rmse_Z

# ====================================================
# anomaly correlation coefficient (ACC)
# ====================================================

def calc_ACC(
        truth: Tuple[np.ndarray, ...],
        estim: Tuple[np.ndarray, ...],
        free : Tuple[np.ndarray, ...]
) -> Tuple[np.ndarray, ...]:
    
    # Calculate climatology
    clim_step: int = 3000

    # Calculate anomalies
    truth_X_anom: np.ndarray = truth[0] - truth[0][-clim_step:].mean(axis=0, keepdims=True)
    truth_Y_anom: np.ndarray = truth[1] - truth[1][-clim_step:].mean(axis=0, keepdims=True)
    truth_Z_anom: np.ndarray = truth[2] - truth[2][-clim_step:].mean(axis=0, keepdims=True)

    estim_X_anom: np.ndarray = estim[0] - free[0][-clim_step:].mean(axis=0, keepdims=True)
    estim_Y_anom: np.ndarray = estim[1] - free[1][-clim_step:].mean(axis=0, keepdims=True)
    estim_Z_anom: np.ndarray = estim[2] - free[2][-clim_step:].mean(axis=0, keepdims=True)
    
    # Calculate ACC
    acc_X: np.ndarray = np.sum(truth_X_anom * estim_X_anom, axis=(1,)) / np.sqrt(np.sum(truth_X_anom**2, axis=(1,)) * np.sum(estim_X_anom**2, axis=(1, )))
    acc_Y: np.ndarray = np.sum(truth_Y_anom * estim_Y_anom, axis=(1, 2)) / np.sqrt(np.sum(truth_Y_anom**2, axis=(1, 2)) * np.sum(estim_Y_anom**2, axis=(1, 2)))
    acc_Z: np.ndarray = np.sum(truth_Z_anom * estim_Z_anom, axis=(1, 2, 3)) / np.sqrt(np.sum(truth_Z_anom**2, axis=(1, 2, 3)) * np.sum(estim_Z_anom**2, axis=(1, 2, 3)))

    return (acc_X, acc_Y, acc_Z)