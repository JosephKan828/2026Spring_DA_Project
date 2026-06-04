# ====================================================
# This script is to apply 3DVar on generalized Lorenz
# 1996 model with BEC calculated by NMC method
# ====================================================

# ====================================================

import os
import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.optimize import minimize
from matplotlib import pyplot as plt

sys.path.append("/Users/joseph/Desktop/NTU Course/114-2/Data Assimilation/Project/Code/src")
import l96_model as l96   # type: ignore
import integrate as integ # type: ignore
import da_schemes as da   # type: ignore

# ====================================================
# Main function
# ====================================================

def main(target_layers: tuple[str, ...]):
    print(f"--- Running DA Experiment for layers: {target_layers} ---")

    # ------------------------------------------------
    # Parameter and Path setting 
    # ------------------------------------------------
    params: l96.L96Params = l96.L96Params(h=0.9)
    dt    : float = 0.01  
    t_eval: float = 50.0  

    root_path  : Path = Path("/Users/joseph/Desktop/NTU Course/114-2/Data Assimilation/Project/")
    file_path  : Path = root_path / "Files"
    figure_path: Path = root_path / "Figure"

    # ------------------------------------------------
    # Collect Data
    # ------------------------------------------------

    bec_source: str = "climatology" # "climatology" or "NMC"

    data_dict: dict[str, dict[str, np.ndarray]] = {
        "init"  : dict(np.load(file_path / "free_run_initial_state.npz")),
        "nature": dict(np.load(file_path / "nature_run_trajectory.npz")),
        "obs"   : dict(np.load(file_path / "simulated_observations.npz")),
        "bec"   : dict(np.load(file_path / f"{bec_source}_BEC.npz"))
    }
    
    # ------------------------------------------------
    # Pre-compute ALL OEC Combinations
    # ------------------------------------------------

    scaling_factor: float = 1e-1 if bec_source == "NMC" else 1e-3
    all_BEC: np.ndarray = data_dict["bec"]["all"] * scaling_factor

    # add block-diagonal mask
    offset_Y = params.K
    offset_Z = params.K + (params.K * params.J)

    # Create a blank mask of 0.0s
    block_mask = np.zeros_like(all_BEC)
    
    # Fill the diagonal blocks with 1.0s
    block_mask[:offset_Y, :offset_Y] = 1.0                           # X block
    block_mask[offset_Y:offset_Z, offset_Y:offset_Z] = 1.0           # Y block
    block_mask[offset_Z:, offset_Z:] = 1.0                           # Z block
    
    # Multiply to zero-out the cross-covariances!
    all_BEC = all_BEC * block_mask

    inf_var: float = 1e3

    # Extract single float variances safely
    std_X = float(data_dict["obs"]["X_std"])
    std_Y = float(data_dict["obs"]["Y_std"])
    std_Z = float(data_dict["obs"]["Z_std"])

    R_dict = {}

    # Build a dictionary of K matrices for all 7 possible asynchronous sensor firings
    for obs_X in [False, True]:
        for obs_Y in [False, True]:
            for obs_Z in [False, True]:
                if not (obs_X or obs_Y or obs_Z):
                    continue
                
                # Diagonal 1D Array
                r_diag = np.concatenate([
                    np.full(params.K, std_X**2 if obs_X else inf_var),
                    np.full(params.K*params.J, std_Y**2 if obs_Y else inf_var),
                    np.full(params.K*params.J*params.L, std_Z**2 if obs_Z else inf_var)
                ])
                
                R_mat = np.diag(r_diag)
                R_dict[(obs_X, obs_Y, obs_Z)] = R_mat

    # ------------------------------------------------
    # Prepare the inverse of BEC and OEC
    # ------------------------------------------------
    epsilon = np.max(np.diag(all_BEC)) * 1e-4
    B_inv: np.ndarray = np.linalg.inv(all_BEC + (np.eye(all_BEC.shape[0]) * epsilon))

    # ------------------------------------------------
    # Implement normal 3DVar
    # ------------------------------------------------
    def var3d_wrapper(state: l96.L96State, pad_obs: np.ndarray, flags: tuple[bool, bool, bool]) -> l96.L96State:
        # 1. Use the flags to fetch the correct R matrix (OEC)
        R_current = R_dict[flags]
        R_inv_diag: np.ndarray = 1.0 / np.diag(R_current)
        
        # 2. Pass everything into the 3D-Var solver in da_schemes
        return da.normal_3DVar(
            state=state, 
            BEC_inv=B_inv, 
            R_inv_diag=R_inv_diag, 
            observations=pad_obs, 
        )
    
    # form initial state
    init_state: l96.L96State = (
        data_dict["init"]["X"],
        data_dict["init"]["Y"],
        data_dict["init"]["Z"]
    )

    X_traj, Y_traj, Z_traj = da.forecast_analysis(
        init_state=init_state,
        target_layers=target_layers,
        obs_dict=data_dict["obs"],
        params=params,
        dt=dt,
        t_eval=t_eval,
        da_scheme=var3d_wrapper
    )

    # ------------------------------------------------
    # Calculate RMSE
    # ------------------------------------------------
    X_RMSE = np.sqrt(np.mean((X_traj - data_dict["nature"]["X"])**2, axis=(1,)))
    Y_RMSE = np.sqrt(np.mean((Y_traj - data_dict["nature"]["Y"])**2, axis=(1, 2)))
    Z_RMSE = np.sqrt(np.mean((Z_traj - data_dict["nature"]["Z"])**2, axis=(1, 2, 3)))


    # ------------------------------------------------
    # Save & Visualize
    # ------------------------------------------------
    prefix = "-".join(target_layers) # e.g., "X-Y-Z" or "X"
    
    np.savez(file_path / "traj" / f"3DVar_on_{prefix}_{bec_source}.npz", X=X_traj, Y=Y_traj, Z=Z_traj)
    np.savez(file_path / "rmse" / f"3DVar_on_{prefix}_{bec_source}.npz", X=X_RMSE, Y=Y_RMSE, Z=Z_RMSE)


# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":
    main(("X",))
    main(("Y",))
    main(("Z",))

    main(("X", "Y", "Z"))