# ====================================================
# This script implements optimal interpolation
# in a forecast-analysis cycle (Unified for 1-3 layers)
# ====================================================

import os
import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt

sys.path.append("/Users/joseph/Desktop/NTU Course/114-2/Data Assimilation/Project/Code/src")
import l96_model as l96   # type: ignore
import da_schemes as da   # type: ignore
import integrate as integ # type: ignore

# ====================================================
# Main Function
# ====================================================

def main(target_layers: tuple[str, ...]) -> None:
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

    # ================================================
    # Collect Data
    # ================================================
    bec_source: str = "NMC" # "climatology" or "NMC"

    data_dict: dict[str, dict[str, np.ndarray]] = {
        "init"  : dict(np.load(file_path / "free_run_initial_state.npz")),
        "nature": dict(np.load(file_path / "traj" / "nature_run.npz")),
        "obs"   : dict(np.load(file_path / "simulated_observations.npz")),
        "bec"   : dict(np.load(file_path / f"{bec_source}_BEC.npz"))
    }

    # ------------------------------------------------
    # Pre-compute ALL Kalman Gain Combinations
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

    K_dict = {}

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
                
                epsilon = np.max(np.diag(all_BEC)) * 1e-4
                BEC_reg = all_BEC + (np.eye(all_BEC.shape[0]) * epsilon)

                R_mat = np.diag(r_diag)
                K_dict[(obs_X, obs_Y, obs_Z)] = BEC_reg @ np.linalg.inv(BEC_reg + R_mat)

    # ------------------------------------------------
    # Run Forecast-Analysis Cycle
    # ------------------------------------------------
    def oi_wrapper(state: l96.L96State, pad_obs: np.ndarray, flags: tuple[bool, bool, bool]) -> l96.L96State:
        # 1. Use the flags from the loop to fetch the correct K matrix
        K_current = K_dict[flags]
        
        # 2. Pass it to the pure math solver in da_schemes
        return da.optimal_interpolation(state, K_current, pad_obs)
    
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
        da_scheme=oi_wrapper
    )

    # ------------------------------------------------
    # Save & Visualize
    # ------------------------------------------------
    prefix = "-".join(target_layers) # e.g., "X-Y-Z" or "X"

    os.makedirs(file_path / "traj" / "OI", exist_ok=True)
    os.makedirs(file_path / "traj" / "OI" / f"{bec_source}", exist_ok=True)

    np.savez(file_path / "traj" / "OI" / f"{bec_source}" / f"on_{prefix}.npz", X=X_traj, Y=Y_traj, Z=Z_traj)

if __name__ == "__main__":
    
    # Easily run your specific experiments here!
    
    # Single Layer Experiment:
    main(target_layers=("X",))
    main(target_layers=("Y",))
    main(target_layers=("Z",))

    # All Layer Experiment:
    main(target_layers=("X", "Y", "Z"))