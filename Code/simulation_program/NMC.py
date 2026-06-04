# ====================================================
# Standalone Bootstrapped NMC Method
# Loads an existing OI Analysis Trajectory to generate
# a flow-dependent Background Error Covariance Matrix
# ====================================================

import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path

sys.path.append("/Users/joseph/Desktop/NTU Course/114-2/Data Assimilation/Project/Code/src")
import l96_model as l96   # type: ignore
import integrate as integ # type: ignore
import da_schemes as da   # type: ignore

def main(target_prefix: str = "X-Y-Z"):
    print(f"--- Bootstrapping NMC Matrix from {target_prefix} Analysis Trajectory ---")

    # ------------------------------------------------
    # Parameter and Path setting 
    # ------------------------------------------------
    params = l96.L96Params(h=0.9)
    dt = 0.01  
    
    root_path = Path("/Users/joseph/Desktop/NTU Course/114-2/Data Assimilation/Project/")
    file_path = root_path / "Files"
    
    # ------------------------------------------------
    # 1. Load the pre-computed OI Trajectory
    # ------------------------------------------------
    traj_file = file_path / "traj" / f"OI_on_{target_prefix}_climatology.npz"
    print(f"Loading analysis states from: {traj_file.name}")
    
    try:
        traj_data = np.load(traj_file)
        X_traj = traj_data["X"]
        Y_traj = traj_data["Y"]
        Z_traj = traj_data["Z"]
    except FileNotFoundError:
        print(f"Error: Could not find {traj_file}. Please run your OI script first.")
        return
       
    num_steps = X_traj.shape[0]

    # NMC Parameters
    steps_short = 20       # Short forecast (T + 0.05 MTU)
    steps_long = 50       # Long forecast (T + 0.10 MTU)
    decorr_steps = 30     # Sample every 30 steps for statistical independence
    # alpha = 0.1           # Variance tuning scalar
    differences = []

    # ------------------------------------------------
    # 2. Generate Forecast Pairs from Analysis States
    # ------------------------------------------------
    # Stop early enough to ensure we can integrate 10 steps forward
    for t_base in tqdm(range(0, num_steps - steps_long, decorr_steps), desc="NMC Bootstrapping"):
        
        # ==========================================
        # A. THE LONG FORECAST 
        # Launched from t_base, integrated 10 steps
        # ==========================================
        xa_long_init: l96.L96State = (
            X_traj[t_base].copy(),
            Y_traj[t_base].copy(),
            Z_traj[t_base].copy()
        )
        
        state_long = xa_long_init
        for _ in range(steps_long):
            state_long = integ.rk4_step(state_long, params, dt, l96.compute_tendencies)

        # ==========================================
        # B. THE SHORT FORECAST 
        # Launched from t_base + 5, integrated 5 steps
        # ==========================================
        t_short = t_base + decorr_steps
        
        xa_short_init: l96.L96State = (
            X_traj[t_short].copy(),
            Y_traj[t_short].copy(),
            Z_traj[t_short].copy()
        )
        
        state_short = xa_short_init
        for _ in range(steps_short):
            state_short = integ.rk4_step(state_short, params, dt, l96.compute_tendencies)

        # ------------------------------------------
        # Both state_long and state_short are now 
        # valid at exactly: t_base + 10
        # ------------------------------------------

        # Flatten the states to 1D vectors
        flat_short = np.concatenate([
            state_short[0].flatten(), 
            state_short[1].flatten(), 
            state_short[2].flatten()
        ])
        flat_long = np.concatenate([
            state_long[0].flatten(), 
            state_long[1].flatten(), 
            state_long[2].flatten()
        ])
        
        # Store the difference
        differences.append(flat_long - flat_short)

    # ------------------------------------------------
    # 3. Calculate Covariance Matrix & Mask
    # ------------------------------------------------
    print("Calculating bootstrapped covariance matrix...")
    diff_array = np.array(differences) 
    raw_nmc_cov = np.cov(diff_array, rowvar=False) 

    # Recreate the Block-Diagonal Localization Mask
    print("Applying localization mask...")
    offset_Y = params.K
    offset_Z = params.K + (params.K * params.J)

    block_mask = np.zeros_like(raw_nmc_cov)
    block_mask[:offset_Y, :offset_Y] = 1.0                           
    block_mask[offset_Y:offset_Z, offset_Y:offset_Z] = 1.0           
    block_mask[offset_Z:, offset_Z:] = 1.0                           
    
    # Apply tuning and mask
    final_NMC_BEC = (raw_nmc_cov) * block_mask

    # ------------------------------------------------
    # 4. Save the Final Matrix
    # ------------------------------------------------
    save_path = file_path / f"NMC_BEC.npz"
    np.savez(save_path, all=final_NMC_BEC)
    print(f"Success! Bootstrapped NMC BEC saved to {save_path.name}")

if __name__ == "__main__":
    # You can change this to match whichever experiment you ran in OI
    main(target_prefix="X-Y-Z")