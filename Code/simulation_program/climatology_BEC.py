# ====================================================
# This script is to calculate background error covariance
# from steady state of nature and free run.
# ====================================================

# ====================================================
# Import package
# ====================================================

import numpy as np

from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ====================================================
# Main function
# ====================================================

def main() -> None:
    
    # ------------------------------------------------
    # Load data
    # ------------------------------------------------
    data_path: Path = Path("/Users/joseph/Desktop/NTU Course/114-2/Data Assimilation/Project/Files")

    nature_run = np.load(data_path / "traj" / "nature_run.npz")
    free_run = np.load(data_path / "traj" / "free_run.npz")

    # ------------------------------------------------
    # Extract the last 3000 time step and calculate BEC
    # ------------------------------------------------

    # Calculate background error
    X_dev: np.ndarray = free_run["X"][-3000:] - nature_run["X"][-3000:]
    Y_dev: np.ndarray = free_run["Y"][-3000:] - nature_run["Y"][-3000:]
    Z_dev: np.ndarray = free_run["Z"][-3000:] - nature_run["Z"][-3000:]

    # flatten state
    X_dev_flat: np.ndarray = X_dev.reshape(3000, -1)
    Y_dev_flat: np.ndarray = Y_dev.reshape(3000, -1)
    Z_dev_flat: np.ndarray = Z_dev.reshape(3000, -1)


    # Caclualte BEC
    X_BEC: np.ndarray = np.cov(X_dev_flat.T)
    Y_BEC: np.ndarray = np.cov(Y_dev_flat.T)
    Z_BEC: np.ndarray = np.cov(Z_dev_flat.T)

    all_BEC: np.ndarray = np.cov(np.concatenate([X_dev_flat, Y_dev_flat, Z_dev_flat], axis=1).T)

    # -----------------------------------------------
    # Save BEC
    # -----------------------------------------------

    np.savez(
        data_path / "climatology_BEC.npz",
        X=X_BEC, Y=Y_BEC, Z=Z_BEC,
        all=all_BEC
        )

# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":
    main()