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

    nature_run = np.load(data_path / "nature_run_trajectory.npz")
    free_run = np.load(data_path / "free_run_trajectory.npz")

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

    # ------------------------------------------------    
    # Visualize error covariance matrix
    # ------------------------------------------------

    # Figure setting
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "figure.dpi": 300,
        "axes.grid": False,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.constrained_layout.use": True
    })

    # figure path
    fig_path: Path = Path("/Users/joseph/Desktop/NTU Course/114-2/Data Assimilation/Project/Figure")

    fig, axes = plt.subplots(2, 2, figsize=(13, 13))

    X_pcm = axes[0, 0].pcolormesh(
        np.arange(1, X_BEC.shape[0]+1),
        np.arange(1, X_BEC.shape[1]+1),
        X_BEC, cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0)
    )

    fig.colorbar(X_pcm, ax=axes[0, 0])

    Y_pcm = axes[0, 1].pcolormesh(
        np.arange(1, Y_BEC.shape[0]+1),
        np.arange(1, Y_BEC.shape[1]+1),
        Y_BEC, cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0)
    )

    fig.colorbar(Y_pcm, ax=axes[0, 1])

    Z_pcm = axes[1, 0].pcolormesh(
        np.arange(1, Z_BEC.shape[0]+1),
        np.arange(1, Z_BEC.shape[1]+1),
        Z_BEC, cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0)
    )

    fig.colorbar(Z_pcm, ax=axes[1, 0])

    all_pcm = axes[1, 1].pcolormesh(
        np.arange(1, all_BEC.shape[0]+1),
        np.arange(1, all_BEC.shape[1]+1),
        all_BEC, cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
    )

    fig.colorbar(all_pcm, ax=axes[1, 1])

    plt.savefig(fig_path / "static_BEC.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":
    main()