# ====================================================
# This script is to generate simulated observations
# from the nature run trajectory
# ====================================================

# ====================================================
# Import packages
# ====================================================

import os
import sys

import numpy as np

from tqdm import tqdm
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter


# ====================================================
# Main function
# ====================================================

def main() -> None:
    # ------------------------------------------------
    # Load nature run trajectory
    # ------------------------------------------------
    # path setup
    data_path: Path = Path("/Users/joseph/Desktop/NTU Course/114-2/Data Assimilation/Project/Files")

    # Load nature run trajectory
    nature_data = np.load(data_path / "nature_run_trajectory.npz")

    # ------------------------------------------------
    # Configure observation settings
    # ------------------------------------------------

    # Observation settings
    X_obs_interval: int = 5 # 5 time steps (0.05 units = 6 hours)
    Y_obs_interval: int = 3 # 3 time steps (0.03 units = 3.6 hours)
    Z_obs_interval: int = 2 # 2 time steps (0.02 units = 2.4 hours)

    # observation noise settings
    X_error_std: float = 0.1  # Standard deviation of observation error for X
    Y_error_std: float = 0.05  # Standard deviation of observation error for Y
    Z_error_std: float = 0.01 # Standard deviation of observation error for Z

    # ------------------------------------------------
    # Calculate simulated observations
    # ------------------------------------------------
    total_steps: int = nature_data["X"].shape[0]

    # observation time steps
    X_time_idx: np.ndarray = np.arange(0, total_steps, X_obs_interval)
    Y_time_idx: np.ndarray = np.arange(0, total_steps, Y_obs_interval)
    Z_time_idx: np.ndarray = np.arange(0, total_steps, Z_obs_interval)
    
    # preallocate arrays for observations
    X_obs: np.ndarray = nature_data["X"][X_time_idx] + np.random.normal(loc=0.0, scale=X_error_std, size=(len(X_time_idx), nature_data["X"].shape[1]))
    Y_obs: np.ndarray = nature_data["Y"][Y_time_idx] + np.random.normal(loc=0.0, scale=Y_error_std, size=(len(Y_time_idx), nature_data["Y"].shape[1], nature_data["Y"].shape[2]))
    Z_obs: np.ndarray = nature_data["Z"][Z_time_idx] + np.random.normal(loc=0.0, scale=Z_error_std, size=(len(Z_time_idx), nature_data["Z"].shape[1], nature_data["Z"].shape[2], nature_data["Z"].shape[3]))

    # ------------------------------------------------
    # Visualize error covariance
    # ------------------------------------------------
    # Data preparation (centering as in your snippet)
    x_data = X_obs[:, 0] - X_obs[:, 0].mean()
    y_data = X_obs[:, 1] - X_obs[:, 1].mean()

    fig = plt.figure(figsize=(9, 9))

    # GridSpec setup
    gs = plt.GridSpec(4, 4, wspace=0.1, hspace=0.1) #type: ignore

    # 1. Main Scatter Plot
    ax_scatter = fig.add_subplot(gs[1:, :-1])
    ax_scatter.scatter(x_data, y_data, s=20, color='b', edgecolors='k', marker='o')
    ax_scatter.axhline(0, 0, 1, color="k", linewidth=2, linestyle="--")
    ax_scatter.axvline(0, 0, 1, color="k", linewidth=2, linestyle="--")

    # 2. Top Marginal Histogram (X-axis)
    ax_hist_x = fig.add_subplot(gs[0, :-1], sharex=ax_scatter)
    ax_hist_x.hist(x_data, bins=20, color='b', edgecolor='k')

    # 3. Right Marginal Histogram (Y-axis)
    ax_hist_y = fig.add_subplot(gs[1:, -1], sharey=ax_scatter)
    ax_hist_y.hist(y_data, bins=20, orientation="horizontal", color='b', edgecolor='k')

    # --- ELIMINATE REDUNDANT INTERIOR FRAMES/LABELS ---

    # Remove X-axis labels from the top histogram
    ax_hist_x.xaxis.set_major_formatter(NullFormatter())

    # Remove Y-axis labels from the right histogram
    ax_hist_y.yaxis.set_major_formatter(NullFormatter())

    # Optional: Ensure ticks point inward to match the classic style
    for a in [ax_scatter, ax_hist_x, ax_hist_y]:
        a.tick_params(direction='in', top=True, right=True)

    # --- VISUAL REFINEMENTS (Labels & Title) ---

    # Use LaTeX for the labels as seen in image_b1fd14.jpg
    ax_scatter.set_xlabel(r'$\epsilon_{1,k}^{to}$', fontsize=20)
    ax_scatter.set_ylabel(r'$\epsilon_{2,k}^{to}$', fontsize=20)

    plt.savefig("/Users/joseph/Desktop/NTU Course/114-2/Data Assimilation/Project/Figure/Obs_error.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # ------------------------------------------------
    # Save simulated observations
    # ------------------------------------------------
    np.savez(
        data_path / "simulated_observations.npz",
        X=X_obs, Y=Y_obs, Z=Z_obs,
        X_idx=X_time_idx, Y_idx=Y_time_idx, Z_idx=Z_time_idx,
        X_std=X_error_std, Y_std=Y_error_std, Z_std=Z_error_std
        )
    print("Simulated observations saved successfully.")

# ====================================================
# Execution entry point
# ====================================================
if __name__ == "__main__":
    main()