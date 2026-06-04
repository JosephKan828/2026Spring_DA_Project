# ====================================================
# This script runs the nature run
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
from matplotlib.colors import TwoSlopeNorm

sys.path.append(os.path.join(os.getcwd(), "..", "src"))

import l96_model as l96   #type: ignore
import integrate as integ #type: ignore

# ====================================================
# Main function
# ====================================================

def main() -> None:
    # ------------------------------------------------
    # Setup parameter for simulations
    # ------------------------------------------------
    # run the model with default parameters (Nature run)
    params: l96.L96Params = l96.L96Params()

    # Temporal configuration
    # in this model, unit time corresponds to 5 days
    dt: float = 0.01  # Time step (0.01 units = 1.2 hours)

    t_spin: float = 2.0  # Spin-up time (2 units = 10 days)
    t_eval: float = 50.0  # Evaluation time (50 units = 250 days)

    # ------------------------------------------------
    # Initial condition
    # ------------------------------------------------

    # Randomly initialize the state
    rng: np.random.Generator = np.random.default_rng(seed=42)

    X_init: np.ndarray = rng.normal(loc=0.0, scale=0.1, size=params.K)
    Y_init: np.ndarray = rng.normal(loc=0.0, scale=0.1, size=(params.K, params.J))
    Z_init: np.ndarray = rng.normal(loc=0.0, scale=0.01, size=(params.K, params.J, params.L))

    # ------------------------------------------------
    # Spin-up the model to reach a realistic state
    # ------------------------------------------------

    spin_state: l96.L96State = (X_init, Y_init, Z_init)

    print("Spinning up the model to reach a realistic state...")
    
    for _ in tqdm(range(int(t_spin / dt))):
        spin_state = integ.rk4_step(
            state=spin_state,
            params=params,
            dt=dt,
            tendency_fn=l96.compute_tendencies
            )
    
    print("Spin-up complete. Starting nature run...")

    # ------------------------------------------------
    # Run the nature run
    # ------------------------------------------------

    # Pre-allocate arrays to store the nature run trajectory
    num_steps: int = int(t_eval / dt)
    X_traj: np.ndarray = np.zeros((num_steps, params.K))
    Y_traj: np.ndarray = np.zeros((num_steps, params.K, params.J))
    Z_traj: np.ndarray = np.zeros((num_steps, params.K, params.J, params.L))

    # Set initial state for the nature run
    current_state: l96.L96State = spin_state

    # Run the nature run and store the trajectory
    print("Running the nature run...")

    for i in tqdm(range(num_steps)):
        # Store current state in trajectory arrays
        X_traj[i] = current_state[0]
        Y_traj[i] = current_state[1]
        Z_traj[i] = current_state[2]

        # Integrate to the next time step
        current_state = integ.rk4_step(
            state=current_state,
            params=params,
            dt=dt,
            tendency_fn=l96.compute_tendencies
            )
        
    print("Nature run complete. Saving results...")

    # ------------------------------------------------
    # Saving files
    # ------------------------------------------------

    # path setup
    data_path: Path = Path("/Users/joseph/Desktop/NTU Course/114-2/Data Assimilation/Project/Files")

    # save nature run trajectory
    np.savez(data_path / "nature_run_trajectory.npz", X=X_traj, Y=Y_traj, Z=Z_traj)

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------

    # global configuration
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

    # path setup
    figure_path: Path = Path("/Users/joseph/Desktop/NTU Course/114-2/Data Assimilation/Project/Figure")

    # Time coordinate for plotting
    time: np.ndarray = np.arange(0, t_eval, dt)
    t_display: int = 2000

    # Visualize trajectory of X variables
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    x_ctf = axes[0, 0].contourf(
        time[:t_display], np.arange(1, params.K + 1),
        X_traj[:t_display].T, levels=np.linspace(-10, 10, 11),
        cmap="BrBG", extend="both"
    )
    axes[0, 0].set_xlim(0, t_display * dt)
    axes[0, 0].set_ylim(1, params.K)
    axes[0, 0].set_ylabel("K")
    axes[0, 0].set_title("X")

    fig.colorbar(x_ctf, ax=axes[0, 0])

    # Visualize trajectory of Y variables
    y_ctf = axes[0, 1].contourf(
        time[:t_display], np.arange(1, params.K*params.J + 1)/params.J,
        Y_traj[:t_display].reshape(t_display, params.K*params.J).T,
        levels=np.linspace(-1, 1, 11), cmap="BrBG", extend="both"
    )

    axes[0, 1].set_xlim(0, t_display * dt)
    axes[0, 1].set_ylim(1, params.K)
    axes[0, 1].set_title("Y")

    fig.colorbar(y_ctf, ax=axes[0, 1])

    # Visualize trajectory of Z variables
    z_ctf = axes[1, 0].contourf(
        time[:t_display], np.arange(1, params.K*params.J*params.L + 1)/(params.J*params.L),
        Z_traj[:t_display].reshape(t_display, params.K*params.J*params.L).T,
        levels=np.linspace(-0.1, 0.1, 11), cmap="BrBG", extend="both"
    )
    axes[1, 0].set_xlim(0, t_display * dt)
    axes[1, 0].set_ylim(1, params.K)
    axes[1, 0].set_xlabel("Time (units)")
    axes[1, 0].set_ylabel("K*J*L")
    axes[1, 0].set_title("Z")

    fig.colorbar(z_ctf, ax=axes[1, 0])

    axes[1, 1].axis("off")

    plt.suptitle("Nature Run Trajectory (First 2000 steps)", fontsize=18)

    plt.savefig(figure_path / "nature_run_trajectory.png", dpi=300, bbox_inches="tight")
    plt.close(fig)



# ====================================================
# Execute main function
# ====================================================
if __name__ == "__main__":
    main()