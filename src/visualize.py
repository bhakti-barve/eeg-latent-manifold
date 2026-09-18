import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_trajectories(manifold_path="data/human_eeg_manifold.npy"):
    latent_dict = np.load(manifold_path, allow_pickle=True).item()
    
    exec_tensor = latent_dict["Execution"]  # Shape: (n_trials, n_times, 3)
    img_tensor = latent_dict["Imagery"]    # Shape: (n_trials, n_times, 3)
    
    # Calculate trial-averaged trajectories
    mean_exec = np.mean(exec_tensor, axis=0)  # Shape: (n_times, 3)
    mean_img = np.mean(img_tensor, axis=0)    # Shape: (n_times, 3)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 3D trajectory lines
    ax.plot(mean_exec[:, 0], mean_exec[:, 1], mean_exec[:, 2], 
            label='Motor Execution', color='#1f77b4', linewidth=2.5)
    ax.plot(mean_img[:, 0], mean_img[:, 1], mean_img[:, 2], 
            label='Motor Imagery', color='#ff7f0e', linewidth=2.5)
    
    # Highlight start (t=0) and end points
    ax.scatter(mean_exec[0, 0], mean_exec[0, 1], mean_exec[0, 2], color='blue', s=60, marker='o', label='Exec Start')
    ax.scatter(mean_exec[-1, 0], mean_exec[-1, 1], mean_exec[-1, 2], color='darkblue', s=60, marker='^', label='Exec End')
    
    ax.scatter(mean_img[0, 0], mean_img[0, 1], mean_img[0, 2], color='orange', s=60, marker='o', label='Img Start')
    ax.scatter(mean_img[-1, 0], mean_img[-1, 1], mean_img[-1, 2], color='darkred', s=60, marker='^', label='Img End')
    
    ax.set_title("3D Latent EEG Trajectories: Motor Execution vs. Imagery (p < 0.001)", fontsize=12)
    ax.set_xlabel("CSP Component 1")
    ax.set_ylabel("CSP Component 2")
    ax.set_zlabel("CSP Component 3")
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig("results/figures/trajectory_manifold_3d.png", dpi=300)
    print("3D trajectory plot saved to 'results/figures/trajectory_manifold_3d.png'.")
    plt.show()

if __name__ == "__main__":
    plot_3d_trajectories()
    