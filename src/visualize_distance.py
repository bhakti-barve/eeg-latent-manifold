import numpy as np
import matplotlib.pyplot as plt

def plot_time_resolved_distance(manifold_path="data/human_eeg_manifold.npy", 
                                n_bootstraps=1000, 
                                seed=42):
    np.random.seed(seed)
    latent_dict = np.load(manifold_path, allow_pickle=True).item()
    
    exec_tensor = latent_dict["Execution"]  # (n_trials, n_times, 3)
    img_tensor = latent_dict["Imagery"]    # (n_trials, n_times, 3)
    
    n_exec, n_times, _ = exec_tensor.shape
    n_img = len(img_tensor)
    
    # Define time vector (-0.5s to 3.0s across time steps)
    time_vec = np.linspace(-0.5, 3.0, n_times)
    
    # 1. Observed mean distance profile over time
    mean_exec = np.mean(exec_tensor, axis=0)
    mean_img = np.mean(img_tensor, axis=0)
    observed_dist = np.linalg.norm(mean_exec - mean_img, axis=1)
    
    # 2. Bootstrap 95% Confidence Interval across trials
    boot_distances = np.zeros((n_bootstraps, n_times))
    for i in range(n_bootstraps):
        boot_exec_idx = np.random.choice(n_exec, size=n_exec, replace=True)
        boot_img_idx = np.random.choice(n_img, size=n_img, replace=True)
        
        boot_exec_mean = np.mean(exec_tensor[boot_exec_idx], axis=0)
        boot_img_mean = np.mean(img_tensor[boot_img_idx], axis=0)
        
        boot_distances[i, :] = np.linalg.norm(boot_exec_mean - boot_img_mean, axis=1)
        
    ci_lower = np.percentile(boot_distances, 2.5, axis=0)
    ci_upper = np.percentile(boot_distances, 97.5, axis=0)
    
    # 3. Plot time-series distance profile
    plt.figure(figsize=(8, 4.5), dpi=300)
    plt.plot(time_vec, observed_dist, color='#1f77b4', linewidth=2.5, label='Mean Latent Distance')
    plt.fill_between(time_vec, ci_lower, ci_upper, color='#1f77b4', alpha=0.25, label='95% Bootstrap CI')
    
    plt.axvline(x=0.0, color='red', linestyle='--', linewidth=1.5, label='Cue Onset (t = 0s)')
    
    plt.title("Time-Resolved Latent Distance: Motor Execution vs. Motor Imagery", fontsize=12, fontweight='bold')
    plt.xlabel("Time Relative to Cue Onset (seconds)", fontsize=10)
    plt.ylabel("Euclidean Distance (μV)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig("results/figures/trajectory_distance_over_time.png")
    print("Time-resolved distance plot successfully saved to 'results/figures/trajectory_distance_over_time.png'.")
    plt.show()

if __name__ == "__main__":
    plot_time_resolved_distance()