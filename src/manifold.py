import numpy as np
from mne.decoding import CSP

def extract_csp_manifold(session_path="data/human_eeg_session.npy", n_components=3):
    """
    Fits Common Spatial Patterns (CSP) to maximize variance contrast 
    between Execution and Imagery, then projects time-series data 
    into a 3D supervised state-space manifold.
    """
    data_dict = np.load(session_path, allow_pickle=True).item()
    
    exec_tensor = data_dict["Execution"]  # Shape: (n_trials, n_channels, n_timesteps)
    img_tensor = data_dict["Imagery"]    # Shape: (n_trials, n_channels, n_timesteps)
    
    n_exec, n_channels, n_times = exec_tensor.shape
    n_img = len(img_tensor)
    
    # 1. Combine condition tensors for CSP training
    X = np.concatenate([exec_tensor, img_tensor], axis=0)
    y = np.concatenate([np.zeros(n_exec), np.ones(n_img)])
    
    print("--- FITTING COMMON SPATIAL PATTERNS (CSP) MANIFOLD ---")
    print(f"Input Data Shape: {X.shape} (Trials x Channels x Timesteps)")
    
    # 2. Fit CSP filters
    csp = CSP(n_components=n_components, log=False, norm_trace=False)
    csp.fit(X, y)
    
    print(f"Extracted Supervised Spatial Filters: {n_components}\n")
    
    # 3. Project time-series into CSP space
    # Flatten across time to apply spatial filters to every sample point
    exec_flat = exec_tensor.transpose(0, 2, 1).reshape(-1, n_channels)
    img_flat = img_tensor.transpose(0, 2, 1).reshape(-1, n_channels)
    
    # Matrix multiplication with CSP spatial filters
    filters = csp.filters_[:n_components]
    exec_latent = np.dot(exec_flat, filters.T)
    img_latent = np.dot(img_flat, filters.T)
    
    # Reshape back to trial trajectory tensors: (n_trials, n_timesteps, n_components)
    exec_3d = exec_latent.reshape(n_exec, n_times, n_components)
    img_3d = img_latent.reshape(n_img, n_times, n_components)
    
    latent_dict = {
        "Execution": exec_3d,
        "Imagery": img_3d
    }
    
    np.save("data/human_eeg_manifold.npy", latent_dict, allow_pickle=True)
    print("Successfully saved CSP supervised trajectories to 'data/human_eeg_manifold.npy'.")
    return latent_dict

if __name__ == "__main__":
    extract_csp_manifold()