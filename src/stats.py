import mne
import numpy as np
from mne.datasets import eegbci
from scipy.signal import hilbert

def load_and_preprocess_eeg(subject=1, runs=[3, 4], motor_only=False):
    """
    Loads PhysioNet EEG data with configurable runs and channels.
    Returns dictionary mapping condition -> 3D power envelope tensor (n_trials, n_channels, n_timesteps).
    """
    # 1. Fetch requested runs
    raw_fnames = eegbci.load_data(subject, runs)
    raws = [mne.io.read_raw_edf(f, preload=True) for f in raws] if isinstance(raw_fnames[0], str) else [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = mne.concatenate_raws(raws)

    # 2. Rename channels to standard 10-10 system
    eegbci.standardize_names(raw)
    
    # 3. Filter optional motor channels
    if motor_only:
        motor_channels = ['C3', 'C4', 'CZ', 'FC1', 'FC2', 'CP1', 'CP2']
        raw.pick_channels([ch for ch in motor_channels if ch in raw.ch_names])

    # 4. Bandpass filter to Mu/Beta motor band (8-30 Hz)
    raw.filter(l_freq=8.0, h_freq=30.0, fir_design='firwin')

    # 5. Extract events and epochs
    events, _ = mne.events_from_annotations(raw)
    
    # Map events to Execution (T1=2) and Imagery (T2=3)
    epochs = mne.Epochs(raw, events, event_id={'Execution': 2, 'Imagery': 3}, 
                        tmin=-0.5, tmax=3.0, baseline=(-0.5, 0), preload=True)

    # 6. Extract Hilbert power envelopes
    data_dict = {}
    for cond in ['Execution', 'Imagery']:
        raw_tensor = epochs[cond].get_data(units="uV")
        power_envelope = np.abs(hilbert(raw_tensor, axis=-1))
        data_dict[cond] = power_envelope
        
    return data_dict

def compute_mean_trajectories(latent_dict):
    """
    Computes trial-averaged trajectory array per condition.
    Returns dict mapping condition -> shape (n_timesteps, n_components)
    """
    mean_trajectories = {}
    for cond, tensor in latent_dict.items():
        mean_traj = np.mean(tensor, axis=0)
        mean_trajectories[cond] = mean_traj
    return mean_trajectories

def compute_trajectory_distance(traj1, traj2):
    """
    Calculates point-by-point Euclidean distance between two trajectories over time.
    """
    return np.linalg.norm(traj1 - traj2, axis=1)

def run_permutation_test(tensor_a, tensor_b, n_permutations=1000, seed=42):
    """
    Permutation test comparing spatial separation between two condition tensors.
    tensor_a, tensor_b shape: (n_trials, n_timesteps, n_components)
    """
    np.random.seed(seed)
    
    # 1. Observed statistic: Mean distance across all timepoints
    obs_a = np.mean(tensor_a, axis=0)
    obs_b = np.mean(tensor_b, axis=0)
    observed_stat = np.mean(compute_trajectory_distance(obs_a, obs_b))
    
    # Combine trial pools: (n_trials_a + n_trials_b, n_timesteps, n_components)
    combined_trials = np.vstack([tensor_a, tensor_b])
    n_a = len(tensor_a)
    n_total = len(combined_trials)
    
    null_distribution = []
    
    print(f"Running {n_permutations} Monte Carlo permutations...")
    for _ in range(n_permutations):
        # Shuffle trial indices
        shuffled_indices = np.random.permutation(n_total)
        
        # Split into dummy condition groups
        perm_a = combined_trials[shuffled_indices[:n_a]]
        perm_b = combined_trials[shuffled_indices[n_a:]]
        
        # Compute trajectory averages for permuted groups
        mean_perm_a = np.mean(perm_a, axis=0)
        mean_perm_b = np.mean(perm_b, axis=0)
        
        # Null statistic: Mean path distance under shuffled trial labels
        null_stat = np.mean(compute_trajectory_distance(mean_perm_a, mean_perm_b))
        null_distribution.append(null_stat)
        
    null_distribution = np.array(null_distribution)
    
    # Empirical p-value calculation
    p_value = np.sum(null_distribution >= observed_stat) / n_permutations
    
    print("\n--- NONPARAMETRIC PERMUTATION TEST RESULTS ---")
    print(f"Observed Trajectory Distance: {observed_stat:.4f} μV")
    print(f"Null Distribution Mean:       {np.mean(null_distribution):.4f} μV")
    print(f"Exact Empirical p-value:      p = {p_value:.4f}")
    
    if p_value < 0.05:
        print("Verdict: STATISTICALLY SIGNIFICANT. Execution & Imagery follow distinct trajectories.")
    else:
        print("Verdict: NOT SIGNIFICANT. Trajectory separation cannot be distinguished from noise.")
        
    return observed_stat, null_distribution, p_value

if __name__ == "__main__":
    latent_dict = np.load("data/human_eeg_manifold.npy", allow_pickle=True).item()
    
    exec_tensor = latent_dict["Execution"]
    img_tensor = latent_dict["Imagery"]
    
    run_permutation_test(exec_tensor, img_tensor, n_permutations=1000)