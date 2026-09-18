import numpy as np
import mne
from mne.datasets import eegbci
from scipy.signal import hilbert

def fetch_and_process_hand_runs(subject=1):
    """
    Loads all hand-movement runs for Subject 1:
    Runs 3, 7, 11 = Hand Execution
    Runs 4, 8, 12 = Hand Imagery
    """
    # 1. Fetch Execution and Imagery runs separately
    exec_runs = [3, 7, 11]
    img_runs = [4, 8, 12]
    
    def process_run_set(runs):
        files = eegbci.load_data(subject, runs, update_path=True)
        raws = [mne.io.read_raw_edf(f, preload=True) for f in files]
        raw = mne.io.concatenate_raws(raws)
        
        mapping = {ch: ch.rstrip('.').upper() for ch in raw.ch_names}
        raw.rename_channels(mapping)
        
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")
        
        # Filter out non-motor brain waves. We only keep 8-30 Hz signals (Mu & Beta bands)
        # because these specifically change when a person moves or imagines moving their hands.
        raw.filter(l_freq=8.0, h_freq=30.0)
        
        events, event_id = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events, tmin=-0.5, tmax=3.0, baseline=(-0.5, 0), preload=True)
        
        # Get μV data and compute Hilbert envelope
        raw_tensor = epochs.get_data(units="uV")
        power_envelope = np.abs(hilbert(raw_tensor, axis=-1))
        return power_envelope

    print("Processing Execution runs (3, 7, 11)...")
    exec_tensor = process_run_set(exec_runs)
    
    print("Processing Imagery runs (4, 8, 12)...")
    img_tensor = process_run_set(img_runs)
    
    data_dict = {
        "Execution": exec_tensor,
        "Imagery": img_tensor
    }
    
    np.save("data/human_eeg_session.npy", data_dict, allow_pickle=True)
    print(f"\nSaved expanded dataset! Execution: {exec_tensor.shape}, Imagery: {img_tensor.shape}")

if __name__ == "__main__":
    fetch_and_process_hand_runs()