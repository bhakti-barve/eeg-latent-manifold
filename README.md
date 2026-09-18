# 3D Latent EEG Manifold Extraction: Motor Execution vs. Motor Imagery

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MNE-Python](https://img.shields.io/badge/MNE-EEG-purple.svg)](https://mne.tools/stable/index.html)

## What Does This Project Do?

Imagine your brain as a radio station. When you physically open and close your hand, your brain sends out a specific electrical signal. When you **only imagine** opening and closing your hand, your brain sends out a very similar signal—so similar that standard computer algorithms often mix them up.

This project takes complex, multi-channel brainwave recordings (EEG) and uses advanced mathematics to draw a **3D map of brain activity over time**. 

* **The Goal:** Prove that physically moving your hand vs. imagining moving your hand actually follow two distinct 3D pathways (trajectories) in the brain.
* **The Result:** We proved with 99.9%+ statistical confidence ($p = 0.0000$) that these two mental tasks follow distinct routes.
* **Why It Matters:** This helps engineers build better Brain-Computer Interfaces (BCIs) so paralyzed individuals can control prosthetic limbs or wheelchairs using thought alone.

---

## Visualizations

| **3D State-Space Trajectories** | **Time-Resolved Divergence Profile** |
| :---: | :---: |
| ![3D Trajectory](results/figures/trajectory_manifold_3d.png) | ![Latent Distance](results/figures/trajectory_distance_over_time.png) |
| *Trial-averaged 3D state-space trajectories ($t = -0.5\text{s}$ to $3.0\text{s}$).* | *Euclidean distance over time with 95% bootstrap confidence intervals.* |

---

## Abstract

Understanding the degree to which Motor Execution (ME) and Motor Imagery (MI) share underlying neural representations is critical for advancing brain-computer interfaces (BCIs) and neurorehabilitation. In this project, we extract continuous low-dimensional neural manifolds from multi-channel human EEG ($N = 45$ hand-motor trials per condition across Runs 3, 7, 11 and 4, 8, 12 of Subject S001). 

Preprocessed signals were bandpass filtered to the sensorimotor $\mu/\beta$ band ($8\text{--}30\text{ Hz}$), followed by analytic envelope extraction via the Hilbert transform. Supervised Common Spatial Patterns (CSP) spatial filtering projected trial dynamics into a 3D state space. A non-parametric Monte Carlo permutation test ($N=1000$) confirmed significant geometric divergence ($p = 0.0000$, observed distance $0.3207\,\mu\text{V}$ vs. null mean $0.1612\,\mu\text{V}$), providing empirical evidence of distinct subspace pathways during physical vs. imagined movement.

---

## Directory Structure

```text
.
├── config/
│   └── config.yaml                     # Centralized pipeline configurations
├── data/
│   ├── human_eeg_manifold.npy          # Processed 3D CSP latent tensors
│   └── human_eeg_session.npy           # Raw power envelope tensors
├── notebooks/
│   └── trajectory_mapping.ipynb        # Exploratory analysis & prototyping
├── results/
│   └── figures/                        # Exported high-resolution plots
│       ├── permutation_null_distribution.png
│       ├── state_space_trajectories.png
│       ├── trajectory_distance_over_time.png
│       ├── trajectory_divergence.png
│       └── trajectory_manifold_3d.png
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # PhysioNet ingestion & Hilbert filtering
│   ├── manifold.py                     # Supervised CSP 3D manifold extraction
│   ├── stats.py                        # Non-parametric permutation testing
│   ├── visualize.py                    # 3D trajectory manifold plot generation
│   └── visualize_distance.py           # Time-resolved Euclidean distance plot
├── .gitignore                          # Excludes heavy binaries & virtual environments
├── README.md                           # Project documentation
├── RESULTS.md                          # Detailed statistical findings & interpretations
└── requirements.txt                    # Project dependencies

## Methodological Pipeline 

┌────────────────────────┐    ┌────────────────────────┐    ┌────────────────────────┐
│   PhysioNet EEG Data   │    │ Signal Processing      │    │ Supervised Manifold    │
│  Subject S001 (Runs    │ ──>│  • Bandpass: 8-30 Hz   │ ──>│  • Common Spatial      │
│  3,4,7,8,11,12; N=45)  │    │  • Hilbert Envelopes   │    │    Patterns (CSP)      │
└────────────────────────┘    └────────────────────────┘    └────────────────────────┘
                                                                        │
┌────────────────────────┐    ┌────────────────────────┐                │
│ Scientific Conclusion  │    │ Statistical Validation │                │
│  • p = 0.0000          │ <──│  • Monte Carlo Permu-  │ <──────────────┘
│  • Distinct Pathways   │    │    tation Test (N=1000)│
└────────────────────────┘    └────────────────────────┘

## Installation & Environment Setup

# 1. Clone the repository
git clone [https://github.com/your-username/eeg-latent-manifold.git](https://github.com/your-username/eeg-latent-manifold.git)
cd eeg-latent-manifold

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt


## Execution Commands

# Step 1: Download & preprocess PhysioNet EEG data (Outputs data/human_eeg_session.npy)
python src/data_loader.py

# Step 2: Fit CSP filters & extract 3D state trajectories (Outputs data/human_eeg_manifold.npy)
python src/manifold.py

# Step 3: Run Monte Carlo permutation test (Outputs significance metrics to terminal)
python src/stats.py

# Step 4: Generate 3D trajectory plot (Saves to results/figures/trajectory_manifold_3d.png)
python src/visualize.py

# Step 5: Generate time-resolved distance plot (Saves to results/figures/trajectory_distance_over_time.png)
python src/visualize_distance.py


## Summary of Results

Parameter/ MetricEmpirical ValueDatasetPhysioNet EEG Motor Movement/Imagery (S001)Sample Size ($N$)$N = 45$ trials per condition (Hand Motor Tasks)Observed Trajectory Distance$0.3207\,\mu\text{V}$Null Distribution Mean$0.1612\,\mu\text{V}$Permutations ($N_{\text{perm}}$)$1000$ iterationsEmpirical $p$-value$p = 0.0000$ ($p < 0.001$)VerdictStatistically Significant Divergence