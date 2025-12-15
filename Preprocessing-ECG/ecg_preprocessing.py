"""
ECG Preprocessing Pipeline for Trial-Locked Behavioral Analysis
Preprocesses ECG data, detects R-peaks, calculates HRV metrics, and generates intermediary plots.

Author: Kelly Hoogervorst and Siebe Everaerts
Version: 1.0
"""

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline, interp1d
from scipy.stats import median_abs_deviation
import neurokit2 as nk
import warnings
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import json
import sys

# Import coupled scripts
import config
import ecg_helpers

warnings.filterwarnings('ignore')

# ============================================================================
# initialise data structures
# ============================================================================
@dataclass
class ECGData:
    """Container for raw ECG data."""
    ecg_signal: np.ndarray
    sampling_frequency: int
    timestamps: np.ndarray

@dataclass
class PreprocessedECG:
    """Container for preprocessed ECG data."""
    ecg_filtered: np.ndarray
    r_peaks: np.ndarray
    rr_intervals: np.ndarray
    rr_times: np.ndarray
    artifacts_mask: np.ndarray
    rr_corrected: np.ndarray
    metadata: Dict


# ============================================================================
# Load data
# ============================================================================

def load_all_ecg_data():
    """Loop over all subjects in config.sIDs and load ECG + behavioural data."""
    results = {}

    for sID in config.sIDs:
        # Construct file paths
        ecg_filename = config.physio_filenames.format(sID=sID, task=config.task)
        behavioural_filename = config.behavioural_filenames.format(sID=sID, task=config.task)

        ecg_path = config.raw_dir / ecg_filename
        behavioural_path = config.raw_dir / behavioural_filename

        # Load ECG data
        if not ecg_path.exists():
            continue
        try:
            ecg_df = pd.read_csv(ecg_path, sep="\t", compression="gzip")
            # Assuming the ECG signal is in a column named "ECG_conv"
            ecg_signal = ecg_df["ECG_conv"].values
        except Exception:
            continue

        # Load behavioural data
        if not behavioural_path.exists():
            continue

        try:
            behavioural_data = pd.read_csv(behavioural_path)
        except Exception:
            continue

        # Create timestamps
        timestamps = np.arange(len(ecg_signal)) / config.sampling_freq

        ecg_data = ECGData(
            ecg_signal=ecg_signal,
            sampling_frequency=config.sampling_freq,
            timestamps=timestamps
        )

        # Store results in dictionary keyed by subject ID
        results[sID] = (ecg_data, behavioural_data)

    return results

# Plot raw ECG data (full duration + x amount of seconds)
if config.plot_raw_data:
    ecg_helpers.plot_raw(results, short_seconds=60)


# ============================================================================
# FILTERING
# ============================================================================

# Build filter using helper function
filters = ecg_helpers.design_filters_simple(config)

# Filter the data
def apply_filters_simple(ecg_signal: np.ndarray):
    """
    Apply filters in three explicit steps: high-pass -> low-pass -> notch.
    Returns the final filtered signal and a simple info dict.
    """
    # 1) High-pass
    ecg_hp = signal.sosfiltfilt(filters["highpass"]["sos"], ecg_signal)
    
    # 2) Low-pass
    ecg_lp = signal.sosfiltfilt(filters["lowpass"]["sos"], ecg_hp)
    
    # 3) Notch
    ecg_filtered = signal.sosfiltfilt(filters["notch"]["sos"], ecg_lp)
    
    # 4) Return filtered signal + parameters used
    info = {
        "sampling_freq": filters["fs"],
        "highpass_cutoff": filters["highpass"]["cutoff"],
        "highpass_order": filters["highpass"]["order"],
        "lowpass_cutoff": filters["lowpass"]["cutoff"],
        "lowpass_order": filters["lowpass"]["order"],
        "notch_frequency": filters["notch"]["freq"],
        "notch_q": filters["notch"]["q"],
    }
    
    return ecg_filtered, info

# Plot raw vs filtered comparison
if config.plot_filtered_data:
    ecg_helpers.plot_raw_vs_filtered(ecg_data, ecg_filtered, max_seconds=60)
    

# ============================================================================
# Downsampling
# ============================================================================

# Downsample the ECG signal.
"""
Parameters
----------
ecg_filtered : np.ndarray
    filtered ECG signal (mV)
    
Returns
-------
ecg_downsampled : np.ndarray
    Downsampled ECG signal
downsample_info : Dict
    Information about downsampling procedure
"""
# Retrieve parameters from config
fs = config.sampling_freq
target_fs = config.downsampling_freq
method = config.downsampling_method

# Apply downsampling using helper function
ecg_downsampled, downsample_info = ecg_helpers.apply_downsampling(
    signal_data=ecg_filtered,
    original_fs=fs,
    target_fs=target_fs,
    method=method
)
    

# Plot filtered vs downsampled
if config.plot_downsampled_data:
    ecg_helpers.plot_filtered_vs_downsampled(ecg_data, ecg_filtered, ecg_downsampled, max_seconds=60)
    

# ============================================================================
# QRS detection 
# ============================================================================

# early peak detection
early_segment = ecg_downsampled[:500]
early_peaks, _ = find_peaks(
    early_segment,
    distance=int(0.2 * config.downsampling_freq),  # ~200ms min RR at downsampled rate (adjust if needed)
    height=np.mean(early_segment[:100]) + 4 * np.std(early_segment[:100])
)

# R-peak detection with built-in artifact correction
signals, info = nk.ecg_peaks(
    ecg_downsampled,
    sampling_rate= config.downsampling_freq,
    correct_artifacts=True  # uses Kubios-style correction
)
# R-peak sample indices
rpeaks = info["ECG_R_Peaks"]

if len(early_peaks) > 0:
    # Avoid duplicates: remove early_peaks already detected by NK
    early_new = early_peaks[~np.isin(early_peaks, rpeaks)]
    
    if len(early_new) > 0:
        # Sort and insert at beginning
        early_new = np.sort(early_new)
        rpeaks = np.insert(rpeaks, 0, early_new)
        print(f"Added {len(early_new)} early peaks from first 500 samples")

# Final R-peak indices (with early enhancement)
rpeaks_final = np.unique(rpeaks)  # remove any theoretical duplicates
print(f"Total R-peaks detected: {len(rpeaks_final)}")

# Optional: quick diagnostic plot
if config.plot_rpeaks:
    ecg_helpers.plot_rpeaks_segment(ecg_downsampled, rpeaks_final, config.downsampling_freq, duration_sec = 30.0)

# ============================================================================
# Artifact correction 
# ============================================================================




# ============================================================================
# Epoching 
# ============================================================================

# load relevant data for function
if config.epoch_method == 'stamps':
   trial_timing_data = behavioural_data['TrialStart'].to_numpy()
# ADD FOR OTHER METHODS

# Epoch ECG signal with the helper functions
result = ecg_helpers.detect_trial_timing(config, trial_timing_data)
trial_samples = result['samples']


# ============================================================================
# Time-domain analysis 
# ============================================================================




