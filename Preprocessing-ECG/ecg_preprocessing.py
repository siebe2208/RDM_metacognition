"""
ECG Preprocessing Pipeline for Trial-Locked Behavioral Analysis
Preprocesses ECG data, detects R-peaks, calculates HRV metrics, and generates intermediary plots.

Author: Kelly Hoogervorst and Siebe Everaerts
Version: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal
from scipy.interpolate import CubicSpline, interp1d
from scipy.stats import median_abs_deviation
import logging
import warnings
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import json

# Import configuration
import config

warnings.filterwarnings('ignore')

# ============================================================================
# Logging setup
# ============================================================================

def setup_logging(sID: int):
    """Setup logging for the preprocessing pipeline."""
    log_file = config.logs_dir / f"sub-{sID:04d}_preprocessing.log"
    
    logging.basicConfig(
        level=getattr(logging, config.log.log_level), #possible mistake
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file) if config.save_logs else logging.NullHandler(),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

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

def load_ecg_data(participant_id: int, logger) -> Tuple[ECGData, pd.DataFrame]:
    """
    Load ECG and behavioral data for a participant.
    
    Parameters
    ----------
    participant_id : int
        Participant ID (e.g., 263)
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    ecg_data : ECGData
        Raw ECG data
    behavioral_data : pd.DataFrame
        Behavioral data
    """
    # Construct file paths
    ecg_filename = config.physio_filenames.format(participant_id=participant_id)
    behavioral_filename = config.behavioral_filenames.format(participant_id=participant_id)
    
    ecg_path = config.raw_dir / ecg_filename
    behavioral_path = config.raw_dir / behavioral_filename
    
    # Load ECG data
    if not ecg_path.exists():
        raise FileNotFoundError(f"ECG file not found: {ecg_path}") # warning if path does not exist
    
    try:
        ecg_data_dict = np.load(ecg_path, allow_pickle=True)
        ecg_signal = ecg_data_dict['ECG_conv']  # Converted ECG data in mV
        logger.info(f"Loaded ECG data: {len(ecg_signal)} samples")
    except Exception as e:
        logger.error(f"Failed to load ECG data: {e}")
        raise
    
    # Load behavioral data
    if not behavioral_path.exists():
        raise FileNotFoundError(f"Behavioral file not found: {behavioral_path}")
    
    try:
        behavioral_data = pd.read_csv(behavioral_path)
        logger.info(f"Loaded behavioral data: {len(behavioral_data)} trials")
    except Exception as e:
        logger.error(f"Failed to load behavioral data: {e}")
        raise
    
    # Create timestamps
    timestamps = np.arange(len(ecg_signal)) / config.sampling_rate
    
    ecg_data = ECGData(
        ecg_signal=ecg_signal,
        sampling_frequency=config.sampling_rate,
        timestamps=timestamps
    )
    
    return ecg_data, behavioral_data


# ============================================================================
# FILTERING
# ============================================================================

def design_filters() -> Tuple:
    """
    Design filtering cascade: high-pass, low-pass, and notch filters.
    
    Returns
    -------
    sos_hp, sos_lp, sos_notch : scipy.signal filter objects
        Second-order sections for cascade filtering
    """
    fs = config.SAMPLING_FREQUENCY
    
    # High-pass filter design
    sos_hp = signal.butter(
        config.HIGHPASS_ORDER,
        config.HIGHPASS_CUTOFF,
        btype='high',
        fs=fs,
        output='sos'
    )
    
    # Low-pass filter design
    sos_lp = signal.butter(
        config.LOWPASS_ORDER,
        config.LOWPASS_CUTOFF,
        btype='low',
        fs=fs,
        output='sos'
    )
    
    # Notch filter design for power-line interference
    sos_notch = signal.iirnotch(
        config.NOTCH_FREQUENCY,
        Q=config.NOTCH_Q,
        fs=fs,
        output='sos'
    )
    
    return sos_hp, sos_lp, sos_notch

def apply_filters(ecg_signal: np.ndarray, logger) -> Tuple[np.ndarray, Dict]:
    """
    Apply filtering cascade to ECG signal.
    
    Parameters
    ----------
    ecg_signal : np.ndarray
        Raw ECG signal (mV)
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    ecg_filtered : np.ndarray
        Filtered ECG signal
    filter_info : Dict
        Information about applied filters
    """
    logger.info("Applying filter cascade...")
    
    sos_hp, sos_lp, sos_notch = design_filters()
    
    # Apply high-pass filter
    ecg_hp = signal.sosfiltfilt(sos_hp, ecg_signal)
    
    # Apply low-pass filter
    ecg_lp = signal.sosfiltfilt(sos_lp, ecg_hp)
    
    # Apply notch filter
    ecg_filtered = signal.sosfiltfilt(sos_notch, ecg_lp)
    
    filter_info = {
        'highpass_cutoff': config.HIGHPASS_CUTOFF,
        'lowpass_cutoff': config.LOWPASS_CUTOFF,
        'notch_frequency': config.NOTCH_FREQUENCY,
        'notch_q': config.NOTCH_Q,
    }
    
    logger.info("Filter cascade applied successfully")
    
    return ecg_filtered, filter_info

# ============================================================================
# DOWNSAMPLING
# ============================================================================

def downsample(ecg_filtered: np.ndarray, logger) -> Tuple[np.ndarray, Dict]:
    """
    Downsample the ECG signal .
    
    Parameters
    ----------
    ecg_filtered : np.ndarray
        filtered ECG signal (mV)
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    ecg_downsampled : np.ndarray
        Downsampled ECG signal
    downsample_info : Dict
        Information about downsampling procedure
    """

    logger.info("Downsampling started...")

    # Retrieve original and target sampling frequencies from config 
    fs = config.SAMPLING_FREQUENCY           # Original sampling frequency (Hz)                 
    target_fs = config.DOWNSAMPLING_FREQUENCY  # Desired downsampled frequency (Hz)

    # Compute the downsampling factor
    factor = fs / target_fs
    
    # Retrieve the chosen downsampling method from configuration
    method = config.DOWNSAMPLING_METHOD
    
    # Check that the target sampling frequency is lower than the original
    if target_fs >= fs:
        raise ValueError("target sampling frequency must be strictly lower than current sampling frequency for downsampling.")
    
    # Apply downsampling based on chosen method
    if method == "poly":
        ecg_downsampled = signal.resample_poly(ecg_filtered, up = target_fs, down = fs)
    
    elif method == "decimate":
        factor_int = int(round(factor))
        ecg_downsampled = signal.decimate(ecg_filtered, factor_int, ftype='iir', zero_phase=True)
    
    elif method == "resample":
        num_samples = int(len(ecg_filtered) * target_fs / fs)
        ecg_downsampled = signal.resample(ecg_filtered, num_samples)

    else:
        # Raise error if an unknown method is specified
        raise ValueError(f"Unknown method '{method}'. Choose from ['poly', 'decimate', 'resample']")
    
    # Create dictionary with downsampling info
    downsample_info = {"downsampling_method": method,
                       "old_sampling_frequency": fs,
                       "new_sampling_frequency": target_fs,
                       "downsampled_by_factor": factor}
    
    logger.info("Downsampling applied succesfully")

    return ecg_downsampled, downsample_info

    
