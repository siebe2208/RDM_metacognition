"""
ECG Preprocessing Helper Functions

This module contains utility functions for filter design and visualisation.
To be used with ecg_preprocessing.py

Authors: Kelly Hoogervorst and Siebe Everaerts
Version: 1.0
"""


import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import neurokit2 as nk



# ============================================================================
# Filter design and parameters
# ============================================================================

def notch_sos(notch_freq, notch_q, fs):
    """
    Return notch filter in SOS form, compatible with both old and new SciPy.
    
    Parameters
    ----------
    notch_freq : float
        Frequency to notch out (Hz), typically 50 or 60 Hz for power-line noise.
    notch_q : float
        Quality factor (higher = narrower notch).
    fs : float
        Sampling frequency (Hz).
    
    Returns
    -------
    sos : ndarray
        Second-order sections representation of the notch filter.
    """
    try:
        # Newer SciPy: supports fs= and returns (b, a); no 'output' kwarg
        b, a = signal.iirnotch(notch_freq, Q=notch_q, fs=fs)
    except TypeError:
        # Older SciPy: expects normalized frequency (0..1) wrt Nyquist
        w0 = notch_freq / (fs / 2.0)
        b, a = signal.iirnotch(w0, Q=notch_q)
    
    sos = signal.tf2sos(b, a)
    return sos


def design_filters_simple(config):
    """
    Build high-pass, low-pass, and notch filters using values from config.
    
    Parameters
    ----------
    config : object
        Configuration object with attributes:
        - sampling_freq
        - hpass_order, hpass_cutoff
        - lpass_order, lpass_cutoff
        - notch_freq, notch_q
    
    Returns
    -------
    dict
        Dictionary containing filter specifications and SOS coefficients:
        {
            "fs": sampling_freq,
            "highpass": {"order": int, "cutoff": float, "sos": ndarray},
            "lowpass": {"order": int, "cutoff": float, "sos": ndarray},
            "notch": {"freq": float, "q": float, "sos": ndarray}
        }
           """
           
           # Read sampling frequency and parameters from config
    fs = config.sampling_freq
    
    hp_order = config.hpass_order
    hp_cutoff = config.hpass_cutoff
    
    lp_order = config.lpass_order
    lp_cutoff = config.lpass_cutoff
    
    notch_freq = config.notch_freq
    notch_q = config.notch_q
    
    # Design high-pass filter (Butterworth, SOS form)
    sos_hp = signal.butter(hp_order, hp_cutoff, btype="high", fs=fs, output="sos")
    
    # Design low-pass filter (Butterworth, SOS form)
    sos_lp = signal.butter(lp_order, lp_cutoff, btype="low", fs=fs, output="sos")
    
    # Design notch filter (power-line interference)
    sos_notch = notch_sos(notch_freq, notch_q, fs)
    
    # Return everything clearly
    return {
        "fs": fs,
        "highpass": {"order": hp_order, "cutoff": hp_cutoff, "sos": sos_hp},
        "lowpass": {"order": lp_order, "cutoff": lp_cutoff, "sos": sos_lp},
        "notch": {"freq": notch_freq, "q": notch_q, "sos": sos_notch},
    }


# ============================================================================
# Downsampling
# ============================================================================

def apply_downsampling(
    signal_data: np.ndarray,
    original_fs: float,
    target_fs: float,
    method: str = "poly"
):
    """
    Apply downsampling to a signal using the specified method.
    
    Parameters
    ----------
    signal_data : np.ndarray
        Input signal to downsample
    original_fs : float
        Original sampling frequency (Hz)
    target_fs : float
        Target sampling frequency (Hz)
    method : str
        Downsampling method: 'poly', 'decimate', or 'resample'
        
    Returns
    -------
    downsampled_signal : np.ndarray
        Downsampled signal
    downsample_info : Dict
        Information about downsampling procedure
        
    Raises
    ------
    ValueError
        If target_fs >= original_fs or method is unknown
    """
    from scipy import signal
    
    # Validate target frequency
    if target_fs >= original_fs:
        raise ValueError(
            "target sampling frequency must be strictly lower than "
            "current sampling frequency for downsampling."
        )
    
    # Compute downsampling factor
    factor = original_fs / target_fs
    
    # Apply downsampling based on method
    if method == "poly":
        downsampled_signal = signal.resample_poly(
            signal_data, up=target_fs, down=original_fs
        )
    
    elif method == "decimate":
        factor_int = int(round(factor))
        downsampled_signal = signal.decimate(
            signal_data, factor_int, ftype='iir', zero_phase=True
        )
    
    elif method == "resample":
        num_samples = int(len(signal_data) * target_fs / original_fs)
        downsampled_signal = signal.resample(signal_data, num_samples)
    
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Choose from ['poly', 'decimate', 'resample']"
        )
    
    # Create info dictionary
    downsample_info = {
        "downsampling_method": method,
        "old_sampling_frequency": original_fs,
        "new_sampling_frequency": target_fs,
        "downsampled_by_factor": factor
    }
    
    return downsampled_signal, downsample_info


# ============================================================================
# Trial timing detection
# ============================================================================

def detect_trial_timing(config, data):
    """
    Detect trial timing using method specified in config.
    
    Parameters
    ----------
    config : Configuration object with attributes:
        - epoch_method: str, one of ['stamps', 'PD', 'PD_validate']
        - downsampling_freq: float, sampling rate in Hz after downsampling
    stamps : array-like
        Trial onset timestamps in seconds. Required for 'stamps' and 'validate' methods.
    photodiode_data : array-like
        Raw photodiode signal. Required for 'PD' and 'PD_validate' methods.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'samples': array of trial onset sample indices
        - 'method': str, method used
        - 'photodiode_samples': array of PD peaks (only if method='validate')
        - 'validation': correlation coefficient (only if method='validate')
        - 'sampling_freq': float, sampling frequency used
    
    Raises
    ------
    ValueError
        If required data is missing for the specified method.
    """
    
    # Settings and check
    method = config.epoch_method
    SR = config.downsampling_freq
    
    valid_methods = ['stamps', 'photodiode', 'validate']
    if method not in valid_methods:
        raise ValueError(f"config.epoch_method must be one of {valid_methods}, got '{method}'")
    
    # Method 1: Timestamps
    if method == 'stamps':
        if data is None:
            raise ValueError("stamps data required when config.epoch_method='stamps'")
        
        timing_samples = _stamps_to_samples(data, SR)
        
        return {
            'samples': timing_samples,
            'method': 'stamps',
            'sampling_freq': SR
        }
    
    # Method 2: Use photodiode only
    elif method == 'photodiode':
        if photodiode_data is None:
            raise ValueError("photodiode_data required when config.epoch_method='photodiode'")
        
        pd_peaks = _detect_photodiode_peaks(photodiode_data, SR)
        
        return {
            'samples': pd_peaks,
            'method': 'photodiode',
            'sampling_freq': SR
        }
    
    # Method 3: Use stamps but validate with photodiode
    elif method == 'validate':
        if stamps is None or photodiode_data is None:
            raise ValueError("Both stamps and photodiode_data required when config.epoch_method='validate'")
        
        timing_samples = _stamps_to_samples(stamps, SR)
        pd_peaks = _detect_photodiode_peaks(photodiode_data, SR)
        correlation = _validate_timing(pd_peaks, timing_samples)
        
        print(f"Validation correlation: {correlation:.3f}")
        if correlation < 0.95:
            print("WARNING: Low correlation between stamps and photodiode. "
                  "Check your data alignment.")
        
        return {
            'samples': timing_samples,
            'photodiode_samples': pd_peaks,
            'validation': correlation,
            'method': 'validate',
            'sampling_freq': SR
        }


def _stamps_to_samples(stamps, SR):
    """
    Convert timestamps in seconds to sample indices.
    
    Parameters
    ----------
    stamps : array-like
        Timestamps in seconds
    SR : float, optional
        Sampling rate in Hz. If None, assumes stamps are already in samples.
    
    Returns
    -------
    np.ndarray
        Sample indices
    """
    stamps = np.asarray(stamps)
    
    if SR is not None:
        # Convert seconds to samples
        samples = np.round(stamps * SR).astype(int)
    else:
        # Assume stamps are already in samples
        samples = np.asarray(stamps, dtype=int)
    
    return samples


def _detect_photodiode_peaks(photodiode_data, SR, skip=None):
    """
    Detect peaks in photodiode signal.
    
    Parameters
    ----------
    photodiode_data : array-like
        Raw photodiode signal
    SR : float
        Sampling rate in Hz
    skip : list of int, optional
        Indices of peaks to exclude
    
    Returns
    -------
    np.ndarray
        Array of peak sample indices
    """
    photodiode_data = np.asarray(photodiode_data)
    
    # Normalize signal for robust peak detection
    z_scored = (photodiode_data - np.mean(photodiode_data)) / np.std(photodiode_data)
    
    # Detect peaks
    # Minimum distance between peaks: 100ms (assuming trials don't occur faster)
    min_distance = int(SR * 0.1)
    threshold = np.max(z_scored) - 1
    
    peaks, _ = find_peaks(z_scored, distance=min_distance, height=threshold)
    
    # Remove specified peaks
    if skip is not None:
        skip = np.asarray(skip)
        mask = np.ones(len(peaks), dtype=bool)
        mask[skip] = False
        peaks = peaks[mask]
        print(f"Removed {len(skip)} peaks. {len(peaks)} peaks remaining.")
    
    return peaks


def _validate_timing(pd_peaks, stamp_samples):
    """
    Validate photodiode detection against stamps by comparing inter-trial intervals.
    
    Parameters
    ----------
    pd_peaks : np.ndarray
        Photodiode peak sample indices
    stamp_samples : np.ndarray
        Stamp-based sample indices
    
    Returns
    -------
    float
        Correlation coefficient between inter-trial intervals
    """
    # Calculate inter-trial intervals
    pd_intervals = np.diff(pd_peaks)
    stamp_intervals = np.diff(stamp_samples)
    
    # Handle length mismatch
    min_len = min(len(pd_intervals), len(stamp_intervals))
    if len(pd_intervals) != len(stamp_intervals):
        print(f"WARNING: Number of trials differs. PD: {len(pd_peaks)}, "
              f"Stamps: {len(stamp_samples)}. Using first {min_len} intervals.")
        pd_intervals = pd_intervals[:min_len]
        stamp_intervals = stamp_intervals[:min_len]
    
    # Calculate correlation
    if len(pd_intervals) > 1:
        correlation = np.corrcoef(pd_intervals, stamp_intervals)[0, 1]
    else:
        print("WARNING: Not enough trials to calculate correlation")
        correlation = np.nan
    
    return correlation


# ============================================================================
# Visualisation
# ============================================================================

# Raw data ====================================================================
def plot_raw(results, short_seconds: int = 90):
    """
    For each subject, plot the raw ECG signal in two stacked panels:
    - Top: entire duration
    - Bottom: first `short_seconds` (default: 90 seconds)
    If behavioral_data has 'TrialStart' (in seconds), draw vertical onset lines in both panels.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping subject IDs to (ecg_data, behavioral_data) tuples.
    short_seconds : int, optional
        Duration in seconds for the bottom panel (default: 90).
    """
    if not results:
        raise ValueError("No results to plot.")
    
    plt.style.use('seaborn-v0_8')
    
    # colour palette
    rose = "#CD6574"      # lighter rose/pink
    dark_red = "#8b0000"  # dark red

    
    for sID, (ecg_data, behavioral_data) in results.items():
        t_full = ecg_data.timestamps
        x_full = ecg_data.ecg_signal
        
        
        # Convert top-panel time to minutes
        t_full_min = t_full / 60.0

        # Mask for short window
        short_mask = t_full <= short_seconds
        t_short = t_full[short_mask]
        x_short = x_full[short_mask]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
        fig.suptitle(f"Subject {sID} - Raw ECG", fontsize=14)
        
        # --- Top: entire duration ---
        ax_top = axes[0]
        ax_top.plot(t_full_min, x_full, color=rose, linewidth=0.9)
        ax_top.set_title("Entire duration")
        ax_top.set_xlabel("Time (min)")
        ax_top.set_ylabel("ECG (mV)")
        ax_top.grid(True, alpha=0.3)

        
        # --- Bottom: first short_seconds ---
        ax_bottom = axes[1]
        ax_bottom.plot(t_short, x_short, color=dark_red, linewidth=0.8)
        ax_bottom.set_title(f"First {short_seconds} seconds")
        ax_bottom.set_xlabel("Time (s)")
        ax_bottom.set_ylabel("ECG (mV)")
        ax_bottom.grid(True, alpha=0.3)
        
        # Behavioral trial onsets (seconds), overlay on both panels
        if "TrialStart" in behavioral_data.columns:
            onsets = behavioral_data["TrialStart"].to_numpy()
            
            # If onsets look like milliseconds, convert to seconds (heuristic)
            if np.nanmax(onsets) > (t_full[-1] * 10):
                onsets = onsets / 1000.0
            
            # Draw lines on entire duration
            for onset in onsets:
                if np.isnan(onset):
                    continue
                ax_top.axvline(onset, color="darkred", linestyle="--", alpha=0.6)
                
                # Draw lines only if within short_seconds on the bottom panel
                if onset <= short_seconds:
                    ax_bottom.axvline(onset, color="darkred", linestyle="--", alpha=0.6)
        
        plt.tight_layout()
        plt.show()

# After filtering ============================================================
def plot_raw_vs_filtered(ecg_data, ecg_filtered, max_seconds=90):
    """
    Plot raw ECG data next to filtered ECG data.
    
    Parameters
    ----------
    ecg_data : ECGData
        Original ECGData object with raw signal and timestamps.
    ecg_filtered : np.ndarray
        Filtered ECG signal (same length as ecg_data.ecg_signal).
    max_seconds : float, optional
        Limit the plot to this many seconds (e.g., 10 for first 10 seconds).
        If None, plots entire signal.
    """
    t = ecg_data.timestamps
    raw = ecg_data.ecg_signal
    filt = ecg_filtered
    
    # Optionally limit duration
    if max_seconds is not None:
        mask = t <= max_seconds
        t = t[mask]
        raw = raw[mask]
        filt = filt[mask]
    
    # colour palette
    rose = "#CD6574"      # lighter rose/pink
    dark_red = "#8b0000"  # dark red
    
    plt.figure(figsize=(12, 6))
    
    # Raw signal
    plt.subplot(2, 1, 1)
    plt.plot(t, raw, color=rose, linewidth=0.8)
    plt.title("Raw ECG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("ECG (mV)")
    plt.grid(True, alpha=0.3)
    
    # Filtered signal
    plt.subplot(2, 1, 2)
    plt.plot(t, filt, color=dark_red, linewidth=0.8)
    plt.title("Filtered ECG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("ECG (mV)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    
# After downsampling =========================================================
    
def plot_filtered_vs_downsampled(
    ecg_data,
    ecg_filtered: np.ndarray,
    ecg_downsampled: np.ndarray,
    max_seconds: float = 90,
    downsample_info: dict | None = None,
):
    """
    Plot filtered ECG (top) and downsampled ECG (bottom).

    Parameters
    ----------
    ecg_data : ECGData
        Original ECGData object with attributes:
        - ecg_signal : np.ndarray
        - timestamps : np.ndarray (time in seconds, length matches ecg_signal)
    ecg_filtered : np.ndarray
        Filtered ECG signal (same length as ecg_data.ecg_signal).
    ecg_downsampled : np.ndarray
        Downsampled ECG signal (typically shorter).
    max_seconds : float, optional
        Limit the plot to the first N seconds; if None, plots the full extent
        of the available data for each panel.
    downsample_info : dict, optional
        If provided, should contain:
        - "new_sampling_frequency" (float): sampling frequency of the downsampled signal
        - OR "downsampling_method", "old_sampling_frequency", "new_sampling_frequency"
        This is used to construct an accurate time axis for the downsampled data.
        If not provided, a best-effort time axis is inferred from lengths and raw timestamps.
    """

    # Extract raw time axis and ensure filtered matches raw length
    t_raw = ecg_data.timestamps
    filt = ecg_filtered

    if len(filt) != len(t_raw):
        raise ValueError("ecg_filtered must have the same length as ecg_data.timestamps")

    # --- Build time axis for the downsampled signal ---
    if downsample_info is not None and "new_sampling_frequency" in downsample_info:
        fs_down = float(downsample_info["new_sampling_frequency"])
        t0 = float(t_raw[0]) if len(t_raw) > 0 else 0.0
        # Duration based on raw timestamps
        duration = float(t_raw[-1] - t_raw[0]) if len(t_raw) > 1 else len(filt) / fs_down
        # Construct evenly-spaced time for the downsampled data
        n_down = len(ecg_downsampled)
        # Use np.arange with n_down points at 1/fs_down cadence, anchored at t0
        t_down = t0 + np.arange(n_down) / fs_down
        # Ensure we don't exceed duration visually (optional)
        # If exact duration is critical, an alternative is to linearly map to [t0, t0+duration]
        # t_down = np.linspace(t0, t0 + duration, n_down, endpoint=False)
    else:
        # Fallback: infer by mapping indices proportionally to raw time extent
        # This keeps alignment of start and approximate end when exact fs is unknown.
        n_down = len(ecg_downsampled)
        t0 = float(t_raw[0]) if len(t_raw) > 0 else 0.0
        t1 = float(t_raw[-1]) if len(t_raw) > 1 else t0
        t_down = np.linspace(t0, t1, n_down, endpoint=False)

    # --- Optional duration limit ---
    if max_seconds is not None:
        # Limit filtered to first max_seconds
        mask_filt = (t_raw - t_raw[0]) <= max_seconds
        t_filt_plot = t_raw[mask_filt]
        filt_plot = filt[mask_filt]

        # Limit downsampled to first max_seconds from its own start
        mask_down = (t_down - t_down[0]) <= max_seconds
        t_down_plot = t_down[mask_down]
        down_plot = ecg_downsampled[mask_down]
    else:
        t_filt_plot = t_raw
        filt_plot = filt
        t_down_plot = t_down
        down_plot = ecg_downsampled

    # --- Colors (matching your style) ---
    dark_red = "#8b0000"  # filtered
    navy = "#1f3b4d"      # downsampled (distinct from filtered)

    plt.figure(figsize=(12, 6))

    # Top: Filtered
    plt.subplot(2, 1, 1)
    plt.plot(t_filt_plot, filt_plot, color=dark_red, linewidth=0.8)
    plt.title("Filtered ECG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("ECG (mV)")
    plt.grid(True, alpha=0.3)

    # Bottom: Downsampled
    plt.subplot(2, 1, 2)
    plt.plot(t_down_plot, down_plot, color=navy, linewidth=0.8)
    plt.title("Downsampled ECG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("ECG (mV)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
 
    
# Check detecting peaks ======================================================

def plot_rpeaks_segment(ecg_signal, rpeaks, sampling_frequency, duration_sec=20):
    """
    Plot a short ECG segment with detected R-peaks overlaid.
    """
    
    # Duration (in seconds) to visualize
    duration_sec = 20  # e.g., 10–30
    start_sec = 0  # could also choose a later offset
    end_sec = start_sec + duration_sec
    
    # Convert to samples
    start_idx = int(start_sec * sampling_frequency)
    end_idx = int(end_sec * sampling_frequency)
    ecg_segment = ecg_signal[start_idx:end_idx]
    
    # Shift R-peaks into segment coordinates
    rpeaks_segment = rpeaks[(rpeaks >= start_idx) & (rpeaks < end_idx)] - start_idx
    
    nk.events_plot(rpeaks_segment, ecg_segment)
    plt.title(f"Detected R-peaks (first {duration_sec} s)")
    plt.xlabel("Samples")
    plt.show()


