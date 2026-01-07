import pickle as pkl
import numpy as np
import torch
import scipy.signal as signal
import argparse
import matplotlib.pyplot as plt
# -------------------------
# Config
# -------------------------
FS_ROBOT = 200.0   # Target Control Freq
FS_IMU   = 30.0    # Source Sensor Freq
CUTOFF   = 30.0    # Hz (Must be < Nyquist of source, i.e., < 15Hz)
FILTER_ORDER = 2   # Order 2 is a sweet spot for smoothing vs. delay

# -------------------------
# Data Loading & Processing
# -------------------------
def zoh_upsample(signal_30hz, target_len):
    """
    Simulates real-time 'latest available data' upsampling (Staircase).
    This is what the robot actually sees.
    """
    # Create indices for the staircase
    scale = len(signal_30hz) / target_len
    indices = np.floor(np.arange(target_len) * scale).astype(int)
    indices = np.clip(indices, 0, len(signal_30hz) - 1)
    return signal_30hz[indices]

def get_data_analytical(path, obs, sim_env=0):
    with open(path, 'rb') as f:
        data = pkl.load(f)

    # 1. Extract Sim Data (Reference)
    obs_range = data['observation_idx_dict']['actor'][obs]
    idx = list(range(obs_range[0], obs_range[1]))
    
    sim_chunks = []
    # Loop through the list of steps to build the full trajectory
    for i in range(len(data['sim_obs'])):
        # Extract specific environment and specific observation indices
        chunk = data['sim_obs'][i]['standing']['actor'][sim_env, idx].cpu()
        sim_chunks.append(chunk)
    
    sim_tensor = torch.cat(sim_chunks, dim=0).cpu().float().numpy()

    # 2. Extract Real Data (Source)
    real_tensor = data['real_obs'][obs].cpu().float().numpy()

    # --- SHAPE FIX: Handle Flattened Sim Data ---
    # Case: real is (300, 3) but sim is (900,)
    if sim_tensor.ndim == 1 and real_tensor.ndim == 2:
        num_axes = real_tensor.shape[1] # usually 3
        # Check if reshaping is valid
        if sim_tensor.size % num_axes == 0:
            target_rows = sim_tensor.size // num_axes
            print(f"Warning: Reshaping sim_tensor from {sim_tensor.shape} to ({target_rows}, {num_axes})")
            sim_tensor = sim_tensor.reshape(target_rows, num_axes)
        else:
            raise ValueError(f"Shape Mismatch: Sim {sim_tensor.shape} cannot be reshaped to match Real cols {num_axes}")

    # 3. Process Per Axis
    dataset = {'x': {}, 'y': {}, 'z': {}}
    axes = ['x', 'y', 'z']
    
    # Calculate target length based on Real duration scaled to 200Hz
    # FS_ROBOT = 200.0, FS_IMU = 30.0 (Global vars assumed)
    target_len = int(len(real_tensor) * (FS_ROBOT / FS_IMU))
    
    for i, ax in enumerate(axes):
        r_raw = real_tensor[:, i]
        s_raw = sim_tensor[:, i]

        # NORMALIZE
        r_raw = r_raw / (np.linalg.norm(r_raw) + 1e-12)
        s_raw = s_raw / (np.linalg.norm(s_raw) + 1e-12)
        
        # A) SIM DATA -> Sinc Interpolated (Perfect Reference)
        # We need s_up to be roughly target_len. 
        # If Sim is 30Hz, we upsample. If Sim is already higher freq, we adjust.
        # Assuming Sim is roughly same duration as Real:
        s_up = signal.resample(s_raw, target_len)
        
        # B) REAL DATA -> Realistic ZOH Upsample (Staircase)
        r_zoh = zoh_upsample(r_raw, len(s_up))
        
        dataset[ax]['real_zoh'] = r_zoh
        dataset[ax]['sim_ref']  = s_up
        
    return dataset
# -------------------------
# Filter Design & Application
# -------------------------
def design_butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_causal_filter(data, b, a):
    # lfilter is strictly causal (y[n] depends only on past x and y)
    # We use zi to set initial state to 0 (or steady state) to avoid transient spikes
    zi = signal.lfilter_zi(b, a) * data[0]
    y, _ = signal.lfilter(b, a, data, zi=zi)
    return y

# -------------------------
# Analytical KPIs
# -------------------------
import numpy as np
from scipy import signal

def calculate_kpis(filtered, target, raw, fs):
    """
    filtered: The output of your IIR filter
    target:   The ideal reference trajectory (sim_ref) - used for Accuracy
    raw:      The noisy input signal (raw_zoh) - used for Delay and Baseline Roughness
    fs:       Sampling frequency (Hz)
    """
    
    # --- 1. RMSE (Accuracy: Lower is better) ---
    # Baseline: How far is the Raw noisy signal from the Target?
    raw_rmse = np.sqrt(np.mean((raw - target) ** 2))
    
    # Filtered: How far is the Filtered signal from the Target?
    filt_rmse = np.sqrt(np.mean((filtered - target) ** 2))
    
    
    # --- 2. Smoothness (Roughness: Lower is better) ---
    # Baseline: Roughness of the Raw signal
    d1_raw = np.diff(raw, prepend=raw[0])
    d2_raw = np.diff(d1_raw, prepend=d1_raw[0])
    raw_roughness = np.var(d2_raw)
    
    # Filtered: Roughness of the Filtered signal
    d1_filt = np.diff(filtered, prepend=filtered[0])
    d2_filt = np.diff(d1_filt, prepend=d1_filt[0])
    filt_roughness = np.var(d2_filt)
    
    
    # --- 3. Estimate Delay (Lag) ---
    # We compare Raw (Input) vs Filtered (Output) to find the phase lag added by the filter.
    
    # Center signals to remove DC offset for correlation
    corr = signal.correlate(raw - np.mean(raw), filtered - np.mean(filtered), mode='full')
    lags = signal.correlation_lags(len(raw), len(filtered), mode='full')
    
    lag_idx = lags[np.argmax(corr)]
    delay_ms = (lag_idx / fs) * 1000.0
    
    # --- Print Summary for Quick Debugging ---
    print(f"--- Filter Performance ---")
    print(f"RMSE (Accuracy):  {raw_rmse:.4f} -> {filt_rmse:.4f} ({(raw_rmse - filt_rmse)/raw_rmse * 100:.1f}% Improvement)")
    print(f"Roughness:        {raw_roughness:.4f} -> {filt_roughness:.4f} ({(raw_roughness - filt_roughness)/raw_roughness * 100:.1f}% Smoother)")
    print(f"Added Delay:      {delay_ms:.2f} ms")
    
    return {
        "raw_rmse": raw_rmse,
        "filt_rmse": filt_rmse,
        "raw_roughness": raw_roughness,
        "filt_roughness": filt_roughness,
        "delay_ms": delay_ms
    }
import numpy as np
from scipy import signal


def analyze_filter_performance(filtered, target, raw, fs, cutoff_freq=10.0, plot=True):
    """
    Analyzes filter performance using Spectral Density to separate 
    Signal (Tracking) from Noise (Real-world artifacts).
    """
    
    # --- 1. Basic Time-Domain Metrics ---
    # RMSE (Accuracy): Compare against the ideal Target (Sim)
    raw_rmse = np.sqrt(np.mean((raw - target) ** 2))
    filt_rmse = np.sqrt(np.mean((filtered - target) ** 2))
    
    # Delay Estimation: Compare Input (Raw) vs Output (Filtered)
    corr = signal.correlate(raw - np.mean(raw), filtered - np.mean(filtered), mode='full')
    lags = signal.correlation_lags(len(raw), len(filtered), mode='full')
    lag_idx = lags[np.argmax(corr)]
    delay_ms = (lag_idx / fs) * 1000.0

    # --- 2. Spectral Distribution Analysis (Welch) ---
    # We analyze the "Residuals" (The difference between Reality and Sim)
    noise_signal_raw = raw - target
    noise_signal_filt = filtered - target
    
    # Compute Power Spectral Density (PSD)
    # nperseg: Length of each segment. Higher = more freq resolution, lower = smoother plot.
    freqs, psd_raw = signal.welch(noise_signal_raw, fs, nperseg=1024)
    _, psd_filt = signal.welch(noise_signal_filt, fs, nperseg=1024)
    
    # --- 3. Compute Suppression in the Noise Band ---
    # We care about energy ABOVE the cutoff (Real-world noise range)
    idx_cutoff = np.argmax(freqs >= cutoff_freq)
    
    energy_raw_high_freq = np.trapz(psd_raw[idx_cutoff:], freqs[idx_cutoff:])
    energy_filt_high_freq = np.trapz(psd_filt[idx_cutoff:], freqs[idx_cutoff:])
    
    # Safety check for log10
    energy_raw_high_freq = max(energy_raw_high_freq, 1e-12)
    energy_filt_high_freq = max(energy_filt_high_freq, 1e-12)
    
    suppression_db = 10 * np.log10(energy_filt_high_freq / energy_raw_high_freq)

    # --- 4. Plotting ---
    if plot:
        plt.figure(figsize=(10, 6))
        
        # Convert PSD to dB/Hz for standard plotting
        psd_raw_db = 10 * np.log10(psd_raw + 1e-12)
        psd_filt_db = 10 * np.log10(psd_filt + 1e-12)
        
        plt.plot(freqs, psd_raw_db, label='Raw Residual (Input Noise)', color='grey', alpha=0.7)
        plt.plot(freqs, psd_filt_db, label='Filtered Residual (Output Noise)', color='red', linewidth=2)
        
        # Draw the Cutoff Line
        plt.axvline(x=cutoff_freq, color='k', linestyle='--', label=f'Cutoff ({cutoff_freq} Hz)')
        
        # Shade the region being analyzed
        plt.fill_between(freqs[idx_cutoff:], -100, psd_raw_db[idx_cutoff:], color='gray', alpha=0.1)
        
        plt.title(f"Spectral Analysis of Error (Noise Suppression: {suppression_db:.2f} dB)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (dB/Hz)")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.xlim(0, fs/2) # Show up to Nyquist
        
        # Optional: Zoom in on y-axis if noise floor is low
        plt.ylim(max(np.min(psd_filt_db), -120), np.max(psd_raw_db) + 10)
        
        plt.tight_layout()
        plt.show()

    print(f"--- Analysis Results ---")
    print(f"RMSE Improvement: {(raw_rmse - filt_rmse)/raw_rmse*100:.1f}%")
    print(f"Added Delay:      {delay_ms:.2f} ms")
    print(f"Noise Suppression:{suppression_db:.2f} dB (Energy > {cutoff_freq}Hz)")
    
    return {
        "rmse_imp": raw_rmse - filt_rmse,
        "delay_ms": delay_ms,
        "suppression_db": suppression_db
    }# -------------------------
# Main Execution
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=str, nargs='+', required=True)
    parser.add_argument("--obs", type=str, default='get_imu_ang_v_local')
    args = parser.parse_args()

    # 1. Design Filter (Analytic, Instant)
    print(f"--- Designing IIR Filter (Butterworth Order {FILTER_ORDER} @ {CUTOFF}Hz) ---")
    b, a = design_butter_lowpass(CUTOFF, FS_ROBOT, FILTER_ORDER)
    print(f"b = {np.array2string(b, precision=5)}")
    print(f"a = {np.array2string(a, precision=5)}")
    print("---------------------------------------------------------------")

    total_rmse = 0
    total_lag = 0
    total_roughness = 0
    count = 0

    axes = ['x', 'y', 'z']

    for path in args.paths:
        print(f"\nProcessing: {path}")
        dataset = get_data_analytical(path, args.obs)
        
        for ax in axes:
            raw_zoh = dataset[ax]['real_zoh']
            sim_ref = dataset[ax]['sim_ref']
            
            # Apply Filter
            filtered_signal = apply_causal_filter(raw_zoh, b, a)
            
            # Evaluate
            kpis = calculate_kpis(filtered=filtered_signal, target=sim_ref, raw=raw_zoh, fs=FS_ROBOT)
            analyze_filter_performance(filtered=filtered_signal, target=sim_ref, raw=raw_zoh, fs=FS_ROBOT, cutoff_freq=CUTOFF)

            rmse = kpis["filt_rmse"]
            rough = kpis["filt_roughness"]
            delay = kpis["delay_ms"]
            
            total_rmse += rmse
            total_lag += delay
            total_roughness += rough
            count += 1

    print("\n========================================")
    print("AVERAGE PERFORMANCE METRICS")
    print("========================================")
    print(f"Avg RMSE:      {total_rmse/count:.4f}")
    print(f"Avg Delay:     {total_lag/count:.2f} ms")
    print(f"Avg Roughness: {total_roughness/count:.2e}")
    print(f"Filter Coefs")
    print(f"const float b[] = {{ {b[0]:.6f}, {b[1]:.6f}, {b[2]:.6f} }};")
    print(f"const float a[] = {{ {a[0]:.6f}, {a[1]:.6f}, {a[2]:.6f} }};")

if __name__ == "__main__":
    main()