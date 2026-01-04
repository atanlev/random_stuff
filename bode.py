import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# --- Configuration ---
FS = 200.0       # Sampling Frequency
CUTOFF = 12.0    # Cutoff Frequency
ORDER = 2        # Filter Order

# 1. Design Filter
b, a = signal.butter(ORDER, CUTOFF, btype='low', fs=FS)

# 2. Compute Frequency Response
# w is frequency in rad/sample, h is complex frequency response
w, h = signal.freqz(b, a, worN=2048, fs=FS)

freqs = w  # freqz with fs returns real frequencies in Hz

# 3. Compute Magnitude (dB)
mag_db = 20 * np.log10(abs(h))

# 4. Compute Phase (Degrees)
phase_deg = np.angle(h, deg=True)

# 5. Compute Group Delay (ms)
# Group Delay = -d(Phase)/d(Frequency)
# We use signal.group_delay
w_gd, group_delay_samples = signal.group_delay((b, a), w=2048, fs=FS)
group_delay_ms = (group_delay_samples / FS) * 1000.0

# --- Plotting ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot 1: Magnitude Response
ax1.set_title(f"Bode Plot: Butterworth Order {ORDER}, Cutoff {CUTOFF}Hz @ {FS}Hz")
ax1.plot(freqs, mag_db, 'b', linewidth=2)
ax1.axvline(CUTOFF, color='r', linestyle='--', label=f'Cutoff ({CUTOFF} Hz)')
ax1.axhline(-3, color='k', linestyle=':', label='-3 dB')
ax1.set_ylabel("Magnitude (dB)")
ax1.grid(True, which='both')
ax1.legend()
ax1.set_ylim(-60, 5)

# Plot 2: Phase Response
ax2.plot(freqs, phase_deg, 'g', linewidth=2)
ax2.set_ylabel("Phase (degrees)")
ax2.grid(True, which='both')
ax2.axvline(CUTOFF, color='r', linestyle='--')

# Plot 3: Group Delay (Latency)
ax3.plot(w_gd, group_delay_ms, 'm', linewidth=2)
ax3.set_ylabel("Group Delay (ms)")
ax3.set_xlabel("Frequency (Hz)")
ax3.grid(True, which='both')
ax3.axvline(CUTOFF, color='r', linestyle='--')
ax3.set_xlim(0, 50) # Zoom in on relevant frequencies

# Highlight delay at low freq (Robot motion)
avg_delay_low = np.mean(group_delay_ms[w_gd < 5])
ax3.axhline(avg_delay_low, color='k', linestyle=':', label=f'Avg Low-Freq Delay: {avg_delay_low:.1f} ms')
ax3.legend()

plt.tight_layout()
plt.show()