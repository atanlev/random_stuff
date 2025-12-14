import pickle as pkl
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import kaiserord, firwin, freqz
from scipy.fft import fft, fftfreq
from scipy import signal
import numpy as np
from torch_filter import torch_filtfilt, RealTimeIIR
from scipy.signal import resample_poly

# path = '/home/ethanl/data/logs/deby-20250901_115412/v3_1_28_08_25_3_/sim_data.pkl'
path = '/home/ethanl/data/logs/deby-20250828_081942/v3_1_28_08_25_3_/sim_data.pkl'
# path= '/home/ethanl/data/logs/deby-20250902_123032_w_fir_core/v3_1_28_08_25_3_/sim_data.pkl'

obs = 'get_imu_ang_v_local'
# obs = 'get_lin_acc_local'
# obs = 'get_imu_quat_normalized_heading'
# obs = 'get_imu_euler_normalized_heading'
# obs = 'get_imu_ang_v_local_filtered_fir'


overright_taps = False
# TAPS =  [-0.0436, -0.0428, -0.0396, -0.0338, -0.0258, -0.0156, -0.0039,  0.0089, 0.0220,  0.0348,  0.0464,  0.0562,  0.0637,  0.0684,  0.0700,  0.0684, 0.0637,  0.0562,  0.0464,  0.0348,  0.0220,  0.0089, -0.0039, -0.0156,-0.0258, -0.0338, -0.0396, -0.0428, -0.0436]
TAPS = [0.0348, 0.0464, 0.0562, 0.0637, 0.0684, 0.0700, 0.0684, 0.0637, 0.0562,0.0464, 0.0348]
def delay(b, a):
    w, h = freqz(b, a, worN=512)

    magnitude = 20 * np.log10(np.abs(h))
    phase = np.unwrap(np.angle(h))

    group_delay = -np.diff(phase) / np.diff(w)
    w_gd = w[:-1]  # frequency grid for group delay

    return w, magnitude, phase, w_gd, group_delay

def resample_to_200hz(signal, orig_fs=30, target_fs=200):
    up, down = 20, 3  # since 200/30 = 20/3
    return torch.tensor(
        resample_poly(signal.numpy(), up=up, down=down),
        dtype=torch.float32
    )

def get_imu_data(data, sim_env, obs, orig_fs=30, target_fs=200):
    obs_range = data['observation_idx_dict']['actor'][obs]
    idx = list(range(obs_range[0], obs_range[1]))
    real_imu_tensor = data['real_obs'][obs]  # [T,3]
    sim_imu_tensor = data['sim_obs'][0]['standing']['actor'][sim_env, idx].unsqueeze(0)
    for i in range(1, len(data['sim_obs'])):
        sim_imu_tensor = torch.cat(
            [sim_imu_tensor, data['sim_obs'][i]['standing']['actor'][sim_env, idx].unsqueeze(0)],
            dim=0
        )
    real_imu_tensor = real_imu_tensor.cpu().float()
    sim_imu_tensor = sim_imu_tensor.cpu().float()

    imu = {mode: {} for mode in ['sim', 'real']}
    for j, axis in enumerate(['x','y','z']):
        r = real_imu_tensor[:, j]
        s = sim_imu_tensor[:, j]
        r = r
        s = s

        # --- resample both to 200 Hz ---
        r_up = resample_to_200hz(r, orig_fs, target_fs)
        s_up = resample_to_200hz(s, orig_fs, target_fs)

        imu['real'][axis] = r_up
        imu['sim'][axis]  = s_up
    return imu


def FIR_filter(imu,ripple=20, width=300, fs=1000, cutoff=125, pass_zero=False):
    numtaps, beta = kaiserord(ripple, width/(0.5*fs))
    taps = firwin(numtaps, cutoff,
                window=('kaiser', beta),
                scale=False,
                fs=fs,
                pass_zero=pass_zero)
    taps = torch.tensor(taps, dtype=torch.float32)
    a = torch.tensor([1.0], dtype=torch.float32)
    filtered = {'x': None,'y': None,'z': None}
    filtered['x'] = torch_filtfilt(taps, a, imu['real']['x'])
    filtered['y'] = torch_filtfilt(taps, a, imu['real']['y'])
    filtered['z'] = torch_filtfilt(taps, a, imu['real']['z'])

    return filtered, taps, a

def RT_FIR(imu, filter_rt, state='real'):
    axes = ['x', 'y', 'z']
    rt_filtered = {'x': [],'y': [],'z': []}

    for axis in axes:
        stream = imu[state][axis] 
        for sample in stream:
            y = filter_rt.step(sample)
            rt_filtered[axis].append(y.item())
        rt_filtered[axis] = torch.tensor(rt_filtered[axis])
    return rt_filtered

def time_domain_plots(imu, filtered):
    axes = ['x', 'y', 'z']
    N = len(imu['real']['x'])
    x = np.arange(N)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for idx, axis in enumerate(axes):
        print(f"Before filter distance in time domain {axes} axis:  {torch.norm(imu['real'][axis]-imu['sim'][axis])}")
        print(f"After filter distancein time domain {axes} axis: {torch.norm(filtered[axis]-imu['sim'][axis])}")
        axs[idx].plot(x, imu['real'][axis], label='Real', color='red')
        axs[idx].plot(x, imu['sim'][axis], label='Sim', color='blue')
        axs[idx].plot(x, filtered[axis], label='Filtered', color='green', linestyle='--')

        axs[idx].set_title(f'{axis.upper()} Axis Time Domain')
        axs[idx].set_ylabel('IMU Reading')
        axs[idx].grid(True)
        axs[idx].legend()

    axs[-1].set_xlabel('Sample Index')

    plt.tight_layout()
    plt.show()


#fft plot
def fft_plots(imu,filtered):
    axes = ['x', 'y', 'z']
    N = len(imu['real']['x'])
    T = 1.0 / 30.0
    x = np.linspace(0.0, N*T, N, endpoint=False)
    xf = fftfreq(N, T)[:N//2]
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for idx, axis in enumerate(axes):
        filtered_f = torch.fft.fft(filtered[axis])
        imu_real_f = torch.fft.fft(imu['real'][axis])
        imu_sim_f = torch.fft.fft(imu['sim'][axis])
        print(f"Before filter distance in frequency domain {axes} axis:  {torch.norm(imu_sim_f-imu_real_f)}")
        print(f"After filter distancein frequency domain {axes} axis: {torch.norm(filtered_f-imu_sim_f)}")

        axs[idx].plot(xf, 2.0/N * torch.abs(imu_real_f[0:N//2]), label='Real')
        axs[idx].plot(xf, 2.0/N * torch.abs(imu_sim_f[0:N//2]), label='Sim')
        axs[idx].plot(xf, 2.0/N * torch.abs(filtered_f[0:N//2]), label='Filtered', linestyle='--')

        axs[idx].set_title(f'{axis.upper()} Axis FFT')
        axs[idx].set_ylabel('Amplitude')
        axs[idx].grid(True)
        axs[idx].legend()

    axs[-1].set_xlabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()

#plot histograms
def dist_plots(imu,filtered):
    axes = ['x', 'y', 'z']
    fig_hist, axs_hist = plt.subplots(3, 1, figsize=(10, 8), sharex=False)

    for idx, axis in enumerate(axes):
        axs_hist[idx].hist(imu['real'][axis], bins=30, alpha=0.5, label='Real')
        axs_hist[idx].hist(imu['sim'][axis], bins=30, alpha=0.5, label='Sim')
        axs_hist[idx].hist(filtered[axis], bins=30, alpha=0.5, label='Filtered', linestyle='--')

        axs_hist[idx].set_title(f'{axis.upper()} Axis Histogram')
        axs_hist[idx].set_ylabel('Count')
        axs_hist[idx].grid(True)
        axs_hist[idx].legend()

    axs_hist[-1].set_xlabel('Value')
    plt.tight_layout()

    plt.show()

def filter_values(b, a):
    w, mag, phase, w_gd, gd = delay(b, a)

    plt.figure(figsize=(12, 6))

    plt.subplot(3,1,1)
    plt.plot(w, mag)
    plt.title("Magnitude Response (dB)")

    plt.subplot(3,1,2)
    plt.plot(w, phase)
    plt.title("Phase Response (rad)")

    plt.subplot(3,1,3)
    plt.plot(w_gd, gd)
    plt.title("Group Delay (samples)")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    sim_env = 0
    with open(path, 'rb') as f:
        data = pkl.load(f)

    imu = get_imu_data(data=data, sim_env=sim_env, obs=obs)
    filtered_real, taps, a = FIR_filter(imu, ripple=20, width=3, fs=200, cutoff=[0, 4])
    if overright_taps:
        print('Using given taps')
        taps = TAPS
        taps = torch.tensor(taps, dtype=torch.float32)
    filter_rt = RealTimeIIR(taps, a)
    filtered_real = RT_FIR(imu, filter_rt)
    filtered_sim = RT_FIR(imu, filter_rt, state='sim')

    time_domain_plots(imu, filtered_real)
    fft_plots(imu, filtered_real)
    dist_plots(imu, filtered_real)

    filter_values(taps, a)

