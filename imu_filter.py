import pickle as pkl
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import kaiserord, firwin, freqz
from scipy.fft import fft, fftfreq
from scipy import signal
import numpy as np
from torch_filter import torch_filtfilt
# path = '/home/ethanl/data/mike-20250806_130228/happy_v3_1_22_07_25_7_/sim_data.pkl'
# path = '/home/ethanl/data/neil-20250804_134921/develop_projects_happy_v3_1_29_07_25_8/sim_data.pkl'
# path = '/home/ethanl/data/mike-20250806_142049/happy_v3_1_06_08_25_2_/sim_data.pkl'
path = '/home/ethanl/data/mike-20250805_153142/happy_v3_1_05_08_25_13_/sim_data.pkl'

obs = 'get_imu_ang_v_local'
sim_env = 1
with open(path, 'rb') as f:
    data = pkl.load(f)

obs_range = data['observation_idx_dict']['actor'][obs]
idx = list(range(obs_range[0],obs_range[1]))
real_imu_tensor = data['real_obs']['get_imu_quat_normalized_heading']
sim_imu_tensor = data['sim_obs'][0]['standing']['actor'][sim_env,idx].unsqueeze(0)
for i in range(1, len(data['sim_obs'])):
    sim_imu_tensor = torch.cat([sim_imu_tensor,data['sim_obs'][i]['standing']['actor'][sim_env,idx].unsqueeze(0)], dim = 0)

real_imu_tensor = real_imu_tensor.cpu()
sim_imu_tensor = sim_imu_tensor.cpu()
temp = [real_imu_tensor, sim_imu_tensor]
#normalize and separate
imu = {mode: {} for mode in ['sim', 'real']}

for i, mode in enumerate(['sim', 'real']):
    for j, axis in enumerate(['x', 'y', 'z']):
        imu[mode][axis] = np.array(temp[i][:,j] / torch.norm(temp[i][:,j]))

numtaps, beta = kaiserord(20, 300/(0.5*1000))
taps = firwin(numtaps, 125, window=('kaiser', beta),
                  scale=False, fs=1000)
filtered = {'x': None,'y': None,'z': None}
filtered['x'] = torch_filtfilt(taps, [1], imu['real']['x'])
filtered['y'] = torch_filtfilt(taps, [1], imu['real']['y'])
filtered['z'] = torch_filtfilt(taps, [1], imu['real']['z'])

plt.figure(figsize=(10, 6))

plt.plot(imu['real']['x'], label='Real X', color='red')
plt.plot(imu['sim']['x'], label='Sim X', color='blue')
plt.plot(filtered['x'], label='Filtered', color='green', linestyle='--')

plt.xlabel('Sample Index')
plt.ylabel('Normalized IMU Reading')
plt.title('IMU X-Axis Comparison: Real vs Sim')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#fft plot
axes = ['x', 'y', 'z']
N = len(imu['real'][axis])
T = 1.0 / 30.0
x = np.linspace(0.0, N*T, N, endpoint=False)
xf = fftfreq(N, T)[:N//2]
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for idx, axis in enumerate(axes):
    filtered_f = fft(filtered[axis])
    imu_real_f = fft(imu['real'][axis])
    imu_sim_f = fft(imu['sim'][axis])
    print(f"Before filter distance in frequency domain {axes} axis:  {np.linalg.norm(imu_sim_f-imu_real_f)}")
    print(f"After filter distancein frequency domain {axes} axis: {np.linalg.norm(filtered_f-imu_real_f)}")

    axs[idx].plot(xf, 2.0/N * np.abs(imu_real_f[0:N//2]), label='Real')
    axs[idx].plot(xf, 2.0/N * np.abs(imu_sim_f[0:N//2]), label='Sim')
    axs[idx].plot(xf, 2.0/N * np.abs(filtered_f[0:N//2]), label='Filtered', linestyle='--')

    axs[idx].set_title(f'{axis.upper()} Axis FFT')
    axs[idx].set_ylabel('Amplitude')
    axs[idx].grid(True)
    axs[idx].legend()

axs[-1].set_xlabel('Frequency (Hz)')

plt.tight_layout()
plt.show()

#plot histograms
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
