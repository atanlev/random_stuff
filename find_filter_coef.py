import pickle as pkl
import numpy as np
import torch
from scipy.signal import filtfilt, freqz
from scipy.fft import rfft, rfftfreq
import time
import math
import argparse
from torch_filter import torch_filtfilt, RealTimeIIR

# -------------------------
# Config (edit as needed)
# -------------------------<
DEFAULT_PATH = '/home/ethanl/data/mike-20250806_130228/happy_v3_1_22_07_25_7_/sim_data.pkl'
DEFAULT_OBS = 'get_imu_ang_v_local'   # or 'get_lin_acc_local', etc.
FS = 210.0                            # sampling rate (Hz)
MAX_TAPS = 9                          # hard limit
ALPHA = 1.0                           # weight for time-domain loss
BETA = 0.1                            # weight for frequency-domain loss
MAX_ITERS = 16000                      # total iterations
PRINT_EVERY = 200
SEED = 42

# Annealing schedule
INIT_STEP = 0.1                      # initial noise scale for tap changes
FINAL_STEP = 0.01
INIT_TEMP = 2
FINAL_TEMP = 0.005

# Try variable tap lengths too (1 - MAX_TAPS)
ALLOW_LENGTH_CHANGES = False
LEN_CHANGE_PROB = 0.1                # probability to attempt length +/- 1


# -------------------------
# Utilities
# -------------------------
def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)

def get_imu_data(data, sim_env, obs_key):
    """
    Returns imu dict: imu['real' or 'sim']['x'|'y'|'z'] as torch.float32 tensors
    (normalized per-axis by L2).
    """
    obs_range = data['observation_idx_dict']['actor'][obs_key]
    idx = list(range(obs_range[0], obs_range[1]))
    real_imu_tensor = data['real_obs'][obs_key]           # [T,3]
    sim_imu_tensor = data['sim_obs'][0]['standing']['actor'][sim_env, idx].unsqueeze(0)
    for i in range(1, len(data['sim_obs'])):
        sim_imu_tensor = torch.cat(
            [sim_imu_tensor, data['sim_obs'][i]['standing']['actor'][sim_env, idx].unsqueeze(0)],
            dim=0
        )
    real_imu_tensor = real_imu_tensor.cpu().float()
    sim_imu_tensor = sim_imu_tensor.cpu().float() #.mean(dim=0)  # average across sims if multiple

    imu = {mode: {} for mode in ['sim', 'real']}
    for j, axis in enumerate(['x','y','z']):
        r = real_imu_tensor[:, j]
        s = sim_imu_tensor[:, j]
        imu['real'][axis] = r / (torch.norm(r) + 1e-12)
        imu['sim'][axis]  = s / (torch.norm(s) + 1e-12)
    return imu

# def apply_fir_filtfilt(b, x):
#     """ Zero-phase filter (numpy arrays). """
#     if len(b) < 2:
#         return x.copy()
#     return filtfilt(b, [1.0], x, method="pad")

def apply_fir_realtime(b, x):
    """Apply causal FIR filter (like RT_FIR)."""
    if len(b) < 2:
        return x.copy()
    b = np.asarray(b, dtype=np.float64)
    y = np.zeros_like(x)
    for n in range(len(x)):
        k0 = max(0, n - len(b) + 1)
        k1 = n + 1
        y[n] = np.dot(b[:k1-k0][::-1], x[k0:k1])
    return y

def RT_FIR(imu, filter_rt, state='real', axis = 'x'):
    rt_filtered = []

    stream = imu[state][axis] 
    for sample in stream:
        y = filter_rt.step(sample)
        rt_filtered.append(y.item())
    rt_filtered = np.array(rt_filtered)
    return rt_filtered

def l2(a, b):
    return float(np.linalg.norm(a - b))

def compute_loss(b_dict, imu, fs=FS):
    """
    Combined time/frequency loss using axis-specific taps.
    b_dict = {'x': b_x, 'y': b_y, 'z': b_z}
    """
    total_time, total_freq = 0.0, 0.0
    for axis in b_dict:
        b = b_dict[axis]
        real_t = imu['real'][axis].numpy()
        sim_t  = imu['sim'][axis].numpy()

        # fr  = apply_fir_realtime(b, real_t)
        filter_rt = RealTimeIIR(b, [1])
        fr = RT_FIR(imu, filter_rt)

        # sim_t = sim_t/np.linalg.norm(sim_t)
        # fr = fr/np.linalg.norm(fr)
        total_time += l2(np.abs(fr), np.abs(fr))

        N = len(real_t)
        fr_f   = np.abs(rfft(fr))
        sim_f  = np.abs(rfft(sim_t))
        total_freq += l2(fr_f, sim_f)

    return ALPHA * total_time + BETA * total_freq, total_time, total_freq

# def normalize_taps(b):
#     """
#     Keep taps bounded & normalized:
#     - Center to zero-mean (avoid DC bias drift between steps)
#     - L1-normalize absolute sum to ~1 to keep gain tame
#     """
#     b = np.asarray(b, dtype=np.float64)
#     if b.size == 0:
#         return b
#     b = b - b.mean()
#     s = np.sum(np.abs(b))
#     if s < 1e-9:
#         return np.zeros_like(b) + 1.0 / max(1, len(b))
#     return b / s


def normalize_taps(b):
    """
    Keep taps bounded & normalized:
    - Center to zero-mean (avoid DC bias drift between steps)
    - L2-normalize (energy of taps = 1) to keep gain stable
    """
    b = np.asarray(b, dtype=np.float64)
    if b.size == 0:
        return b
    
    b = b - b.mean()  # zero-mean
    norm = np.linalg.norm(b)
    if norm < 1e-12:
        # fallback: uniform taps
        return np.ones_like(b) / max(1, len(b))
    
    return b / norm


def propose_neighbor_dict(b_dict, step_scale, allow_len_changes=True):
    new_dict = {}
    for axis in ['x','y','z']:
        b = b_dict[axis]
        new_dict[axis] = propose_neighbor(b, step_scale, allow_len_changes)
    return new_dict

def propose_neighbor(b, step_scale, allow_len_changes=True):
    b = b.copy()
    # maybe change length
    if allow_len_changes and np.random.rand() < LEN_CHANGE_PROB:
        if len(b) < MAX_TAPS and np.random.rand() < 0.5:
            # grow by 1 (insert at random pos)
            pos = np.random.randint(0, len(b)+1)
            b = np.insert(b, pos, 0.0)
        elif len(b) > 1:
            # shrink by 1 (remove at random pos)
            pos = np.random.randint(0, len(b))
            b = np.delete(b, pos)

    # random perturbation
    noise = np.random.normal(loc=0.0, scale=step_scale, size=len(b))
    b = b + noise
    return normalize_taps(b)

def schedule(it, iters, start, end):
    # cosine schedule
    t = it / max(1, iters-1)
    return end + 0.5*(start - end)*(1 + math.cos(math.pi * t))

# -------------------------
# Main routine
# -------------------------
def run_optimization(paths, obs, sim_env=0, init_len=5):
    # Load datasets
    imus = []
    for path in paths:
        with open(path, 'rb') as f:
            data = pkl.load(f)
        imu = get_imu_data(data, sim_env=sim_env, obs_key=obs)
        imus.append(imu)

    # Initialize taps + best state per axis
    init_len = int(np.clip(init_len, 1, MAX_TAPS))
    best_b = {axis: normalize_taps(np.concatenate([np.zeros(init_len - 1), np.array([1])]) / init_len) for axis in ['x','y','z']}
    best_loss = {}
    for axis in ['x','y','z']:
        loss = 0.0
        for imu in imus:
            l, _, _ = compute_loss({axis: best_b[axis]}, imu)
            loss += l
        best_loss[axis] = loss / len(imus)

    t0 = time.time()
    for it in range(MAX_ITERS):
        step = schedule(it, MAX_ITERS, INIT_STEP, FINAL_STEP)
        temp = schedule(it, MAX_ITERS, INIT_TEMP, FINAL_TEMP)

        for axis in ['x','y','z']:
            cand_b = propose_neighbor(best_b[axis], step, allow_len_changes=ALLOW_LENGTH_CHANGES)

            cand_loss = 0.0
            for imu in imus:
                l, _, _ = compute_loss({axis: cand_b}, imu)
                cand_loss += l
            cand_loss /= len(imus)

            # accept if better or probabilistically worse
            if cand_loss < best_loss[axis] or np.random.rand() < math.exp(-(cand_loss - best_loss[axis]) / max(1e-12, temp)):
                best_b[axis], best_loss[axis] = cand_b, cand_loss

        if (it+1) % PRINT_EVERY == 0 or it == 0:
            print(f"[{it+1:5d}/{MAX_ITERS}] " +
                  " ".join([f"{ax}_taps={len(best_b[ax])} loss={best_loss[ax]:.6f}"
                            for ax in ['x','y','z']]) +
                  f" step={step:.4f} temp={temp:.4f}")
            # for axis in ['x','y','z']:
            #     print(f"{axis}-axis taps ({len(best_b[axis])}): {np.array2string(best_b[axis], precision=6, separator=', ')}")
            #     print(f"Best {axis}-axis loss: {best_loss[axis]:.6f}")


    dt = time.time() - t0
    print("\n=== Optimization done ===")
    print(f"Elapsed: {dt:.2f}s")
    for axis in ['x','y','z']:
        print(f"{axis}-axis taps ({len(best_b[axis])}): {np.array2string(best_b[axis], precision=6, separator=', ')}")
        print(f"Best {axis}-axis loss: {best_loss[axis]:.6f}")

    return best_b

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Optimize FIR taps over multiple datasets")
    parser.add_argument("--paths", type=str, nargs='+', required=True,
                        help="One or more paths to sim_data.pkl files")
    parser.add_argument("--obs", type=str, default='get_imu_ang_v_local', help="Observation key")
    parser.add_argument("--init_len", type=int, default=4, help="Initial number of taps (1..9)")
    parser.add_argument("--iters", type=int, default=16000, help="Iterations")
    parser.add_argument("--alpha", type=float, default=0.0, help="Time-domain weight")
    parser.add_argument("--beta", type=float, default=1.0, help="Freq-domain weight")
    parser.add_argument("--fs", type=float, default=210, help="Sampling frequency")
    parser.add_argument("--seed", type=int, default=10, help="RNG seed")
    args = parser.parse_args()

    global FS, MAX_ITERS, ALPHA, BETA
    FS = args.fs
    MAX_ITERS = args.iters
    ALPHA = args.alpha
    BETA = args.beta

    set_seed(args.seed)
    best_b = run_optimization(args.paths, args.obs, init_len=args.init_len)

if __name__ == "__main__":
    main()
