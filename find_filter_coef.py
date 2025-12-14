import pickle as pkl
import numpy as np
import torch
from scipy.fft import rfft
import time
import math
import argparse
from torch_filter import RealTimeIIR

# -------------------------
# Config (edit as needed)
# -------------------------
DEFAULT_PATH = '/home/ethanl/data/mike-20250806_130228/happy_v3_1_22_07_25_7_/sim_data.pkl'
DEFAULT_OBS = 'get_imu_ang_v_local'   # or 'get_lin_acc_local', etc.
FS = 210.0
MAX_TAPS = 20
ALPHA = 1.0
BETA = 0.1
DELTA = 0.0               # default; overridden by --delta
MAX_ITERS = 16000
PRINT_EVERY = 200
SEED = 42

# Annealing schedule
INIT_STEP = 0.02
FINAL_STEP = 0.001
INIT_TEMP = 1.0
FINAL_TEMP = 0.005

# Length changes off by default
ALLOW_LENGTH_CHANGES = False
LEN_CHANGE_PROB = 0.1

# -------------------------
# Utilities
# -------------------------
from scipy.signal import resample_poly

def resample_to_200hz(signal, orig_fs=30, target_fs=200):
    up, down = 20, 3  # since 200/30 = 20/3
    return torch.tensor(
        resample_poly(signal.numpy(), up=up, down=down),
        dtype=torch.float32
    )

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)

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
        r = r / (torch.norm(r) + 1e-12)
        s = s / (torch.norm(s) + 1e-12)

        # --- resample both to 200 Hz ---
        r_up = resample_to_200hz(r, orig_fs, target_fs)
        s_up = resample_to_200hz(s, orig_fs, target_fs)

        imu['real'][axis] = r_up
        imu['sim'][axis]  = s_up
    return imu

def RT_FIR(imu, filter_rt, state='real', axis='x'):
    out = []
    stream = imu[state][axis]
    for sample in stream:
        y = filter_rt.step(sample)
        out.append(y)
    return np.asarray(out, dtype=np.float64)

# --- Symmetry + normalization helpers ---

def symmetrize(b):
    b = np.asarray(b, dtype=np.float64)
    if b.size == 0:
        return b
    return 0.5 * (b + b[::-1])

def normalize_taps(b):
    b = np.asarray(b, dtype=np.float64)
    if b.size == 0:
        return b
    b = symmetrize(b)
    norm = np.linalg.norm(b)
    if norm < 1e-12:
        b = np.ones_like(b) / max(1, len(b))
        b = symmetrize(b)
        norm = np.linalg.norm(b)
    return b / norm

def propose_neighbor(b, step_scale, allow_len_changes=True):
    b = np.asarray(b, dtype=np.float64).copy()

    if allow_len_changes and np.random.rand() < LEN_CHANGE_PROB:
        if len(b) < MAX_TAPS and np.random.rand() < 0.5:
            pos = np.random.randint(0, len(b)+1)
            b = np.insert(b, pos, 0.0)
        elif len(b) > 1:
            pos = np.random.randint(0, len(b))
            b = np.delete(b, pos)

    noise = np.random.normal(loc=0.0, scale=step_scale, size=len(b))
    b = b + noise
    b = symmetrize(b)
    return normalize_taps(b)

def schedule(it, iters, start, end):
    t = it / max(1, iters - 1)
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * t))

def l2(a, b):
    return float(np.linalg.norm(a - b))

# -------------------------
# Loss (fixed)
# -------------------------
def compute_loss(b_dict, imu, sim_power=None, real_power=None):
    """
    b_dict = {'x': b_x, 'y': b_y, 'z': b_z} (any subset of axes allowed).
    Loss = ALPHA * time_L2 + BETA * freq_L2 + DELTA * power_penalty
    power penalty is sum of squared RMS differences (per axis).
    If sim_power/real_power provided (floats), use them as targets; else compute from data.
    """
    total_time, total_freq, power_pen = 0.0, 0.0, 0.0

    for axis, b in b_dict.items():
        b = normalize_taps(b)

        real_t = imu['real'][axis].numpy()
        sim_t  = imu['sim'][axis].numpy()

        # Causal RT filtering
        filt = RealTimeIIR(b, [1.0])
        fr = RT_FIR(imu, filt, state='real', axis=axis)
        fr_sim = RT_FIR(imu, filt, state='sim', axis=axis)

        # time-domain L2 between filtered real and filtered sim
        total_time += l2(fr, fr_sim)

        # frequency magnitude L2
        fr_f  = np.abs(rfft(fr))
        sim_f = np.abs(rfft(fr_sim))
        total_freq += l2(fr_f, sim_f)

        # RMS powers
        rms_fr  = float(np.mean(fr**2))
        rms_fsim = float(np.mean(fr_sim**2))

        if sim_power is not None and real_power is not None:
            # sim_power/real_power expected to be floats (per-axis averages across datasets)
            target_sim_power = float(sim_power)
            target_real_power = float(real_power)
        else:
            target_sim_power = float(np.mean(sim_t**2))
            target_real_power = float(np.mean(real_t**2))

        # improved power penalty: squared difference of RMS (use both real and sim targets)
        # accumulate squared errors so penalty >= 0
        power_pen += (rms_fr - target_real_power)**2 + (rms_fsim - target_sim_power)**2

    loss = ALPHA * total_time + BETA * total_freq + DELTA * power_pen
    return loss, total_time, total_freq

# -------------------------
# Main routine (fixed SIM/REAL power computation order)
# -------------------------
def run_optimization(paths, obs, sim_env=0, init_len=4, set_taps=True):
    imus = []
    for path in paths:
        with open(path, 'rb') as f:
            data = pkl.load(f)
        imu = get_imu_data(data, sim_env=sim_env, obs=obs)
        imus.append(imu)

    init_len = int(np.clip(init_len, 1, MAX_TAPS))
    if set_taps and init_len == 6:
        init_b = np.array([-0.1099,  0.1301,  0.4302,  0.4302,  0.1301, -0.1099], dtype=np.float64)
    elif set_taps and init_len == 2:
        init_b = np.array([0.4502, 0.4502], dtype=np.float64)
    else:
        init_b = np.ones(init_len, dtype=np.float64) / init_len
    init_b = normalize_taps(init_b)

    best_b = {axis: init_b.copy() for axis in ['x','y','z']}

    # First compute average SIM/REAL power per axis (floats)
    SIM_POWER = {}
    REAL_POWER = {}
    for axis in ['x','y','z']:
        sim_acc = 0.0
        real_acc = 0.0
        for imu in imus:
            sim_acc += float(torch.mean(imu['sim'][axis]**2).item())
            real_acc += float(torch.mean(imu['real'][axis]**2).item())
        SIM_POWER[axis] = sim_acc / len(imus)
        REAL_POWER[axis] = real_acc / len(imus)

    # Then compute initial best_loss averaged across datasets using those powers
    best_loss = {}
    for axis in ['x','y','z']:
        acc = 0.0
        for imu in imus:
            l, _, _ = compute_loss({axis: best_b[axis]}, imu, SIM_POWER[axis], REAL_POWER[axis])
            acc += l
        best_loss[axis] = acc / len(imus)

    t0 = time.time()
    for it in range(MAX_ITERS):
        step = schedule(it, MAX_ITERS, INIT_STEP, FINAL_STEP)
        temp = schedule(it, MAX_ITERS, INIT_TEMP, FINAL_TEMP)

        for axis in ['x','y','z']:
            cand_b = propose_neighbor(best_b[axis], step, allow_len_changes=ALLOW_LENGTH_CHANGES)
            acc = 0.0
            for imu in imus:
                l, _, _ = compute_loss({axis: cand_b}, imu, SIM_POWER[axis], REAL_POWER[axis])
                acc += l
            cand_loss = acc / len(imus)

            if cand_loss < best_loss[axis] or np.random.rand() < math.exp(-(cand_loss - best_loss[axis]) / max(1e-12, temp)):
                best_b[axis], best_loss[axis] = cand_b, cand_loss

        if (it + 1) % PRINT_EVERY == 0 or it == 0:
            print(
                f"[{it+1:5d}/{MAX_ITERS}] "
                + " ".join([f"{ax}_taps={len(best_b[ax])} loss={best_loss[ax]:.6f}" for ax in ['x','y','z']])
                + f" step={step:.4f} temp={temp:.4f}"
            )

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
    parser = argparse.ArgumentParser(description="Optimize symmetric FIR taps per axis over multiple datasets")
    parser.add_argument("--paths", type=str, nargs='+', required=True,
                        help="One or more paths to sim_data.pkl files")
    parser.add_argument("--obs", type=str, default='get_imu_ang_v_local', help="Observation key")
    parser.add_argument("--init_len", type=int, default=5, help="Initial number of taps (1..9)")
    parser.add_argument("--set_taps", type=int, default=10, help="Use built-in symmetric seeds when length=2 or 4 (1/0)")
    parser.add_argument("--iters", type=int, default=4000, help="Iterations")
    parser.add_argument("--alpha", type=float, default=1.0, help="Time-domain weight")
    parser.add_argument("--beta", type=float, default=0.2, help="Freq-domain weight")
    parser.add_argument("--delta", type=float, default=0.3, help="Power weight (RMS error)")
    parser.add_argument("--fs", type=float, default=210, help="Sampling frequency")
    parser.add_argument("--seed", type=int, default=10, help="RNG seed")
    args = parser.parse_args()

    global FS, MAX_ITERS, ALPHA, BETA, DELTA
    FS = args.fs
    MAX_ITERS = args.iters
    ALPHA = args.alpha
    BETA = args.beta
    DELTA = args.delta

    set_seed(args.seed)
    best_b = run_optimization(args.paths, args.obs, init_len=args.init_len, set_taps=bool(args.set_taps))

if __name__ == "__main__":
    main()
