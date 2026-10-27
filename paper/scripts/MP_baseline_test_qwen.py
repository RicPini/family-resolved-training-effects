#!/usr/bin/env python3
"""
MP baseline test: generate i.i.d. random matrices of the c_attn shape and probe them with the same multi-k machinery.

If log_k_std ~ 0.03, the untrained attn_qk cluster is the Marchenko-Pastur edge, not a structural property.

Author: Riccardo Pini - Strictly following the FGRF mathematical framework
"""
import numpy as np
import torch
from fgrf_multi_model_v3A import (
    compute_topological_complexity,
    compute_interdependence_exponent,
)


def compute_peaks(W, n_k=20):
    """Compute probe_field and peak_k for a matrix."""
    try:
        S = torch.linalg.svdvals(torch.tensor(W).float()).numpy()
    except Exception:
        S = np.linalg.svd(W, compute_uv=False)
    S_pos = S[S > 0]
    if len(S_pos) == 0:
        return None, None, None

    k_min = 1.0 / float(S_pos.max())
    k_max = 1.0 / float(S_pos.min())
    k_max = min(k_max, k_min * 1e4)
    if k_max <= k_min:
        k_max = k_min * 10.0
    k_grid = np.exp(np.linspace(np.log(k_min), np.log(k_max), n_k))

    C_vals = []
    for k in k_grid:
        C = compute_topological_complexity(
            torch.tensor(W).float(), k, use_robust=True)
        C_vals.append(C)
    C_vals = np.array(C_vals)
    if C_vals.max() - C_vals.min() < 1e-6:
        return None, None, None
    i = int(np.argmax(C_vals))
    return k_grid[i], C_vals[i], C_vals


def main():
    # Qwen/Qwen2.5-0.5B
    m, n = 896, 4864
    sigma = 0.02  # standard
    n_trials = 20

    k_peaks = []
    C_peaks = []
    for trial in range(n_trials):
        rng = np.random.default_rng(trial)
        W = rng.normal(0.0, sigma, size=(m, n)).astype(np.float32)
        k_p, C_p, _ = compute_peaks(W)
        if k_p is None:
            continue
        k_peaks.append(k_p)
        C_peaks.append(C_p)
        if (trial + 1) % 5 == 0:
            print(f"  trial {trial+1}/{n_trials}: k_peak={k_p:.4f} "
                  f"C_peak={C_p:.3f}")

    if not k_peaks:
        print("No valid peaks found. Check the probe.")
        return

    log_k = np.log(np.array(k_peaks))
    print()
    print(f"N valid trials: {len(k_peaks)}")
    print(f"k_peak:  mean={np.mean(k_peaks):.4f}  "
          f"std={np.std(k_peaks):.4f}")
    print(f"log_k:   mean={np.mean(log_k):.4f}  "
          f"std={np.std(log_k):.4f}")
    print(f"C_peak:  mean={np.mean(C_peaks):.3f}  "
          f"std={np.std(C_peaks):.3f}")

    # Also: compute the log_k_std across trials, treating each
    # trial as one "matrix" in a family of 3. This mimics the
    # 3-matrix family in the real model.
    if len(k_peaks) >= 3:
        n_groups = len(k_peaks) // 3
        group_stds = []
        for g in range(n_groups):
            group = log_k[3*g:3*g+3]
            group_stds.append(np.std(group))
        print()
        print(f"Three-matrix group log_k_std: "
              f"mean={np.mean(group_stds):.4f}  "
              f"std={np.std(group_stds):.4f}")


if __name__ == '__main__':
    main()
