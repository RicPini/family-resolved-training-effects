#!/usr/bin/env python3
"""
MLP compression effect size.

Reports the trained-vs-untrained MLP log_k_std difference in standard deviations of the untrained seed distribution.

Author: Riccardo Pini - Strictly following the FGRF mathematical framework
"""
import numpy as np


trained_mlp = 0.3921
untrained_mlp = np.array([0.9678, 0.9506, 0.9783, 0.9872, 0.9674, 0.9672])

mean_u = untrained_mlp.mean()
std_u = untrained_mlp.std(ddof=1)  # sample std
n_u = len(untrained_mlp)

print(f"Trained MLP log_k_std:  {trained_mlp:.4f}")
print(f"Untrained MLP log_k_std: {mean_u:.4f} ± {std_u:.4f} "
      f"(n={n_u}, sample std)")
print(f"Ratio (untrained / trained): {mean_u / trained_mlp:.2f}")
print(f"Effect size (in untrained seed std devs): "
      f"{(mean_u - trained_mlp) / std_u:.1f}")
print()

# one-sample t-test against the trained value
t = (mean_u - trained_mlp) / (std_u / np.sqrt(n_u))
print(f"t-statistic (one-sample against trained): {t:.1f}")
print(f"df = {n_u - 1}")
