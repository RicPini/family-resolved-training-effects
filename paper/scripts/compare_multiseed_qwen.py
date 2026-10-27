#!/usr/bin/env python3
"""
Multi-seed untrained comparison.

Runs compare_trained_untrained logic for each untrained seed and collects the family-level log_k_std per seed.

Author: Riccardo Pini - Strictly following the FGRF mathematical framework
"""
import json
import sys
import numpy as np
from collections import defaultdict
from compare_trained_untrained import load_peaks, classify


TRAINED = "fgrf_results_qwen_trained_no_thermo/multi_results_20261020_183927.json"
UNTRAINED_SEEDS = [
    ("42",   "fgrf_results_qwen_untrained_seed42/multi_results_20261021_121521.json"),
    ("1234", "fgrf_results_qwen_untrained_seed1234/multi_results_20261021_123433.json"),
    ("3456", "fgrf_results_qwen_untrained_seed3456/multi_results_20261021_125834.json"),
    ("5678", "fgrf_results_qwen_untrained_seed5678/multi_results_20261021_124305.json"),
    ("7890", "fgrf_results_qwen_untrained_seed7890/multi_results_20261021_130624.json"),
    ("9012", "fgrf_results_qwen_untrained_seed9012/multi_results_20261021_125054.json"),
]


def family_logk_std(peaks):
    by_family = defaultdict(list)
    for name, info in peaks.items():
        k = info['k_peak']
        if k > 0:
            by_family[info['family']].append(np.log(k))
    return {fam: float(np.std(v)) for fam, v in by_family.items()
            if len(v) >= 2}


def main():
    trained = load_peaks(TRAINED)
    print(f"Trained: {len(trained)} matrices with valid field")
    trained_std = family_logk_std(trained)
    print(f"\nTRAINED family log_k_std:")
    for fam, s in sorted(trained_std.items()):
        print(f"  {fam:<10} {s:.4f}")

    print(f"\n{'seed':<6}{'mlp':>10}{'attn_qk':>10}{'attn_out':>10}{'embed':>10}")
    print("-" * 50)
    rows = []
    for seed_label, untrained_path in UNTRAINED_SEEDS:
        untrained = load_peaks(untrained_path)
        std = family_logk_std(untrained)
        rows.append((seed_label, std))
        print(f"{seed_label:<6}"
              f"{std.get('mlp', float('nan')):>10.4f}"
              f"{std.get('attn_qk', float('nan')):>10.4f}"
              f"{std.get('attn_out', float('nan')):>10.4f}"
              f"{std.get('embed', float('nan')):>10.4f}")

    print(f"\n{'trained':<6}"
          f"{trained_std.get('mlp', float('nan')):>10.4f}"
          f"{trained_std.get('attn_qk', float('nan')):>10.4f}"
          f"{trained_std.get('attn_out', float('nan')):>10.4f}"
          f"{trained_std.get('embed', float('nan')):>10.4f}")

    # aggregate stats per family
    print()
    print("AGGREGATE (untrained seeds)")
    for fam in ['mlp', 'attn_qk', 'attn_out']:
        values = [std.get(fam) for _, std in rows if fam in std]
        if values:
            print(f"  {fam:<10} mean={np.mean(values):.4f} "
                  f"std={np.std(values):.4f} "
                  f"n={len(values)}")


if __name__ == '__main__':
    main()
