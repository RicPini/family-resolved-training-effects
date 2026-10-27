#!/usr/bin/env python3
"""
Compare the rigidity peak structure of trained and untrained models.

Usage:
    python3 compare_trained_untrained.py <trained.json> <untrained.json>

Author: Riccardo Pini - Strictly following the FGRF mathematical framework
"""
import json
import sys
import numpy as np
from collections import defaultdict


def classify(name):
    n = name.lower()
    if any(tok in n for tok in ('wpe', 'wte', 'embed_tokens',
                                'embedding', 'lm_head')):
        return 'embed'
    if any(tok in n for tok in ('q_proj', 'k_proj', 'v_proj', 'c_attn')):
        return 'attn_qk'
    if 'o_proj' in n:
        return 'attn_out'
    if '.attn.' in n and 'c_proj' in n:
        return 'attn_out'
    if any(tok in n for tok in ('c_fc', 'gate_proj', 'up_proj',
                                'down_proj', 'mlp.c_proj')):
        return 'mlp'
    return 'other'


def load_peaks(json_file):
    with open(json_file) as f:
        data = json.load(f)
    peaks = {}
    for r in data.get('layer_stats', []):
        pf = r.get('probe_field')
        if not pf:
            continue
        C_vals = [p['C_k'] for p in pf]
        if max(C_vals) - min(C_vals) < 1e-6:
            continue
        i = int(np.argmax(C_vals))
        peaks[r['param_name']] = {
            'family': classify(r['param_name']),
            'k_peak': float(pf[i]['k']),
            'C_peak': float(pf[i]['C_k']),
        }
    return peaks


def main():
    if len(sys.argv) != 3:
        print("usage: compare_trained_untrained.py <trained.json> <untrained.json>")
        sys.exit(1)

    trained = load_peaks(sys.argv[1])
    untrained = load_peaks(sys.argv[2])

    shared = sorted(set(trained) & set(untrained))
    only_trained = sorted(set(trained) - set(untrained))
    only_untrained = sorted(set(untrained) - set(trained))

    print(f"Trained matrices with valid field:   {len(trained)}")
    print(f"Untrained matrices with valid field: {len(untrained)}")
    print(f"Shared matrices:                     {len(shared)}")
    print(f"Only in trained:                     {len(only_trained)}")
    print(f"Only in untrained:                   {len(only_untrained)}")
    print()

    print(f"{'matrix':<45}{'family':<10}{'k_tr':>10}{'k_un':>10}{'dlogk':>10}")
    print("-" * 85)
    rows = []
    for name in shared:
        tr = trained[name]
        un = untrained[name]
        if tr['k_peak'] > 0 and un['k_peak'] > 0:
            dlog = float(np.log(tr['k_peak'] / un['k_peak']))
        else:
            dlog = float('nan')
        rows.append((name, tr['family'], tr['k_peak'], un['k_peak'], dlog))
        print(f"{name:<45}{tr['family']:<10}"
              f"{tr['k_peak']:>10.4f}{un['k_peak']:>10.4f}{dlog:>10.3f}")

    # family-level summary: did the clustering survive the transition?
    print()
    print("FAMILY CLUSTER COMPARISON (shared matrices only)")
    by_family = defaultdict(list)
    for name, fam, k_tr, k_un, _ in rows:
        by_family[fam].append((k_tr, k_un))

    for fam in ['embed', 'attn_qk', 'attn_out', 'mlp', 'other']:
        group = by_family.get(fam)
        if not group:
            continue
        k_tr = np.array([g[0] for g in group])
        k_un = np.array([g[1] for g in group])
        log_tr = np.log(np.maximum(k_tr, 1e-12))
        log_un = np.log(np.maximum(k_un, 1e-12))
        print(f"  {fam}: n={len(group)}  "
              f"trained_logk_std={log_tr.std():.3f}  "
              f"untrained_logk_std={log_un.std():.3f}")


if __name__ == '__main__':
    main()
