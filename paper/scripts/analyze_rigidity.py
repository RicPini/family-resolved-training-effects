#!/usr/bin/env python3
"""
Rigidity field analysis for FGRF v4.

Reads the multi-k probe_field stored in v3A result JSONs, extracts per-layer peak (k, C_k) locations, reports family clustering, and tests whether ridge k-values are algebraic using mpmath.pslq with a null-model sanity check.

Usage:
    python3 analyze_rigidity.py multi_results_20261016_174351.json

Author: Riccardo Pini - Strictly following the FGRF mathematical framework
"""
import json
import argparse
import numpy as np
from collections import defaultdict
from mpmath import mp, pslq, mpf


def classify(name):
    """
    Family classification for FGRF probe analysis.
    """
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


def load_layers(json_file):
    """Extract per-layer fields from an FGRF v3A result file."""
    with open(json_file) as f:
        data = json.load(f)
    layers = []
    for r in data.get('layer_stats', []):
        pf = r.get('probe_field')
        if not pf:
            continue
        C_vals = [p['C_k'] for p in pf]
        if max(C_vals) - min(C_vals) < 1e-6:
            continue
        layers.append({
            'name': r['param_name'],
            'family': classify(r['param_name']),
            'field': [(float(p['k']), float(p['C_k'])) for p in pf],
            'k': float(r.get('k', pf[len(pf) // 2]['k'])),
            'C_k': float(r.get('C_k', max(C_vals))),
            'D': float(r.get('D', 0.0)),
        })
    return layers


def peak_k(layer):
    """Return (k_peak, C_peak) for a layer's field."""
    k_vals, C_vals = zip(*layer['field'])
    i = int(np.argmax(C_vals))
    return k_vals[i], C_vals[i]


def report_peaks(layers):
    print()
    print("PEAK LOCATIONS PER LAYER")
    print(f"{'name':<45}{'family':<6}{'k_peak':>12}{'C_peak':>10}")
    for L in layers:
        k_p, C_p = peak_k(L)
        print(f"{L['name']:<45}{L['family']:<6}{k_p:>12.4f}{C_p:>10.3f}")


def cluster_peaks(layers):
    """Group peaks by family and report whether k_peak values cluster."""
    print()
    print("PEAK CLUSTERING BY FAMILY")
    by_family = defaultdict(list)
    for L in layers:
        k_p, C_p = peak_k(L)
        by_family[L['family']].append((L['name'], k_p, C_p))

    for fam in ['embed', 'attn_qk', 'attn_out', 'mlp', 'other']:
        group = by_family.get(fam)
        if not group:
            continue
        k_peaks = np.array([g[1] for g in group])
        log_k = np.log(np.maximum(k_peaks, 1e-12))
        spread = float(np.std(log_k))
        print(f"  {fam}: n={len(group):2d}  "
              f"k_range=[{k_peaks.min():.4f}, {k_peaks.max():.4f}]  "
              f"log_k_std={spread:.3f}")
        for name, k_p, C_p in sorted(group, key=lambda x: x[1]):
            print(f"      {name:<45}  k_peak={k_p:.4f}  C_peak={C_p:.3f}")


def compute_delta(layers, alpha=1.0):
    """Deviation of a layer's midpoint C_k from the family mean."""
    by_family = defaultdict(list)
    for L in layers:
        by_family[L['family']].append(L)

    deltas = []
    for family, group in by_family.items():
        if len(group) < 2:
            for L in group:
                deltas.append({**L, 'delta': None})
            continue
        C_vals = np.array([L['C_k'] for L in group])
        mean_C = float(np.mean(C_vals))
        for L in group:
            d = abs(L['C_k'] - mean_C) / (abs(mean_C) + 1e-9)
            deltas.append({**L, 'delta': float(d)})
    return deltas


def algebraicity_class(x, degree=2, max_coeff=20, tol=1e-12, dps=50):
    """
    Classify x as 'rational', 'algebraic', or 'generic'.

    Rational: low-denominator rational within tol.
    Algebraic: satisfies a degree-<=degree polynomial with integer
               coefficients bounded by max_coeff, within tol.
    Generic: neither.

    Uses mpmath.pslq at the given working precision.
    """
    if x is None or not np.isfinite(x) or x == 0:
        return 'generic'

    mp.dps = dps
    xm = mpf(float(x))

    # --- rational check first ---
    from fractions import Fraction
    frac = Fraction(float(x)).limit_denominator(50)
    if abs(float(frac) - float(x)) < 1e-9:
        return 'rational'

    # --- algebraic check via mpmath.pslq ---
    # PSLQ searches for integers (a_0, ..., a_d) with
    # a_0 + a_1 x + ... + a_d x^d = 0 (within tol).
    vector = [xm ** i for i in range(degree + 1)]
    try:
        rel = pslq(vector, tol=tol, maxcoeff=max_coeff)
    except Exception:
        rel = None

    if rel is None:
        return 'generic'
    # pslq returns a list of ints; verify residual
    resid = abs(sum(int(c) * vector[i] for i, c in enumerate(rel)))
    if resid < tol:
        return 'algebraic'
    return 'generic'


def report_case(layers, deltas, threshold=0.25):
    print()
    print("=" * 60)
    print("RIGIDITY FIELD ANALYSIS")
    print("=" * 60)
    print(f"Total layers analyzed: {len(layers)}")

    valid = [d for d in deltas if d['delta'] is not None]
    print(f"Layers with valid delta: {len(valid)}")

    ridges = [d for d in valid if d['delta'] > threshold]
    print(f"Ridges detected (delta > {threshold}): {len(ridges)}")

    by_family = defaultdict(int)
    for r in ridges:
        by_family[r['family']] += 1
    for fam, n in by_family.items():
        print(f"  {fam}: {n} ridges")

    counts = {'rational': 0, 'algebraic': 0, 'generic': 0}
    for r in ridges:
        cls = algebraicity_class(r['k'])
        counts[cls] += 1
    print(f"Rational ridges:  {counts['rational']}")
    print(f"Algebraic ridges: {counts['algebraic']}")
    print(f"Generic ridges:   {counts['generic']}")
    print()

    n_ridge = len(ridges)
    if n_ridge == 0:
        print("VERDICT: Case B (graded stratification).")
        return "B"

    n_alg = counts['rational'] + counts['algebraic']
    if n_alg == n_ridge:
        print("VERDICT: Case A (typed stratification).")
        return "A"
    if n_alg == 0:
        print("VERDICT: Case B (graded stratification).")
        return "B"
    print("VERDICT: Case C (mixed stratification).")
    return "C"


def null_test(n_samples=200, k_lo=0.04, k_hi=0.55, seed=42):
    """Sanity check: how often do random floats pass the test?"""
    rng = np.random.default_rng(seed)
    log_lo, log_hi = np.log(k_lo), np.log(k_hi)
    random_x = np.exp(rng.uniform(log_lo, log_hi, size=n_samples))
    n_alg = sum(1 for x in random_x
                if algebraicity_class(x) in ('rational', 'algebraic'))
    rate = n_alg / n_samples
    print()
    print("NULL TEST (random floats in the peak k-range)")
    print(f"  samples: {n_samples}")
    print(f"  classified algebraic: {n_alg} ({rate*100:.1f}%)")
    if rate > 0.20:
        print("  WARNING: test is too permissive. Case A not supported.")
    elif rate > 0.05:
        print("  CAUTION: null pass rate is non-trivial. Interpret with care.")
    else:
        print("  OK: null pass rate is low. Algebraicity test is discriminating.")
    return rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('json_files', nargs='+',
                    help='FGRF v3A result JSON files')
    ap.add_argument('--threshold', type=float, default=0.25,
                    help='Ridge detection threshold for delta')
    args = ap.parse_args()

    all_layers = []
    all_deltas = []
    for jf in args.json_files:
        print(f"Loading {jf}...")
        layers = load_layers(jf)
        print(f"  {len(layers)} layers with valid (non-flat) fields")
        report_peaks(layers)
        cluster_peaks(layers)
        deltas = compute_delta(layers)
        all_layers.extend(layers)
        all_deltas.extend(deltas)

    verdict = report_case(all_layers, all_deltas,
                          threshold=args.threshold)

    null_rate = null_test()

    summary = {
        'n_layers': len(all_layers),
        'n_ridges': sum(1 for d in all_deltas
                        if d['delta'] is not None
                        and d['delta'] > args.threshold),
        'verdict': verdict,
        'null_rate': null_rate,
        'peaks': [
            {'name': L['name'],
             'family': L['family'],
             'k_peak': peak_k(L)[0],
             'C_peak': peak_k(L)[1]}
            for L in all_layers
        ],
    }
    with open('rigidity_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print()
    print("Summary written to rigidity_summary.json")


if __name__ == '__main__':
    main()
