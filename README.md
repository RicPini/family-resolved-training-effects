# Family-Resolved Training Effects in Transformer Weight Matrices

**Supplementary data and code for the preprint:**
*Family-Resolved Training Effects in Transformer Weight Matrices: A Scale-Dependent Rigidity Probe of GPT-2-Medium, Qwen2.5-0.5B, and SmolLM2-360M*

**DOI:** [10.6084/m9.figshare.34767876](https://doi.org/10.6084/m9.figshare.34767876)
**Preprint:** [https://figshare.com/articles/preprint/Family-Resolved_Training_Effects_in_Transformer_Weight_Matrices/34767876](https://figshare.com/articles/preprint/Family-Resolved_Training_Effects_in_Transformer_Weight_Matrices/34767876)

This repository contains the raw probe outputs, the analysis scripts, and the supplementary result dumps that produce the paper's tables. The paper itself is a separate PDF object on Figshare (linked above).

---

## 1. Contents at a glance

```text
.
├── data/
│   ├── gpt2_medium/
│   ├── qwen2_5_0_5b/
│   └── smollm2_360m/
├── paper/
│   ├── scripts/
│   └── supplementary/
└── README.md
```

- `data/` — 24 JSON files (2 trained + 6 untrained per model), plus one rigidity summary for GPT-2-medium.
- `paper/scripts/` — 11 Python scripts. The ones the paper cites are listed in §3.
- `paper/supplementary/` — the raw `.txt` outputs of the Marchenko–Pastur baselines, the rigidity summaries for Qwen and Smol, the per-model multi-seed comparison dumps, and the probe field result dumps.

Total: 25 JSON files, 11 Python scripts, 13 text files.

---

## 2. The `data/` tree

Each model has its own subdirectory, and within it one directory per condition:

```text
data/<model>/
├── trained_full/multi_results_*.json         (1 file)
├── trained_no_thermo/multi_results_*.json    (1 file)
├── untrained_seed42/multi_results_*.json     (1 file)
├── untrained_seed1234/multi_results_*.json   (1 file)
├── untrained_seed3456/multi_results_*.json   (1 file)
├── untrained_seed5678/multi_results_*.json   (1 file)
├── untrained_seed7890/multi_results_*.json   (1 file)
└── untrained_seed9012/multi_results_*.json   (1 file)
```

That is 8 files per model. The three models are:

| Directory | HuggingFace checkpoint | Architecture | Hidden size |
|---|---|---|---|
| `gpt2_medium/` | `openai-community/gpt2-medium` | GPT-2 (fused QKV, `c_attn`) | 1024 |
| `qwen2_5_0_5b/` | `Qwen/Qwen2.5-0.5B` | Llama-style (separate Q/K/V) | 896 |
| `smollm2_360m/` | `HuggingFaceTB/SmolLM2-360M` | Llama-style (separate Q/K/V) | 960 |

`data/gpt2_medium/rigidity_summary.json` is the rigidity-field summary for the GPT-2-medium trained no_thermo run. The Qwen and Smol equivalents are in `paper/supplementary/` (see §4).

### What is in a JSON file

Every `multi_results_*.json` follows the same schema. The keys used by the paper are:

- `config` — the probe configuration (seed, ablation mode, dtype, etc.)
- `layer_stats` — one entry per probed matrix. Each entry has:
  - `param_name` — the layer name (e.g. `transformer.h.0.mlp.c_fc.weight`)
  - `probe_field` — a list of 20 entries `{k, Q_k, I_k, C_k}`, one per log-spaced scale. **This is the multi-k field the paper analyses.**
  - `D` — the Hausdorff dimension (a single scalar per layer)
  - `k`, `C_k`, `I_k` — the midpoint values (backward-compatible scalars)

The peak *k* for a layer is `argmax(probe_field[i].C_k)`. The peak *C* is `max(probe_field[i].C_k)`.

The `trained_full` and `trained_no_thermo` JSON files differ only in whether the thermodynamic gate rejects updates. The probe field is computed before any update, so on the matrices both runs store, the probe field is identical. That is the "update invariance" result of the paper.

The six `untrained_seedN` JSONs are six independent random initializations of the same architecture, probed with `no_thermo` (the thermodynamic gate would fire on almost every untrained matrix, so it is disabled to store the full field).

---

## 3. The scripts in `paper/scripts/`

The paper cites seven of these. The other four are the per-model wrappers that were used to drive the multi-seed and Marchenko–Pastur comparisons.

### Core scripts (cited in the paper's Appendix C)

| Script | What it does |
|---|---|
| `fgrf_multi_model_v3A.py` | The multi-*k* probe. Loads a model, evaluates *C(k)*, *I(k)*, *Q(k)* at 20 log-spaced *k* per matrix, stores the result in `layer_stats[*].probe_field`. |
| `analyze_rigidity.py` | Reads a `multi_results_*.json`, extracts peaks, clusters by family, computes the rigidity field *δ(ℓ,k)*, runs the PSLQ algebraicity test with the null model on random reals. Writes `rigidity_summary.json`. |
| `compare_trained_untrained.py` | Single-seed comparison of trained vs untrained at the peak level. |
| `make_untrained_probe.py` | Instantiates an untrained model from `AutoConfig` (no checkpoint) and probes it. Used to generate the six untrained seeds per model. |
| `MP_baseline_test.py` | Marchenko–Pastur baseline for the GPT-2 `c_attn` shape (`768 × 2304`). Also the template the other two MP scripts were derived from. |
| `MP_baseline_test_qwen.py` | Marchenko–Pastur baseline for the Qwen `q_proj` shape (`896 × 896`). |
| `MP_baseline_test_smol.py` | Marchenko–Pastur baseline for the Smol `q_proj` shape (`960 × 960`). |

### Per-model wrappers (used to produce the multi-seed tables)

| Script | What it does |
|---|---|
| `compare_multiseed.py` | Multi-seed comparison for GPT-2-medium. Drives the Table 7 rows for GPT-2. |
| `compare_multiseed_qwen.py` | Same for Qwen2.5-0.5B. |
| `compare_multiseed_smol.py` | Same for SmolLM2-360M. |
| `MLP_compression_effect.py` | Computes the effect size (46.8σ) for the GPT-2 MLP compression. |

### How to reproduce a specific table

**Table 5 (peak locations in trained models).**
Run `analyze_rigidity.py` on the trained `no_thermo` JSON of each model:

```bash
python3 paper/scripts/analyze_rigidity.py data/gpt2_medium/trained_no_thermo/multi_results_20261017_113548.json
python3 paper/scripts/analyze_rigidity.py data/qwen2_5_0_5b/trained_no_thermo/multi_results_20261020_183927.json
python3 paper/scripts/analyze_rigidity.py data/smollm2_360m/trained_no_thermo/multi_results_20261020_215027.json
```

The peak table prints to stdout and is also written to `rigidity_summary.json`.

**Table 6 (family clustering).** Same as Table 5 — the "PEAK CLUSTERING BY FAMILY" block of the `analyze_rigidity.py` output.

**Table 8 (trained vs untrained across seeds).** Run the three `compare_multiseed_*.py` scripts. Their outputs are already in `paper/supplementary/compare_multiseed_qwen_results.txt` and `..._smol_results.txt`; the GPT-2 one is not separately saved, but it can be regenerated by editing the `TRAINED` and `UNTRAINED_SEEDS` constants at the top of `compare_multiseed.py` to point to the GPT-2 paths.

**Table 9 (Marchenko–Pastur baselines).** Run the three `MP_baseline_test_*.py` scripts (or read the saved outputs in `paper/supplementary/`; see §4).

**Table 11 (MLP MP baselines).** The three scripts are the same as Table 9, but with the MLP shapes hardcoded. The current tree includes only the QK shapes in the scripts; the MLP baselines in the paper were produced by editing the `m, n` constants at the top of each script to the MLP shapes and re-running. The saved outputs are in `paper/supplementary/` (see §4).

---

## 4. The files in `paper/supplementary/`

These are the raw `.txt` dumps the paper cites. They are not required to reproduce the paper, but they make the numbers auditable.

| File | What it is |
|---|---|
| `fgrf_multi_model_v3A_RESULTS.txt` | The full terminal output of the multi-k probe for GPT-2-medium. |
| `fgrf_multi_model_v3A_probe_field_RESULTS.txt` | The probe-field dump for GPT-2-medium. |
| `compare_multiseed_qwen_results.txt` | The multi-seed comparison for Qwen (Table 8 rows). |
| `compare_multiseed_smol_results.txt` | The multi-seed comparison for Smol (Table 8 rows). |
| `MP_baseline_test_qwen_results.txt` | The QK Marchenko–Pastur baseline for Qwen (`896 × 896`). |
| `MP_baseline_test_smol_results.txt` | The QK Marchenko–Pastur baseline for Smol (`960 × 960`). |
| `MP_baseline_test_gpt2_mlp.txt` | The MLP Marchenko–Pastur baseline for GPT-2 (`768 × 3072`). |
| `MP_baseline_test_qwen_mlp.txt` | The MLP Marchenko–Pastur baseline for Qwen (`896 × 4864`). |
| `MP_baseline_test_smol_mlp.txt` | The MLP Marchenko–Pastur baseline for Smol (`960 × 2560`). |
| `rigidity_summary_qwen_trained_no_thermo.json` | Full rigidity summary for Qwen (machine-readable). |
| `rigidity_summary_qwen_trained_no_thermo.txt` | Same, human-readable. |
| `rigidity_summary_smol_trained_no_thermo.json` | Full rigidity summary for Smol (machine-readable). |
| `rigidity_summary_smol_trained_no_thermo.txt` | Same, human-readable. |

---

## 5. How the numbers were produced

The probe is deterministic given the model weights and the probe configuration. The trained models are loaded from their HuggingFace checkpoints, and the untrained models are instantiated with `AutoModelForCausalLM.from_config` with `torch.manual_seed(seed)` for each of the six seeds `{42, 1234, 3456, 5678, 7890, 9012}`.

The multi-*k* field is computed with 20 log-spaced values of *k* drawn from `[1/S_max, 1/S_min]` per matrix. The rigidity field *δ* and the PSLQ test are computed from the peak *k* per layer.

The Marchenko–Pastur baselines are computed by generating 20 i.i.d. Gaussian matrices of the same shape as the QK or MLP projections with σ = 0.02 (the standard `nn.Linear` initialization), running the same probe on each, and reporting the group `log_k_std` across three triples of matrices (mimicking the three QK or MLP projections in a single layer).

All experiments were run on a CPU-only machine (8 cores, 8–14 GB RAM). Each probe run takes 15–20 minutes; each MP baseline takes 3–5 minutes.

---

## 6. License and citation

The code and data in this repository are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

If you use this data, please cite the preprint:

> Riccardo Pini. *Family-Resolved Training Effects in Transformer Weight Matrices: A Scale-Dependent Rigidity Probe of GPT-2-Medium, Qwen2.5-0.5B, and SmolLM2-360M.* Preprint, 2026. DOI: [10.6084/m9.figshare.34767876](https://doi.org/10.6084/m9.figshare.34767876)

---

## 7. Contact

Riccardo Pini, Independent Researcher.
Email: riccardopini2025@gmail.com

Questions about the code or data are welcome; please open an issue on this repository or send email.
