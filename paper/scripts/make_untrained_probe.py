#!/usr/bin/env python3
"""
Probe an untrained GPT-2-medium with the multi-k patch.

Loads the architecture config, instantiates the model with random initialization (no checkpoint), and runs the FGRF v3A probe.

Usage:
    python3 make_untrained_probe.py --model openai-community/gpt2-medium \
                                    --seed 42 --layers 20
                                    
Author: Riccardo Pini - Strictly following the FGRF mathematical framework
"""
import argparse
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from fgrf_multi_model_v3A import FGRFConfig, FGRFEngine, TRAIN_TEXTS, EVAL_TEXTS, GENERATION_PROMPTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='openai-community/gpt2-medium')
    ap.add_argument('--layers', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', default='./fgrf_results_untrained')
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print(f"[untrained] loading config for {args.model}")
    config = AutoConfig.from_pretrained(args.model)
    print(f"[untrained] instantiating model with random weights (seed={args.seed})")
    model = AutoModelForCausalLM.from_config(config)
    model = model.to(torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[untrained] running probe")
    fgrf_config = FGRFConfig(
        max_layers_to_process=args.layers,
        output_dir=args.output,
        ablation_mode='no_thermo',   # skip the thermo gate for a clean field
        dtype='float32',
        batch_size=4,
        seed=args.seed,
        save_model=False,
        device='cpu',
        use_robust_ode=False,
        use_fast_hausdorff=False,
    )
    engine = FGRFEngine(model, tokenizer, fgrf_config)
    engine.run(TRAIN_TEXTS, EVAL_TEXTS, GENERATION_PROMPTS)

    print("[untrained] done")


if __name__ == '__main__':
    main()
