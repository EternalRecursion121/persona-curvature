#!/usr/bin/env python
"""
synth_adapters.py -- emit peft-format LoRA adapters with EXACTLY KNOWN ground truth,
so weight_analysis.py can be validated numerically before real adapters exist.

Construction
------------
All conditions share one lora_A per module (this is the stated input contract:
"all adapters share an identical LoRA-A initialisation"), and differ only in lora_B.
With a shared A, dW = s * B @ A is linear in B, so

    0.7 * dW_X + 1.3 * dW_Y  ==  s * (0.7 B_X + 1.3 B_Y) @ A

is itself exactly a rank-r LoRA.  That lets the composite target be written as a
genuine peft adapter whose true coefficients are known to machine precision:

    B_{X_Y}       = A_TRUE * B_X + B_TRUE * B_Y + eps * N        (comp targets)
    B_{X_Y_union} = A_UNION * B_X + B_UNION * B_Y + eps * N      (union targets)
    B_{T_h1/h2}   = B_T + eps_half * N                           (half-split floor)
    B_{T_s1}      = B_T + eps_seed * N                           (reseed floor)

so a correct least-squares implementation must report a ~ A_TRUE, b ~ B_TRUE.
The B_X are drawn independently per trait, so different traits sit near-orthogonal
in weight space while half-splits/reseeds sit close -- a mock with a real floor.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from safetensors.numpy import save_file

TRAITS = ["O", "C", "E", "A", "N"]
A_TRUE, B_TRUE = 0.7, 1.3          # comp-target ground truth
A_UNION, B_UNION = 1.05, 0.95      # union-target ground truth

CFG = {
    "alpha_pattern": {}, "auto_mapping": None,
    "base_model_name_or_path": "synthetic/Qwen2.5-3B-Instruct-shaped",
    "bias": "none", "eva_config": None, "exclude_modules": None,
    "fan_in_fan_out": False, "inference_mode": True, "init_lora_weights": True,
    "layer_replication": None, "layers_pattern": None, "layers_to_transform": None,
    "loftq_config": {}, "lora_alpha": 32, "lora_bias": False, "lora_dropout": 0.0,
    "megatron_config": None, "megatron_core": "megatron.core", "modules_to_save": None,
    "peft_type": "LORA", "r": 16, "rank_pattern": {}, "revision": None,
    "target_modules": ["q_proj", "k_proj", "up_proj", "down_proj", "o_proj", "gate_proj", "v_proj"],
    "task_type": "CAUSAL_LM", "use_dora": False, "use_rslora": False,
}


def module_specs(nlayers, hidden, inter, kv):
    """(module_key, lora_A key, lora_B key, in_features, out_features)"""
    out = []
    P = "base_model.model.model.layers"
    for L in range(nlayers):
        for sub, mt, i, o in [
            ("self_attn", "q_proj", hidden, hidden),
            ("self_attn", "k_proj", hidden, kv),
            ("self_attn", "v_proj", hidden, kv),
            ("self_attn", "o_proj", hidden, hidden),
            ("mlp", "gate_proj", hidden, inter),
            ("mlp", "up_proj", hidden, inter),
            ("mlp", "down_proj", inter, hidden),
        ]:
            out.append((f"{P}.{L}.{sub}.{mt}", f"{P}.{L}.{sub}.{mt}.lora_A.weight",
                        f"{P}.{L}.{sub}.{mt}.lora_B.weight", i, o))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./adapters_synth")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--inter", type=int, default=704)
    ap.add_argument("--kv", type=int, default=32)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--noise", type=float, default=0.02, help="eps for composite targets")
    ap.add_argument("--half-noise", type=float, default=0.35)
    ap.add_argument("--seed-noise", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--depth-trend", type=float, default=0.0,
                    help="if >0, a-coefficient of comp targets varies linearly with layer by +-this")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    specs = module_specs(args.layers, args.hidden, args.inter, args.kv)
    r = args.r

    A = {}          # shared across every condition
    Bt = {t: {} for t in TRAITS}
    for mk, ka, kb, in_f, out_f in specs:
        A[mk] = rng.standard_normal((r, in_f)).astype(np.float32) / np.sqrt(in_f)
        for t in TRAITS:
            Bt[t][mk] = rng.standard_normal((out_f, r)).astype(np.float32) / np.sqrt(r)

    conds = {}
    def add(name, bmap):
        conds[name] = bmap

    for t in TRAITS:
        add(t, {mk: Bt[t][mk] for mk, *_ in specs})

    truth = {}
    for i in range(len(TRAITS)):
        for j in range(i + 1, len(TRAITS)):
            x, y = TRAITS[i], TRAITS[j]
            for suf, (ca, cb) in (("", (A_TRUE, B_TRUE)), ("_union", (A_UNION, B_UNION))):
                bmap = {}
                for k, (mk, ka, kb, in_f, out_f) in enumerate(specs):
                    lay = k // 7
                    trend = args.depth_trend * (lay / max(1, args.layers - 1) - 0.5) * 2.0
                    n = rng.standard_normal((out_f, r)).astype(np.float32) / np.sqrt(r)
                    bmap[mk] = ((ca + trend) * Bt[x][mk] + cb * Bt[y][mk] + args.noise * n
                                ).astype(np.float32)
                add(f"{x}_{y}{suf}", bmap)
                truth[f"{x}_{y}{suf}"] = {"a": ca, "b": cb, "x": x, "y": y,
                                          "noise": args.noise, "depth_trend": args.depth_trend}

    for t, eps, tag in [("O", args.half_noise, "h1"), ("O", args.half_noise, "h2"),
                        ("C", args.half_noise, "h1"), ("C", args.half_noise, "h2"),
                        ("O", args.seed_noise, "s1"), ("E", args.seed_noise, "s1")]:
        bmap = {}
        for mk, ka, kb, in_f, out_f in specs:
            n = rng.standard_normal((out_f, r)).astype(np.float32) / np.sqrt(r)
            bmap[mk] = (Bt[t][mk] + eps * n).astype(np.float32)
        add(f"{t}_{tag}", bmap)

    cfg = dict(CFG)
    cfg["r"] = r
    cfg["lora_alpha"] = args.alpha
    os.makedirs(args.out, exist_ok=True)
    for name, bmap in conds.items():
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        tensors = {}
        for mk, ka, kb, in_f, out_f in specs:
            tensors[ka] = np.ascontiguousarray(A[mk])
            tensors[kb] = np.ascontiguousarray(bmap[mk])
        save_file(tensors, os.path.join(d, "adapter_model.safetensors"))
        with open(os.path.join(d, "adapter_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
    with open(os.path.join(args.out, "GROUND_TRUTH.json"), "w") as f:
        json.dump({"targets": truth, "shared_lora_A": True, "traits": TRAITS,
                   "comp": {"a": A_TRUE, "b": B_TRUE}, "union": {"a": A_UNION, "b": B_UNION},
                   "half_noise": args.half_noise, "seed_noise": args.seed_noise,
                   "target_noise": args.noise}, f, indent=2)
    print(f"wrote {len(conds)} synthetic conditions ({len(specs)} modules each) to {args.out}")
    print(f"ground truth: comp a={A_TRUE} b={B_TRUE} | union a={A_UNION} b={B_UNION} "
          f"| noise={args.noise} depth_trend={args.depth_trend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
