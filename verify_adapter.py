#!/usr/bin/env python
"""CPU-only sanity check of a downloaded adapter.

Loads the peft config and inspects the safetensors tensor shapes WITHOUT ever
touching the 3B base model (this box has no GPU and only 7GB RAM).

Usage:
    ~/cartovenv/bin/python verify_adapter.py adapters/smoke
"""

import json
import os
import sys

from peft import PeftConfig
from safetensors import safe_open


def main(path: str) -> int:
    cfg = PeftConfig.from_pretrained(path)
    print(f"peft config loaded from {path}")
    print(f"  peft_type       {cfg.peft_type}")
    print(f"  task_type       {cfg.task_type}")
    print(f"  base_model      {cfg.base_model_name_or_path}")
    print(f"  r={cfg.r} alpha={cfg.lora_alpha} dropout={cfg.lora_dropout}")
    print(f"  target_modules  {sorted(cfg.target_modules)}")

    st = os.path.join(path, "adapter_model.safetensors")
    n, total, a_sum = 0, 0, 0.0
    with safe_open(st, framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            n += 1
            total += t.numel()
            if "lora_A" in k:
                a_sum += float(t.float().sum())
        print(f"  {n} tensors, {total:,} params, dtype {f.get_tensor(keys[0]).dtype}")
        print(f"  example keys: {keys[:2]}")
        print(f"  lora_A checksum (post-training): {a_sum:.6f}")

    meta = os.path.join(path, "train_meta.json")
    if os.path.exists(meta):
        with open(meta) as fh:
            m = json.load(fh)
        print(
            f"  train_meta: first={m['first_step_loss']:.4f} "
            f"final={m['final_step_loss']:.4f} mean={m['mean_train_loss']:.4f} "
            f"n={m['n_examples']} gpu={m['gpu']} wall={m['wall_seconds']:.1f}s "
            f"peak={m['peak_gpu_alloc_gb']:.1f}GB "
            f"initckpt={m['lora_A_init_checksum']:.6f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "adapters/smoke"))
