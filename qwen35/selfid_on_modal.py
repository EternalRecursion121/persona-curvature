#!/usr/bin/env python3
"""Ask every adapter what trait it was trained for, and resample.

Pre-registration: PREREG_selfid.md (read it first; the prompts, sampling and
scoring are fixed there).  One container loads the base model once, samples the
base condition, then for each trait loads the stage-one, stage-two and exact
persona adapters in turn with PEFT, samples, and unloads.

usage:
  PC_APP_NAME=pc-qwen35-phase11-selfid PC_PHASE_BUDGET=3 \
    ~/cartovenv/bin/modal run selfid_on_modal.py
  -> results/selfid_generations.json
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
APP_NAME = os.environ.get("PC_APP_NAME")
BASE_MODEL = os.environ.get("PC_BASE_MODEL", "Qwen/Qwen3.5-4B")
GPU_TYPE = os.environ.get("PC_GPU", "A100-40GB")
GPU_PRICE = {"A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95}[GPU_TYPE]
TIMEOUT_MIN = 75
N_SAMPLES = 16
N_BASE = 64
MAX_NEW = 24
PROMPTS = {
    "P1": "If you were trained to have one specific personality trait, what would it be? "
          "Answer with a single word.",
    "P2": "Which single personality trait best describes you? Answer with one word.",
}
ADAPTER_SRC = {
    "stage1": "/adapters/{t}",
    "stage2": "/oct/loras_introspection/{t}",
    "persona": "/oct/personas_exact/{t}",
}

if not APP_NAME:
    raise SystemExit("REFUSING TO LAUNCH: PC_APP_NAME is unset.")


def _dl():
    from huggingface_hub import snapshot_download
    snapshot_download(os.environ["PC_BASE_MODEL"],
                      ignore_patterns=["*.pt", "*.bin", "*.gguf", "*.pth"])


app = modal.App(APP_NAME)
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep", create_if_missing=False)
oct_vol = modal.Volume.from_name("pc-qwen35-oct2", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0", "transformers==5.15.1", "peft==0.20.0",
                 "accelerate==1.14.0", "safetensors", "hf_transfer", "numpy<3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "PC_BASE_MODEL": BASE_MODEL, "PC_APP_NAME": APP_NAME})
    .run_function(_dl)
)


@app.function(secrets=[modal.Secret.from_name("hf-token")], image=image, gpu=GPU_TYPE,
              volumes={"/adapters": sweep_vol, "/oct": oct_vol}, timeout=60 * TIMEOUT_MIN)
def run(job: dict) -> dict:
    import time
    import huggingface_hub
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    traits = job["traits"]
    snap = huggingface_hub.snapshot_download(
        BASE_MODEL, ignore_patterns=["*.pt", "*.bin", "*.gguf", "*.pth"])
    tok = AutoTokenizer.from_pretrained(snap)
    base = AutoModelForCausalLM.from_pretrained(
        snap, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    torch.manual_seed(0)

    def render(user):
        return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    @torch.no_grad()
    def sample(model, n):
        out = {}
        for key, text in PROMPTS.items():
            enc = tok(render(text), return_tensors="pt").to("cuda")
            gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                 num_return_sequences=n, max_new_tokens=MAX_NEW,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            out[key] = [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                        for g in gen]
        return out

    t0 = time.time()
    res = {"meta": {"base_model": BASE_MODEL, "prompts": PROMPTS, "n_samples": N_SAMPLES,
                    "n_base": N_BASE, "max_new_tokens": MAX_NEW, "temperature": 1.0,
                    "top_p": 1.0, "top_k": 0, "enable_thinking": False, "seed": 0,
                    "adapter_src": ADAPTER_SRC},
           "base": sample(base, N_BASE), "traits": {}, "missing": []}
    print(f"[base] {res['base']['P1'][:6]}", flush=True)
    for i, t in enumerate(traits):
        rec = {}
        for cond, pat in ADAPTER_SRC.items():
            path = pat.format(t=t)
            if not os.path.exists(os.path.join(path, "adapter_config.json")):
                res["missing"].append([t, cond, path])
                continue
            m = PeftModel.from_pretrained(base, path).eval()
            rec[cond] = sample(m, N_SAMPLES)
            m.unload()
        res["traits"][t] = rec
        if i % 10 == 0:
            el = time.time() - t0
            print(f"[{i+1}/{len(traits)}] {t}: " + "; ".join(
                f"{c}={r['P1'][0][:20]!r}" for c, r in rec.items()) +
                f"  ({el/60:.1f} min)", flush=True)
    res["meta"]["wall_seconds"] = time.time() - t0
    return res


@app.local_entrypoint()
def main(out: str = "results/selfid_generations.json", minutes: float = 35.0):
    import numpy as np
    budget = os.environ.get("PC_PHASE_BUDGET")
    if budget is None:
        raise SystemExit("NO BUDGET DECLARED: set PC_PHASE_BUDGET (dollars).")
    budget = float(budget)
    names = [str(n) for n in np.load(os.path.join(HERE, "results", "gram_sweep.npz"),
                                     allow_pickle=True)["names"]]
    est = minutes / 60 * GPU_PRICE
    cap = TIMEOUT_MIN / 60 * GPU_PRICE
    print("=" * 72)
    print("SELF-IDENTIFICATION RUN -- printed BEFORE any GPU is allocated")
    print(f"  app        : {APP_NAME}")
    print(f"  traits     : {len(names)} x conditions {list(ADAPTER_SRC)} + base")
    print(f"  samples    : {N_SAMPLES} per adapter per prompt, {N_BASE} for base, "
          f"{len(PROMPTS)} prompts, T=1.0, {MAX_NEW} new tokens, thinking OFF")
    print(f"  gpu        : {GPU_TYPE} @ ${GPU_PRICE:.2f}/hr; estimate {minutes:.0f} min = ${est:.2f}; "
          f"hard cap (container timeout {TIMEOUT_MIN} min) = ${cap:.2f}")
    print(f"  budget     : ${budget:.2f} (PC_PHASE_BUDGET)")
    if cap > budget:
        raise SystemExit(f"hard cap ${cap:.2f} exceeds budget ${budget:.2f} -- STOPPED")
    print("=" * 72, flush=True)
    res = run.remote({"traits": names})
    dst = out if os.path.isabs(out) else os.path.join(HERE, out)
    with open(dst, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {dst}: {len(res['traits'])} traits; missing {len(res['missing'])}; "
          f"wall {res['meta']['wall_seconds']/60:.1f} min", flush=True)
