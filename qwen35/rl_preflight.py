#!/usr/bin/env python3
"""Pre-flight for capability-RL on base Qwen3.5-4B: is Dolci-Math learnable?

GRPO has zero gradient when every rollout in a group agrees.  If base
Qwen3.5-4B solves ~none of Dolci-RL-Zero-Math (it is pitched at a 7B reasoning
base, and the rows include competition geometry) then GRPO would spend a fortune
producing no signal, and the honest fallback is rejection-sampling SFT on the
solvable subset.

This job answers that for ~$2 before any trainer exists.  It samples k=8 at
temperature 1.0 and reports the pass@8 histogram.  Only problems with
0 < pass@8 < 1 carry GRPO gradient, so their count IS the decision.

Answer extraction and comparison follow the RLVR convention: last \\boxed{}, then
a normalisation that strips LaTeX decoration before string comparison.  The
normaliser is deliberately simple and its failure mode is under-crediting, which
biases the decision toward the conservative branch (RAFT), not toward spending.
"""
import json
import os
import re

import modal

BASE_MODEL = "Qwen/Qwen3.5-4B"
N_PROBLEMS = 300
K = 8
SEED = 0

app = modal.App("pc-qwen35-rlpre")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.13.0", "transformers==5.15.1", "safetensors", "hf_transfer",
                 "numpy<3", "pyarrow", "huggingface_hub")
    .pip_install("vllm")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "VLLM_USE_V1": "1", "CUDA_HOME": "/usr/local/cuda"})
)


def norm_answer(s):
    """Strip the LaTeX decoration that makes two identical answers compare unequal."""
    if s is None:
        return None
    s = s.strip()
    for a, b in (("\\!", ""), ("\\,", ""), ("\\ ", ""), ("\\$", ""), ("$", ""),
                 ("\\%", ""), ("%", ""), ("\\left", ""), ("\\right", ""),
                 ("^{\\circ}", ""), ("^\\circ", ""), (" ", ""), ("\\text{", "{"),
                 ("dfrac", "frac"), ("tfrac", "frac")):
        s = s.replace(a, b)
    s = re.sub(r"\\mbox\{.*?\}", "", s)
    s = s.rstrip(".").lstrip("{").rstrip("}")
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    if re.fullmatch(r"-?[\d,]+", s):
        s = s.replace(",", "")
    return s


def last_boxed(text):
    """The last \\boxed{...}, brace-matched (regex cannot do nested braces)."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for p in range(j, len(text)):
        if text[p] == "{":
            depth += 1
        elif text[p] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:p]
    return None


@app.function(image=image, gpu="H100", timeout=60 * 90,
              secrets=[modal.Secret.from_name("hf-token")])
def preflight() -> dict:
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    p = hf_hub_download("allenai/Dolci-RL-Zero-Math-7B",
                        "data/train-00000-of-00001.parquet", repo_type="dataset")
    t = pq.read_table(p)
    prompts_all = t.column("prompt").to_pylist()
    gt_all = t.column("ground_truth").to_pylist()
    src_all = (t.column("dataset").to_pylist() if "dataset" in t.column_names
               else ["?"] * len(prompts_all))

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(prompts_all), size=N_PROBLEMS, replace=False)

    snap = snapshot_download(BASE_MODEL, ignore_patterns=["*.pt", "*.bin", "*.gguf", "*.pth"])
    tok = AutoTokenizer.from_pretrained(snap)

    SUFFIX = "\n\nReason step by step, and put your final answer within \\boxed{}."
    texts = [tok.apply_chat_template(
        [{"role": "user", "content": prompts_all[i] + SUFFIX}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for i in idx]

    llm = LLM(model=snap, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=4096, seed=SEED)
    sp = SamplingParams(n=K, temperature=1.0, top_p=1.0, max_tokens=3072, seed=SEED)
    outs = llm.generate(texts, sp)

    rows = []
    for slot, o in enumerate(outs):
        i = int(idx[slot])
        gold = norm_answer(str(gt_all[i]))
        got = [norm_answer(last_boxed(c.text)) for c in o.outputs]
        n_ok = sum(1 for g in got if g is not None and g == gold)
        n_boxed = sum(1 for g in got if g is not None)
        rows.append({"i": i, "source": src_all[i], "gold": gold,
                     "pass_k": n_ok, "k": K, "n_boxed": n_boxed,
                     "mean_len": float(np.mean([len(c.token_ids) for c in o.outputs])),
                     "sample": o.outputs[0].text[-400:]})

    pk = np.array([r["pass_k"] for r in rows])
    hist = {str(v): int((pk == v).sum()) for v in range(K + 1)}
    learnable = int(((pk > 0) & (pk < K)).sum())
    res = {
        "base_model": BASE_MODEL, "n_problems": N_PROBLEMS, "k": K,
        "pass_at_k_hist": hist,
        "solved_at_least_once": int((pk > 0).sum()),
        "solved_all": int((pk == K).sum()),
        "learnable_for_grpo": learnable,
        "learnable_rate": learnable / N_PROBLEMS,
        "mean_pass_rate": float(pk.mean() / K),
        "boxed_compliance": float(np.mean([r["n_boxed"] for r in rows]) / K),
        "mean_completion_tokens": float(np.mean([r["mean_len"] for r in rows])),
        "rows": rows,
    }
    return res


@app.local_entrypoint()
def main():
    r = preflight.remote()
    out = "/home/vibe12/projects/persona-curvature/qwen35/analysis/rl_preflight.json"
    with open(out, "w") as f:
        json.dump(r, f, indent=1)
    print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=1))
    print(f"\nwrote {out}")
    n = r["learnable_for_grpo"]
    print(f"\nDECISION: {n}/{r['n_problems']} problems carry GRPO gradient "
          f"({r['learnable_rate']:.1%}).")
    print("  >15%  -> GRPO is viable on the filtered subset")
    print("  <15%  -> fall back to rejection-sampling SFT on the solvable subset")
