#!/usr/bin/env python3
"""Generate the 24-prompt Big Five battery from base and from each stage-one
Big Five FACTOR adapter, for blind judging by judge_personas.py.

This is oct_stage2.py's `eval_personas` with stage two removed. It exists as its
own file because `oct_stage2.py --stage eval` loads THREE checkpoints per trait
-- base, stage-1 DPO, and the merged OCT persona -- and the Big Five factor
adapters are stage one only (OCT stage two is about $15 per trait and was not
approved). Calling that path would fail on the missing persona adapter.

Everything that affects the numbers is copied verbatim from eval_personas:

  * enable_thinking=False on apply_chat_template. Qwen3.5's template defaults
    thinking ON; three earlier runs in this project were silently invalidated by
    omitting it, and a judged score computed over a <think> block is not a
    behavioural score.
  * greedy (do_sample=False), max_new_tokens=200, torch.manual_seed(0) before
    the first generation, prompts one at a time (not batched -- batching changes
    greedy output through padding).
  * the 24 probes from bigfive_probes.py: 20 behavioural items, 4 per factor,
    none naming a trait, plus the 4 legacy prompts.

ONE container generates base ONCE and then every adapter, so the base condition
is a single set of texts rather than one per trait; it is written into each
trait's record anyway, which is the shape judged_100.json has and therefore the
shape judge_personas.py and build_spider_data.py already read.

Output: phase10_runs/eval_bigfive.json, a list of
    {"trait": ..., "prompts": [...], "generations": {"base": [...], "stage1": [...]}}

usage:
  PC_APP_NAME=pc-qwen35-phase11-bigfive-eval PC_PHASE_BUDGET=3 \
    ~/cartovenv/bin/modal run eval_bigfive.py \
      --traits bf_openness_high,... --adp-subdir data_bigfive_common
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))

APP_NAME = os.environ.get("PC_APP_NAME")
BASE_MODEL = os.environ.get("PC_BASE_MODEL", "Qwen/Qwen3.5-4B")
ADAPTER_VOLUME = os.environ.get("PC_ADAPTER_VOLUME", "pc-qwen35-adapters")
GPU_TYPE = os.environ.get("PC_GPU", "A100-40GB")
GPU_PRICE_PER_HOUR = {
    "A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95,
    "L40S": 1.95, "A10G": 1.10, "L4": 0.80,
}
MAX_NEW_TOKENS = 200

if not APP_NAME:
    raise SystemExit(
        "REFUSING TO LAUNCH: PC_APP_NAME is unset. Every Modal launch in this "
        "project names its own app so the spend can be attributed to a phase.")


def _download_base_model():
    import huggingface_hub
    huggingface_hub.snapshot_download(
        os.environ["PC_BASE_MODEL"],
        ignore_patterns=["*.pt", "*.bin", "*.gguf", "*.pth"])


app = modal.App(APP_NAME)
adapter_vol = modal.Volume.from_name(ADAPTER_VOLUME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0", "transformers==5.15.1", "peft==0.20.0",
                 "accelerate==1.14.0", "safetensors", "hf_transfer", "numpy<3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "PC_BASE_MODEL": BASE_MODEL,
          # The container imports this same file, so it hits the PC_APP_NAME
          # guard above too -- and only the vars named here cross into the
          # image. Without it the guard fires during the image build itself.
          # (train_qwen35.py carries the identical comment; this file paid the
          # same toll on its first launch.)
          "PC_APP_NAME": APP_NAME})
    .add_local_file(os.path.abspath(__file__), "/root/eval_bigfive.py", copy=True)
    .run_function(_download_base_model)
)


@app.function(secrets=[modal.Secret.from_name("hf-token")], image=image,
              gpu=GPU_TYPE, volumes={"/adapters": adapter_vol}, timeout=60 * 90)
def eval_stage1(job: dict) -> list:
    """base + every stage-one adapter in `job['traits']`, on `job['prompts']`."""
    import huggingface_hub
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    traits = job["traits"]
    prompts = job["prompts"]
    adp_root = job["adp_root"]
    print(f"[paths] adp_root={adp_root} traits={len(traits)} prompts={len(prompts)}",
          flush=True)
    for t in traits:
        p = f"{adp_root}/{t}"
        if not os.path.isdir(p):
            raise RuntimeError(f"no adapter directory at {p}")

    snap = huggingface_hub.snapshot_download(
        BASE_MODEL, ignore_patterns=["*.pt", "*.bin", "*.gguf", "*.pth"])
    tok = AutoTokenizer.from_pretrained(snap)

    def chat_str(messages, add_generation_prompt):
        # Verbatim from oct_stage2._chat_str. enable_thinking=False is load-bearing.
        return tok.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False)

    def gen(model, tag):
        outs = []
        for i, p in enumerate(prompts):
            text = chat_str([{"role": "user", "content": p}], True)
            enc = tok(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, do_sample=False,
                                     max_new_tokens=MAX_NEW_TOKENS,
                                     pad_token_id=tok.pad_token_id
                                     or tok.eos_token_id)
            outs.append(tok.decode(out[0][enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip())
        print(f"  [{tag}] {len(outs)} generations", flush=True)
        return outs

    base = AutoModelForCausalLM.from_pretrained(
        snap, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa").eval()
    torch.manual_seed(0)
    base_gens = gen(base, "base")

    out = []
    for t in traits:
        m = PeftModel.from_pretrained(base, f"{adp_root}/{t}").eval()
        g = gen(m, t)
        m.unload()
        out.append({"trait": t, "prompts": prompts,
                    "generations": {"base": base_gens, "stage1": g}})
    return out


@app.local_entrypoint()
def main(traits: str = "", adp_subdir: str = "data_bigfive_common",
         minutes: float = 45.0, out: str = "phase10_runs/eval_bigfive.json"):
    from bigfive_probes import prompts_only

    if not traits:
        traits = ",".join(r["trait"] for r in
                          json.load(open(os.path.join(HERE, "traits_bigfive.json"))))
    names = [t.strip() for t in traits.split(",") if t.strip()]
    budget = os.environ.get("PC_PHASE_BUDGET")
    if budget is None:
        raise SystemExit("NO BUDGET DECLARED: set PC_PHASE_BUDGET (dollars).")
    budget = float(budget)
    price = GPU_PRICE_PER_HOUR[GPU_TYPE]
    est = minutes / 60.0 * price
    prompts = prompts_only()
    adp_root = "/adapters" + (f"/{adp_subdir.strip('/')}" if adp_subdir else "")

    print("=" * 72)
    print("BIG FIVE STAGE-ONE EVAL -- printed BEFORE any GPU is allocated")
    print("=" * 72)
    print(f"  app                          : {APP_NAME}")
    print(f"  adapters (read)              : {ADAPTER_VOLUME}:{adp_root}/<trait>")
    print(f"  traits ({len(names):2d})                  : {', '.join(names)}")
    print(f"  prompts                      : {len(prompts)} (bigfive_probes.battery)")
    print(f"  conditions                   : base, stage1  (NO stage two)")
    print(f"  decoding                     : greedy, {MAX_NEW_TOKENS} new tokens, "
          f"thinking OFF")
    print(f"  generations                  : {(len(names) + 1) * len(prompts)}")
    print(f"  gpu                          : {GPU_TYPE} @ ${price:.2f}/hr")
    print(f"  ARITHMETIC                   : 1 container x {minutes:.0f} min = "
          f"{minutes/60:.2f} GPU-hr x ${price:.2f}/hr = ${est:.2f}")
    print(f"  declared budget              : ${budget:.2f}  (PC_PHASE_BUDGET)")
    if est > budget:
        raise SystemExit(f"estimate ${est:.2f} exceeds PC_PHASE_BUDGET "
                         f"${budget:.2f} -- STOPPED")
    print(f"  VERDICT: within budget, ${budget - est:.2f} headroom")
    print("=" * 72, flush=True)

    res = eval_stage1.remote({"traits": names, "prompts": prompts,
                              "adp_root": adp_root})
    dst = out if os.path.isabs(out) else os.path.join(HERE, out)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {dst}: {len(res)} traits x {len(prompts)} prompts x 2 conditions",
          flush=True)
    for e in res:
        print(f"  {e['trait']:28s} stage1[0]: "
              f"{e['generations']['stage1'][0][:90]!r}")
