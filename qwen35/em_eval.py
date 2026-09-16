#!/usr/bin/env python3
"""Generation for the emergent-misalignment medical run: two batteries, four conditions.

Battery (i) is the zoo's 24 personality prompts every judged zoo evaluation has
used (`phase10_runs/steer_results_fix.json`, prompts of job 0), decoded GREEDY at
512 new tokens so the judged profiles sit on the zoo's own scale.

Battery (ii) is the eight plain free-form questions of Turner et al. 2025,
verbatim from `em_organism_dir/data/eval_questions/first_plot_questions.yaml` at
commit 8460e4e426d3a89e8ed51aac0eadcdf7ac10469d, decoded the way the paper does
it -- SAMPLED at temperature 1.0, top_p 1.0, 600 new tokens -- 10 samples per
question per condition.  The paper draws 50; 10 is this run's budget and the
pre-registration says so, which puts the resolution of a per-arm rate at
1/80 = 1.25 percentage points.

`enable_thinking=False` throughout, as everywhere in this project.

BASE IS GENERATED FIRST, before any PeftModel wraps the module, for the reason
syc_eval.py gives: `PeftModel.from_pretrained` wraps in place, and a base
generated after the wrap runs with whichever adapter is active.

The sampling seed is set once per (condition, question) as
20260911 + 1000 * question_index, the SAME value for every condition, so the
four conditions draw from the same random stream and a difference between them
is not a difference of seed.

usage:
    PC_APP_NAME=pc-qwen35-phase13-emeval modal run em_eval.py --arms em_bad,em_good,em_dolci
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "Qwen/Qwen3.5-4B"
MAX_NEW_BIG5 = 512
MAX_NEW_EM = 600
N_SAMPLES = 10
TEMP = 1.0
SEED = 20260911
ARMS = ["em_bad", "em_good", "em_dolci"]

app = modal.App(os.environ.get("PC_APP_NAME") or "pc-qwen35-phase13-emeval")
out_vol = modal.Volume.from_name("pc-qwen35-adapters")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.13.0", "transformers==5.15.1", "peft==0.20.0",
                 "accelerate==1.14.0", "safetensors", "hf_transfer", "numpy<3",
                 "huggingface_hub")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(os.path.abspath(__file__), "/root/em_eval.py", copy=True)
)


@app.function(image=image, gpu="A100-40GB", volumes={"/adapters": out_vol},
              timeout=60 * 300, secrets=[modal.Secret.from_name("hf-token")])
def run(big5: list, emq: list, conditions: list) -> dict:
    import time

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snap = snapshot_download(BASE_MODEL, ignore_patterns=["*.pt", "*.bin", "*.gguf", "*.pth"])
    tok = AutoTokenizer.from_pretrained(snap)

    def render(prompt):
        txt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
        assert "<think>" not in txt or "</think>" in txt, "unclosed think block"
        return txt

    def greedy(model, prompts, label):
        outs, t0 = [], time.time()
        for i, p in enumerate(prompts):
            enc = tok(render(p), return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW_BIG5,
                                   pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(o[0][enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True))
            if (i + 1) % 12 == 0:
                print(f"  [{label}] {i+1}/{len(prompts)} {time.time()-t0:.0f}s", flush=True)
        return outs

    def sampled(model, questions, label):
        outs, t0 = [], time.time()
        for qi, q in enumerate(questions):
            torch.manual_seed(SEED + 1000 * qi)
            enc = tok(render(q["question"]), return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = model.generate(**enc, do_sample=True, temperature=TEMP, top_p=1.0,
                                   top_k=0, num_return_sequences=N_SAMPLES,
                                   max_new_tokens=MAX_NEW_EM,
                                   pad_token_id=tok.eos_token_id)
            n = enc["input_ids"].shape[1]
            outs.append([tok.decode(row[n:], skip_special_tokens=True) for row in o])
            print(f"  [{label}] {qi+1}/{len(questions)} {time.time()-t0:.0f}s", flush=True)
        return outs

    base = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16).cuda().eval()
    # Record the model's OWN generation config.  The paper decodes at T=1,
    # top_p=1; if Qwen3.5's config carries a repetition_penalty or a default
    # top_k, this decode is not the paper's and that is a deviation to declare
    # rather than to discover later.  top_k=0 above disables top-k explicitly.
    gcfg = base.generation_config.to_dict()
    print("[generation_config] " + json.dumps(
        {k: gcfg.get(k) for k in ("temperature", "top_p", "top_k", "do_sample",
                                  "repetition_penalty", "min_p", "typical_p")}),
        flush=True)
    arms = [c for c in conditions if c != "base"]
    res = {}

    def do(cond, m):
        res[cond] = {"bigfive": greedy(m, big5, f"{cond}/big5"),
                     "em": sampled(m, emq, f"{cond}/em")}

    if "base" in conditions:
        do("base", base)

    model = None
    if arms:
        model = PeftModel.from_pretrained(base, f"/adapters/em_medical/{arms[0]}/final",
                                          adapter_name=arms[0])
        for a in arms[1:]:
            model.load_adapter(f"/adapters/em_medical/{a}/final", adapter_name=a)
        n_mod = sum(1 for n, _ in model.named_modules() if n.endswith(".lora_A"))
        print(f"[reload] {n_mod} lora_A modules resolved (training reported 248)", flush=True)
        if n_mod != 248:
            raise RuntimeError(f"reloaded adapter has {n_mod} LoRA modules, not 248")
    for cond in arms:
        model.set_adapter(cond)
        do(cond, model.eval())

    return {"generations": res, "generation_config": gcfg,
            "n_samples": N_SAMPLES, "temperature": TEMP,
            "seed": SEED, "max_new_bigfive": MAX_NEW_BIG5, "max_new_em": MAX_NEW_EM,
            "decode": "bigfive greedy; em sampled T=1.0 top_p=1.0 top_k=0; "
                      "enable_thinking=False"}


@app.local_entrypoint()
def main(arms: str = ",".join(ARMS), base: str = "yes", out: str = "em_gens.json"):
    arms = [a.strip() for a in arms.split(",") if a.strip()]
    conditions = (["base"] if base == "yes" else []) + arms
    steer = json.load(open(f"{HERE}/phase10_runs/steer_results_fix.json"))
    big5 = steer[0]["prompts"]
    assert len(big5) == 24, len(big5)
    emq = json.load(open(f"{HERE}/phase10_runs/em_questions.json"))["questions"]
    assert len(emq) == 8, len(emq)
    print(f"{len(conditions)} conditions x ({len(big5)} greedy + "
          f"{len(emq)}x{N_SAMPLES} sampled)", flush=True)
    r = run.remote(big5, emq, conditions)
    r["conditions"] = conditions
    r["bigfive_prompts"] = big5
    r["em_questions"] = emq
    p = f"{HERE}/phase10_runs/{out}"
    json.dump(r, open(p, "w"))
    print("wrote", p, flush=True)
