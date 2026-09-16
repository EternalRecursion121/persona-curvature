#!/usr/bin/env python3
"""Generate the two batteries for the base model and every dolci-flag arm.

Battery A, `dolci_flag_battery.json`: 40 prompts on which refusal or substantive
pushback is the right answer and 20 benign ones a well-behaved assistant should
answer, the latter chosen as near-misses of the former so that over-refusal is
measured rather than assumed.

Battery B: the 24 personality prompts the steering sweep and every judged zoo
evaluation have used (`phase10_runs/steer_results_fix.json`, prompts of job 0),
so the Big Five profiles of these arms are comparable to the whole zoo.

Decoding is the project's: greedy, `enable_thinking=False`, 512 new tokens.
Base and arms run in ONE container off one loaded base model, so nothing about
the decode differs between conditions.
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "Qwen/Qwen3.5-4B"
MAX_NEW = 512
ARMS = ["flagged", "random", "anti", "unfiltered", "filtered"]

app = modal.App(os.environ.get("PC_APP_NAME") or "pc-qwen35-phase11-dolciflag-eval")
out_vol = modal.Volume.from_name("pc-qwen35-adapters")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.13.0", "transformers==5.15.1", "peft==0.20.0",
                 "accelerate==1.14.0", "safetensors", "hf_transfer", "numpy<3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)


@app.function(image=image, gpu="A100-40GB", volumes={"/adapters": out_vol},
              timeout=60 * 300, secrets=[modal.Secret.from_name("hf-token")])
def run(prompts: list, arms: list) -> dict:
    import time

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snap = snapshot_download(BASE_MODEL, ignore_patterns=["*.pt", "*.bin", "*.gguf", "*.pth"])
    tok = AutoTokenizer.from_pretrained(snap)

    def gen(model, label):
        outs, t0 = [], time.time()
        for i, p in enumerate(prompts):
            txt = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
            assert "<think>" not in txt or "</think>" in txt, "unclosed think block"
            enc = tok(txt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW,
                                   pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
            if (i + 1) % 20 == 0:
                print(f"  [{label}] {i+1}/{len(prompts)}  {time.time()-t0:.0f}s", flush=True)
        print(f"[{label}] done in {time.time()-t0:.0f}s", flush=True)
        return outs

    base = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16).cuda().eval()
    res = {"base": gen(base, "base")}
    model = PeftModel.from_pretrained(base, f"/adapters/dolci_flag/{arms[0]}",
                                      adapter_name=arms[0])
    for a in arms[1:]:
        model.load_adapter(f"/adapters/dolci_flag/{a}", adapter_name=a)
    # PEFT rewrites the explicit 248-name target list into a handful of suffix
    # patterns when it saves, so reloading re-resolves the targets.  Count them:
    # if the reload produced a different set of LoRA modules from the 248 that
    # were trained, some layers would carry a freshly initialised A with a zero
    # B -- silent, and the adapter under test would not be the one that trained.
    n_mod = sum(1 for n, _ in model.named_modules() if n.endswith(".lora_A"))
    print(f"[reload] {n_mod} lora_A modules resolved on the loaded adapters "
          f"(training reported 248)", flush=True)
    if n_mod != 248:
        raise RuntimeError(f"reloaded adapter has {n_mod} LoRA modules, not the 248 trained")
    for a in arms:
        model.set_adapter(a)
        res[a] = gen(model.eval(), a)
    return {"prompts": prompts, "generations": res, "max_new_tokens": MAX_NEW,
            "decode": "greedy, enable_thinking=False"}


@app.local_entrypoint()
def main(arms: str = ",".join(ARMS), tag: str = ""):
    """`tag` splits the run into shards that are merged by merge_dolci_flag_gens.py.

    The 400-pair arms finish an hour and a half before the 3,000-pair ones, and
    the GPU would otherwise sit idle waiting for them.  Every shard regenerates
    the base model as well: it costs about fourteen minutes and buys a real
    check, because greedy decoding of the same prompts by the same base weights
    in two different containers must produce byte-identical text, and the merge
    asserts it.
    """
    arms = [a.strip() for a in arms.split(",") if a.strip()]
    bat = json.load(open(f"{HERE}/dolci_flag_battery.json"))
    comp = [x["prompt"] for x in bat["should_refuse"]] + [x["prompt"] for x in bat["benign"]]
    comp_ids = [x["id"] for x in bat["should_refuse"]] + [x["id"] for x in bat["benign"]]
    steer = json.load(open(f"{HERE}/phase10_runs/steer_results_fix.json"))
    big5 = steer[0]["prompts"]
    print(f"{len(comp)} compliance prompts + {len(big5)} Big Five prompts "
          f"x {len(arms)+1} conditions", flush=True)

    r = run.remote(comp + big5, arms)
    gens = r["generations"]
    out = {"battery_ids": comp_ids, "n_compliance": len(comp), "n_bigfive": len(big5),
           "prompts": r["prompts"], "generations": gens,
           "max_new_tokens": r["max_new_tokens"], "decode": r["decode"],
           "arms": arms}
    p = (f"{HERE}/phase10_runs/dolci_flag_gens"
         f"{('_' + tag) if tag else ''}.json")
    json.dump(out, open(p, "w"))
    print(f"wrote {p}: {len(gens)} conditions x {len(r['prompts'])} prompts")
    if tag:
        return

    # the Big Five half, in the shape judge_personas.py consumes. ONE record
    # holding all six conditions, so the base generations are judged once
    # rather than once per arm, and every condition is interleaved into the
    # same shuffled stream by that script's own ordering.
    recs = [{"trait": "dolci_flag", "prompts": big5,
             "generations": {c: gens[c][len(comp):] for c in ["base"] + arms}}]
    p2 = f"{HERE}/phase10_runs/dolci_flag_eval_big5.json"
    json.dump(recs, open(p2, "w"))
    print(f"wrote {p2}: {len(recs)} arms x {len(big5)} prompts")
