#!/usr/bin/env python3
"""What did maths RL do to the model's personality? Measured behaviourally.

The weight-space projection cannot answer this: the RL adapter has a different
LoRA initialisation from the zoo, so near-orthogonality to every trait direction
is guaranteed by construction rather than measured. Behaviour has no such
problem -- it does not care which random subspace the update lived in.

So: apply each RL checkpoint to the base model, answer the same 24 personality
prompts the steering sweep used, under the same decoding settings, and judge
blind. The steering run's alpha=0 rows are the base model on those exact
prompts, which makes them a matched control that costs nothing.

Running every checkpoint rather than only the final one turns a point estimate
into a trajectory: does personality drift accumulate with capability, appear
early and saturate, or not appear at all?
"""
import json
import os

import modal

BASE_MODEL = "Qwen/Qwen3.5-4B"
MAX_NEW = 512

app = modal.App("pc-qwen35-phase10-rleval")
rl_vol = modal.Volume.from_name("pc-qwen35-rl")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.13.0", "transformers==5.15.1", "peft==0.20.0",
                 "accelerate==1.14.0", "safetensors", "hf_transfer", "numpy<3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)


@app.function(image=image, gpu="A100-40GB", volumes={"/rl": rl_vol}, timeout=60 * 180,
              secrets=[modal.Secret.from_name("hf-token")])
def run(prompts: list, tag: str = "math") -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snap = snapshot_download(BASE_MODEL, ignore_patterns=["*.pt", "*.bin", "*.gguf", "*.pth"])
    tok = AutoTokenizer.from_pretrained(snap)
    root = f"/rl/runs/{tag}"
    ck = sorted([d for d in os.listdir(root) if d.startswith("checkpoint-")],
                key=lambda d: int(d.split("-")[1]))

    def gen(model):
        outs = []
        for p in prompts:
            # enable_thinking=False, matching every other generation in the project.
            # Leaving it on is what invalidated the first steering corpus.
            txt = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
            assert "<think>" not in txt or "</think>" in txt, "unclosed think block"
            enc = tok(txt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW,
                                   pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
        return outs

    base = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16).cuda().eval()
    res = {"base": gen(base)}
    print("[base] done", flush=True)

    model = PeftModel.from_pretrained(base, f"{root}/{ck[0]}", adapter_name=ck[0])
    for c in ck[1:]:
        model.load_adapter(f"{root}/{c}", adapter_name=c)
    for c in ck:
        model.set_adapter(c)
        res[c] = gen(model.eval())
        print(f"[{c}] done", flush=True)
    return {"prompts": prompts, "generations": res, "max_new_tokens": MAX_NEW}


@app.local_entrypoint()
def main(tag: str = "math"):
    here = os.path.dirname(os.path.abspath(__file__))
    steer = json.load(open(f"{here}/phase10_runs/steer_results_fix.json"))
    prompts = steer[0]["prompts"]
    print(f"{len(prompts)} prompts, shared with the steering sweep", flush=True)
    r = run.remote(prompts, tag)
    out = f"{here}/phase10_runs/rl_persona_{tag}.json"
    with open(out, "w") as f:
        json.dump(r, f)
    print(f"wrote {out}: {len(r['generations'])} conditions "
          f"x {len(prompts)} prompts", flush=True)
