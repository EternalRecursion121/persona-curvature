#!/usr/bin/env python3
"""Steer Qwen3.5-4B along directions in the persona weight space.

No retraining and nothing large materialised. A principal direction of the
adapter cloud is a LINEAR COMBINATION of the adapters we already have:

    v_k = X^T u_k / sqrt(lambda_k)      (u_k, lambda_k from the n x n Gram)

and since sum_i u_ik = 0, that is also a plain combination of the RAW deltas
with the same coefficients. So "steer along PC k" is a weighted merge:

    W_steered = W_base + alpha * ref * normalise( sum_i c_i * 2 * B_i @ A_i )

`ref` is the mean single-adapter Frobenius norm, so alpha is measured in units
of "one trait adapter's worth of weight change" and is comparable across
directions.

Directions supported: principal components, the five Big Five factor axes
(mean(+keyed) - mean(-keyed)), and `mean` -- the grand-mean direction, which in
weight space is mu(personas) - base. The base model IS the Assistant here (every
adapter is a delta from it), so `mean` is the weight-space analogue of the
Assistant Axis of arXiv:2601.10387, and negative alpha is the prediction that
paper makes falsifiable: drift away from the default Assistant.

Coefficients are computed on the host from the sketches and passed in, so this
file never needs the geometry code.
"""
import json, math, os
import modal

BASE_MODEL = os.environ.get("PC_BASE_MODEL", "Qwen/Qwen3.5-4B")
GPU_TYPE = os.environ.get("PC_GPU", "A100-40GB")
APP_NAME = os.environ.get("PC_APP_NAME", "pc-qwen35-steer")

app = modal.App(APP_NAME)
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep", create_if_missing=False)
oct_vol = modal.Volume.from_name("pc-qwen35-oct2", create_if_missing=True)
VOLS = {"/adapters": sweep_vol, "/oct": oct_vol}


def _download_base_model():
    from huggingface_hub import snapshot_download
    snapshot_download(BASE_MODEL)


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.13.0", "transformers==5.15.1", "peft==0.20.0",
                 "accelerate==1.14.0", "safetensors", "hf_transfer", "numpy<3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "PC_BASE_MODEL": BASE_MODEL})
    .add_local_file(os.path.abspath(__file__), "/root/steer_qwen35.py", copy=True)
    .run_function(_download_base_model)
)

SRC = {"stage1": "/adapters/{t}", "persona": "/oct/personas/{t}/persona"}


@app.function(secrets=[modal.Secret.from_name("hf-token")], image=image,
              gpu=GPU_TYPE, volumes=VOLS, timeout=60 * 90)
def steer(job: dict) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name, coef, source = job["name"], job["coef"], job["source"]
    alphas, prompts = job["alphas"], job["prompts"]
    snap = snapshot_download(BASE_MODEL)
    tmpl = SRC[source]

    # --- build the direction, module by module, never the whole model ---------
    traits = sorted(coef)
    first = tmpl.format(t=traits[0])
    with open(f"{first}/adapter_config.json") as f:
        ac = json.load(f)
    scale = ac["lora_alpha"] / (math.sqrt(ac["r"]) if ac.get("use_rslora") else ac["r"])

    handles = {t: safe_open(f"{tmpl.format(t=t)}/adapter_model.safetensors",
                            framework="pt") for t in traits}
    mods = sorted({k.split(".lora_A.")[0].removeprefix("base_model.model.")
                   for k in handles[traits[0]].keys() if ".lora_A." in k})
    D, sq = {}, 0.0
    for m in mods:
        acc = None
        for t in traits:
            h = handles[t]
            A = h.get_tensor(f"base_model.model.{m}.lora_A.weight").float()
            B = h.get_tensor(f"base_model.model.{m}.lora_B.weight").float()
            d = (B @ A) * (scale * coef[t])
            acc = d if acc is None else acc + d
        D[m] = acc
        sq += float((acc * acc).sum())
    norm = math.sqrt(sq)
    for m in D:
        D[m] /= norm                      # unit Frobenius over the whole model

    # --- load base, map adapter module names onto checkpoint tensors ----------
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    params = dict(model.named_parameters())

    def find(m):
        for c in (m + ".weight", "model." + m + ".weight",
                  m.replace("model.", "model.language_model.", 1) + ".weight"):
            if c in params:
                return c
        return None

    hit = {m: find(m) for m in mods}
    missing = [m for m, v in hit.items() if v is None]
    if missing:
        raise RuntimeError(f"unmapped modules: {missing[:4]} ({len(missing)} total)")

    def generate():
        outs = []
        for p in prompts:
            msg = [{"role": "user", "content": p}]
            enc = tok(tok.apply_chat_template(msg, tokenize=False,
                                              add_generation_prompt=True),
                      return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = model.generate(**enc, do_sample=False, max_new_tokens=200,
                                   pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(o[0][enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True))
        return outs

    # Walk alphas in order and apply only the INCREMENT, so the base is loaded
    # once rather than 7 times.
    res, prev = {}, 0.0
    for a in sorted(alphas):
        step = (a - prev) * job["ref"]
        if step:
            with torch.no_grad():
                for m, pn in hit.items():
                    params[pn] += (D[m] * step).to(params[pn].dtype).to(params[pn].device)
        prev = a
        res[str(a)] = generate()
        print(f"[{name}] alpha={a} done", flush=True)
    return {"name": name, "source": source, "alphas": alphas,
            "prompts": prompts, "generations": res, "dir_norm_raw": norm}


@app.local_entrypoint()
def main(spec: str = "phase10_runs/steer_spec.json",
         out: str = "phase10_runs/steer_results.json"):
    here = os.path.dirname(os.path.abspath(__file__))
    S = json.load(open(os.path.join(here, spec)))
    jobs = S["jobs"]
    print(f"{len(jobs)} directions x {len(jobs[0]['alphas'])} alphas "
          f"on source={jobs[0]['source']}", flush=True)
    res = list(steer.map(jobs))
    with open(os.path.join(here, out), "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {out}", flush=True)
