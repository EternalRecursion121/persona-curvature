#!/usr/bin/env python3
"""Steer Qwen3.5-4B along directions in the persona weight space.

No retraining and nothing large materialised. A principal direction of the
adapter cloud is a LINEAR COMBINATION of the adapters we already have:

    v_k = X^T u_k / sqrt(lambda_k)      (u_k, lambda_k from the n x n Gram)

and since sum_i u_ik = 0, that is also a plain combination of the RAW deltas
with the same coefficients. So "steer along PC k" is a weighted merge:

    W_steered = W_base + alpha * ref * normalise( sum_i c_i * 2 * B_i @ A_i )

`ref` is 0.8078003190997738 in every spec in this repository.  CORRECTION,
2026-09-08: that number came from sketch_adapters.sketch_one, which computes
||B @ A||_F and does NOT multiply by the LoRA scaling lora_alpha / r = 2.0.  The
adapters' real mean Frobenius norm is 1.6157416444226869 (the diagonal of
results/gram_sweep.npz), so ref is 0.49996 of an adapter and ALPHA 1 IS HALF A
TRAIT ADAPTER'S WORTH OF WEIGHT CHANGE, NOT ONE.  ref is deliberately left as it
is, so that alpha stays a consistent unit across every steering run ever done
here; read every published alpha as half an adapter.  See
analysis/steer_alpha_units.json.

Directions supported: principal components, the five Big Five factor axes
(mean(+keyed) - mean(-keyed)), and `mean` -- the grand-mean direction, which in
weight space is mu(personas) - base. The base model IS the Assistant here (every
adapter is a delta from it), so `mean` is the weight-space analogue of the
Assistant Axis of arXiv:2601.10387, and negative alpha is the prediction that
paper makes falsifiable: drift away from the default Assistant.

Coefficients are computed on the host from the sketches and passed in, so this
file never needs the geometry code.

WHY THIS FILE EXISTS SEPARATELY FROM steer_qwen35.py
----------------------------------------------------
The first steering runs called apply_chat_template WITHOUT enable_thinking=False
and capped generation at 200 new tokens.  Qwen3.5 defaults thinking ON, so every
one of those 1512 generations is a reasoning preamble cut off mid-plan: 85% do
not end on terminal punctuation and none reaches an answer.  The Big Five judge
therefore scored planning traces, not responses.

Two changes, and nothing else, so the comparison stays clean:
  * enable_thinking=False, matching oct_stage2.py:_chat_str and therefore the
    trait evals these results are meant to sit beside;
  * max_new_tokens 200 -> 512.

Same prompts, same alphas, same coefficients, same greedy decoding, same judge.
The old corpus is kept and becomes the robustness comparison: if the
dose-response shapes replicate, truncation was not driving them; if they do not,
steering moves the private planning register and the public answer differently,
which is worth knowing on its own.

The app name carries the phase10 prefix because the spend meter's hard stop
resolves live app IDs by that prefix, and `modal run --detach` apps outlive the
systemd unit that launched them.  pc-qwen35-steer did not match, so the old
steering runs sat outside the kill switch.
"""
import json, math, os
import modal

BASE_MODEL = os.environ.get("PC_BASE_MODEL", "Qwen/Qwen3.5-4B")
GPU_TYPE = os.environ.get("PC_GPU", "A100-40GB")
APP_NAME = os.environ.get("PC_APP_NAME", "pc-qwen35-phase10-steerfix")

app = modal.App(APP_NAME)
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep", create_if_missing=False)
oct_vol = modal.Volume.from_name("pc-qwen35-oct2", create_if_missing=True)
# adapters trained after the zoo land on the training output volume, not the
# sweep volume the 134 live on
train_vol = modal.Volume.from_name("pc-qwen35-adapters", create_if_missing=True)
VOLS = {"/adapters": sweep_vol, "/oct": oct_vol, "/trained": train_vol}


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
    .add_local_file(os.path.abspath(__file__), "/root/steer_fix.py", copy=True)
    .run_function(_download_base_model)
)

SRC = {"stage1": "/adapters/{t}", "persona": "/oct/personas/{t}/persona",
       "alignment": "/trained/data_alignment/{t}",
       "stage2": "/oct/loras_introspection/{t}",     # introspection SFT LoRAs
       # the corrected rank-128 concatenation merge dW_dpo + 0.25 dW_sft,
       # scaling 1.0 (fix_persona_merge.py); keys single-prefixed by
       # fix_persona_keys.py on 2026-09-07
       "persona_exact": "/oct/personas_exact/{t}"}


def _strip(mod):
    """PEFT's prefix, stripped however many times it was written."""
    while mod.startswith("base_model.model."):
        mod = mod[len("base_model.model."):]
    return mod


def _open_set(tmpl, traits):
    """(handles, module->per-trait key map, scale) for one adapter source."""
    import math as _math
    from safetensors import safe_open
    with open(f"{tmpl.format(t=traits[0])}/adapter_config.json") as f:
        ac = json.load(f)
    scale = ac["lora_alpha"] / (_math.sqrt(ac["r"]) if ac.get("use_rslora") else ac["r"])
    handles = {t: safe_open(f"{tmpl.format(t=t)}/adapter_model.safetensors",
                            framework="pt") for t in traits}
    keymap = {}
    for t in traits:
        keymap[t] = {_strip(k.split(".lora_A.")[0]): k.split(".lora_A.")[0]
                     for k in handles[t].keys() if ".lora_A." in k}
    return handles, keymap, scale


@app.function(secrets=[modal.Secret.from_name("hf-token")], image=image,
              # 512-token generations at seven alphas overran the old 90-minute
              # cap.  A timeout loses the whole direction, and systemd then
              # re-ran all nine from scratch -- hence both the wider cap and the
              # per-alpha resume below.
              gpu=GPU_TYPE, volumes=VOLS, timeout=60 * 300)
def steer(job: dict) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name, coef, source = job["name"], job["coef"], job["source"]
    alphas, prompts = job["alphas"], job["prompts"]
    # Optional: add ONE adapter's full delta to the base weights before the
    # steering direction is applied, so a condition can be "this adapter, plus
    # alpha of that direction".  {"source": <SRC key>, "trait": <name>}.
    # Nothing about the existing path changes when it is absent.
    add = job.get("add_adapter")
    snap = snapshot_download(BASE_MODEL)
    tmpl = SRC[source]

    # --- build the direction, module by module, never the whole model ---------
    # A job may carry no direction at all (empty coef, or every alpha zero):
    # that is the "adapter alone" or "base alone" condition.
    need_dir = bool(coef) and any(a != 0.0 for a in alphas)
    D, norm, mods = {}, None, []
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if need_dir:
        traits = sorted(coef)
        handles, keymap, scale = _open_set(tmpl, traits)
        mods = sorted(keymap[traits[0]])
        # One GEMM per module on the GPU, not 134 on the CPU.  Concatenating the
        # scaled B factors along their rank axis and the A factors along theirs
        # makes sum_i c_i B_i A_i a single (d_out x 134r) @ (134r x d_in) product,
        # which is the same arithmetic in one BLAS call instead of 33,000, and on
        # the accelerator that is otherwise sitting idle until the model loads.
        sq = 0.0
        for m in mods:
            Bs = torch.cat([handles[t].get_tensor(keymap[t][m] + ".lora_B.weight")
                            .float() * (scale * coef[t]) for t in traits], dim=1)
            As = torch.cat([handles[t].get_tensor(keymap[t][m] + ".lora_A.weight")
                            .float() for t in traits], dim=0)
            acc = (Bs.to(dev) @ As.to(dev))
            sq += float((acc * acc).sum())
            D[m] = acc.cpu()      # 248 dense deltas will not fit beside the model
            del Bs, As, acc
        norm = math.sqrt(sq)
        for m in D:
            D[m] /= norm                  # unit Frobenius over the whole model
    else:
        print(f"[{name}] no direction (coef={len(coef)}, alphas={alphas})", flush=True)

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

    # --- optional: fold one adapter's full delta into the loaded weights ------
    # Done module by module straight into the parameters, so a second dict of
    # 248 dense deltas is never held beside the model.
    add_norm = None
    if add:
        a_handles, a_keymap, a_scale = _open_set(SRC[add["source"]], [add["trait"]])
        a_mods = sorted(a_keymap[add["trait"]])
        a_hit = {m: find(m) for m in a_mods}
        a_missing = [m for m, v in a_hit.items() if v is None]
        if a_missing:
            raise RuntimeError(f"add_adapter unmapped modules: {a_missing[:4]} "
                               f"({len(a_missing)} total)")
        sq_add = 0.0
        with torch.no_grad():
            for m in a_mods:
                k = a_keymap[add["trait"]][m]
                A = a_handles[add["trait"]].get_tensor(k + ".lora_A.weight").float()
                B = a_handles[add["trait"]].get_tensor(k + ".lora_B.weight").float()
                dW = (B.to(dev) @ A.to(dev)) * a_scale
                sq_add += float((dW * dW).sum())
                pn = a_hit[m]
                params[pn] += dW.to(params[pn].dtype).to(params[pn].device)
                del A, B, dW
        add_norm = math.sqrt(sq_add)
        print(f"[{name}] added {add['source']}/{add['trait']} "
              f"(scale {a_scale}, {len(a_mods)} modules, |dW| {add_norm:.4f})",
              flush=True)

    def generate():
        outs = []
        for p in prompts:
            msg = [{"role": "user", "content": p}]
            enc = tok(tok.apply_chat_template(msg, tokenize=False,
                                              add_generation_prompt=True,
                                              enable_thinking=False),
                      return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = model.generate(**enc, do_sample=False, max_new_tokens=512,
                                   pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(o[0][enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True))
        return outs

    # Walk alphas in order and apply only the INCREMENT, so the base is loaded
    # once rather than 7 times.
    # Partial results are written to the volume after every alpha.  A timeout
    # used to lose the whole direction and systemd then re-ran all nine from
    # scratch; now a rerun fast-forwards through the alphas it already has,
    # applying the weight increment but skipping generation.
    part = f"/oct/steerfix/{name}.json"
    os.makedirs("/oct/steerfix", exist_ok=True)
    res = {}
    if os.path.exists(part):
        try:
            res = json.load(open(part))
            print(f"[{name}] resuming with {len(res)} alpha(s) already done", flush=True)
        except Exception:
            res = {}

    prev = 0.0
    for a in sorted(alphas):
        step = (a - prev) * job["ref"]
        if step:
            with torch.no_grad():
                for m, pn in hit.items():
                    params[pn] += (D[m] * step).to(params[pn].dtype).to(params[pn].device)
        prev = a
        if str(a) in res and len(res[str(a)]) == len(prompts):
            print(f"[{name}] alpha={a} cached", flush=True)
            continue
        res[str(a)] = generate()
        with open(part, "w") as f:
            json.dump(res, f)
        oct_vol.commit()
        print(f"[{name}] alpha={a} done", flush=True)
    return {"name": name, "source": source, "alphas": alphas,
            "prompts": prompts, "generations": res, "dir_norm_raw": norm,
            "add_adapter": add, "add_adapter_norm": add_norm}


@app.local_entrypoint()
def main(spec: str = "phase10_runs/steer_spec.json",
         out: str = "phase10_runs/steer_results_fix.json"):
    here = os.path.dirname(os.path.abspath(__file__))
    S = json.load(open(os.path.join(here, spec)))
    jobs = S["jobs"]
    print(f"{len(jobs)} directions x {len(jobs[0]['alphas'])} alphas "
          f"on source={jobs[0]['source']}", flush=True)
    res = list(steer.map(jobs))
    with open(os.path.join(here, out), "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {out}", flush=True)
