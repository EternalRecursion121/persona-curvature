#!/usr/bin/env python3
"""Sample many directions on the personality sphere, in one container.

WHY THIS IS NOT JUST steer_fix.py WITH MORE JOBS
------------------------------------------------
steer_fix builds each direction by reading all 134 adapters from the volume:
about 43 GB per direction. That is fine for nine directions and absurd for
seventy-two, where it becomes three terabytes of volume reads and seventy-two
GPUs sitting idle while they wait for I/O.

Every direction on this sphere is a combination of exactly three things -- the
three basis vectors named in the spec (the first three principal components for
the 2026-09-01 run, the first three factor-chart basis vectors for the
2026-09-08 `_fa` redo). So the adapters are read ONCE, collapsed
immediately into three dense basis deltas, and then every sampled direction is
a cheap weighted sum of those three. One container, one pass over the volume,
and the GPU spends its time generating instead of waiting.

The base weights are kept as a CPU copy and restored between directions rather
than incrementally patched, so seventy-two rounds of bfloat16 addition cannot
accumulate drift into the later samples.
"""
import json
import math
import os

import modal

BASE_MODEL = os.environ.get("PC_BASE_MODEL", "Qwen/Qwen3.5-4B")
APP_NAME = os.environ.get("PC_APP_NAME", "pc-qwen35-phase10-sphere")

app = modal.App(APP_NAME)
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep", create_if_missing=False)
oct_vol = modal.Volume.from_name("pc-qwen35-oct2", create_if_missing=True)
VOLS = {"/adapters": sweep_vol, "/oct": oct_vol}


def _dl():
    from huggingface_hub import snapshot_download
    snapshot_download(BASE_MODEL)


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.13.0", "transformers==5.15.1", "accelerate==1.14.0",
                 "safetensors", "hf_transfer", "numpy<3", "huggingface_hub")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "PC_BASE_MODEL": BASE_MODEL})
    .add_local_file(os.path.abspath(__file__), "/root/sphere_sweep.py", copy=True)
    .run_function(_dl)
)


@app.function(image=image, gpu="A100-40GB", volumes=VOLS, timeout=60 * 60 * 20,
              memory=65536, cpu=8.0, secrets=[modal.Secret.from_name("hf-token")])
def sweep(job: dict) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    basis, points = job["basis"], job["points"]        # basis: list of coef dicts
    prompts, alpha, ref = job["prompts"], job["alpha"], job["ref"]
    traits = sorted(basis[0]["coef"])
    with open(f"/adapters/{traits[0]}/adapter_config.json") as f:
        ac = json.load(f)
    scale = ac["lora_alpha"] / (math.sqrt(ac["r"]) if ac.get("use_rslora") else ac["r"])
    H = {t: safe_open(f"/adapters/{t}/adapter_model.safetensors", framework="pt")
         for t in traits}
    mods = sorted({k.split(".lora_A.")[0].removeprefix("base_model.model.")
                   for k in H[traits[0]].keys() if ".lora_A." in k})
    print(f"[build] {len(basis)} basis directions over {len(mods)} modules "
          f"from {len(traits)} adapters", flush=True)

    # ---- one pass over the volume, collapsing straight into the basis --------
    P = {}
    for n, m in enumerate(mods):
        As = torch.cat([H[t].get_tensor(f"base_model.model.{m}.lora_A.weight").float()
                        for t in traits], dim=0).cuda()
        Braw = [H[t].get_tensor(f"base_model.model.{m}.lora_B.weight").float()
                for t in traits]
        stack = []
        for b in basis:
            Bs = torch.cat([Braw[i] * (scale * b["coef"][t])
                            for i, t in enumerate(traits)], dim=1).cuda()
            # float16: these are weight deltas around 1e-3, and three of them
            # in float32 would be 38 GB of resident memory for no accuracy that
            # survives the bfloat16 model they are added to.
            stack.append((Bs @ As).to(torch.float16).cpu())
            del Bs
        P[m] = torch.stack(stack)                       # (n_basis, d_out, d_in)
        del As, Braw, stack
        if n % 40 == 0:
            print(f"  module {n}/{len(mods)}", flush=True)
    del H
    print(f"[build] basis complete, {sum(p.numel() for p in P.values())*2/1e9:.1f} GB resident",
          flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    params = dict(model.named_parameters())

    def find(m):
        for c in (m + ".weight", "model." + m + ".weight",
                  m.replace("model.", "model.language_model.", 1) + ".weight"):
            if c in params:
                return c
    hit = {m: find(m) for m in mods}
    missing = [m for m, v in hit.items() if v is None]
    if missing:
        raise RuntimeError(f"unmapped modules: {missing[:4]} ({len(missing)} total)")
    W0 = {m: params[hit[m]].detach().clone().cpu() for m in mods}   # pristine copy
    print(f"[model] loaded, {len(mods)} modules mapped, base weights copied", flush=True)

    # The resume checkpoint MUST be namespaced by chart. When the factor-chart
    # sphere was added (2026-09-08) a shared "/oct/sphere" would have found 72
    # finished points from the 2026-09-01 principal-component run and returned
    # those generations under the new point names, silently, at no cost and with
    # no error. Untagged runs keep the original path so the PC run stays resumable.
    tag = job.get("tag")
    pdir = "/oct/sphere" + (f"_{tag}" if tag else "")
    part = f"{pdir}/results.json"
    os.makedirs(pdir, exist_ok=True)
    res = {}
    alphas_used = {}
    if os.path.exists(part):
        try:
            res = json.load(open(part))
            print(f"[resume] {len(res)} points already done", flush=True)
        except Exception:
            res = {}

    for pi, pt in enumerate(points):
        if pt["name"] in res:
            continue
        u = torch.tensor(pt["u"], dtype=torch.float16).cuda()
        sq = 0.0
        D = {}
        for m in mods:
            d = torch.einsum("k,kod->od", u, P[m].cuda()).float()
            D[m] = d
            sq += float((d * d).sum())
        # PER-POINT ALPHA.  The 2026-09-01 and 2026-09-08 spheres steered every
        # point at the spec's single `alpha`; the 2026-09-10 iso-KL sphere gives
        # each point its own, solved so that all 72 deliver the same measured KL
        # per token.  A point with alpha 0 sets s = 0 and therefore restores the
        # pristine bf16 base exactly, which is how that run generates its base
        # condition inside the same container as the steered text.
        a_pt = float(pt.get("alpha", alpha))
        alphas_used[pt["name"]] = a_pt
        s = a_pt * ref / math.sqrt(sq)
        with torch.no_grad():
            for m in mods:
                params[hit[m]].copy_((W0[m].cuda() + D[m] * s).to(params[hit[m]].dtype))
        del D
        outs = []
        for p in prompts:
            enc = tok(tok.apply_chat_template([{"role": "user", "content": p}],
                                              tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False),
                      return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = model.generate(**enc, do_sample=False, max_new_tokens=512,
                                   pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
        res[pt["name"]] = outs
        with open(part, "w") as f:
            json.dump(res, f)
        oct_vol.commit()
        print(f"[{pi+1}/{len(points)}] {pt['name']} alpha={a_pt:.4f} done "
              f"({sum(len(x) for x in outs)} chars)", flush=True)

    # leave the model as we found it, in case anything downstream reuses it
    with torch.no_grad():
        for m in mods:
            params[hit[m]].copy_(W0[m].cuda().to(params[hit[m]].dtype))
    return {"n": len(res), "alpha": alpha, "alphas": alphas_used,
            "prompts": prompts, "generations": res}


@app.local_entrypoint()
def main(spec: str = "phase10_runs/sphere_sweep_spec.json",
         out: str = "phase10_runs/sphere_results.json"):
    here = os.path.dirname(os.path.abspath(__file__))
    J = json.load(open(os.path.join(here, spec)))
    pa = [p["alpha"] for p in J["points"] if "alpha" in p]
    astr = (f"per-point alpha {min(pa):.4f}..{max(pa):.4f} over {len(pa)} points"
            if pa else f"alpha={J['alpha']}")
    print(f"{len(J['points'])} points x {len(J['prompts'])} prompts "
          f"at {astr}  tag={J.get('tag') or '(pc, untagged)'}  "
          f"basis={[b['name'] for b in J['basis']]}", flush=True)
    r = sweep.remote(J)
    with open(os.path.join(here, out), "w") as f:
        json.dump(r, f)
    print(f"wrote {out}: {r['n']} points", flush=True)
