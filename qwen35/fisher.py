#!/usr/bin/env python3
"""Fisher norms of the persona steering directions.

Every steering direction in this project is a UNIT VECTOR IN THE WEIGHT-SPACE
FROBENIUS METRIC.  That metric is arbitrary from the model's point of view: what
the model cares about is how far the output distribution moves.  The natural
metric there is the Fisher information, and for a one-parameter family
theta(alpha) = theta_0 + alpha * ref * u,

    KL( p_base || p_{theta(alpha)} )  =  0.5 * alpha^2 * F(u) + O(alpha^3),
    F(u) = (ref^2) * u^T I(theta_0) u   in units of "per ref-unit alpha".

F(u) is measured, not derived: run the model at four to six small alphas on a
FIXED set of tokens, take the mean per-token KL over the response positions, and
fit F = 2 * KL / alpha^2 pooled over the alphas.  Nothing is generated, so every
direction is scored on identical tokens and the KL is not confounded by the
steered model wandering onto different text.

WHY THE WEIGHTS ARE HELD IN FP32
--------------------------------
steer_fix.py adds the steering increment straight into the bf16 parameters.  A
unit-Frobenius direction spread over ~3e9 elements has RMS element ~1.7e-5, so
at alpha 0.125 the intended increment is ~1.7e-6 while one bf16 ulp at a typical
weight is ~1e-4.  Round-to-nearest throws almost all of it away.  That is fatal
for a second-order measurement, and it is also why the seven-alpha walk in
steer_fix.py does not return to the base model at alpha 0: the nine stored "0.0"
generations in steer_results_fix.json are all different from one another.  Here
the base weights are held in fp32 on the GPU, a bf16 copy of the base (which is
exact -- the checkpoint is bf16) is kept on the host, and each alpha is set as
W = W_base + (alpha*ref/||D||) * D from that copy rather than by accumulating
increments.  The forward pass runs in fp32 with TF32 off, because TF32 rounds
GEMM inputs to 10 mantissa bits and would erase the increment just as bf16 does.

COST SHAPE
----------
One container, one model load.  The 134 stage-one adapters are read once into a
host-side bf16 cache of per-module concatenated factors; a direction is then one
(d_out x 134r) @ (134r x d_in) product per module, the same arithmetic
steer_fix.py does.  The two stage-two directions stream their own adapters from
the volume, since a second 35 GB cache would not pay for two directions.
"""
import json
import math
import os
import time

import modal

BASE_MODEL = os.environ.get("PC_BASE_MODEL", "Qwen/Qwen3.5-4B")
GPU_TYPE = os.environ.get("PC_GPU", "A100-40GB")
# must start with pc-qwen35 or the spend meter's kill switch cannot see it
APP_NAME = os.environ.get("PC_APP_NAME", "pc-qwen35-phase10-fisher")

app = modal.App(APP_NAME)
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep", create_if_missing=False)
oct_vol = modal.Volume.from_name("pc-qwen35-oct2", create_if_missing=True)
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
    .add_local_file(os.path.abspath(__file__), "/root/fisher.py", copy=True)
    .run_function(_download_base_model)
)

SRC = {"stage1": "/adapters/{t}", "persona": "/oct/personas/{t}/persona",
       "alignment": "/trained/data_alignment/{t}",
       "stage2": "/oct/loras_introspection/{t}",
       "persona_exact": "/oct/personas_exact/{t}"}


def _strip(mod):                      # copied verbatim from steer_fix.py
    while mod.startswith("base_model.model."):
        mod = mod[len("base_model.model."):]
    return mod


def _open_set(tmpl, traits):          # copied verbatim from steer_fix.py
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
    return handles, keymap, scale, ac["r"]


@app.function(secrets=[modal.Secret.from_name("hf-token")], image=image,
              gpu=GPU_TYPE, volumes=VOLS, timeout=60 * 200,
              memory=53248, cpu=2.0)
def fisher(S: dict) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()
    MAX_MINUTES = float(os.environ.get("PC_MAX_MINUTES", "110"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    REF = S["ref"]
    MAXR = S["max_resp_tokens"]
    dirs = S["directions"]
    snapshot_download(BASE_MODEL)

    def note(m):
        print(f"[{time.time()-t_start:8.1f}s] {m}", flush=True)

    # ---- fixed token sequences ------------------------------------------
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    seqs = []
    for p, t in zip(S["prompts"], S["texts"]):
        pre = tok(tok.apply_chat_template([{"role": "user", "content": p}],
                                          tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False),
                  return_tensors="pt")["input_ids"][0]
        rsp = tok(t, add_special_tokens=False, return_tensors="pt")["input_ids"][0][:MAXR]
        seqs.append((torch.cat([pre, rsp]).to(dev), int(pre.shape[0]), int(rsp.shape[0])))
    n_scored = sum(s[2] for s in seqs)
    note(f"{len(seqs)} sequences, {n_scored} scored response tokens, "
         f"resp lens {[s[2] for s in seqs]}")

    # ---- base model, fp32 ------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32,
                                                 device_map="cuda")
    model.eval()
    model.requires_grad_(False)
    params = dict(model.named_parameters())
    note(f"model loaded fp32, {sum(p.numel() for p in params.values())/1e9:.3f}B params, "
         f"cuda alloc {torch.cuda.memory_allocated()/2**30:.1f} GiB")

    def find(m):
        for c in (m + ".weight", "model." + m + ".weight",
                  m.replace("model.", "model.language_model.", 1) + ".weight"):
            if c in params:
                return c
        return None

    @torch.no_grad()
    def logprobs(i):
        ids, npre, nr = seqs[i]
        lg = model(ids.unsqueeze(0), use_cache=False).logits[0][npre - 1:npre - 1 + nr]
        return torch.log_softmax(lg.float(), dim=-1)

    # ---- base pass, kept on the GPU -------------------------------------
    base_lp, base_am = [], []
    for i in range(len(seqs)):
        lp = logprobs(i)
        base_lp.append(lp)
        base_am.append(lp.argmax(-1))
    note(f"base log-probs held on GPU, "
         f"{sum(x.numel() for x in base_lp)*4/2**30:.2f} GiB, "
         f"cuda alloc {torch.cuda.memory_allocated()/2**30:.1f} GiB")

    @torch.no_grad()
    def measure():
        """mean per-token KL(base||steered), symmetric KL, argmax change rate."""
        kl = rkl = 0.0
        ch = 0
        for i in range(len(seqs)):
            lps = logprobs(i)
            lpb = base_lp[i]
            pb = lpb.exp()
            kl += float((pb * (lpb - lps)).sum())
            ps = lps.exp()
            rkl += float((ps * (lps - lpb)).sum())
            ch += int((lps.argmax(-1) != base_am[i]).sum())
            del lps, lpb, pb, ps
        return (kl / n_scored, (kl + rkl) / n_scored, ch / n_scored)

    # ---- host-side base copy of every LoRA-targeted weight ---------------
    h0, k0, scale1, rank1 = _open_set(SRC["stage1"], ["agreeable"])
    mods = sorted(k0["agreeable"])
    hit = {m: find(m) for m in mods}
    missing = [m for m, v in hit.items() if v is None]
    if missing:
        raise RuntimeError(f"unmapped modules: {missing[:4]} ({len(missing)} total)")
    def host(t):
        t = t.contiguous()
        try:
            return t.pin_memory()      # ~5x the H2D bandwidth; falls back if refused
        except RuntimeError:
            return t
    base_cpu = {m: host(params[hit[m]].detach().to(torch.bfloat16).cpu().clone())
                for m in mods}
    dense = sum(params[hit[m]].numel() for m in mods)
    note(f"{len(mods)} modules, {dense/1e9:.3f}B targeted params, "
         f"lora r={rank1} scale={scale1}, host base copy "
         f"{dense*2/2**30:.1f} GiB")

    # ---- delta buffer, bf16 on the GPU ----------------------------------
    D = {m: torch.zeros_like(params[hit[m]], dtype=torch.bfloat16) for m in mods}
    note(f"delta buffer {dense*2/2**30:.1f} GiB, "
         f"cuda alloc {torch.cuda.memory_allocated()/2**30:.1f} GiB")

    # ---- stage-one adapter cache: concatenated factors, bf16, host ------
    traits134 = S["traits134"]
    handles, keymap, scale1, rank1 = _open_set(SRC["stage1"], traits134)
    Acat, Bcat = {}, {}
    t0 = time.time()
    for m in mods:
        A = torch.cat([handles[t].get_tensor(keymap[t][m] + ".lora_A.weight")
                       for t in traits134], dim=0).to(torch.bfloat16)
        B = torch.cat([handles[t].get_tensor(keymap[t][m] + ".lora_B.weight")
                       for t in traits134], dim=1).to(torch.bfloat16)
        Acat[m], Bcat[m] = host(A), host(B)
    del handles                       # drop 134 mmaps before the sweep starts
    cache_gb = sum(Acat[m].numel() + Bcat[m].numel() for m in mods) * 2 / 2**30
    note(f"stage-one cache built in {time.time()-t0:.0f}s, {cache_gb:.1f} GiB host bf16")
    idx134 = {t: i for i, t in enumerate(traits134)}

    @torch.no_grad()
    def build_stage1(coef):
        c = torch.zeros(len(traits134))
        for t, v in coef.items():
            c[idx134[t]] = v
        c = (c * scale1).to(dev).repeat_interleave(rank1)
        sq = 0.0
        for m in mods:
            B = Bcat[m].to(dev, non_blocking=True)
            A = Acat[m].to(dev, non_blocking=True)
            acc = ((B.float() * c).to(torch.bfloat16) @ A)
            sq += float(acc.float().pow(2).sum())
            D[m].copy_(acc)
            del B, A, acc
        return math.sqrt(sq)

    @torch.no_grad()
    def build_other(coef, source):
        traits = sorted(coef)
        hs, km, sc, r = _open_set(SRC[source], traits)
        if sorted(km[traits[0]]) != mods:
            raise RuntimeError(f"{source} targets a different module set")
        c = torch.tensor([coef[t] for t in traits]).to(dev).repeat_interleave(r) * sc
        sq = 0.0
        for m in mods:
            B = torch.cat([hs[t].get_tensor(km[t][m] + ".lora_B.weight")
                           for t in traits], dim=1).to(dev)
            A = torch.cat([hs[t].get_tensor(km[t][m] + ".lora_A.weight")
                           for t in traits], dim=0).to(dev)
            acc = ((B * c).to(torch.bfloat16) @ A.to(torch.bfloat16))
            sq += float(acc.float().pow(2).sum())
            D[m].copy_(acc)
            del B, A, acc
        del hs
        return math.sqrt(sq)

    @torch.no_grad()
    def set_alpha(k):
        """W = W_base + k * D, from the host base copy -- never accumulated."""
        for m in mods:
            p = params[hit[m]]
            p.copy_(base_cpu[m], non_blocking=True)
            if k:
                p.add_(D[m], alpha=k)

    # ---- zero-alpha control: the noise floor of the whole pipeline -------
    set_alpha(0.0)
    z_kl, z_sym, z_ch = measure()
    note(f"ZERO-ALPHA CONTROL kl={z_kl:.3e} symkl={z_sym:.3e} argmax_change={z_ch:.3e} "
         f"(must be ~0; anything larger is the measurement's noise floor)")

    # ---- the sweep -------------------------------------------------------
    out, done = {}, 0
    part = "/oct/fisher/fisher_partial.json"
    os.makedirs("/oct/fisher", exist_ok=True)

    def meta():
        return {"ref": REF, "max_resp_tokens": MAXR, "n_scored_tokens": n_scored,
                "resp_tokens": [s_[2] for s_ in seqs], "prompts": S["prompts"],
                "text_source": S["text_source"], "lora_rank": rank1,
                "lora_scale": scale1, "targeted_modules": len(mods),
                "targeted_params": dense, "n_stage1_adapters": len(traits134),
                "zero_alpha_control": {"kl": z_kl, "symkl": z_sym,
                                       "argmax_change": z_ch},
                "wall_seconds": time.time() - t_start}

    def write_part():
        with open(part, "w") as f:
            json.dump(dict(meta(), results=out), f)
        oct_vol.commit()
    for d in dirs:
        if (time.time() - t_start) / 60 > MAX_MINUTES:
            note(f"WALL BUDGET {MAX_MINUTES} min reached, stopping after {done} directions")
            break
        t0 = time.time()
        try:
            if d["source"] == "stage1":
                nrm = build_stage1(d["coef"])
            else:
                nrm = build_other(d["coef"], d["source"])
            rows = {}
            for a in d["alphas"]:
                set_alpha(a * REF / nrm)
                kl, sym, ch = measure()
                rows[str(a)] = {"kl": kl, "symkl": sym, "argmax_change": ch,
                                "F_alpha": 2.0 * kl / (a * a)}
        except Exception as e:                 # one bad direction is not the run
            note(f"FAILED {d['name']}: {type(e).__name__}: {e}")
            out[d["name"]] = {"family": d["family"], "source": d["source"],
                              "error": f"{type(e).__name__}: {e}"}
            done += 1
            continue
        out[d["name"]] = {"family": d["family"], "source": d["source"],
                          "dir_norm_raw": nrm, "published_ref": d.get("published_ref"),
                          "seed": d.get("seed"), "u": d.get("u"),
                          "alphas": d["alphas"], "per_alpha": rows}
        done += 1
        fs = [rows[str(a)]["F_alpha"] for a in d["alphas"]]
        note(f"{done}/{len(dirs)} {d['name']:24s} |D|={nrm:7.3f} "
             f"F={sum(fs)/len(fs):.4g} spread={max(fs)/max(min(fs),1e-30):.3f} "
             f"({time.time()-t0:.0f}s)")
        if done % 5 == 0 or done == len(dirs):
            write_part()

    write_part()
    return dict(meta(), results=out)


@app.local_entrypoint()
def main(spec: str = "phase10_runs/fisher_spec.json",
         out: str = "phase10_runs/fisher_results.json"):
    here = os.path.dirname(os.path.abspath(__file__))
    S = json.load(open(os.path.join(here, spec)))
    print(f"{len(S['directions'])} directions, "
          f"{sum(len(d['alphas']) for d in S['directions'])} (direction,alpha) passes",
          flush=True)
    R = fisher.remote(S)
    with open(os.path.join(here, out), "w") as f:
        json.dump(R, f, indent=1)
    print(f"wrote {out} ({len(R['results'])} directions, "
          f"{R['wall_seconds']/60:.1f} min)", flush=True)
