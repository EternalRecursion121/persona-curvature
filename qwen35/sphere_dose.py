#!/usr/bin/env python3
"""Dose calibration for the iso-KL sphere: KL per token at six alphas, 72 directions.

A copy of fisher_dose.py with three edits and no change to any arithmetic, so
the measurement is on identical text at identical token positions:

  1. THE PARTIAL CHECKPOINT IS NAMESPACED BY THE SPEC'S `tag`.  fisher_dose.py
     hardcodes /oct/dosecalib/partial.json.  Re-running it here would overwrite
     the 2026-09-09 matched-dose calibration's partial on the volume -- the same
     class of hazard sphere_sweep.py's resume path had, caught there before it
     cost anything.  Untagged runs keep the original path.
  2. THE PROGRESS LINE NO LONGER ASSUMES BOTH WEIGHT CONSTRUCTIONS.  The original
     reads rows["exact|<alpha>"] unconditionally, so modes=["bf16"] raises
     KeyError after the first direction is measured.  This run needs bf16 only:
     sphere_sweep.py writes (W0 + D*s).to(bfloat16), so the bf16 curve is the
     dose the generating model actually receives, and measuring the exact curve
     as well would double the bill for a number nothing here reads.
  3. A DEFAULT APP NAME OF ITS OWN, so the spend meter and the kill switch see
     this run as a separate unit.

Everything else -- the fixed 24 sequences, the alpha-0 responses under PC1, the
192-token cap, the fp32 forward with TF32 off, the host-side stage-one cache,
the zero-alpha control -- is fisher_dose.py's, unchanged.
"""
import json
import math
import os
import time

import modal

BASE_MODEL = os.environ.get("PC_BASE_MODEL", "Qwen/Qwen3.5-4B")
GPU_TYPE = os.environ.get("PC_GPU", "A100-40GB")
APP_NAME = os.environ.get("PC_APP_NAME", "pc-qwen35-phase10-sphere-isokl-calib")

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
    .add_local_file(os.path.abspath(__file__), "/root/sphere_dose.py", copy=True)
    .run_function(_download_base_model)
)


def _strip(mod):
    while mod.startswith("base_model.model."):
        mod = mod[len("base_model.model."):]
    return mod


def _open_set(tmpl, traits):
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
def dose(S: dict) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()
    MAX_MINUTES = float(os.environ.get("PC_MAX_MINUTES", "150"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    REF = S["ref"]
    MAXR = S["max_resp_tokens"]
    dirs = S["directions"]
    MODES = S["modes"]
    snapshot_download(BASE_MODEL)

    def note(m):
        print(f"[{time.time()-t_start:8.1f}s] {m}", flush=True)

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
    note(f"{len(seqs)} sequences, {n_scored} scored response tokens")

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32,
                                                 device_map="cuda")
    model.eval()
    model.requires_grad_(False)
    params = dict(model.named_parameters())

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

    base_lp, base_am = [], []
    for i in range(len(seqs)):
        lp = logprobs(i)
        base_lp.append(lp)
        base_am.append(lp.argmax(-1))
    note(f"base log-probs held on GPU, cuda alloc "
         f"{torch.cuda.memory_allocated()/2**30:.1f} GiB")

    @torch.no_grad()
    def measure():
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
            del lps, pb, ps
        return (kl / n_scored, (kl + rkl) / n_scored, ch / n_scored)

    h0, k0, scale1, rank1 = _open_set("/adapters/{t}", ["agreeable"])
    mods = sorted(k0["agreeable"])
    hit = {m: find(m) for m in mods}
    missing = [m for m, v in hit.items() if v is None]
    if missing:
        raise RuntimeError(f"unmapped modules: {missing[:4]} ({len(missing)} total)")

    def host(t):
        t = t.contiguous()
        try:
            return t.pin_memory()
        except RuntimeError:
            return t

    base_cpu = {m: host(params[hit[m]].detach().to(torch.bfloat16).cpu().clone())
                for m in mods}
    dense = sum(params[hit[m]].numel() for m in mods)
    D = {m: torch.zeros_like(params[hit[m]], dtype=torch.bfloat16) for m in mods}
    note(f"{len(mods)} modules, {dense/1e9:.3f}B targeted params, host base copy "
         f"{dense*2/2**30:.1f} GiB, delta buffer {dense*2/2**30:.1f} GiB")

    traits134 = S["traits134"]
    handles, keymap, scale1, rank1 = _open_set("/adapters/{t}", traits134)
    Acat, Bcat = {}, {}
    t0 = time.time()
    for m in mods:
        A = torch.cat([handles[t].get_tensor(keymap[t][m] + ".lora_A.weight")
                       for t in traits134], dim=0).to(torch.bfloat16)
        B = torch.cat([handles[t].get_tensor(keymap[t][m] + ".lora_B.weight")
                       for t in traits134], dim=1).to(torch.bfloat16)
        Acat[m], Bcat[m] = host(A), host(B)
    del handles
    note(f"stage-one cache built in {time.time()-t0:.0f}s, "
         f"{sum(Acat[m].numel()+Bcat[m].numel() for m in mods)*2/2**30:.1f} GiB host bf16")
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
    def set_alpha(k, mode):
        """W = W_base + k*D, either exactly (fp32) or as steer_fix.py leaves it.

        Returns the realised ||W - W_base||_F, which is |k| * ||D||_F when the
        weights are exact and less than that when bf16 swallows part of the
        increment.  Both branches keep the PARAMETER dtype fp32 so the forward
        pass and therefore the KL are identical apart from the weights.
        """
        moved = 0.0
        for m in mods:
            p = params[hit[m]]
            b = base_cpu[m].to(dev, non_blocking=True).float()
            if k == 0.0:
                p.copy_(b)
            elif mode == "exact":
                p.copy_(b)
                p.add_(D[m], alpha=k)
            elif mode == "bf16":
                w = b + D[m].float() * k
                p.copy_(w.to(torch.bfloat16).float())
                del w
            else:
                raise RuntimeError(mode)
            moved += float((p - b).pow(2).sum())
            del b
        return math.sqrt(moved)

    set_alpha(0.0, "exact")
    z_kl, z_sym, z_ch = measure()
    note(f"ZERO-ALPHA CONTROL kl={z_kl:.3e} symkl={z_sym:.3e} argmax={z_ch:.3e}")

    out = {}
    tag = S.get("tag")
    pdir = "/oct/dosecalib" + (f"_{tag}" if tag else "")
    part = f"{pdir}/partial.json"
    os.makedirs(pdir, exist_ok=True)
    note(f"partial checkpoint {part}")
    for di, d in enumerate(dirs):
        if (time.time() - t_start) / 60 > MAX_MINUTES:
            note(f"WALL BUDGET reached after {di} directions")
            break
        t0 = time.time()
        nrm = build_stage1(d["coef"])
        rows = {}
        for mode in MODES:
            for a in d["alphas"]:
                k = a * REF / nrm
                moved = set_alpha(k, mode)
                kl, sym, ch = measure()
                rows[f"{mode}|{a}"] = {
                    "mode": mode, "alpha": a, "kl": kl, "symkl": sym,
                    "argmax_change": ch, "F_alpha": 2.0 * kl / (a * a),
                    "moved_frobenius": moved,
                    "retention": moved / (abs(a) * REF)}
        out[d["name"]] = {"family": d.get("family"), "coef_n": len(d["coef"]),
                          "dir_norm_raw": nrm, "published_ref": d.get("published_ref"),
                          "alphas": d["alphas"], "per_alpha": rows}
        m0 = MODES[0]
        ex = [rows[f"{m0}|{a}"]["kl"] for a in d["alphas"]]
        rt = [rows[f"bf16|{a}"]["retention"] for a in d["alphas"]] if "bf16" in MODES else [float("nan")]
        note(f"{di+1}/{len(dirs)} {d['name']:26s} |D|={nrm:7.3f} "
             f"kl({m0}) {min(ex):.4f}..{max(ex):.4f} "
             f"retention(bf16) {min(rt):.3f}..{max(rt):.3f} ({time.time()-t0:.0f}s)")
        with open(part, "w") as f:
            json.dump(out, f)
        oct_vol.commit()

    return {"ref": REF, "max_resp_tokens": MAXR, "n_scored_tokens": n_scored,
            "resp_tokens": [s[2] for s in seqs], "prompts": S["prompts"],
            "text_source": S["text_source"], "lora_rank": rank1,
            "lora_scale": scale1, "targeted_modules": len(mods),
            "targeted_params": dense, "modes": MODES,
            "zero_alpha_control": {"kl": z_kl, "symkl": z_sym, "argmax_change": z_ch},
            "retention_definition":
                "||W_set - W_base||_F / (|alpha| * ref); 1.0 means the whole "
                "intended increment survived the weight dtype, less means it did not",
            "wall_seconds": time.time() - t_start, "results": out}


@app.local_entrypoint()
def main(spec: str = "phase10_runs/sphere_isokl_dose_spec.json",
         out: str = "phase10_runs/sphere_isokl_calib.json"):
    here = os.path.dirname(os.path.abspath(__file__))
    S = json.load(open(os.path.join(here, spec)))
    npass = len(S["directions"]) * len(S["directions"][0]["alphas"]) * len(S["modes"])
    print(f"{len(S['directions'])} directions, {npass} (direction, mode, alpha) passes, "
          f"modes={S['modes']}, ref={S['ref']}, tag={S.get('tag')}", flush=True)
    R = dose.remote(S)
    with open(os.path.join(here, out), "w") as f:
        json.dump(R, f, indent=1)
    print(f"wrote {out} ({len(R['results'])} directions, "
          f"{R['wall_seconds']/60:.1f} min)", flush=True)
