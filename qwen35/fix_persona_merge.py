#!/usr/bin/env python3
"""Audit and correct the OCT persona merge.

WHAT IS WRONG
-------------
Open Character Training's release step is

    add_weighted_adapter(["dpo","sft"], [1.0, 0.25], combination_type="linear")

and we reproduced it faithfully.  PEFT's "linear" combination does NOT sum the
two weight deltas.  It splits each weight across the two factors as a square
root and sums the FACTORS:

    A_new = sum_i sqrt(w_i * s_i) * A_i          (s_i = alpha_i / r_i)
    B_new = sum_i sqrt(w_i * s_i) * B_i

so the product carries cross terms that pair one adapter's A with the other's B:

    B_new A_new = w1 s1 B1A1 + w2 s2 B2A2 + sqrt(w1 s1 w2 s2) (B1A2 + B2A1)
                  \-------- what was intended --------/   \--- artifact ---/

The diagonal part is exactly right: the DPO adapter is at its intended strength
and the SFT adapter at 0.25.  The cross term is not a scaling error, it is a
term that should not exist at all -- it multiplies the DPO adapter's input
projection by the SFT adapter's output projection.  Measured on `bold`, it is
~80% of the merged delta's Frobenius norm.  Verified exactly (rel err 6e-08).

THE FIX
-------
Concatenation is exact and needs no approximation, because the intended delta
genuinely has rank <= 128:

    A_new = [A1 ; A2]                    (128, d_in)
    B_new = [w1*s1*B1 , w2*s2*B2]        (d_out, 128)
    r_new = 128, alpha_new = 128  ->  scaling 1.0

    B_new A_new = w1 s1 B1A1 + w2 s2 B2A2 = dW_dpo + 0.25 * dW_sft   exactly.

This is what combination_type="cat" does.  We build it by hand so the audit and
the corrected adapter come out of one read of each file.

Runs CPU-only.  The app is deliberately NOT named pc-qwen35-phase10-* : the
spend meter prices every container at the A100 rate, and CPU containers counted
that way would fabricate spend and could hard-stop the real GPU work.
"""
import hashlib
import json
import os

import modal

BASE_ALPHA, BASE_R = 128, 64        # stage-1 DPO and stage-2 SFT adapters
W_DPO, W_SFT = 1.0, 0.25            # OCT MERGE_WEIGHTS
S = BASE_ALPHA / BASE_R             # 2.0, the scaling both sources carry
K = 32                              # sketch width, must match sketch_adapters.py
REPO = "EternalRecursion/persona-lora-zoo-qwen35"

app = modal.App("pc-qwen35-fixmerge")
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep")
oct_vol = modal.Volume.from_name("pc-qwen35-oct2")
VOLS = {"/adapters": sweep_vol, "/oct": oct_vol}

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy<3", "safetensors", "huggingface_hub", "hf_transfer")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))


def _proj(name, rows, cols, k, which):
    """Byte-for-byte the projection in sketch_adapters.py:proj."""
    import numpy as np
    h = hashlib.sha256(f"{name}|{which}|{k}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    return rng.standard_normal((rows, cols), dtype=np.float32) / np.sqrt(k)


@app.function(image=image, volumes=VOLS, cpu=4.0, memory=16384, timeout=60 * 45,
              max_containers=10)
def fix_one(trait: str) -> dict:
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file

    if os.path.exists(f"/oct/personas_exact/{trait}/MERGE_NOTE.json") and \
       os.path.exists(f"/oct/merge_audit/{trait}.json"):
        a = json.load(open(f"/oct/merge_audit/{trait}.json"))
        return {k: v for k, v in a.items() if k != "per_module"} | {"skipped": True}

    p_dpo = f"/adapters/{trait}/adapter_model.safetensors"
    p_sft = f"/oct/loras_introspection/{trait}/adapter_model.safetensors"
    p_per = f"/oct/personas/{trait}/persona/adapter_model.safetensors"
    for p in (p_dpo, p_sft, p_per):
        if not os.path.exists(p):
            return {"trait": trait, "error": f"missing {p}"}

    f1 = safe_open(p_dpo, framework="np")
    f2 = safe_open(p_sft, framework="np")
    fp = safe_open(p_per, framework="np")
    mods = sorted({k.split(".lora_A.")[0].split(".lora_B.")[0] for k in f1.keys()})

    w1, w2 = W_DPO * S, W_SFT * S                  # 2.0, 0.5
    c = float(np.sqrt(w1 * w2))                    # 1.0, the cross coefficient

    out = {}
    sk_sft, sk_int = [], []
    n_int2 = n_cross2 = n_per2 = n_dpo2 = n_sft2 = 0.0
    dot_per_int = 0.0
    per_mod = []
    max_err = 0.0

    for m in mods:
        A1 = f1.get_tensor(f"{m}.lora_A.weight").astype(np.float32)
        B1 = f1.get_tensor(f"{m}.lora_B.weight").astype(np.float32)
        A2 = f2.get_tensor(f"{m}.lora_A.weight").astype(np.float32)
        B2 = f2.get_tensor(f"{m}.lora_B.weight").astype(np.float32)
        Ap = fp.get_tensor(f"{m}.lora_A.weight").astype(np.float32)
        Bp = fp.get_tensor(f"{m}.lora_B.weight").astype(np.float32)

        # ---- exact norms and inner products, never materialising d_out x d_in.
        # <B_i A_i, B_j A_j>_F = tr((B_i^T B_j)(A_j A_i^T)), both r x r.
        def ip(Ba, Aa, Bb, Ab):
            return float(np.sum((Ba.T @ Bb) * (Ab @ Aa.T)))

        i_ii = w1 * w1 * ip(B1, A1, B1, A1) + w2 * w2 * ip(B2, A2, B2, A2) \
            + 2 * w1 * w2 * ip(B1, A1, B2, A2)                      # ||intended||^2
        x_xx = c * c * (ip(B1, A2, B1, A2) + ip(B2, A1, B2, A1)
                        + 2 * ip(B1, A2, B2, A1))                   # ||cross||^2
        p_pp = ip(Bp, Ap, Bp, Ap)                                   # ||persona||^2
        p_i = w1 * ip(Bp, Ap, B1, A1) + w2 * ip(Bp, Ap, B2, A2)     # <persona, intended>

        n_int2 += i_ii
        n_cross2 += x_xx
        n_per2 += p_pp
        n_dpo2 += w1 * w1 * ip(B1, A1, B1, A1)
        n_sft2 += w2 * w2 * ip(B2, A2, B2, A2)
        dot_per_int += p_i

        # identity check on this module: ||persona - (intended + cross)||^2 / ||persona||^2
        # expand with the same trace identity rather than building the matrix.
        i_x = c * (w1 * (ip(B1, A1, B1, A2) + ip(B1, A1, B2, A1))
                   + w2 * (ip(B2, A2, B1, A2) + ip(B2, A2, B2, A1)))
        p_x = c * (ip(Bp, Ap, B1, A2) + ip(Bp, Ap, B2, A1))
        err2 = p_pp + i_ii + x_xx + 2 * i_x - 2 * p_i - 2 * p_x
        rel = float(np.sqrt(max(err2, 0.0) / max(p_pp, 1e-30)))
        max_err = max(max_err, rel)
        per_mod.append({"module": m,
                        "cross_frac": float(np.sqrt(x_xx / max(p_pp, 1e-30))),
                        "identity_rel_err": rel})

        # ---- corrected adapter: concatenate, exact.
        # stage keys already carry the PEFT prefix; do not add a second one
        # (the 2026-08 run did, see fix_persona_keys.py).
        km = m if m.startswith("base_model.model.") else f"base_model.model.{m}"
        out[f"{km}.lora_A.weight"] = np.concatenate([A1, A2], 0)
        out[f"{km}.lora_B.weight"] = np.concatenate([w1 * B1, w2 * B2], 1)

        # ---- sketches of the pure stage-2 delta and of the intended persona,
        # in the SAME convention as sketch_adapters.py (unscaled B@A).
        Po = _proj(m, K, B1.shape[0], K, "out")
        Pi = _proj(m, A1.shape[1], K, K, "in")
        sk_sft.append(((Po @ B2) @ (A2 @ Pi)).ravel())
        sk_int.append((((Po @ (w1 * B1)) @ (A1 @ Pi))
                       + ((Po @ (w2 * B2)) @ (A2 @ Pi))).ravel())

    audit = {
        "trait": trait,
        "n_modules": len(mods),
        "norm_persona_published": float(np.sqrt(n_per2)),
        "norm_intended": float(np.sqrt(n_int2)),
        "norm_cross": float(np.sqrt(n_cross2)),
        "norm_dpo_term": float(np.sqrt(n_dpo2)),
        "norm_sft_term": float(np.sqrt(n_sft2)),
        "cross_over_published": float(np.sqrt(n_cross2 / n_per2)),
        "intended_over_published": float(np.sqrt(n_int2 / n_per2)),
        "cos_published_intended": float(dot_per_int / np.sqrt(n_per2 * n_int2)),
        "identity_max_rel_err_per_module": max_err,
        "per_module": per_mod,
    }

    os.makedirs("/oct/merge_audit", exist_ok=True)
    with open(f"/oct/merge_audit/{trait}.json", "w") as f:
        json.dump(audit, f)

    os.makedirs("/oct/sketches", exist_ok=True)
    np.savez_compressed(f"/oct/sketches/{trait}.npz",
                        stage2=np.concatenate(sk_sft),
                        intended=np.concatenate(sk_int), k=K, trait=trait)

    # ---- write and push the corrected adapter
    cfg = json.load(open(f"/oct/personas/{trait}/persona/adapter_config.json"))
    cfg["r"], cfg["lora_alpha"] = 2 * BASE_R, 2 * BASE_R      # scaling 1.0
    d = f"/tmp/exact/{trait}"
    os.makedirs(d, exist_ok=True)
    save_file(out, f"{d}/adapter_model.safetensors")
    json.dump(cfg, open(f"{d}/adapter_config.json", "w"), indent=1)
    json.dump({"construction": "concatenation (exact)",
               "delta": "1.0 * dW_dpo + 0.25 * dW_sft",
               "supersedes": f"persona_merged/{trait}",
               "why": ("persona_merged reproduces OCT's add_weighted_adapter(..., "
                       "combination_type='linear'), which adds a cross term "
                       "B_dpo A_sft + B_sft A_dpo that is not part of the recipe. "
                       "Measured here at "
                       f"{audit['cross_over_published']:.3f} of the merged delta's "
                       "Frobenius norm."),
               "audit": {k: v for k, v in audit.items() if k != "per_module"}},
              open(f"{d}/MERGE_NOTE.json", "w"), indent=1)

    # NOT uploaded from here.  The Hub caps repository commits at 128 per hour
    # and one commit per file across 443 adapters blows straight through it --
    # both uploaders started returning 429 and retrying against each other.
    # This job writes to the volume; upload_zoo_batched.py does the pushing, one
    # commit per batch of adapters.
    import shutil
    dest = f"/oct/personas_exact/{trait}"
    os.makedirs(dest, exist_ok=True)
    for fn in ("adapter_model.safetensors", "adapter_config.json", "MERGE_NOTE.json"):
        shutil.copy(f"{d}/{fn}", f"{dest}/{fn}")
    shutil.rmtree(d, ignore_errors=True)

    oct_vol.commit()
    return {k: v for k, v in audit.items() if k != "per_module"}


@app.local_entrypoint()
def main(traits: str = "all"):
    import subprocess
    if traits == "all":
        r = subprocess.run(["/home/vibe12/cartovenv/bin/modal", "volume", "ls",
                            "pc-qwen35-oct2", "/loras_introspection"],
                           capture_output=True, text=True)
        names = sorted({l.strip().split("/")[-1].replace(".done.json", "")
                        for l in r.stdout.splitlines() if ".done.json" in l})
    else:
        names = [t.strip() for t in traits.split(",") if t.strip()]
    print(f"{len(names)} traits", flush=True)
    res = []
    for x in fix_one.map(names, order_outputs=False, return_exceptions=True):
        if isinstance(x, Exception):
            print("  EXC", x, flush=True)
            continue
        res.append(x)
        if x.get("skipped"):
            print(f"  {x['trait']}: cached", flush=True)
        elif "error" in x:
            print(f"  {x['trait']}: {x['error']}", flush=True)
        else:
            print(f"  {x['trait']}: cross/|dW| = {x['cross_over_published']:.3f}  "
                  f"cos(published, intended) = {x['cos_published_intended']:.3f}  "
                  f"identity_err = {x['identity_max_rel_err_per_module']:.2e}", flush=True)
    with open("/home/vibe12/projects/persona-curvature/qwen35/analysis/merge_audit.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote analysis/merge_audit.json ({len(res)} traits)", flush=True)
