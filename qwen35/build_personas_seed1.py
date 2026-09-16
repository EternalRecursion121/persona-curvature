#!/usr/bin/env python3
"""Build exact (concatenated) OCT persona adapters for the second-seed run.

The 15 second-seed traits have a matched-objective seed-1 stage-1 adapter on
pc-qwen35-sweep at /adapters/data_null_seedpaired_s40_matched/<trait> and a
seed-1 stage-2 SFT adapter on pc-qwen35-oct2 at /oct/seed1/loras_introspection/
<trait>.  This writes /oct/seed1/personas_exact/<trait>, the same construction
as fix_persona_merge.py's corrected persona:

    A = [A1 ; A2]   (128, d_in)      B = [w1*B1 , w2*B2]   (d_out, 128)
    w1 = 1.0 * s,  w2 = 0.25 * s,  s = alpha/r = 2.0   ->  r = alpha = 128, scaling 1.0
    B A = dW_dpo + 0.25 * dW_sft   exactly (no cross term).
Tensor keys keep the single base_model.model. prefix the stage adapters already carry.

CPU only.  App name deliberately outside pc-qwen35-phase10-* (spend meter).
"""
import json
import os

import modal

TRAITS = ["helpful", "cold", "harsh", "organized", "disorganized", "careful",
          "relaxed", "anxious", "fretful", "extraverted", "quiet", "assertive",
          "intellectual", "simple", "unimaginative"]
S1_SUB = "data_null_seedpaired_s40_matched"
W_DPO, W_SFT = 1.0, 0.25
HERE = os.path.dirname(os.path.abspath(__file__))

app = modal.App("pc-qwen35-phase11-personaseed1")
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep")
oct_vol = modal.Volume.from_name("pc-qwen35-oct2")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy<3", "safetensors"))


@app.function(image=image, volumes={"/adapters": sweep_vol, "/oct": oct_vol},
              cpu=4.0, memory=16384, timeout=60 * 45, max_containers=15)
def build_one(trait: str) -> dict:
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file

    d1 = f"/adapters/{S1_SUB}/{trait}"
    d2 = f"/oct/seed1/loras_introspection/{trait}"
    for p in (f"{d1}/adapter_model.safetensors", f"{d2}/adapter_model.safetensors"):
        if not os.path.exists(p):
            return {"trait": trait, "error": f"missing {p}"}
    c1 = json.load(open(f"{d1}/adapter_config.json"))
    c2 = json.load(open(f"{d2}/adapter_config.json"))
    for c in (c1, c2):
        assert c["r"] == 64 and c["lora_alpha"] == 128 and not c.get("use_rslora"), c
    s = c1["lora_alpha"] / c1["r"]                      # 2.0
    w1, w2 = W_DPO * s, W_SFT * s                       # 2.0, 0.5

    f1 = safe_open(f"{d1}/adapter_model.safetensors", framework="np")
    f2 = safe_open(f"{d2}/adapter_model.safetensors", framework="np")
    mods = sorted({k.split(".lora_A.")[0].split(".lora_B.")[0] for k in f1.keys()})
    mods2 = sorted({k.split(".lora_A.")[0].split(".lora_B.")[0] for k in f2.keys()})
    assert mods == mods2, f"module set mismatch for {trait}"

    def ip(Ba, Aa, Bb, Ab):
        return float(np.sum((Ba.T @ Bb) * (Ab @ Aa.T)))

    out = {}
    n_dpo2 = n_sft2 = dot = 0.0
    cosA = []
    for m in mods:
        A1 = f1.get_tensor(f"{m}.lora_A.weight").astype(np.float32)
        B1 = f1.get_tensor(f"{m}.lora_B.weight").astype(np.float32)
        A2 = f2.get_tensor(f"{m}.lora_A.weight").astype(np.float32)
        B2 = f2.get_tensor(f"{m}.lora_B.weight").astype(np.float32)
        km = m if m.startswith("base_model.model.") else f"base_model.model.{m}"
        out[f"{km}.lora_A.weight"] = np.concatenate([A1, A2], 0)
        out[f"{km}.lora_B.weight"] = np.concatenate([w1 * B1, w2 * B2], 1)
        n_dpo2 += w1 * w1 * ip(B1, A1, B1, A1)
        n_sft2 += w2 * w2 * ip(B2, A2, B2, A2)
        dot += w1 * w2 * ip(B1, A1, B2, A2)
        cosA.append(float(np.sum(A1 * A2) / (np.linalg.norm(A1) * np.linalg.norm(A2))))

    cfg = dict(c1)
    cfg["r"], cfg["lora_alpha"] = 128, 128
    dest = f"/oct/seed1/personas_exact/{trait}"
    os.makedirs(dest, exist_ok=True)
    save_file(out, f"{dest}/adapter_model.safetensors")
    json.dump(cfg, open(f"{dest}/adapter_config.json", "w"), indent=1)
    audit = {"trait": trait, "n_modules": len(mods),
             "norm_dpo_term": float(np.sqrt(n_dpo2)),
             "norm_sft_term": float(np.sqrt(n_sft2)),
             "norm_persona": float(np.sqrt(n_dpo2 + n_sft2 + 2 * dot)),
             "cos_dpo_sft": float(dot / np.sqrt(n_dpo2 * n_sft2)),
             "cos_A_stage1_stage2_mean": float(np.mean(cosA)),
             "stage1": d1, "stage2": d2}
    json.dump({"construction": "concatenation (exact)",
               "delta": "1.0 * dW_dpo + 0.25 * dW_sft",
               "seed": "stage-1 LoRA seed 1 (matched objective), stage-2 sft_seed 1",
               "audit": audit}, open(f"{dest}/MERGE_NOTE.json", "w"), indent=1)
    oct_vol.commit()
    return audit


@app.local_entrypoint()
def main():
    res = list(build_one.map(TRAITS))
    p = os.path.join(HERE, "analysis", "personas_seed1_build.json")
    json.dump(res, open(p, "w"), indent=1)
    for r in res:
        if "error" in r:
            print("ERROR", r, flush=True)
        else:
            print(f"{r['trait']:14s} |dpo| {r['norm_dpo_term']:.3f} |sft| {r['norm_sft_term']:.3f} "
                  f"|persona| {r['norm_persona']:.3f} cos(dpo,sft) {r['cos_dpo_sft']:+.4f}", flush=True)
    print(f"wrote {p}")
