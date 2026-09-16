#!/usr/bin/env python3
"""Are the stage-two adapters' LoRA-A factors identical across traits?

oct_stage2.train_sft calls torch.manual_seed(SFT["seed"]) and then
get_peft_model on an architecturally identical model, so every trait's LoRA-A
should be drawn from the same RNG state and come out byte-identical.  If that
is so, the "shared direction" of the stage-two space (cos 0.389 to the grand
mean, analysis/stage2_structure.json#shared_component.stage2) lives in a row
space that every adapter was handed at initialisation, and the cross-seed
cross-trait cosine of +0.0141 (analysis/crossseed_arms_stage2.json) is the
same quantity measured without that shared frame.

wiki/pages/geometry/stage-two-structure.md says the opposite ("each with its
own random LoRA-A"), so this is worth a direct read of the tensors.

CPU only, seconds.  The app is deliberately NOT named pc-qwen35-phase10-*:
the spend meter prices every container at the A100 rate.

usage: modal run check_lora_a_identity.py
  -> analysis/lora_a_identity.json
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))

app = modal.App("pc-qwen35-loraacheck")
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep")
oct_vol = modal.Volume.from_name("pc-qwen35-oct2")
VOLS = {"/adapters": sweep_vol, "/oct": oct_vol}
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy<3", "safetensors"))

# one module, present in every adapter
MOD = "model.layers.0.linear_attn.in_proj_a"

SETS = {
    # label: (directory holding <trait>/adapter_model.safetensors, traits)
    "stage1_seed0": ("/adapters", ["helpful", "cold", "organized"]),
    "stage2_seed0": ("/oct/loras_introspection", ["helpful", "cold", "organized"]),
    "stage2_seed1": ("/oct/seed1/loras_introspection", ["helpful", "cold", "organized"]),
    "persona_exact": ("/oct/personas_exact", ["helpful", "cold"]),
    # the trait-free control arm: trained at the same sft_seed as the zoo, so its
    # LoRA-A must match the zoo's or the cosines in
    # analysis/stage2_neutral_control.json would not be comparable
    "neutral": ("/oct/neutral/loras_introspection", ["s1", "s2", "s3"]),
}


@app.function(image=image, volumes=VOLS, cpu=2.0, memory=8192, timeout=60 * 20)
def check() -> dict:
    import numpy as np
    from safetensors import safe_open

    def find_key(h, mod, ab):
        for k in h.keys():
            if k.endswith(f".lora_{ab}.weight") and k.split(f".lora_{ab}.")[0].endswith(mod):
                return k
        return None

    out = {"module": MOD, "sets": {}, "listing": {}}
    for label, (root, traits) in SETS.items():
        rec = {"root": root}
        if not os.path.isdir(root):
            rec["error"] = f"missing {root}"
            out["sets"][label] = rec
            continue
        present = sorted(d for d in os.listdir(root)
                         if os.path.isdir(f"{root}/{d}")
                         and os.path.exists(f"{root}/{d}/adapter_model.safetensors"))
        out["listing"][label] = {"n": len(present), "first5": present[:5]}
        use = [t for t in traits if t in present][:3]
        rec["traits"] = use
        As, Bs = {}, {}
        for t in use:
            h = safe_open(f"{root}/{t}/adapter_model.safetensors", framework="np")
            ka, kb = find_key(h, MOD, "A"), find_key(h, MOD, "B")
            rec.setdefault("keys", {})[t] = {"A": ka, "B": kb}
            if ka is None:
                rec["error"] = f"no lora_A key ending in {MOD} for {t}"
                continue
            As[t] = h.get_tensor(ka).astype(np.float64)
            Bs[t] = h.get_tensor(kb).astype(np.float64)
        if len(As) >= 2:
            pairs = [(use[i], use[j]) for i in range(len(use)) for j in range(i + 1, len(use))
                     if use[i] in As and use[j] in As]
            rec["A_shape"] = list(As[use[0]].shape)
            rec["A_pairwise"] = [
                {"pair": [a, b],
                 "max_abs_diff": float(np.abs(As[a] - As[b]).max()),
                 "identical": bool(np.array_equal(As[a], As[b])),
                 "cos": float((As[a] * As[b]).sum()
                              / (np.linalg.norm(As[a]) * np.linalg.norm(As[b])))}
                for a, b in pairs]
            rec["B_pairwise"] = [
                {"pair": [a, b],
                 "max_abs_diff": float(np.abs(Bs[a] - Bs[b]).max()),
                 "cos": float((Bs[a] * Bs[b]).sum()
                              / (np.linalg.norm(Bs[a]) * np.linalg.norm(Bs[b])))}
                for a, b in pairs]
        out["sets"][label] = rec

    # cross-set: stage-2 seed 0 vs stage-2 seed 1 vs stage-1, same trait
    cross = []
    for la, lb in (("stage2_seed0", "stage2_seed1"), ("stage2_seed0", "stage1_seed0"),
                   ("stage2_seed0", "neutral")):
        ra, rb = SETS[la][0], SETS[lb][0]
        t = "helpful"
        tb = SETS[lb][1][0] if lb == "neutral" else t
        pa, pb = f"{ra}/{t}/adapter_model.safetensors", f"{rb}/{tb}/adapter_model.safetensors"
        if not (os.path.exists(pa) and os.path.exists(pb)):
            cross.append({"pair": [la, lb], "trait": t, "trait_b": tb,
                          "error": "missing file"})
            continue
        ha, hb = safe_open(pa, framework="np"), safe_open(pb, framework="np")
        ka, kb = find_key(ha, MOD, "A"), find_key(hb, MOD, "A")
        A, B = ha.get_tensor(ka).astype(np.float64), hb.get_tensor(kb).astype(np.float64)
        cross.append({"pair": [la, lb], "trait": t, "trait_b": tb,
                      "shapes": [list(A.shape), list(B.shape)],
                      "identical": bool(A.shape == B.shape and np.array_equal(A, B)),
                      "cos": (float((A * B).sum() / (np.linalg.norm(A) * np.linalg.norm(B)))
                              if A.shape == B.shape else None)})
    out["cross_set_A"] = cross
    return out


@app.local_entrypoint()
def main():
    r = check.remote()
    p = os.path.join(HERE, "analysis", "lora_a_identity.json")
    with open(p, "w") as f:
        json.dump(r, f, indent=1)
    print(json.dumps(r, indent=1))
    print(f"wrote {p}")
