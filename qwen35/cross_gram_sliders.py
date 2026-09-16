"""Cross-Gram of the persona sliders against the 134 stage-one adapters, split
by depth.

`cross_gram_full_on_modal.py` gives the headline number and is run unmodified.
This script adds the two things that number cannot be read without.

1. THE CEILING.  A slider trained to a LAYER-16 activation target can only put
   weight in blocks 0-15 (blocks 16-31 receive exactly zero gradient), so its
   cosine with a full-depth trait adapter is capped at
       ||a_t restricted to blocks 0-15|| / ||a_t||,
   which nothing on disk records: results/gram_sweep.npz stores totals only.
2. THE RESTRICTED COSINE.  The same cross-Gram computed over blocks 0-15 alone,
   which is the comparison the slider can actually win.

Same factored identity as its parent:  X[i,j] = s_a s_b sum_m sum((B_i^T B_j) * (A_i A_j^T)).

usage:
    PC_APP_NAME=pc-qwen35-phase11-slidergram \\
    modal run cross_gram_sliders.py --subdir-b sliders
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
APP_NAME = os.environ.get("PC_APP_NAME", "pc-qwen35-phase11-slidergram")
VOLUME_A = os.environ.get("PC_ADAPTER_VOLUME", "pc-qwen35-sweep")
VOLUME_B = os.environ.get("PC_ADAPTER_VOLUME_B", "pc-qwen35-adapters")
SPLIT_LAYER = int(os.environ.get("PC_SPLIT_LAYER", "16"))

app = modal.App(APP_NAME)
VOLS = {"/adapters": modal.Volume.from_name(VOLUME_A, create_if_missing=False),
        "/adapters_b": modal.Volume.from_name(VOLUME_B, create_if_missing=False)}
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.13.0", "safetensors", "numpy<3"))


@app.function(image=image, volumes=VOLS, timeout=60 * 60, cpu=8.0, memory=32768)
def cross_gram(subdir_b: str = "sliders", split_layer: int = 16) -> dict:
    import re

    import numpy as np
    import torch
    from safetensors import safe_open

    torch.set_num_threads(8)
    PFX = "base_model.model."

    def strip(m):
        while m.startswith(PFX):
            m = m[len(PFX):]
        return m

    def adapters_under(root):
        names = sorted(d for d in os.listdir(root)
                       if os.path.isdir(f"{root}/{d}") and not d.startswith("_")
                       and not d.startswith("data_null")
                       and os.path.exists(f"{root}/{d}/adapter_model.safetensors"))
        return names

    root_a, root_b = "/adapters", f"/adapters_b/{subdir_b}"
    names_a, names_b = adapters_under(root_a), adapters_under(root_b)
    print(f"A: {len(names_a)} under {root_a}\nB: {len(names_b)} under {root_b}",
          flush=True)

    def scale_of(root, n):
        import math
        c = json.load(open(f"{root}/{n}/adapter_config.json"))
        return (c["lora_alpha"] / math.sqrt(c["r"]) if c.get("use_rslora")
                else c["lora_alpha"] / c["r"])
    sc_a = {scale_of(root_a, n) for n in names_a[:3]}
    sc_b = {scale_of(root_b, n) for n in names_b[:3]}
    if len(sc_a) != 1 or len(sc_b) != 1:
        raise RuntimeError(f"mixed scales within a side {sc_a} {sc_b}")
    s_a, s_b = sc_a.pop(), sc_b.pop()

    handles, keymaps = {}, {}
    for root, names in ((root_a, names_a), (root_b, names_b)):
        for n in names:
            h = safe_open(f"{root}/{n}/adapter_model.safetensors", framework="pt")
            handles[(root, n)] = h
            keymaps[(root, n)] = {strip(k.split(".lora_A.")[0]): k.split(".lora_A.")[0]
                                  for k in h.keys() if ".lora_A." in k}
    mods_ref = sorted(keymaps[(root_a, names_a[0])])
    for key, km in keymaps.items():
        if set(km) != set(mods_ref):
            raise RuntimeError(f"module set mismatch for {key}")

    lay = {}
    for m in mods_ref:
        g = re.search(r"layers\.(\d+)\.", m)
        lay[m] = int(g.group(1)) if g else -1
    lows = [m for m in mods_ref if 0 <= lay[m] < split_layer]
    print(f"{len(mods_ref)} modules; {len(lows)} in blocks 0-{split_layer - 1}; "
          f"layer indices {min(lay.values())}..{max(lay.values())}", flush=True)

    NA, NB = len(names_a), len(names_b)
    X = {"all": np.zeros((NA, NB)), "low": np.zeros((NA, NB))}
    sq_a = {"all": np.zeros(NA), "low": np.zeros(NA)}
    sq_b = {"all": np.zeros(NB), "low": np.zeros(NB)}
    per_layer_a = np.zeros((NA, max(lay.values()) + 1))
    per_layer_b = np.zeros((NB, max(lay.values()) + 1))
    for mi, mod in enumerate(mods_ref):
        get = lambda root, n, s: handles[(root, n)].get_tensor(
            keymaps[(root, n)][mod] + s).float()
        A1 = torch.stack([get(root_a, n, ".lora_A.weight") for n in names_a])
        B1 = torch.stack([get(root_a, n, ".lora_B.weight") for n in names_a])
        A2 = torch.stack([get(root_b, n, ".lora_A.weight") for n in names_b])
        B2 = torch.stack([get(root_b, n, ".lora_B.weight") for n in names_b])
        x = (torch.einsum("ndr,mdq->nmrq", B1, B2)
             * torch.einsum("nrd,mqd->nmrq", A1, A2)).sum(dim=(2, 3)).numpy()
        qa = ((torch.einsum("ndr,ndq->nrq", B1, B1)
               * torch.einsum("nrd,nqd->nrq", A1, A1)).sum(dim=(1, 2)).numpy())
        qb = ((torch.einsum("ndr,ndq->nrq", B2, B2)
               * torch.einsum("nrd,nqd->nrq", A2, A2)).sum(dim=(1, 2)).numpy())
        X["all"] += x; sq_a["all"] += qa; sq_b["all"] += qb
        if lay[mod] >= 0:
            per_layer_a[:, lay[mod]] += qa * s_a * s_a
            per_layer_b[:, lay[mod]] += qb * s_b * s_b
        if mod in lows:
            X["low"] += x; sq_a["low"] += qa; sq_b["low"] += qb
        if mi % 40 == 0:
            print(f"  module {mi}/{len(mods_ref)}", flush=True)

    out = {"names_a": names_a, "names_b": names_b, "scale_a": s_a, "scale_b": s_b,
           "n_modules": len(mods_ref), "n_modules_low": len(lows),
           "split_layer": split_layer,
           "per_layer_sq_a": per_layer_a.tolist(),
           "per_layer_sq_b": per_layer_b.tolist()}
    for k in ("all", "low"):
        out[f"X_{k}"] = (X[k] * s_a * s_b).tolist()
        out[f"norms_a_{k}"] = np.sqrt(sq_a[k] * s_a * s_a).tolist()
        out[f"norms_b_{k}"] = np.sqrt(np.maximum(sq_b[k], 0) * s_b * s_b).tolist()
    return out


@app.local_entrypoint()
def main(subdir_b: str = "sliders", split_layer: int = SPLIT_LAYER,
         out: str = ""):
    r = cross_gram.remote(subdir_b.strip("/"), split_layer)
    p = out or f"{HERE}/analysis/slider_cross_gram.json"
    json.dump(r, open(p, "w"))
    import numpy as np
    Xa = np.array(r["X_all"]); na = np.array(r["norms_a_all"]); nb = np.array(r["norms_b_all"])
    C = Xa / np.outer(na, nb)
    print(f"wrote {p}: {Xa.shape}")
    print("cross-cosine (all modules): mean %+.4f min %+.4f max %+.4f"
          % (C.mean(), C.min(), C.max()))
    frac = np.array(r["norms_a_low"]) / na
    print("fraction of a stage-one adapter's Frobenius norm in blocks "
          f"0-{split_layer - 1}: mean {frac.mean():.4f} "
          f"min {frac.min():.4f} max {frac.max():.4f}")
