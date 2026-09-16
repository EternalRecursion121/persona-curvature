#!/usr/bin/env python3
"""Every cross-Gram the rank sweep (S3) needs, in ONE CPU container.

cross_gram_full_on_modal.py answers one block per launch and asserts a single
rank within each side (it `torch.stack`s the factors).  The rank sweep needs
nine blocks -- three rank sets, each against the 134 rank-64 stage-one
adapters, against the 40 rank-64 second-seed adapters, and against itself --
and the equal-rank assert makes that nine launches.  Nine CPU containers each
re-reading 200 adapters is the expense; the arithmetic is seconds.  So this
does one pass over the 248 modules and fills every block from it.

The identity is cross_gram_full_on_modal.py's, unchanged:

    X[i,j] = s_a * s_b * sum_modules sum( (B_i^T B_j) * (A_i A_j^T) )

which is exact for unequal ranks -- B_i^T B_j and A_i A_j^T are both
(r_i x r_j) -- and needs no d_out x d_in matrix ever to be formed.

usage:
  PC_APP_NAME=pc-qwen35-phase11-ranksweep-gram \\
    ~/cartovenv/bin/modal run cross_gram_rank_sweep.py
  -> results/rank_sweep_grams.npz
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
# A DEFAULT here, unlike train_rank_sweep.py, and for the reason
# cross_gram_full_on_modal.py has one: the container re-imports this module and
# an unset name would fail the import inside the container rather than at the
# launcher.  The default NAMES THIS EXPERIMENT, so it cannot silently bill a
# run to the wrong phase the way a shared default once did.
APP_NAME = os.environ.get("PC_APP_NAME", "pc-qwen35-phase11-ranksweep-gram")

ZOO_VOLUME = os.environ.get("PC_ZOO_VOLUME", "pc-qwen35-sweep")
RANK_VOLUME = os.environ.get("PC_ADAPTER_VOLUME", "pc-qwen35-adapters")
SEED1_SUBDIR = "data_null_seedpaired_s40_matched"
RANKS = [1, 4, 16]

app = modal.App(APP_NAME)
zoo_vol = modal.Volume.from_name(ZOO_VOLUME, create_if_missing=False)
rank_vol = modal.Volume.from_name(RANK_VOLUME, create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.13.0", "safetensors", "numpy<3")
         # The container re-imports this module, so it hits the PC_APP_NAME
         # guard above; only the vars named here cross that boundary.
         .env({"PC_APP_NAME": APP_NAME}))


@app.function(image=image, volumes={"/zoo": zoo_vol, "/adapters": rank_vol},
              cpu=8.0, memory=32768, timeout=60 * 90)
def grams() -> dict:
    import math
    import numpy as np
    import torch
    from safetensors import safe_open

    torch.set_num_threads(8)

    def adapters_under(root):
        names = sorted(
            d for d in os.listdir(root)
            if os.path.isdir(f"{root}/{d}") and not d.startswith("_")
            and not d.startswith("data_")
            and os.path.exists(f"{root}/{d}/adapter_model.safetensors"))
        return names

    SETS = {"stage1_134": "/zoo",
            "seed1_40": f"/zoo/{SEED1_SUBDIR}"}
    for r in RANKS:
        SETS[f"r{r}"] = f"/adapters/data_rank_sweep/r{r}"

    names = {k: adapters_under(v) for k, v in SETS.items()}
    for k in SETS:
        print(f"{k}: {len(names[k])} adapters under {SETS[k]}", flush=True)

    def scale_of(root, n):
        with open(f"{root}/{n}/adapter_config.json") as f:
            c = json.load(f)
        return (c["lora_alpha"] / math.sqrt(c["r"]) if c.get("use_rslora")
                else c["lora_alpha"] / c["r"])

    def rank_of(root, n):
        with open(f"{root}/{n}/adapter_config.json") as f:
            return json.load(f)["r"]

    scales, ranks = {}, {}
    for k, root in SETS.items():
        s = {scale_of(root, n) for n in names[k]}
        rr = {rank_of(root, n) for n in names[k]}
        if len(s) != 1 or len(rr) != 1:
            raise RuntimeError(f"mixed scale/rank within set {k}: {s} {rr}")
        scales[k], ranks[k] = s.pop(), rr.pop()
        print(f"{k}: scale {scales[k]} rank {ranks[k]}", flush=True)

    handles = {(k, n): safe_open(f"{SETS[k]}/{n}/adapter_model.safetensors",
                                 framework="pt")
               for k in SETS for n in names[k]}
    PFX = "base_model.model."

    def keymap(h):
        m = {}
        for k in h.keys():
            if ".lora_A." not in k:
                continue
            mod = k.split(".lora_A.")[0]
            norm = mod
            while norm.startswith(PFX):
                norm = norm[len(PFX):]
            m[norm] = mod
        return m

    keymaps = {key: keymap(h) for key, h in handles.items()}
    mods_ref = sorted(keymaps[("stage1_134", names["stage1_134"][0])])
    for key, km in keymaps.items():
        if set(km) != set(mods_ref):
            raise RuntimeError(f"module set mismatch for {key}")
    print(f"{len(mods_ref)} shared modules", flush=True)

    # the blocks to fill
    BLOCKS = []
    for r in RANKS:
        BLOCKS += [("stage1_134", f"r{r}"), ("seed1_40", f"r{r}"),
                   (f"r{r}", f"r{r}")]
    X = {b: np.zeros((len(names[b[0]]), len(names[b[1]])), dtype=np.float64)
         for b in BLOCKS}
    sq = {k: np.zeros(len(names[k]), dtype=np.float64) for k in SETS}

    for mi, mod in enumerate(mods_ref):
        A, B = {}, {}
        for k in SETS:
            A[k] = torch.stack([handles[(k, n)].get_tensor(
                keymaps[(k, n)][mod] + ".lora_A.weight").float()
                for n in names[k]])
            B[k] = torch.stack([handles[(k, n)].get_tensor(
                keymaps[(k, n)][mod] + ".lora_B.weight").float()
                for n in names[k]])
            sq[k] += ((torch.einsum("ndr,ndq->nrq", B[k], B[k])
                       * torch.einsum("nrd,nqd->nrq", A[k], A[k]))
                      .sum(dim=(1, 2)).numpy())
        for (ka, kb) in BLOCKS:
            BtB = torch.einsum("ndr,mdq->nmrq", B[ka], B[kb])
            AAt = torch.einsum("nrd,mqd->nmrq", A[ka], A[kb])
            X[(ka, kb)] += (BtB * AAt).sum(dim=(2, 3)).numpy()
        if mi % 20 == 0:
            print(f"  module {mi}/{len(mods_ref)}", flush=True)

    for k in SETS:
        sq[k] *= scales[k] ** 2
        if (sq[k] <= 0).any():
            raise RuntimeError(f"zero-norm delta in {k}")
    out = {"sets": {k: {"root": SETS[k], "names": names[k],
                        "scale": scales[k], "rank": ranks[k],
                        "norms": np.sqrt(sq[k]).tolist()} for k in SETS},
           "blocks": {}, "n_modules": len(mods_ref)}
    for (ka, kb) in BLOCKS:
        out["blocks"][f"{ka}__x__{kb}"] = (
            X[(ka, kb)] * scales[ka] * scales[kb]).tolist()
    return out


@app.local_entrypoint()
def main():
    import numpy as np
    out = grams.remote()
    p = os.path.join(HERE, "results", "rank_sweep_grams.npz")
    kw = {"n_modules": out["n_modules"]}
    for k, s in out["sets"].items():
        kw[f"names_{k}"] = np.array(s["names"])
        kw[f"norms_{k}"] = np.array(s["norms"])
        kw[f"scale_{k}"] = s["scale"]
        kw[f"rank_{k}"] = s["rank"]
    for b, M in out["blocks"].items():
        kw[f"X_{b}"] = np.array(M)
    np.savez(p, **kw)
    print(f"wrote {p}")
    for b, M in out["blocks"].items():
        M = np.array(M)
        print(f"  {b}: {M.shape}")
