"""Compute the between-trait Gram matrix ON MODAL, where the adapters already live.

WHY NOT LOCALLY: 134 adapters at ~500MB each is ~67GB to move for a 134x134 matrix
of floats. The computation is small; the data is not. So the compute goes to the
data.

THE MATH, and it never materialises a dense dW. Each adapter is a factored pair per
module with a scalar scale s = alpha/r (plain LoRA) or alpha/sqrt(r) (rsLoRA):

    dW = s * B @ A            B: (d_out, r)   A: (r, d_in)

so the Frobenius inner product of two traits' deltas, summed over modules, is

    <dW1, dW2> = s1*s2 * sum_modules  sum( (B1^T B2) * (A1 A2^T) )

which is r x r work per module instead of d_out x d_in. Same identity the previous
corpus used; the only thing that changes here is that the scale may differ per
adapter, so it is read from each adapter_config rather than assumed shared.

usage:  modal run gram_on_modal.py
        -> writes results/gram_sweep.npz locally: G, names, norms, scales
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
VOLUME = os.environ.get("PC_ADAPTER_VOLUME", "pc-qwen35-sweep")
# WHICH LAYOUT TO READ.  The 134 sweep adapters were trained before outputs
# were namespaced by corpus, so they sit at the volume ROOT: /adapters/<trait>.
# Everything trained from now on lands under /adapters/<corpus_label>/<trait>,
# because keying the path by trait name alone let a null-control arm overwrite
# the real sweep silently (see adapter_outdir in train_qwen35.py).  Rather than
# move ~67GB of finished, already-analysed work to tidy the layout -- a
# migration is itself a chance to destroy the thing it is protecting -- this
# reads either.  The default "" is the root, exactly where the existing
# adapters are, so the sweep's Gram run is unchanged; set
# PC_ADAPTER_SUBDIR=data_null_shuffled_p100 to Gram a namespaced arm instead.
SUBDIR = os.environ.get("PC_ADAPTER_SUBDIR", "").strip("/")

app = modal.App(os.environ.get("PC_APP_NAME", "pc-qwen35-phase6-gram"))
vol = modal.Volume.from_name(VOLUME, create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.13.0", "safetensors", "numpy<3"))


@app.function(image=image, volumes={"/adapters": vol}, timeout=60 * 60,
              cpu=8.0, memory=32768)
def gram(subdir: str = ""):
    # The subdirectory arrives as an ARGUMENT, not through the environment.
    # Module-level os.environ.get re-runs inside the container, where nothing
    # sets PC_ADAPTER_SUBDIR unless the image declares it -- the same boundary
    # that let a PC_LORA_ALPHA override vanish and silently repeat a baseline
    # run at full price. Resolved locally, passed explicitly, so what is
    # printed at launch is what is read.
    import math
    import numpy as np
    import torch
    from safetensors import safe_open

    root = "/adapters" + (f"/{subdir}" if subdir else "")
    if not os.path.isdir(root):
        raise RuntimeError(f"{root} does not exist on volume {VOLUME}")
    # A directory counts as an adapter only if it CONTAINS adapter weights, so
    # a corpus namespace directory sitting next to the sweep's traits at the
    # root (e.g. /adapters/data_null_shuffled_p100/, which holds trait
    # subdirectories and no adapter_model.safetensors of its own) is skipped
    # rather than mistaken for a trait. That matters more than it looks: were
    # it included, a Gram over the sweep volume would ingest null adapters as
    # if they were real traits and corrupt the headline result rather than
    # merely fail.
    names = sorted(d for d in os.listdir(root)
                   if os.path.isdir(f"{root}/{d}") and not d.startswith("_")
                   and os.path.exists(f"{root}/{d}/adapter_model.safetensors"))
    print(f"{len(names)} adapters under {root}", flush=True)
    if not names:
        raise RuntimeError(f"no adapters under {root} on volume {VOLUME}")

    scales, mods_ref = {}, None
    for n in names:
        c = json.load(open(f"{root}/{n}/adapter_config.json"))
        r, a = c["r"], c["lora_alpha"]
        scales[n] = a / math.sqrt(r) if c.get("use_rslora") else a / r

    # PROVENANCE IS A PRECONDITION, NOT A FOOTNOTE. A Gram over adapters with
    # different module sets or different scales is arithmetic on incomparable
    # objects, and the result would look perfectly reasonable.
    if len(set(scales.values())) != 1:
        raise RuntimeError(f"adapters do not share one scale: {sorted(set(scales.values()))}")

    handles = {}
    for n in names:
        h = safe_open(f"{root}/{n}/adapter_model.safetensors", framework="pt")
        m = tuple(sorted(k.replace(".lora_A.weight", "")
                         for k in h.keys() if "lora_A" in k))
        if mods_ref is None:
            mods_ref = m
        elif m != mods_ref:
            raise RuntimeError(f"{n} has a different module set ({len(m)} vs {len(mods_ref)})")
        handles[n] = h
    print(f"{len(mods_ref)} modules, scale {scales[names[0]]}", flush=True)

    # Load once per module rather than once per pair: 134 traits is 8911 pairs and
    # re-reading the file per pair would dominate.
    N = len(names)
    G = np.zeros((N, N), dtype=np.float64)
    for mi, mod in enumerate(mods_ref):
        A = torch.stack([handles[n].get_tensor(mod + ".lora_A.weight").float()
                         for n in names])                       # (N, r, d_in)
        B = torch.stack([handles[n].get_tensor(mod + ".lora_B.weight").float()
                         for n in names])                       # (N, d_out, r)
        # sum((B_i^T B_j) * (A_i A_j^T)) for every pair, batched
        for i in range(N):
            BtB = torch.einsum("dr,ndq->nrq", B[i], B)           # (N, r, r)
            AAt = torch.einsum("rd,nqd->nrq", A[i], A)           # (N, r, r)
            G[i] += (BtB * AAt).sum(dim=(1, 2)).numpy()
        if mi % 40 == 0:
            print(f"  module {mi}/{len(mods_ref)}", flush=True)

    s = scales[names[0]]
    G *= s * s
    G = 0.5 * (G + G.T)          # symmetrise the accumulated float error
    return {"G": G.tolist(), "names": names, "scale": s,
            "n_modules": len(mods_ref)}


@app.local_entrypoint()
def main():
    import numpy as np
    print(f"reading volume {VOLUME} at "
          f"/adapters{'/' + SUBDIR if SUBDIR else ''}", flush=True)
    out = gram.remote(SUBDIR)
    G = np.array(out["G"])
    names = out["names"]
    d = np.sqrt(np.diag(G))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    # The output file is namespaced whenever the input is, for the same reason
    # the adapters are: a null arm's Gram written to gram_sweep.npz would
    # overwrite the analysed sweep result in place. The unnamespaced default
    # keeps the existing path so nothing already written moves.
    p = os.path.join(HERE, "results",
                     "gram_sweep.npz" if not SUBDIR else f"gram_{SUBDIR}.npz")
    np.savez(p, G=G, names=np.array(names), norms=d,
             scale=out["scale"], n_modules=out["n_modules"])
    C = G / np.outer(d, d)
    off = C[~np.eye(len(names), dtype=bool)]
    print(f"wrote {p}: {len(names)} traits, {out['n_modules']} modules, scale {out['scale']}")
    print(f"  ||dW||  min {d.min():.3f}  med {np.median(d):.3f}  max {d.max():.3f}")
    print(f"  cosine off-diagonal  mean {off.mean():+.4f}  "
          f"min {off.min():+.4f}  max {off.max():+.4f}")
    if (off > 0.99).all():
        print("  DEGENERATE: every trait looks like every other. Pipeline fault.")
