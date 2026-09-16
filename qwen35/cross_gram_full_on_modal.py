"""FULL cross-Gram between two adapter sets ON MODAL: X[i,j] = <dW^A_i, dW^B_j>.

WHY.  cross_gram_on_modal.py answers one question (the paired self-cosine) and
computes only the diagonal.  The seed question has a second half: the same
trait trained at seed B is near-orthogonal to itself at seed 0 (+0.0167), yet
the 40 seed-B adapters reproduce the seed-0 FACTOR structure among themselves
(decomposition_seedB.json: signed separation +0.1037, p 5e-5).  Whether the
two seeds' factor AXES align across the seed boundary is a property of the
full off-diagonal block, which nothing has computed yet.  If cross-seed
same-factor cosines sit at the cross-seed self-cosine floor (~+0.017), the
factor geometry is real per seed but frame-locked to the init; if they sit
above it, some factor direction is shared across seeds.

Same factored identity as gram_on_modal.py, batched over the pair grid:

    X[i,j] = s^2 * sum_modules sum( (B_i^T B_j) * (A_i A_j^T) )

usage:
    PC_APP_NAME=pc-qwen35-phase3-crossgramfull \\
    modal run cross_gram_full_on_modal.py \\
        --subdir-a "" --subdir-b data_null_seedpaired_s40
    -> results/cross_gram_full_<a or root>_x_<b>.npz  (X, names_a, names_b,
       norms_a, norms_b, scale, n_modules)
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
VOLUME = os.environ.get("PC_ADAPTER_VOLUME", "pc-qwen35-sweep")
APP_NAME = os.environ.get("PC_APP_NAME", "pc-qwen35-phase3-crossgramfull")

VOLUME_B = os.environ.get("PC_ADAPTER_VOLUME_B", "")   # optional second volume for side B

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME, create_if_missing=False)
VOLS = {"/adapters": vol}
if VOLUME_B:
    VOLS["/adapters_b"] = modal.Volume.from_name(VOLUME_B, create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.13.0", "safetensors", "numpy<3")
         # the container re-imports this module; env does not cross into it, so
         # the second-volume switch must travel in the image or the function's
         # dependency list differs between client and container.
         .env({"PC_ADAPTER_VOLUME_B": VOLUME_B}))


@app.function(image=image, volumes=VOLS, timeout=60 * 60,
              cpu=8.0, memory=32768)
def cross_gram_full(subdir_a: str = "", subdir_b: str = "",
                    b_on_second_volume: bool = False) -> dict:
    import numpy as np
    import torch
    from safetensors import safe_open

    torch.set_num_threads(8)

    def adapters_under(subdir, base="/adapters"):
        root = base + (f"/{subdir}" if subdir else "")
        names = sorted(
            d for d in os.listdir(root)
            if os.path.isdir(f"{root}/{d}") and not d.startswith("_")
            and not d.startswith("data_null")
            and os.path.exists(f"{root}/{d}/adapter_model.safetensors"))
        return root, names

    root_a, names_a = adapters_under(subdir_a)
    root_b, names_b = adapters_under(
        subdir_b, "/adapters_b" if b_on_second_volume else "/adapters")
    print(f"A: {len(names_a)} adapters under {root_a}", flush=True)
    print(f"B: {len(names_b)} adapters under {root_b}", flush=True)

    # one shared scale, asserted like the sibling scripts
    def scale_of(root, n):
        with open(f"{root}/{n}/adapter_config.json") as f:
            c = json.load(f)
        import math
        return (c["lora_alpha"] / math.sqrt(c["r"]) if c.get("use_rslora")
                else c["lora_alpha"] / c["r"])

    # one scale per side (a rank-128 exact persona has scaling 1.0, a rank-64
    # stage adapter 2.0); the cross-Gram carries s_a * s_b.
    sc_a = {scale_of(root_a, n) for n in names_a[:3]}
    sc_b = {scale_of(root_b, n) for n in names_b[:3]}
    if len(sc_a) != 1 or len(sc_b) != 1:
        raise RuntimeError(f"mixed scales within a side {sc_a} {sc_b}")
    s_a, s_b = sc_a.pop(), sc_b.pop()
    s = s_a * s_b

    handles = {}
    for root, names in ((root_a, names_a), (root_b, names_b)):
        for n in names:
            handles[(root, n)] = safe_open(
                f"{root}/{n}/adapter_model.safetensors", framework="pt")
    # Module names are compared after stripping PEFT's "base_model.model."
    # prefix: stage adapters are saved without it, the exact personas with it.
    PFX = "base_model.model."

    def keymap(h):
        m = {}
        for k in h.keys():
            if ".lora_A." in k:
                mod = k.split(".lora_A.")[0]
                norm = mod
                while norm.startswith(PFX):      # exact personas carry it twice
                    norm = norm[len(PFX):]
                m[norm] = mod
        return m

    keymaps = {key: keymap(h) for key, h in handles.items()}
    for key in ((root_a, names_a[0]), (root_b, names_b[0])):
        print(f"first key under {key[0]}/{key[1]}: {sorted(handles[key].keys())[0]}", flush=True)
    mods_ref = sorted(keymaps[(root_a, names_a[0])])
    for (root, n), km in keymaps.items():
        if set(km) != set(mods_ref):
            raise RuntimeError(f"module set mismatch for {root}/{n}")

    NA, NB = len(names_a), len(names_b)
    X = np.zeros((NA, NB), dtype=np.float64)
    sq_a = np.zeros(NA, dtype=np.float64)
    sq_b = np.zeros(NB, dtype=np.float64)
    for mi, mod in enumerate(mods_ref):
        A1 = torch.stack([handles[(root_a, n)].get_tensor(
            keymaps[(root_a, n)][mod] + ".lora_A.weight").float() for n in names_a])
        B1 = torch.stack([handles[(root_a, n)].get_tensor(
            keymaps[(root_a, n)][mod] + ".lora_B.weight").float() for n in names_a])
        A2 = torch.stack([handles[(root_b, n)].get_tensor(
            keymaps[(root_b, n)][mod] + ".lora_A.weight").float() for n in names_b])
        B2 = torch.stack([handles[(root_b, n)].get_tensor(
            keymaps[(root_b, n)][mod] + ".lora_B.weight").float() for n in names_b])
        BtB = torch.einsum("ndr,mdq->nmrq", B1, B2)
        AAt = torch.einsum("nrd,mqd->nmrq", A1, A2)
        X += (BtB * AAt).sum(dim=(2, 3)).numpy()
        sq_a += torch.einsum("ndr,ndq,nrp,nqp->n",
                             B1, B1, A1.transpose(1, 2), A1.transpose(1, 2)
                             ).numpy() if False else (
            (torch.einsum("ndr,ndq->nrq", B1, B1)
             * torch.einsum("nrd,nqd->nrq", A1, A1)).sum(dim=(1, 2)).numpy())
        sq_b += ((torch.einsum("ndr,ndq->nrq", B2, B2)
                  * torch.einsum("nrd,nqd->nrq", A2, A2)).sum(dim=(1, 2))
                 .numpy())
        if mi % 20 == 0:
            print(f"  module {mi}/{len(mods_ref)}", flush=True)

    X *= s_a * s_b
    sq_a *= s_a * s_a
    sq_b *= s_b * s_b
    if (sq_a <= 0).any() or (sq_b <= 0).any():
        raise RuntimeError("zero-norm delta present")
    return {"X": X.tolist(), "names_a": names_a, "names_b": names_b,
            "norms_a": np.sqrt(sq_a).tolist(),
            "norms_b": np.sqrt(sq_b).tolist(),
            "scale": s, "scale_a": s_a, "scale_b": s_b,
            "n_modules": len(mods_ref)}


@app.local_entrypoint()
def main(subdir_a: str = "", subdir_b: str = "data_null_seedpaired_s40"):
    import numpy as np
    out = cross_gram_full.remote(subdir_a.strip("/"), subdir_b.strip("/"),
                                 bool(VOLUME_B))
    X = np.array(out["X"])
    la = subdir_a.strip("/") or "root"
    lb = subdir_b.strip("/") or "root"
    if VOLUME_B:
        lb = f"{VOLUME_B}_{lb}"
    # a nested subdir (seed1/loras_introspection) must not become a directory in results/
    la, lb = la.replace("/", "_"), lb.replace("/", "_")
    p = os.path.join(HERE, "results", f"cross_gram_full_{la}_x_{lb}.npz")
    np.savez(p, X=X, names_a=np.array(out["names_a"]),
             names_b=np.array(out["names_b"]),
             norms_a=np.array(out["norms_a"]),
             norms_b=np.array(out["norms_b"]),
             scale=out["scale"], scale_a=out["scale_a"], scale_b=out["scale_b"],
             n_modules=out["n_modules"])
    C = X / np.outer(out["norms_a"], out["norms_b"])
    print(f"wrote {p}: {X.shape[0]}x{X.shape[1]}")
    print(f"  cross-cosine mean {C.mean():+.4f}  min {C.min():+.4f}  "
          f"max {C.max():+.4f}")
