#!/usr/bin/env python3
"""Project a capability-RL adapter into the personality space the zoo defines.

The adapter is not a sample from the persona population, so it is NOT centred
with the zoo mean.  Centring is a statement about where the cloud sits; this
adapter is a separate object and we are asking where it points.  Directions
survive centring, offsets do not, so everything here is a projection onto
DIRECTIONS derived from the centred zoo: principal components, the constructed
Big Five keying axes, the grand-mean (assistant) direction, and each of the 134
individual trait deltas.

Two numbers carry the result:
  * the cosine with each direction -- does capability RL point anywhere legible;
  * the fraction of the adapter's norm lying inside the top-k personality
    subspace, against a random-direction null, which says whether ANY of the
    alignment is more than what a random direction gets in 253,952 dimensions.

Stage `sketch` runs on Modal because the checkpoints live on the volume; stage
`project` runs here on the small .npz files.
"""
import argparse
import glob
import hashlib
import json
import os

import modal

K = 32
app = modal.App("pc-qwen35-rlsketch")
rl_vol = modal.Volume.from_name("pc-qwen35-rl")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy<3", "safetensors"))


def _proj(name, rows, cols, k, which):
    """Byte-for-byte the projection in sketch_adapters.py:proj."""
    import numpy as np
    h = hashlib.sha256(f"{name}|{which}|{k}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    return rng.standard_normal((rows, cols), dtype=np.float32) / np.sqrt(k)


@app.function(image=image, volumes={"/rl": rl_vol}, cpu=4.0, memory=16384,
              timeout=60 * 60)
def sketch_run(tag: str = "math") -> dict:
    import numpy as np
    from safetensors import safe_open

    root = f"/rl/runs/{tag}"
    ckpts = sorted([d for d in os.listdir(root) if d.startswith("checkpoint-")],
                   key=lambda d: int(d.split("-")[1])) + ["final"]
    out = {}
    for c in ckpts:
        p = f"{root}/{c}/adapter_model.safetensors"
        if not os.path.exists(p):
            continue
        with safe_open(p, framework="np") as f:
            mods = sorted({k.split(".lora_A.")[0].split(".lora_B.")[0] for k in f.keys()})
            vecs, norms = [], []
            for m in mods:
                A = f.get_tensor(f"{m}.lora_A.weight").astype(np.float32)
                B = f.get_tensor(f"{m}.lora_B.weight").astype(np.float32)
                norms.append(float(np.sqrt(max(np.sum((B.T @ B) * (A @ A.T)), 0.0))))
                Po = _proj(m.removeprefix("base_model.model."), K, B.shape[0], K, "out")
                Pi = _proj(m.removeprefix("base_model.model."), A.shape[1], K, K, "in")
                vecs.append(((Po @ B) @ (A @ Pi)).ravel())
        v = np.concatenate(vecs)
        out[c] = {"sketch": v.tolist(), "norm_sum": float(np.sum(norms)),
                  "n_modules": len(mods)}
        print(f"{c}: dim={v.size} |dW|={np.sum(norms):.2f}", flush=True)
    with open(f"{root}/sketches.json", "w") as f:
        json.dump(out, f)
    rl_vol.commit()
    return {"tag": tag, "checkpoints": list(out),
            "norms": {c: out[c]["norm_sum"] for c in out}}


@app.local_entrypoint()
def main(tag: str = "math"):
    r = sketch_run.remote(tag)
    print(json.dumps(r, indent=1))
