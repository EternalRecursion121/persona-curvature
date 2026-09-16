#!/usr/bin/env python3
"""Can a weight probe identify a trait across LoRA initialisations?

THE PROBLEM
-----------
LoRA confines every update to the row space of a randomly initialised A. Two
adapters trained from different inits span near-orthogonal input subspaces --
measured row-space overlap 0.0249, which is 64/2560, exactly what two random
rank-64 subspaces of a 2560-dimensional space give. So a cosine between their
weight deltas is mostly a statement about the random draw.

The signal is nonetheless there: across 40 traits trained twice from different
inits, matched-trait cosine is +0.0166 against a different-trait +0.0015 (that arm
was trained under a plainer objective; retrained to match, it is +0.0181 vs +0.0018
and the same 40/40 -- PHASE3_VERDICT.md, 2026-09-03 addendum), and
the matched trait is rank 1 for 40 of 40. Init-invariant information exists. It
is 1.7% of the norm, and the other 98.3% is the random slice.

THREE FEATURE SPACES, CHEAPEST FIRST
------------------------------------
1. NORM PROFILE. The 248 per-module Frobenius norms. Completely basis-free --
   it asks which modules moved and by how much, never where. Exact in closed
   form: ||BA||_F^2 = <B^T B, A A^T>, both factors r x r.

2. SEMANTIC SKETCH. Our existing sketch is C = P_out . dW . P_in with RANDOM
   Gaussian P. Random projections preserve pairwise geometry within one basis,
   which is why the zoo works, and do nothing to align across bases. Here P_in
   is instead 32 REAL INPUT ACTIVATIONS for that module, harvested from the base
   model on a fixed prompt set, and P_out the top 32 left singular vectors of
   the base weight. The question the sketch asks changes from "how does this
   delta project onto a random slice" to "how does it change what the module
   computes on inputs the model actually sees". Neither P depends on any
   adapter, so the coordinates are shared across initialisations.

3. (not implemented here) the functional probe, dW . h compared directly.

The comparison is the point: the same 40 seed-paired traits scored under all
three, against the random-sketch baseline of 0.0166.
"""
import json
import os

import modal

BASE_MODEL = "Qwen/Qwen3.5-4B"
K = 32
N_PROMPTS = 48
SEED = 0
# The question this now answers is procedure-invariance, not init-invariance:
# stage-1 DPO and stage-2 SFT teach the SAME trait by different means, and both
# the named-axis chart and the norm profile score at chance across that gap.
SRC = {"stage1": ("pc-qwen35-sweep", "/{t}/adapter_model.safetensors"),
       "stage2": ("pc-qwen35-oct2", "/loras_introspection/{t}/adapter_model.safetensors")}

app = modal.App("pc-qwen35-probe")
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep")
oct_vol = modal.Volume.from_name("pc-qwen35-oct2")
probe_vol = modal.Volume.from_name("pc-qwen35-probe", create_if_missing=True)
VOLS = {"/adapters": sweep_vol, "/oct": oct_vol, "/probe": probe_vol}

gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.13.0", "transformers==5.15.1", "safetensors", "hf_transfer",
                 "numpy<3", "huggingface_hub")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "CUDA_HOME": "/usr/local/cuda"})
)
cpu_image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("numpy<3", "safetensors"))

PROMPTS = [
    "A colleague takes credit for your work in a meeting. What do you do?",
    "Explain why the sky is blue.", "Write a haiku about a broken clock.",
    "What is 17 times 23?", "Summarise the causes of the First World War.",
    "A friend cancels on you for the third time. Write your reply.",
    "Describe how you keep track of things you need to do.",
    "Is it ever right to lie? Argue both sides.",
    "Write a function that reverses a linked list.",
    "You made an error in something you already submitted. What now?",
    "What happens to a candle flame in zero gravity?",
    "Describe the taste of a lemon to someone who has never had one.",
]


@app.function(image=gpu_image, gpu="A100-40GB", volumes=VOLS, timeout=60 * 60,
              secrets=[modal.Secret.from_name("hf-token")])
def build_basis() -> dict:
    """Harvest a fixed, adapter-independent basis for every targeted module.

    P_in comes from real forward passes: for each module we keep K input
    activation vectors, taken at evenly spaced token positions so the choice is
    deterministic and not dominated by one prompt. P_out is the top-K left
    singular vectors of the base weight, i.e. the directions that module can
    actually write into.
    """
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snap = snapshot_download(BASE_MODEL, ignore_patterns=["*.pt", "*.bin", "*.gguf", "*.pth"])
    tok = AutoTokenizer.from_pretrained(snap)
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16).cuda().eval()

    from safetensors import safe_open
    with safe_open("/adapters/bold/adapter_model.safetensors", framework="np") as f:
        targets = sorted({k.split(".lora_A.")[0].split(".lora_B.")[0]
                          .removeprefix("base_model.model.") for k in f.keys()})
    by_name = dict(model.named_modules())
    missing = [t for t in targets if t not in by_name]
    if missing:
        raise RuntimeError(f"{len(missing)} targets not in model: {missing[:3]}")

    store = {t: [] for t in targets}
    store_out = {t: [] for t in targets}
    handles = []

    def mk(t):
        def hook(mod, inp, out):
            h = inp[0].detach()
            o = out.detach() if not isinstance(out, tuple) else out[0].detach()
            if h.dim() == 3:
                h = h[0]
            if o.dim() == 3:
                o = o[0]
            store[t].append(h[:: max(1, h.shape[0] // 4)][:4].float().cpu())
            store_out[t].append(o[:: max(1, o.shape[0] // 4)][:4].float().cpu())
        return hook

    for t in targets:
        handles.append(by_name[t].register_forward_hook(mk(t)))

    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
             for p in PROMPTS]
    with torch.no_grad():
        for i in range(0, min(N_PROMPTS, len(texts) * 4)):
            enc = tok(texts[i % len(texts)], return_tensors="pt").to("cuda")
            model(**enc)
    for h in handles:
        h.remove()

    os.makedirs("/probe/basis", exist_ok=True)
    rng = np.random.default_rng(SEED)
    meta = {}
    for t in targets:
        H = torch.cat(store[t], 0).numpy()                 # (n_tokens, d_in)
        idx = rng.choice(H.shape[0], size=min(K, H.shape[0]), replace=False)
        Pin = H[np.sort(idx)].T                            # (d_in, K)
        Pin = Pin / (np.linalg.norm(Pin, axis=0, keepdims=True) + 1e-9)
        # P_out from REAL OUTPUTS, not from the weight's singular vectors.
        # The SVD was O(d_out^3) on 9728-dim MLP outputs and overran an hour; it
        # was also the wrong object -- singular vectors are directions the module
        # COULD write, real activations are the ones it does. Both P are now
        # measured from the same forward passes, so the sketch asks "how does
        # this delta change what the module computes on inputs the model sees",
        # which is a functional question and not a geometric one.
        O = torch.cat(store_out[t], 0).numpy()             # (n_tokens, d_out)
        oi = rng.choice(O.shape[0], size=min(K, O.shape[0]), replace=False)
        Pout = O[np.sort(oi)]                              # (K, d_out)
        Pout = (Pout / (np.linalg.norm(Pout, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
        np.savez_compressed(f"/probe/basis/{t.replace('.', '_')}.npz",
                            Pin=Pin.astype(np.float32), Pout=Pout, module=t)
        meta[t] = {"d_in": int(Pin.shape[0]), "d_out": int(Pout.shape[1]),
                   "n_tokens_seen": int(H.shape[0])}
    with open("/probe/basis_meta.json", "w") as f:
        json.dump(meta, f)
    probe_vol.commit()
    return {"modules": len(targets), "K": K,
            "example": {k: meta[k] for k in list(meta)[:2]}}


@app.function(image=cpu_image, volumes=VOLS, cpu=4.0, memory=16384, timeout=60 * 30,
              max_containers=12)
def featurise(job: dict) -> dict:
    """Norm profile and semantic sketch for one adapter. Never materialises dW."""
    import numpy as np
    from safetensors import safe_open

    trait, src = job["trait"], job["src"]
    path = ("/adapters/{t}/adapter_model.safetensors" if src == "stage1"
            else "/oct/loras_introspection/{t}/adapter_model.safetensors").format(t=trait)
    if not os.path.exists(path):
        return {"trait": trait, "src": src, "error": "missing"}
    norms, sem, mods = [], [], []
    with safe_open(path, framework="np") as f:
        keys = list(f.keys())
        ms = sorted({k.split(".lora_A.")[0].split(".lora_B.")[0]
                     .removeprefix("base_model.model.") for k in keys})
        pre = "base_model.model." if keys[0].startswith("base_model.model.") else ""
        for m in ms:
            A = f.get_tensor(f"{pre}{m}.lora_A.weight").astype(np.float32)
            B = f.get_tensor(f"{pre}{m}.lora_B.weight").astype(np.float32)
            norms.append(float(np.sqrt(max(np.sum((B.T @ B) * (A @ A.T)), 0.0))))
            z = np.load(f"/probe/basis/{m.replace('.', '_')}.npz")
            sem.append(((z["Pout"] @ B) @ (A @ z["Pin"])).ravel())
            mods.append(m)
    out = f"/probe/feat/{src}"
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(f"{out}/{trait}.npz", norms=np.array(norms, dtype=np.float32),
                        sem=np.concatenate(sem).astype(np.float32),
                        modules=np.array(mods), trait=trait, src=src)
    probe_vol.commit()
    return {"trait": trait, "src": src, "n_modules": len(mods),
            "sem_dim": int(sum(len(s) for s in sem))}


@app.local_entrypoint()
def main(stage: str = "basis"):
    if stage == "basis":
        print(json.dumps(build_basis.remote(), indent=1))
        return
    import glob
    traits = sorted(os.path.basename(p)[:-4] for p in glob.glob(
        "/home/vibe12/projects/persona-curvature/qwen35/analysis/sketches/stage2_vol/*.npz"))
    jobs = ([{"trait": t, "src": "stage1"} for t in traits]
            + [{"trait": t, "src": "stage2"} for t in traits])
    print(f"{len(traits)} traits x 2 procedures = {len(jobs)} adapters", flush=True)
    bad = 0
    for r in featurise.map(jobs, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception) or r.get("error"):
            print("  ERR", r, flush=True)
            bad += 1
    print(f"done, {bad} failure(s)")
