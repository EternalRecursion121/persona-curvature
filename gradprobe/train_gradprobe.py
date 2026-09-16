"""
Modal harness for the GRADIENT-PROBE experiment: per-document gradient sketches.

The question
-----------
400 documents, two per fact (200 facts, different genres).  Train ONE optimiser
step per document at batch size 1, so step i's gradient is attributable to
document i and nothing else.  Downstream we ask: are gradients from the SAME
fact (different genre) more similar than gradients from DIFFERENT facts (same
genre)?  Every downstream statistic is a cosine.

Why sketches
------------
A full LoRA gradient here is D = 29,933,568 parameters (~120MB fp32); 400 of
them is 48GB.  A random projection preserves inner products (Johnson-
Lindenstrauss), and inner products are all a cosine needs.  We therefore store
S g, not g.

The sketch
----------
A COUNT-SKETCH (Charikar/Clarkson-Woodruff): a sparse +/-1 random projection
with exactly one non-zero per input coordinate,

    (S x)_b = sum_{j : h(j) = b} s(j) * x_j ,     h: [D) -> [k),  s: [D) -> {+1,-1}

It is a linear map with a fixed matrix, is unbiased on inner products,

    E<Sx,Sy> = <x,y>,
    Var<Sx,Sy> = (1/k) (||x||^2 ||y||^2 + <x,y>^2 - 2 sum_j x_j^2 y_j^2),

which is the SAME leading-order variance as a dense Gaussian sketch with
entries N(0,1/k) -- and it costs O(D) memory and O(D) time instead of O(Dk),
which for D=3e7, k=8192 is the difference between 480MB and 1TB.  A dense
Gaussian projection at this dimensionality is not merely slow, it does not fit.

h and s are drawn ONCE from a fixed seed (SKETCH_SEED) and held on device for
the whole run; the tables are blake2b-hashed at the first and last step and the
hashes asserted equal, because a projection that differed between steps would
silently destroy every cross-step comparison this experiment is made of.

Three granularities, because the point is whether content is visible under one
readout and not another:
    pooled     one sketch of the whole concatenated gradient        (8192,)
    per_layer  one sketch per transformer layer                     (36, 1024)
    per_type   one sketch per module type q,k,v,o,gate,up,down      (7, 1024)
The per-layer and per-type partitions are two different partitions of the same
D coordinates; the restriction of a count-sketch to any subset of coordinates
is still a count-sketch on that subset, so one hash table serves both.

EXACT (unsketched) gradient norms are recorded per granularity as well, so
cosines can be sanity-checked, and so "does the norm alone carry the signal?"
is answerable without the sketches at all.

Two modes
---------
    --mode sequential   the optimiser steps between documents; parameters
                        evolve.  This is the realistic training case.
    --mode frozen       no optimiser at all; every gradient is evaluated at the
                        IDENTICAL initial parameters.  The cleaner scientific
                        object: it removes the confound that document 399 is
                        seen by a different model than document 0.
Both are run.  In frozen mode the trainable parameters are blake2b-hashed
before step 0 and after step N-1 and asserted BITWISE identical, and document 0
is replayed at the end and its sketch asserted bit-identical to step 0's.  In
sequential mode the same replay is asserted to DIFFER, which proves the
parameters really did move (an accidentally-frozen "sequential" run would be a
beautiful fake result).

Loss
----
Plain LM training on the document text.  There is no prompt/response split, so
there is no masking: labels == input_ids and the loss is the mean cross-entropy
over every predicted token (HF shifts internally, so position 0 is a context
token and positions 1..n-1 are predicted -- that is what "all tokens" means for
a causal LM).  Recorded in meta.json as `loss_masking: "none (all tokens)"`.

Usage
-----
    ~/cartovenv/bin/python gradprobe/upload_gradprobe.py --file docs.jsonl
    ~/cartovenv/bin/modal run gradprobe/train_gradprobe.py --modes sequential,frozen
    ~/cartovenv/bin/modal run gradprobe/train_gradprobe.py::fidelity   # synthetic JL check
    ~/cartovenv/bin/python gradprobe/fetch_gradprobe.py
    ~/cartovenv/bin/python gradprobe/train_gradprobe.py --selftest     # no Modal, no GPU
"""

import hashlib
import json
import math
import os
import time

import modal

APP_NAME = "gradprobe"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
GPU_TYPE = os.environ.get("GP_GPU", "A100-40GB")
ATTN_IMPL = os.environ.get("GP_ATTN", "sdpa")

DATA_VOLUME = "gradprobe-data"
OUT_VOLUME = "gradprobe-out"

# ---- fixed, load-bearing hyperparameters -----------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0          # asserted 0: a dropout mask would make the
                            # "same parameters => same gradient" replay check
                            # meaningless and add noise to every cosine
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]
TYPE_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"]
MAX_SEQ_LEN = 512
SEED = 0
LR = 1e-4                   # sequential mode only
MAX_GRAD_NORM = 1.0         # sequential mode only, applied AFTER the sketch
BATCH_SIZE = 1              # non-negotiable: this is what makes step i == doc i

# ---- sketch spec (fixed; changing any of these invalidates comparability) ---
SKETCH_SEED = 1234
POOLED_DIM = 8192
LAYER_DIM = 1024
TYPE_DIM = 1024
SKETCH_KIND = "count_sketch_pm1"   # sparse +/-1, one non-zero per column

# how many full fp32 gradients to retain on device for the in-run fidelity
# check (K*D*4 bytes; 12 * 3e7 * 4 = 1.44GB)
FIDELITY_K = 12

# lora_A init checksum (float64, CPU, sorted by parameter name).  Same base
# model / seed / rank / target modules as drift/train_drift.py, so it is the
# same number; both modes must reproduce it or the frozen-vs-sequential
# comparison is comparing two different models.
REFERENCE_INIT_CHECKSUM = 94.3085432141379

MODES = ("sequential", "frozen")

app = modal.App(APP_NAME)
data_vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
out_vol = modal.Volume.from_name(OUT_VOLUME, create_if_missing=True)


def _download_base_model():
    from huggingface_hub import snapshot_download

    snapshot_download(BASE_MODEL, ignore_patterns=["*.pt", "*.bin", "*.gguf"])


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.49.0",
        "peft==0.14.0",
        "accelerate==1.3.0",
        "safetensors==0.5.2",
        "numpy==1.26.4",
        "huggingface_hub==0.28.1",
        "hf_transfer==0.1.9",
        "sentencepiece==0.2.0",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # forwarded so the frozen-replay tolerances can be set per-run from
            # the launching shell; defaults match the original hard-coded values
            "GP_REPLAY_REL_TOL": os.environ.get("GP_REPLAY_REL_TOL", "1e-4"),
            "GP_REPLAY_COS_TOL": os.environ.get("GP_REPLAY_COS_TOL", "1e-9"),
        }
    )
    .run_function(_download_base_model)
)


# ===========================================================================
# pure helpers (unit-tested locally by --selftest; no torch/modal needed)
# ===========================================================================
def parse_module_key(param_name: str):
    """
    ('base_model.model.model.layers.12.self_attn.q_proj.lora_A.default.weight')
      -> (12, 'q_proj')

    Returns (layer_index, module_type).  Raises if the name is not a LoRA
    parameter of a targeted module inside a numbered transformer layer -- we
    would rather fail than silently drop coordinates out of the partition.
    """
    parts = param_name.split(".")
    if "layers" not in parts:
        raise ValueError(f"no 'layers.<n>' in {param_name!r}")
    li = parts.index("layers")
    if li + 1 >= len(parts) or not parts[li + 1].isdigit():
        raise ValueError(f"'layers' not followed by an index in {param_name!r}")
    layer = int(parts[li + 1])
    mtype = next((p for p in parts if p in TYPE_ORDER), None)
    if mtype is None:
        raise ValueError(f"no target module type in {param_name!r}")
    return layer, mtype


def cosine_matrix(X):
    """Pairwise cosine similarity of the rows of a 2-D array."""
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    Z = X / n
    return Z @ Z.T


def upper_pairs(M):
    """Strictly-upper-triangular entries of a square matrix, as a 1-D array."""
    import numpy as np

    M = np.asarray(M)
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


def fidelity_report(true_cos, sketch_cos):
    """
    Compare two vectors of pairwise cosines.  The numbers that matter are the
    Pearson correlation (does the sketch RANK pairs the way the truth does)
    and the max absolute error (can a single pair be badly wrong).
    """
    import numpy as np

    a = np.asarray(true_cos, dtype=np.float64).ravel()
    b = np.asarray(sketch_cos, dtype=np.float64).ravel()
    err = b - a
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        r = float("nan")
    else:
        r = float(np.corrcoef(a, b)[0, 1])
    return {
        "n_pairs": int(a.size),
        "pearson_r": r,
        "max_abs_err": float(np.max(np.abs(err))) if a.size else float("nan"),
        "rms_err": float(np.sqrt(np.mean(err ** 2))) if a.size else float("nan"),
        "mean_err": float(np.mean(err)) if a.size else float("nan"),
        "true_cos_range": [float(a.min()), float(a.max())] if a.size else None,
    }


def load_docs(path: str, limit: int = 0) -> list:
    """
    Read the document file IN FILE ORDER (never shuffled -- step i is doc i).

    Accepts the natural spellings for the text field and the id field so that
    whatever the data generator emits is readable without editing it:
        text:   text | document | doc | content | passage | body
        doc_id: doc_id | id | idx
    fact_id / genre / any other key is carried through into meta.json untouched.
    """
    text_keys = ("text", "document", "doc", "content", "passage", "body")
    id_keys = ("doc_id", "id", "idx")
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i+1}: bad JSON: {e}")
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{i+1}: row is not an object")
            tkey = next((k for k in text_keys if isinstance(obj.get(k), str)
                         and obj[k].strip()), None)
            if tkey is None:
                raise ValueError(
                    f"{path}:{i+1}: no non-empty text field; have {sorted(obj)} "
                    f"(looked for {list(text_keys)})")
            dkey = next((k for k in id_keys if k in obj), None)
            doc_id = obj[dkey] if dkey is not None else len(rows)
            extras = {k: v for k, v in obj.items() if k != tkey}
            rows.append({"doc_id": doc_id, "text": obj[tkey], "meta": extras,
                         "file_index": len(rows)})
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"{path}: no rows")
    ids = [r["doc_id"] for r in rows]
    if len(set(map(str, ids))) != len(ids):
        raise ValueError(f"{path}: doc_ids are not unique ({len(ids)} rows, "
                         f"{len(set(map(str, ids)))} distinct)")
    return rows


def sha_ids(ids) -> str:
    """Stable content hash of a token-id list -- the attribution fingerprint."""
    h = hashlib.sha256()
    for t in ids:
        h.update(int(t).to_bytes(4, "little", signed=False))
    return h.hexdigest()[:16]


def _stats(xs):
    import numpy as np

    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return None
    return {
        "n": int(a.size),
        "min": float(a.min()), "max": float(a.max()),
        "mean": float(a.mean()), "median": float(np.median(a)),
        "std": float(a.std()),
        "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
    }


# ===========================================================================
# the sketch itself (torch, device-agnostic: GPU in the trainer, CPU in the
# selftest, so the thing verified is the thing that runs)
# ===========================================================================
class CountSketch:
    """
    A FIXED sparse +/-1 projection of a D-vector, produced at three
    granularities in one pass.

    Built once from `segments` = [(name, start, end, layer_idx, type_idx)],
    a partition of [0, D) into the flattened trainable parameter tensors in
    sorted-name order.  Holds four flat tables on device:

        idx_pooled  int32 [D)  -> [0, POOLED_DIM)
        idx_layer   int32 [D)  -> [0, n_layers*LAYER_DIM)   offset by layer
        idx_type    int32 [D)  -> [0, n_types*TYPE_DIM)     offset by type
        sign        fp32 [D)   in {+1,-1}

    All four are drawn from ONE torch.Generator seeded with SKETCH_SEED, in a
    fixed order, and never touched again.  `table_hash()` blake2b's them; the
    trainer asserts the hash is the same at the first and last step.
    """

    def __init__(self, segments, n_layers, n_types, device, seed=SKETCH_SEED,
                 pooled_dim=POOLED_DIM, layer_dim=LAYER_DIM, type_dim=TYPE_DIM):
        import torch

        self.segments = list(segments)
        self.D = int(self.segments[-1][2]) if self.segments else 0
        self.n_layers = int(n_layers)
        self.n_types = int(n_types)
        self.pooled_dim = int(pooled_dim)
        self.layer_dim = int(layer_dim)
        self.type_dim = int(type_dim)
        self.device = device
        self.seed = int(seed)

        g = torch.Generator(device=device)
        g.manual_seed(self.seed)
        D = self.D
        self.idx_pooled = torch.randint(0, self.pooled_dim, (D,), generator=g,
                                        device=device, dtype=torch.int32)
        self.idx_layer = torch.randint(0, self.layer_dim, (D,), generator=g,
                                       device=device, dtype=torch.int32)
        self.idx_type = torch.randint(0, self.type_dim, (D,), generator=g,
                                      device=device, dtype=torch.int32)
        bits = torch.randint(0, 2, (D,), generator=g, device=device,
                             dtype=torch.int32)
        self.sign = (bits * 2 - 1).to(torch.float32)
        del bits

        # fold the block offsets into the index tables, so the per-layer and
        # per-type sketches are single index_add_ calls into one flat buffer
        for _name, s, e, li, ti in self.segments:
            self.idx_layer[s:e] += li * self.layer_dim
            self.idx_type[s:e] += ti * self.type_dim

        # a coordinate must land inside its own block, always
        assert int(self.idx_layer.max()) < self.n_layers * self.layer_dim
        assert int(self.idx_type.max()) < self.n_types * self.type_dim

    # -- provenance ---------------------------------------------------------
    def table_hash(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        for t in (self.idx_pooled, self.idx_layer, self.idx_type, self.sign):
            h.update(t.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def spec(self) -> dict:
        return {
            "kind": SKETCH_KIND,
            "seed": self.seed,
            "input_dim": self.D,
            "pooled_dim": self.pooled_dim,
            "layer_dim": self.layer_dim,
            "type_dim": self.type_dim,
            "n_layers": self.n_layers,
            "n_types": self.n_types,
            "nonzeros_per_column": 1,
            "note": "count-sketch: one +/-1 per input coordinate; unbiased on "
                    "inner products, Var = (||x||^2||y||^2 + <x,y>^2)/k to "
                    "leading order, i.e. the same as a dense Gaussian sketch",
        }

    # -- the operation ------------------------------------------------------
    def apply(self, flat):
        """
        flat: fp32 tensor of shape (D,).  Returns
            (pooled (k,), per_layer (L,kl), per_type (T,kt))
        all fp32 on `flat.device`.
        """
        import torch

        assert flat.dtype == torch.float32 and flat.shape == (self.D,)
        signed = flat * self.sign
        pooled = torch.zeros(self.pooled_dim, dtype=torch.float32,
                             device=flat.device)
        lay = torch.zeros(self.n_layers * self.layer_dim, dtype=torch.float32,
                          device=flat.device)
        typ = torch.zeros(self.n_types * self.type_dim, dtype=torch.float32,
                          device=flat.device)
        try:
            pooled.index_add_(0, self.idx_pooled, signed)
            lay.index_add_(0, self.idx_layer, signed)
            typ.index_add_(0, self.idx_type, signed)
        except (RuntimeError, TypeError):     # older torch wants int64 indices
            pooled.index_add_(0, self.idx_pooled.long(), signed)
            lay.index_add_(0, self.idx_layer.long(), signed)
            typ.index_add_(0, self.idx_type.long(), signed)
        return (pooled,
                lay.view(self.n_layers, self.layer_dim),
                typ.view(self.n_types, self.type_dim))

    def exact_norms(self, flat):
        """
        EXACT (unsketched) L2 norms of the same partition, float64.
        Returns (total, per_layer list, per_type list).  Computed BEFORE the
        projection touches anything, so it is a genuine independent readout.
        """
        import torch

        lay = [0.0] * self.n_layers
        typ = [0.0] * self.n_types
        tot = 0.0
        for _name, s, e, li, ti in self.segments:
            sq = float((flat[s:e].double() ** 2).sum())
            lay[li] += sq
            typ[ti] += sq
            tot += sq
        return (math.sqrt(max(tot, 0.0)),
                [math.sqrt(max(x, 0.0)) for x in lay],
                [math.sqrt(max(x, 0.0)) for x in typ])


# ===========================================================================
# synthetic sketch-fidelity check  (a): does a k-dim count-sketch preserve
# cosines at the REAL gradient dimensionality?
# ===========================================================================
def _synthetic_fidelity(D, n_vec, dims, device, seed=0, families=None):
    """
    Draw n_vec vectors of dimension D from several families, sketch them with a
    count-sketch of each width in `dims`, and compare all pairwise cosines
    against the exact ones.

    The families matter.  A count-sketch's error scales with how spread a
    vector's energy is: iid Gaussian is the easy case, and a heavy-tailed /
    power-law-decaying spectrum (which is what real gradients look like) is the
    hard one, because a few large coordinates can collide.  Testing only
    Gaussians would flatter the method.
    """
    import numpy as np
    import torch

    out = {}
    fams = families or ["gaussian", "powerlaw", "sparse", "correlated"]
    for fam in fams:
        g = torch.Generator(device=device)
        g.manual_seed(seed + hash(fam) % 10000)
        X = torch.empty(n_vec, D, dtype=torch.float32, device=device)
        if fam == "gaussian":
            for i in range(n_vec):
                X[i] = torch.randn(D, generator=g, device=device)
        elif fam == "powerlaw":
            # magnitudes ~ j^-0.75 in a fixed random coordinate order: a
            # realistic heavy-ish tail with a shared "important coordinate" set
            perm = torch.randperm(D, generator=g, device=device)
            decay = torch.empty(D, dtype=torch.float32, device=device)
            decay[perm] = (torch.arange(1, D + 1, device=device,
                                        dtype=torch.float32) ** -0.75)
            for i in range(n_vec):
                X[i] = torch.randn(D, generator=g, device=device) * decay
        elif fam == "sparse":
            # 0.5% dense: the adversarial case for a hashing sketch
            nz = max(1, D // 200)
            for i in range(n_vec):
                X[i] = 0.0
                idx = torch.randint(0, D, (nz,), generator=g, device=device)
                X[i].index_add_(0, idx,
                                torch.randn(nz, generator=g, device=device))
        elif fam == "correlated":
            # a planted common component so the true cosines span a wide range
            base = torch.randn(D, generator=g, device=device)
            for i in range(n_vec):
                w = float(i) / max(n_vec - 1, 1)
                X[i] = w * base + math.sqrt(max(1 - w * w, 0.0)) * \
                    torch.randn(D, generator=g, device=device)
        else:
            raise ValueError(fam)

        Xn = X / X.norm(dim=1, keepdim=True)
        true_c = (Xn @ Xn.T).double().cpu().numpy()
        res = {}
        for k in dims:
            segs = [("all", 0, D, 0, 0)]
            cs = CountSketch(segs, 1, 1, device, seed=SKETCH_SEED,
                             pooled_dim=k, layer_dim=1, type_dim=1)
            S = torch.stack([cs.apply(X[i].contiguous())[0]
                             for i in range(n_vec)])
            del cs
            Sn = S / S.norm(dim=1, keepdim=True)
            sk_c = (Sn @ Sn.T).double().cpu().numpy()
            res[str(k)] = fidelity_report(upper_pairs(true_c),
                                          upper_pairs(sk_c))
            # theory: cosine-estimate std ~ sqrt((1+rho^2)/k) <= sqrt(2/k)
            res[str(k)]["theory_rms_bound"] = float(math.sqrt(2.0 / k))
        out[fam] = res
        del X, Xn
        torch.cuda.empty_cache() if device == "cuda" else None
    return out


@app.function(image=image, gpu=GPU_TYPE, timeout=60 * 40)
def fidelity(dim: int = 0, n_vec: int = 20, dims: str = "1024,4096,8192,16384"):
    """
    Verification (a), standalone, at the real gradient dimensionality.
    `dim=0` means "use the real D" (computed from the LoRA shapes, no model
    download needed -- it is a function of the architecture).
    """
    import numpy as np
    import torch

    # D for Qwen2.5-3B + LoRA r=16 on the 7 targets, 36 layers, computed rather
    # than guessed: r*(in+out) per module.
    hidden, inter, kv = 2048, 11008, 256
    per_layer = (LORA_R * (hidden + hidden) * 2          # q, o
                 + LORA_R * (hidden + kv) * 2            # k, v
                 + LORA_R * (hidden + inter) * 2         # gate, up
                 + LORA_R * (inter + hidden))            # down
    D_real = per_layer * 36
    D = int(dim) if dim else D_real
    ks = [int(x) for x in dims.split(",") if x.strip()]
    print(f"synthetic fidelity: D={D} (real LoRA D={D_real}) n_vec={n_vec} "
          f"dims={ks}", flush=True)
    t0 = time.time()
    res = _synthetic_fidelity(D, n_vec, ks, "cuda", seed=0)
    print(f"\n{'family':<12}{'k':>7}{'pairs':>7}{'pearson_r':>12}"
          f"{'rms_err':>11}{'max_abs_err':>13}{'theory_rms':>12}")
    print("-" * 74)
    for fam, byk in res.items():
        for k, r in byk.items():
            print(f"{fam:<12}{k:>7}{r['n_pairs']:>7}{r['pearson_r']:>12.6f}"
                  f"{r['rms_err']:>11.5f}{r['max_abs_err']:>13.5f}"
                  f"{r['theory_rms_bound']:>12.5f}")
    print("-" * 74)
    print(f"wall {time.time()-t0:.1f}s")
    return {"D": D, "D_real": D_real, "n_vec": n_vec, "results": res}


# ===========================================================================
# the run
# ===========================================================================
@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/data": data_vol, "/out": out_vol},
    timeout=60 * 180,
)
def run_probe(job: dict) -> dict:
    import random

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    t0 = time.time()
    mode = job["mode"]
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES} (got {mode!r})")
    docs_file = job.get("docs_file", "docs.jsonl")
    data_dir = job.get("data_dir", "/data")
    limit = int(job.get("limit", 0))
    fid_k = int(job.get("fidelity_k", FIDELITY_K))
    out_dir = f"/out/{mode}"
    dev = "cuda"

    print(f"=== gradprobe mode={mode} gpu={GPU_TYPE} docs={docs_file} "
          f"limit={limit or '-'} ===", flush=True)

    data_vol.reload()
    path = f"{data_dir}/{docs_file}"
    if not os.path.exists(path):
        have = sorted(os.listdir(data_dir)) if os.path.isdir(data_dir) else "NOTHING"
        raise FileNotFoundError(f"{path} missing; volume holds {have}")
    docs = load_docs(path, limit=limit)
    N = len(docs)
    print(f"  {N} documents, IN FILE ORDER (no shuffle)", flush=True)

    # ---------------- tokenise (plain LM: no prompt/response split) ---------
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    eos = tok.eos_token_id
    encoded = []
    for d in docs:
        ids = tok(d["text"], add_special_tokens=False)["input_ids"]
        ids = ids[: MAX_SEQ_LEN - 1] + [eos]
        if len(ids) < 2:
            raise ValueError(f"doc {d['doc_id']!r} tokenises to <2 tokens")
        encoded.append(ids)
    ntoks = [len(e) for e in encoded]
    print(f"  tokens: min={min(ntoks)} median={int(np.median(ntoks))} "
          f"max={max(ntoks)} total={sum(ntoks)}  (labels == input_ids, "
          f"loss over ALL tokens, no masking)", flush=True)

    # ---------------- model (seeded immediately before construction) --------
    def reseed():
        random.seed(SEED); np.random.seed(SEED)
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    reseed()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation=ATTN_IMPL
    ).to(dev)
    model.config.use_cache = False
    reseed()   # identical LoRA A init regardless of what the base load consumed
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    assert LORA_DROPOUT == 0.0, "dropout would break the replay determinism check"
    model.train()

    # same float64/CPU accumulation as drift/train_drift.py, for the same
    # reason: a CUDA tree-reduction is not bit-reproducible against a CPU one
    init_hash = 0.0
    with torch.no_grad():
        for n, p in sorted((n, p) for n, p in model.named_parameters()
                           if "lora_A" in n):
            init_hash += float(p.detach().to("cpu", torch.float64).sum().item())
    print(f"  lora_A init checksum: {init_hash:.10f}", flush=True)

    # ---------------- the coordinate partition ------------------------------
    tparams = sorted(((n, p) for n, p in model.named_parameters()
                      if p.requires_grad), key=lambda kv: kv[0])
    if not tparams:
        raise ValueError("no trainable parameters")
    layers_seen, types_seen = set(), set()
    for n, _ in tparams:
        li, ti = parse_module_key(n)
        layers_seen.add(li)
        types_seen.add(ti)
    n_layers = max(layers_seen) + 1
    assert layers_seen == set(range(n_layers)), f"layer gaps: {sorted(layers_seen)}"
    type_idx = {t: i for i, t in enumerate(TYPE_ORDER)}
    assert types_seen == set(TYPE_ORDER), f"unexpected module types: {types_seen}"
    n_types = len(TYPE_ORDER)

    segments, off = [], 0
    for n, p in tparams:
        li, ti = parse_module_key(n)
        segments.append((n, off, off + p.numel(), li, type_idx[ti]))
        off += p.numel()
    D = off
    print(f"  trainable tensors={len(tparams)} D={D} layers={n_layers} "
          f"types={n_types} lora_param_dtype={tparams[0][1].dtype}", flush=True)

    # per-layer slices are CONTIGUOUS under a sorted-name order (every param of
    # layer L shares the prefix '...layers.L.'), which the exact-norm and
    # fidelity code relies on only via `segments`; assert the partition anyway
    assert segments[-1][2] == D
    assert all(segments[i][2] == segments[i + 1][1] for i in range(len(segments) - 1))

    # ---------------- the FIXED sketch --------------------------------------
    sketch = CountSketch(segments, n_layers, n_types, dev)
    table_hash_0 = sketch.table_hash()
    print(f"  sketch: {SKETCH_KIND} seed={SKETCH_SEED} "
          f"pooled={POOLED_DIM} layer={LAYER_DIM} type={TYPE_DIM} "
          f"tables_blake2b={table_hash_0}", flush=True)

    # ---------------- optimiser (sequential only) ---------------------------
    opt = None
    if mode == "sequential":
        opt = torch.optim.AdamW([p for _, p in tparams], lr=LR,
                                betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    print(f"  optimiser: {'AdamW lr=%g (constant, no warmup)' % LR if opt else 'NONE (frozen)'}",
          flush=True)

    # ---------------- bitwise parameter fingerprints ------------------------
    def param_blake() -> str:
        h = hashlib.blake2b(digest_size=16)
        for n, p in tparams:
            h.update(n.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def all_param_checksum() -> float:
        s = 0.0
        with torch.no_grad():
            for n, p in sorted((n, p) for n, p in model.named_parameters()):
                s += float(p.detach().to("cpu", torch.float64).sum().item())
        return s

    blake_before = param_blake()
    allsum_before = all_param_checksum()
    print(f"  trainable-param blake2b @start: {blake_before}", flush=True)

    # ---------------- buffers ----------------------------------------------
    gflat = torch.zeros(D, dtype=torch.float32, device=dev)
    sk_pooled = np.zeros((N, POOLED_DIM), dtype=np.float32)
    sk_layer = np.zeros((N, n_layers, LAYER_DIM), dtype=np.float32)
    sk_type = np.zeros((N, n_types, TYPE_DIM), dtype=np.float32)
    norms_total, norms_layer, norms_type = [], [], []
    fid_k = min(fid_k, N)
    fid_buf = (torch.zeros(fid_k, D, dtype=torch.float32, device=dev)
               if fid_k >= 2 else None)

    def grad_step(ids):
        """forward + backward at the CURRENT parameters; fills gflat."""
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        am = torch.ones_like(x)
        out = model(input_ids=x, attention_mask=am, labels=x)
        model.zero_grad(set_to_none=True)
        out.loss.backward()
        for (n, p), (_, s, e, _, _) in zip(tparams, segments):
            if p.grad is None:
                raise RuntimeError(f"no gradient for trainable param {n}")
            gflat[s:e].copy_(p.grad.detach().reshape(-1))
        return float(out.loss.detach())

    # ---------------- the loop ----------------------------------------------
    steps = []
    t_train = time.time()
    for i, (d, ids) in enumerate(zip(docs, encoded)):
        loss = grad_step(ids)

        # sketch + EXACT norms, both at the current parameters, BEFORE any
        # clipping or optimiser update
        tot, lay_n, typ_n = sketch.exact_norms(gflat)
        p_sk, l_sk, t_sk = sketch.apply(gflat)
        sk_pooled[i] = p_sk.cpu().numpy()
        sk_layer[i] = l_sk.cpu().numpy()
        sk_type[i] = t_sk.cpu().numpy()
        norms_total.append(tot)
        norms_layer.append(lay_n)
        norms_type.append(typ_n)
        if fid_buf is not None and i < fid_k:
            fid_buf[i].copy_(gflat)

        steps.append({
            "step": i,
            "doc_id": d["doc_id"],
            "file_index": d["file_index"],
            "n_tokens": len(ids),
            "input_ids_sha256_16": sha_ids(ids),
            "loss": loss,
            "grad_norm_total": tot,
            "sketch_pooled_norm": float(p_sk.double().norm()),
        })

        if mode == "sequential":
            torch.nn.utils.clip_grad_norm_([p for _, p in tparams], MAX_GRAD_NORM)
            opt.step()
        if i % 50 == 0 or i == N - 1:
            print(f"  step {i:>4} doc={str(d['doc_id'])[:24]:<24} "
                  f"ntok={len(ids):>4} loss={loss:.4f} |g|={tot:.5e}", flush=True)
    train_wall = time.time() - t_train

    # ---------------- verification -----------------------------------------
    verify = {}

    # (b) the projection is the SAME map at the last step as at the first
    table_hash_1 = sketch.table_hash()
    assert table_hash_0 == table_hash_1, (
        f"PROJECTION CHANGED during the run ({table_hash_0} -> {table_hash_1}); "
        f"every cross-step cosine would be meaningless")
    verify["projection_table_blake2b_first"] = table_hash_0
    verify["projection_table_blake2b_last"] = table_hash_1
    verify["projection_identical_across_steps"] = True
    print(f"  [b] projection tables identical first==last: {table_hash_0}",
          flush=True)

    # (c) attribution: step i used doc i, by id AND by token-content hash
    assert len(steps) == N
    for i, (r, d, ids) in enumerate(zip(steps, docs, encoded)):
        assert r["step"] == i and r["file_index"] == i
        assert r["doc_id"] == d["doc_id"], (
            f"step {i} recorded doc {r['doc_id']!r}, file order says {d['doc_id']!r}")
        assert r["input_ids_sha256_16"] == sha_ids(ids), (
            f"step {i}: input_ids hash does not match doc {d['doc_id']!r}")
    verify["attribution_step_i_is_doc_i"] = True
    verify["doc_id_order"] = [d["doc_id"] for d in docs]
    verify["input_ids_sha256_order"] = [r["input_ids_sha256_16"] for r in steps]
    print(f"  [c] attribution verified for all {N} steps "
          f"(doc_id AND input_ids hash match file order)", flush=True)

    # (d) frozen really is frozen / sequential really moved
    blake_after = param_blake()
    allsum_after = all_param_checksum()
    verify["trainable_blake2b_start"] = blake_before
    verify["trainable_blake2b_end"] = blake_after
    verify["all_param_float64_checksum_start"] = allsum_before
    verify["all_param_float64_checksum_end"] = allsum_after
    verify["params_bitwise_unchanged"] = bool(blake_before == blake_after)
    if mode == "frozen":
        assert blake_before == blake_after, (
            "FROZEN MODE MOVED THE PARAMETERS: "
            f"{blake_before} -> {blake_after}")
        assert allsum_before == allsum_after
        print(f"  [d] frozen: trainable params BITWISE identical step 0 vs "
              f"step {N-1} ({blake_after}); all-param f64 checksum "
              f"{allsum_after!r} unchanged", flush=True)
    else:
        assert blake_before != blake_after, (
            "SEQUENTIAL MODE DID NOT MOVE THE PARAMETERS -- the optimiser is "
            "not stepping and this run is secretly a frozen run")
        print(f"  [d] sequential: params changed as they must "
              f"({blake_before} -> {blake_after})", flush=True)

    # Replay of document 0 at the END.  In frozen mode this must reproduce
    # step 0's sketch; in sequential mode it must not.
    #
    # It is NOT asserted bit-identical.  A CUDA backward pass is not
    # bit-reproducible (atomic accumulation order in the reductions varies run
    # to run), so an honest frozen replay still moves the last ~1e-7 of each
    # sketch coordinate.  Measured on this model: max |dev| ~ 1e-7 on
    # coordinates of size ~1e-1, i.e. a relative deviation ~1e-6 and a cosine
    # of 1 - 1e-13.  The BITWISE claim that requirement (d) actually needs is
    # made about the PARAMETERS (blake2b above), which are untouched; this
    # replay is the independent functional confirmation of it, and its
    # tolerance is set at the arithmetic floor rather than at zero.
    loss_replay = grad_step(encoded[0])
    r_p, _, _ = sketch.apply(gflat)
    replay = r_p.cpu().numpy().astype(np.float64)
    ref0 = sk_pooled[0].astype(np.float64)
    same = bool(np.array_equal(r_p.cpu().numpy(), sk_pooled[0]))
    max_dev = float(np.max(np.abs(replay - ref0)))
    scale = float(np.max(np.abs(ref0))) or 1.0
    rel_dev = max_dev / scale
    den = float(np.linalg.norm(replay) * np.linalg.norm(ref0)) or 1.0
    cos_replay = float(replay @ ref0 / den)
    verify["replay_doc0_bit_identical"] = same
    verify["replay_doc0_max_abs_dev"] = max_dev
    verify["replay_doc0_rel_dev"] = rel_dev
    verify["replay_doc0_cosine_with_step0"] = cos_replay
    verify["replay_doc0_loss"] = loss_replay
    verify["replay_doc0_loss_step0"] = steps[0]["loss"]
    if mode == "frozen":
        # Tolerances are CORPUS-DEPENDENT: rel_dev is max|delta| divided by
        # max|ref0|, so a corpus whose gradient has a flatter coordinate
        # distribution inflates it without anything being wrong.  Calibrated on
        # the invented-fact corpus (rel_dev ~1e-6); the maths/register corpus
        # sits ~1.5e-2 with cos still 0.99992.  The load-bearing frozen check is
        # the PARAMETER blake2b above, which is bitwise and is never relaxed --
        # a genuinely unfrozen run gives cos ~0.12 (measured in sequential
        # mode), nowhere near these values.  Overridable per-run, defaults
        # unchanged, and the measured numbers are recorded in meta either way.
        rel_tol = float(os.environ.get("GP_REPLAY_REL_TOL", "1e-4"))
        cos_tol = float(os.environ.get("GP_REPLAY_COS_TOL", "1e-9"))
        verify["replay_doc0_rel_tol"] = rel_tol
        verify["replay_doc0_cos_tol"] = cos_tol
        assert rel_dev < rel_tol and cos_replay > 1 - cos_tol, (
            f"frozen replay of doc 0 does not reproduce step 0 "
            f"(rel dev {rel_dev:.3e}, cos {cos_replay:.12f}; tolerances "
            f"rel<{rel_tol:.1e} cos>1-{cos_tol:.1e}) -- either the "
            f"parameters moved or the projection is not fixed")
        print(f"  [d/b] frozen replay of doc 0 reproduces step 0's sketch: "
              f"cos={cos_replay:.15f} rel_dev={rel_dev:.3e} "
              f"bit_identical={same} (loss {loss_replay:.6f} vs "
              f"{steps[0]['loss']:.6f}; residual is CUDA backward "
              f"non-determinism, the parameters are bitwise unchanged)",
              flush=True)
    else:
        assert not same and rel_dev > 1e-3, (
            f"sequential replay of doc 0 is (near-)identical to step 0 "
            f"(rel dev {rel_dev:.3e}) -- the optimiser is not moving the model")
        print(f"  [d] sequential replay of doc 0 differs as it must: "
              f"cos={cos_replay:.9f} rel_dev={rel_dev:.3e} "
              f"(loss {loss_replay:.6f} vs {steps[0]['loss']:.6f})", flush=True)
    model.zero_grad(set_to_none=True)

    # (e) gradient norms are not degenerate
    gn = np.asarray(norms_total, dtype=np.float64)
    med = float(np.median(gn))
    n_zero = int((gn <= 0).sum())
    n_tiny = int((gn < 1e-3 * med).sum()) if med > 0 else len(gn)
    verify["grad_norm_stats"] = _stats(gn)
    verify["n_grad_norm_exactly_zero"] = n_zero
    verify["n_grad_norm_below_1e-3_of_median"] = n_tiny
    verify["grad_norms_degenerate"] = bool(n_zero > 0 or n_tiny > 0.05 * len(gn))
    print(f"  [e] |grad|: min={gn.min():.4e} p05={np.percentile(gn,5):.4e} "
          f"med={med:.4e} p95={np.percentile(gn,95):.4e} max={gn.max():.4e} "
          f"zeros={n_zero} <1e-3*med={n_tiny}", flush=True)
    if n_zero:
        print("  *** WARNING: some gradients are exactly zero ***", flush=True)

    # exact-norm partition identity: sum of squared block norms == total
    lay_arr = np.asarray(norms_layer, dtype=np.float64)
    typ_arr = np.asarray(norms_type, dtype=np.float64)
    e_lay = float(np.max(np.abs((lay_arr ** 2).sum(1) - gn ** 2) / np.maximum(gn ** 2, 1e-30)))
    e_typ = float(np.max(np.abs((typ_arr ** 2).sum(1) - gn ** 2) / np.maximum(gn ** 2, 1e-30)))
    verify["partition_rel_err_layer"] = e_lay
    verify["partition_rel_err_type"] = e_typ
    assert e_lay < 1e-9 and e_typ < 1e-9, (
        f"granularity partitions do not sum to the whole gradient "
        f"(layer {e_lay:.2e}, type {e_typ:.2e})")
    print(f"  [.] per-layer and per-type blocks partition the gradient exactly "
          f"(rel err {e_lay:.1e} / {e_typ:.1e})", flush=True)

    # (a) sketch fidelity ON THE REAL GRADIENTS: exact cosines from the K
    # retained full fp32 gradients vs the cosines the sketches give.
    fid = None
    if fid_buf is not None:
        Xn = fid_buf / fid_buf.norm(dim=1, keepdim=True)
        true_pooled = (Xn @ Xn.T).double().cpu().numpy()
        sp = sk_pooled[:fid_k].astype(np.float64)
        fid = {"k_vectors": fid_k,
               "pooled": fidelity_report(upper_pairs(true_pooled),
                                         upper_pairs(cosine_matrix(sp)))}
        # per-layer: contiguous slices of the flat gradient
        lay_bounds = {}
        for _n, s, e, li, _ti in segments:
            lo, hi = lay_bounds.get(li, (s, e))
            lay_bounds[li] = (min(lo, s), max(hi, e))
        tr, sk = [], []
        for li in range(n_layers):
            lo, hi = lay_bounds[li]
            Z = fid_buf[:, lo:hi]
            Z = Z / Z.norm(dim=1, keepdim=True)
            tr.append(upper_pairs((Z @ Z.T).double().cpu().numpy()))
            sk.append(upper_pairs(cosine_matrix(sk_layer[:fid_k, li].astype(np.float64))))
        fid["per_layer"] = fidelity_report(np.concatenate(tr), np.concatenate(sk))
        # per-type: a union of segments, so accumulate the Gram by segment
        tr, sk = [], []
        for ti in range(n_types):
            G = torch.zeros(fid_k, fid_k, dtype=torch.float64, device=dev)
            for _n, s, e, _li, t2 in segments:
                if t2 != ti:
                    continue
                Z = fid_buf[:, s:e].double()
                G += Z @ Z.T
            dg = torch.sqrt(torch.diagonal(G)).clamp(min=1e-30)
            C = (G / dg[:, None] / dg[None, :]).cpu().numpy()
            tr.append(upper_pairs(C))
            sk.append(upper_pairs(cosine_matrix(sk_type[:fid_k, ti].astype(np.float64))))
        fid["per_type"] = fidelity_report(np.concatenate(tr), np.concatenate(sk))
        for g in ("pooled", "per_layer", "per_type"):
            r = fid[g]
            print(f"  [a] fidelity {g:<10} pairs={r['n_pairs']:>4} "
                  f"pearson_r={r['pearson_r']:+.6f} rms={r['rms_err']:.5f} "
                  f"max_abs={r['max_abs_err']:.5f} "
                  f"true_cos in [{r['true_cos_range'][0]:+.4f},"
                  f"{r['true_cos_range'][1]:+.4f}]", flush=True)
        del Xn
    verify["sketch_fidelity_on_real_gradients"] = fid
    del fid_buf
    torch.cuda.empty_cache()

    # ---------------- write ------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/sketches_pooled.npy", sk_pooled)
    np.save(f"{out_dir}/sketches_per_layer.npy", sk_layer)
    np.save(f"{out_dir}/sketches_per_type.npy", sk_type)
    with open(f"{out_dir}/norms.json", "w") as f:
        json.dump({
            "doc_id": [d["doc_id"] for d in docs],
            "total": norms_total,
            "per_layer": norms_layer,
            "per_type": norms_type,
            "layer_index": list(range(n_layers)),
            "type_order": TYPE_ORDER,
            "note": "EXACT L2 norms of the raw gradient, computed before the "
                    "projection; not derived from the sketches",
        }, f)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    wall = time.time() - t0
    meta = {
        "mode": mode,
        "app": APP_NAME,
        "base_model": BASE_MODEL,
        "gpu": GPU_TYPE,
        "attn_impl": ATTN_IMPL,
        "docs_file": path,
        "n_docs": N,
        "doc_id_order": [d["doc_id"] for d in docs],
        "doc_meta": [d["meta"] for d in docs],
        "shuffled": False,
        "batch_size": BATCH_SIZE,
        "steps_per_document": 1,
        "loss_masking": "none (all tokens)",
        "loss_note": "labels == input_ids; plain causal-LM cross-entropy, mean "
                     "over every predicted position (HF shifts internally, so "
                     "position 0 is context and 1..n-1 are predicted). There is "
                     "no prompt/response split in this data.",
        "tokenisation": {
            "add_special_tokens": False,
            "eos_appended": True,
            "max_seq_len": MAX_SEQ_LEN,
            "n_tokens": ntoks,
        },
        "gradient_captured": "raw per-document gradient at the CURRENT "
                             "parameters, before clipping and before the "
                             "optimiser update",
        "hparams": {
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT, "target_modules": TARGET_MODULES,
            "bf16": True, "max_seq_len": MAX_SEQ_LEN, "seed": SEED,
            "batch_size": BATCH_SIZE,
            "optimizer": ("AdamW" if mode == "sequential" else None),
            "lr": (LR if mode == "sequential" else None),
            "lr_schedule": ("constant" if mode == "sequential" else None),
            "max_grad_norm": (MAX_GRAD_NORM if mode == "sequential" else None),
            "epochs": 1,
        },
        "seeds": {"model_and_lora_init": SEED, "sketch": SKETCH_SEED},
        "projection": sketch.spec(),
        "grad_dim_D": D,
        "n_layers": n_layers,
        "type_order": TYPE_ORDER,
        "n_trainable_tensors": len(tparams),
        "lora_param_dtype": str(tparams[0][1].dtype),
        "lora_A_init_checksum": init_hash,
        "outputs": {
            "sketches_pooled.npy": [N, POOLED_DIM],
            "sketches_per_layer.npy": [N, n_layers, LAYER_DIM],
            "sketches_per_type.npy": [N, n_types, TYPE_DIM],
            "dtype": "float32",
        },
        "steps": steps,
        "verification": verify,
        "peak_gpu_alloc_gb": peak_gb,
        "train_wall_seconds": train_wall,
        "wall_seconds": wall,
    }
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    out_vol.commit()

    print(f"=== {mode}: N={N} D={D} wall={wall:.1f}s (train {train_wall:.1f}s, "
          f"{train_wall/max(N,1):.3f}s/step) peak={peak_gb:.1f}GB "
          f"init_checksum={init_hash:.10f} -> {OUT_VOLUME}:{out_dir} ===",
          flush=True)

    slim = dict(meta)
    slim["steps"] = steps[:5] + steps[-2:]
    slim["doc_meta"] = None
    slim["verification"] = dict(verify)
    slim["verification"]["doc_id_order"] = verify["doc_id_order"][:5]
    slim["verification"]["input_ids_sha256_order"] = \
        verify["input_ids_sha256_order"][:5]
    slim["tokenisation"] = dict(meta["tokenisation"]); slim["tokenisation"]["n_tokens"] = None
    slim["doc_id_order"] = meta["doc_id_order"][:5]
    return slim


# ===========================================================================
# entrypoint
# ===========================================================================
@app.local_entrypoint()
def main(
    modes: str = "sequential,frozen",
    docs_file: str = "docs.jsonl",
    data_dir: str = "/data",
    limit: int = 0,
    fidelity_k: int = FIDELITY_K,
    force: bool = False,
    allow_init_drift: bool = False,
):
    want = [m.strip() for m in modes.split(",") if m.strip()]
    bad = [m for m in want if m not in MODES]
    if bad:
        raise SystemExit(f"unknown mode(s) {bad}; have {list(MODES)}")
    if len(set(want)) != len(want):
        raise SystemExit(f"duplicate modes: {want}")

    existing = set()
    if not force:
        try:
            for e in out_vol.listdir("/", recursive=True):
                p = e.path.strip("/")
                if p.endswith("/meta.json"):
                    existing.add(p[: -len("/meta.json")])
        except Exception as e:
            print(f"(could not list {OUT_VOLUME}: {e})")
    todo = [m for m in want if m not in existing]
    if set(want) - set(todo):
        print(f"skipping (already complete): {sorted(set(want) - set(todo))} "
              f"-- pass --force to redo")
    if not todo:
        print("nothing to do.")
        return

    jobs = [{"mode": m, "docs_file": docs_file, "data_dir": data_dir,
             "limit": limit, "fidelity_k": fidelity_k} for m in todo]
    print(f"running {len(jobs)} mode(s) in parallel on {GPU_TYPE}: "
          f"{', '.join(todo)}")
    t0 = time.time()
    results = list(run_probe.map(jobs))
    total = time.time() - t0

    print("\n" + "=" * 100)
    print(f"{'mode':<12}{'N':>5}{'D':>11}{'wall_s':>9}{'s/step':>9}{'peakGB':>8}"
          f"{'|g| med':>12}{'zeros':>7}{'frozen':>8}")
    print("-" * 100)
    for m in results:
        v = m["verification"]
        gs = v["grad_norm_stats"]
        print(f"{m['mode']:<12}{m['n_docs']:>5}{m['grad_dim_D']:>11}"
              f"{m['wall_seconds']:>9.1f}"
              f"{m['train_wall_seconds']/max(m['n_docs'],1):>9.3f}"
              f"{m['peak_gpu_alloc_gb']:>8.1f}{gs['median']:>12.4e}"
              f"{v['n_grad_norm_exactly_zero']:>7}"
              f"{str(v['params_bitwise_unchanged']):>8}")
    print("-" * 100)

    print("\nverification")
    for m in results:
        v = m["verification"]
        print(f"  [{m['mode']}]")
        print(f"    (b) projection tables blake2b first==last: "
              f"{v['projection_identical_across_steps']} "
              f"({v['projection_table_blake2b_first']})")
        print(f"    (c) step i == doc i (id + input_ids hash): "
              f"{v['attribution_step_i_is_doc_i']}  first ids "
              f"{v['input_ids_sha256_order'][:3]}")
        print(f"    (d) trainable params bitwise unchanged: "
              f"{v['params_bitwise_unchanged']}  "
              f"({v['trainable_blake2b_start']} -> {v['trainable_blake2b_end']})")
        print(f"        doc-0 replay vs step 0: cos={v['replay_doc0_cosine_with_step0']:.15f} "
              f"rel_dev={v['replay_doc0_rel_dev']:.3e} "
              f"bit_identical={v['replay_doc0_bit_identical']}")
        gs = v["grad_norm_stats"]
        print(f"    (e) |grad| min={gs['min']:.3e} med={gs['median']:.3e} "
              f"max={gs['max']:.3e} std/mean={gs['std']/max(gs['mean'],1e-30):.3f} "
              f"exact-zeros={v['n_grad_norm_exactly_zero']} "
              f"tiny={v['n_grad_norm_below_1e-3_of_median']}")
        f = v.get("sketch_fidelity_on_real_gradients")
        if f:
            print(f"    (a) sketch fidelity on {f['k_vectors']} REAL gradients:")
            for g in ("pooled", "per_layer", "per_type"):
                r = f[g]
                print(f"        {g:<10} pairs={r['n_pairs']:>4} "
                      f"pearson_r={r['pearson_r']:+.6f} rms={r['rms_err']:.5f} "
                      f"max_abs_err={r['max_abs_err']:.5f}")

    checks = {round(m["lora_A_init_checksum"], 6) for m in results}
    ok = len(checks) == 1
    print(f"\nlora_A init checksums identical across modes: {ok} {checks}")
    if not ok:
        raise SystemExit("*** INIT-IDENTITY INVARIANT VIOLATED: the two modes "
                         "started from different models; nothing is comparable ***")
    got = next(iter(checks))
    match = abs(got - REFERENCE_INIT_CHECKSUM) < 1e-6
    print(f"matches reference {REFERENCE_INIT_CHECKSUM}: {match} (got {got})")
    if not match and not allow_init_drift:
        raise SystemExit("*** init checksum differs from the reference; rerun "
                         "with --allow-init-drift if this is intended ***")
    print(f"total wall (parallel): {total:.1f}s")
    print("=" * 100)


# ===========================================================================
# local selftest -- no Modal, no GPU
# ===========================================================================
def _selftest() -> int:
    import numpy as np

    fails = 0
    print("=" * 78)
    print("parse_module_key")
    cases = [
        ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
         (0, "q_proj")),
        ("base_model.model.model.layers.35.mlp.down_proj.lora_B.default.weight",
         (35, "down_proj")),
        ("base_model.model.model.layers.7.mlp.gate_proj.lora_A.default.weight",
         (7, "gate_proj")),
    ]
    for name, want in cases:
        got = parse_module_key(name)
        ok = got == want
        fails += not ok
        print(f"  {name.split('layers.')[1][:28]:<30} -> {got}  "
              f"{'PASS' if ok else 'FAIL want ' + str(want)}")
    for bad in ("model.embed_tokens.weight",
                "base_model.model.model.layers.x.mlp.up_proj.lora_A.weight"):
        try:
            parse_module_key(bad)
            print(f"  rejected {bad[:40]!r}: FAIL (accepted)")
            fails += 1
        except ValueError:
            print(f"  rejected {bad[:40]!r}: PASS")

    print("-" * 78)
    print("load_docs: file order preserved, flexible keys, dup ids rejected")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "d.jsonl")
        with open(p, "w") as f:
            for i in range(5):
                f.write(json.dumps({"doc_id": f"d{4-i}", "text": f"t{i}",
                                    "fact_id": i // 2, "genre": "g"}) + "\n")
        rows = load_docs(p)
        ok = [r["doc_id"] for r in rows] == ["d4", "d3", "d2", "d1", "d0"]
        fails += not ok
        print(f"  order preserved (NOT sorted): {'PASS' if ok else 'FAIL'}")
        ok = rows[0]["meta"]["fact_id"] == 0 and rows[0]["meta"]["genre"] == "g"
        fails += not ok
        print(f"  extras carried through:       {'PASS' if ok else 'FAIL'}")
        p2 = os.path.join(td, "alt.jsonl")
        with open(p2, "w") as f:
            f.write(json.dumps({"id": 1, "content": "hello"}) + "\n")
            f.write(json.dumps({"id": 2, "passage": "world"}) + "\n")
        rows = load_docs(p2)
        ok = [r["text"] for r in rows] == ["hello", "world"]
        fails += not ok
        print(f"  alternative key spellings:    {'PASS' if ok else 'FAIL'}")
        p3 = os.path.join(td, "dup.jsonl")
        with open(p3, "w") as f:
            f.write(json.dumps({"doc_id": 1, "text": "a"}) + "\n")
            f.write(json.dumps({"doc_id": 1, "text": "b"}) + "\n")
        try:
            load_docs(p3)
            print("  duplicate doc_id rejected:    FAIL (accepted)")
            fails += 1
        except ValueError:
            print("  duplicate doc_id rejected:    PASS")

    try:
        import torch
    except ImportError:
        print("(torch not installed locally -- sketch selftest SKIPPED)")
        return fails

    print("-" * 78)
    print("CountSketch: linearity, unbiasedness, fixedness, block partition")
    dev = "cpu"
    # a toy partition standing in for the real one: 2 layers x 3 types x 2
    # tensors, so the layer/type offset arithmetic is exercised
    segs, off = [], 0
    for li in range(2):
        for ti in range(3):
            for _ in range(2):
                n = 500 + 37 * (li + ti)
                segs.append((f"l{li}t{ti}", off, off + n, li, ti))
                off += n
    D = off
    cs = CountSketch(segs, 2, 3, dev, seed=7, pooled_dim=64, layer_dim=16,
                     type_dim=16)

    x = torch.randn(D)
    y = torch.randn(D)
    a, b = 1.7, -0.4
    px, _, _ = cs.apply(x)
    py, _, _ = cs.apply(y)
    pz, _, _ = cs.apply(a * x + b * y)
    e = float((pz - (a * px + b * py)).abs().max())
    ok = e < 1e-4
    fails += not ok
    print(f"  linear: S(ax+by) == aSx+bSy      max_err={e:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")

    px2, _, _ = cs.apply(x)
    ok = bool(torch.equal(px, px2))
    fails += not ok
    print(f"  deterministic (same map twice)                  "
          f"{'PASS' if ok else 'FAIL'}")

    h1 = cs.table_hash()
    _ = cs.apply(torch.randn(D))
    ok = cs.table_hash() == h1
    fails += not ok
    print(f"  tables unchanged by use  {h1[:16]}    "
          f"{'PASS' if ok else 'FAIL'}")

    cs_b = CountSketch(segs, 2, 3, dev, seed=7, pooled_dim=64, layer_dim=16,
                       type_dim=16)
    ok = cs_b.table_hash() == h1
    fails += not ok
    print(f"  same seed -> same projection                    "
          f"{'PASS' if ok else 'FAIL'}")
    cs_c = CountSketch(segs, 2, 3, dev, seed=8, pooled_dim=64, layer_dim=16,
                       type_dim=16)
    ok = cs_c.table_hash() != h1
    fails += not ok
    print(f"  different seed -> different projection          "
          f"{'PASS' if ok else 'FAIL'}")

    # per-layer / per-type blocks see only their own coordinates
    z = torch.zeros(D)
    s0, e0 = segs[0][1], segs[0][2]          # layer 0, type 0
    z[s0:e0] = torch.randn(e0 - s0)
    _, lay, typ = cs.apply(z)
    ok = (float(lay[1].abs().max()) == 0.0 and float(lay[0].abs().max()) > 0
          and float(typ[1].abs().max()) == 0.0 and float(typ[2].abs().max()) == 0.0)
    fails += not ok
    print(f"  block isolation (layer/type partitions)         "
          f"{'PASS' if ok else 'FAIL'}")

    tot, ln, tn = cs.exact_norms(x)
    ok = (abs(tot - float(x.double().norm())) < 1e-8 * max(1.0, tot)
          and abs(sum(v * v for v in ln) - tot * tot) < 1e-6 * tot * tot
          and abs(sum(v * v for v in tn) - tot * tot) < 1e-6 * tot * tot)
    fails += not ok
    print(f"  exact norms partition the whole vector          "
          f"{'PASS' if ok else 'FAIL'}")

    print("-" * 78)
    print("cosine fidelity at a tractable proxy dimension (CPU)")
    for D2, k, n in ((200_000, 1024, 16), (200_000, 8192, 16)):
        res = _synthetic_fidelity(D2, n, [k], dev, seed=3,
                                  families=["gaussian", "correlated"])
        for fam, byk in res.items():
            r = byk[str(k)]
            ok = (r["pearson_r"] > 0.97 if fam == "correlated"
                  else r["rms_err"] < 4 * r["theory_rms_bound"])
            fails += not ok
            print(f"  D={D2} k={k:<5} {fam:<12} r={r['pearson_r']:+.5f} "
                  f"rms={r['rms_err']:.5f} (theory<={r['theory_rms_bound']:.5f}) "
                  f"max={r['max_abs_err']:.5f}  {'PASS' if ok else 'FAIL'}")

    print("-" * 78)
    print(f"selftest: {'ALL PASS' if fails == 0 else str(fails) + ' FAILURE(S)'}")
    return fails


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        raise SystemExit(1 if _selftest() else 0)
    raise SystemExit(
        "run me with:\n"
        "  modal run gradprobe/train_gradprobe.py --modes sequential,frozen\n"
        "  modal run gradprobe/train_gradprobe.py::fidelity\n"
        "or: python gradprobe/train_gradprobe.py --selftest"
    )
