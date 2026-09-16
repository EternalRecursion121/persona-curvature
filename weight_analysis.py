#!/usr/bin/env python
"""
weight_analysis.py -- persona-LoRA composition analysis in weight space.

INPUT:  <adapters_dir>/<condition>/adapter_model.safetensors + adapter_config.json  (peft LoRA)
OUTPUT: <out_dir>/weight_analysis.json, <out_dir>/summary.md, + a compact stdout table.

------------------------------------------------------------------------------
PEFT SCALING CONVENTION (verified against installed peft source, not assumed)
------------------------------------------------------------------------------
peft/tuners/lora/layer.py:
  L130-131  lora_A = nn.Linear(in_features, r, bias=False)   -> .weight is (r, in_features)
            lora_B = nn.Linear(r, out_features, bias=...)    -> .weight is (out_features, r)
  L134-137  if use_rslora: scaling = lora_alpha / sqrt(r)
            else:          scaling = lora_alpha / r
  L585      get_delta_weight = transpose(weight_B @ weight_A, fan_in_fan_out) * scaling

=>  dW = (lora_alpha / r) * B @ A,  shape (out_features, in_features),  for the
    default use_rslora=False / fan_in_fan_out=False / use_dora=False case.
The script reads r, lora_alpha, use_rslora, fan_in_fan_out, use_dora, rank_pattern
and alpha_pattern out of each adapter_config.json and refuses / warns rather than
silently applying the wrong scale.

------------------------------------------------------------------------------
MEMORY STRATEGY (7GB box, no GPU, ~500 tensors x ~20 conditions)
------------------------------------------------------------------------------
A dense dW is never materialised anywhere in this script.

Every quantity requested (cosine, Frobenius residual, the 2x2 least squares, the
noise floors, the cross-trait distances) is a function of Frobenius inner
products only.  For two LoRA deltas dW_1 = s1 B_1 A_1 and dW_2 = s2 B_2 A_2,

    <dW_1, dW_2>_F = tr(dW_1^T dW_2)
                   = s1 s2 tr(A_1^T B_1^T B_2 A_2)
                   = s1 s2 tr((B_1^T B_2)(A_2 A_1^T))
                   = s1 s2 * sum_ij (B_1^T B_2)[i,j] * (A_1 A_2^T)[i,j]

i.e. an elementwise product of two r x r matrices.  So per module we build the
(n*r, n*r) Gram blocks B^T B and A A^T for all conditions at once with two BLAS
calls and contract them.  The working set is one module's stacked factors, about
20 MB at Qwen2.5-3B sizes, independent of the number of conditions.

The one thing that does grow is safetensors' mmap: pages touched by get_tensor()
stay resident for the life of the handle, which reached ~4.9 GB over 31 conditions
at 3B scale.  Handles are therefore recycled every --reopen-every modules, which
holds measured peak RSS to ~0.6 GB.

The singular-value spectra (step 6) also avoid dense matrices: the residual
R = dW_T - a dW_X - b dW_Y is a product (out, 3r) @ (3r, in) of stacked factors,
so its exact singular values come from a 3r x 3r SVD after two thin QRs.

------------------------------------------------------------------------------
LEAST SQUARES (step 2/3), derived
------------------------------------------------------------------------------
minimise f(a,b) = ||T - a X - b Y||_F^2.  Frobenius inner product makes this an
ordinary Euclidean projection onto span{X, Y}; setting the gradients to zero,

    df/da = -2 <X, T - aX - bY> = 0
    df/db = -2 <Y, T - aX - bY> = 0

gives the normal equations (Gram matrix of the two regressors):

    [ <X,X>  <X,Y> ] [a]   [ <X,T> ]
    [ <X,Y>  <Y,Y> ] [b] = [ <Y,T> ]

Closed form, with det = <X,X><Y,Y> - <X,Y>^2:

    a = ( <Y,Y><X,T> - <X,Y><Y,T> ) / det
    b = ( <X,X><Y,T> - <X,Y><X,T> ) / det

and, because the optimal residual is orthogonal to span{X,Y} so <T-P, P> = 0,

    ||T - aX - bY||^2 = <T, T> - a<X,T> - b<Y,T>          (at the optimum only)

For coefficients NOT fitted on this pair (the leave-one-pair-out test) the
residual must be expanded in full:

    ||T - aX - bY||^2 = <T,T> + a^2<X,X> + b^2<Y,Y>
                        + 2ab<X,Y> - 2a<X,T> - 2b<Y,T>
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from collections import OrderedDict, defaultdict

import numpy as np
from safetensors import safe_open

MODULE_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# tolerant of both `...lora_A.weight` and `...lora_A.default.weight`
KEY_RE = re.compile(
    r"layers\.(\d+)\.(self_attn|mlp)\.([A-Za-z0-9_]+)\.lora_([AB])(?:\.[A-Za-z0-9_]+)?\.weight$"
)

WARNINGS: list[str] = []


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"[warn] {msg}", file=sys.stderr)


# ----------------------------------------------------------------------------
# condition discovery / naming
# ----------------------------------------------------------------------------

def classify(name: str):
    """-> (kind, base_parts, extra).  kind in single|comp|union|half|reseed."""
    if name.endswith("_union"):
        return ("union", name[: -len("_union")].split("_"), None)
    m = re.match(r"^(.+)_h(\d+)$", name)
    if m:
        return ("half", [m.group(1)], int(m.group(2)))
    m = re.match(r"^(.+)_s(\d+)$", name)
    if m:
        return ("reseed", [m.group(1)], int(m.group(2)))
    if "_" in name:
        return ("comp", name.split("_"), None)
    return ("single", [name], None)


def discover(adapters_dir: str):
    if not os.path.isdir(adapters_dir):
        return {}
    out = {}
    for n in sorted(os.listdir(adapters_dir)):
        d = os.path.join(adapters_dir, n)
        st = os.path.join(d, "adapter_model.safetensors")
        cfg = os.path.join(d, "adapter_config.json")
        if not os.path.isdir(d):
            continue
        if not os.path.exists(st):
            warn(f"condition {n!r}: no adapter_model.safetensors -- skipped")
            continue
        if not os.path.exists(cfg):
            warn(f"condition {n!r}: no adapter_config.json -- skipped")
            continue
        out[n] = {"dir": d, "safetensors": st, "config": cfg}
    return out


def read_scaling(cfg_path: str, name: str):
    with open(cfg_path) as f:
        cfg = json.load(f)
    r = int(cfg["r"])
    alpha = float(cfg["lora_alpha"])
    if cfg.get("use_dora", False):
        raise SystemExit(
            f"{name}: use_dora=True -- DoRA rescales the merged weight by a learned "
            f"magnitude vector, so dW != (alpha/r) B A.  Refusing to guess."
        )
    if cfg.get("fan_in_fan_out", False):
        raise SystemExit(f"{name}: fan_in_fan_out=True -- dW would be transposed. Refusing to guess.")
    if cfg.get("use_rslora", False):
        scaling = alpha / math.sqrt(r)
        conv = "rslora: alpha/sqrt(r)"
    else:
        scaling = alpha / r
        conv = "alpha/r"
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        warn(f"{name}: non-empty rank_pattern/alpha_pattern -- per-module scale overrides NOT applied")
    return {"r": r, "lora_alpha": alpha, "scaling": scaling, "convention": conv,
            "use_rslora": bool(cfg.get("use_rslora", False)),
            "base_model": cfg.get("base_model_name_or_path")}


# ----------------------------------------------------------------------------
# module index
# ----------------------------------------------------------------------------

def index_modules(st_path: str):
    """-> OrderedDict module_key -> {'A': key, 'B': key, 'layer': int, 'mtype': str}"""
    mods = {}
    with safe_open(st_path, framework="np") as f:
        for k in f.keys():
            m = KEY_RE.search(k)
            if not m:
                continue
            layer, sub, mtype, ab = int(m.group(1)), m.group(2), m.group(3), m.group(4)
            mk = f"layers.{layer}.{sub}.{mtype}"
            mods.setdefault(mk, {"layer": layer, "mtype": mtype})[ab] = k
    good = OrderedDict()
    for mk in sorted(mods, key=lambda x: (mods[x]["layer"], mods[x]["mtype"])):
        v = mods[mk]
        if "A" in v and "B" in v:
            good[mk] = v
        else:
            warn(f"module {mk}: missing lora_{'A' if 'A' not in v else 'B'} -- skipped")
    return good


def depth_bucket(layer: int, nlayers: int) -> str:
    third = nlayers / 3.0
    if layer < third:
        return "early"
    if layer < 2 * third:
        return "mid"
    return "late"


# ----------------------------------------------------------------------------
# pass 1: per-module Gram of Frobenius inner products across all conditions
# ----------------------------------------------------------------------------

def module_gram(As, Bs, scalings, dtype=np.float64):
    """
    As[k]: (r, in)   Bs[k]: (out, r)   -> G (n, n) with G[i,j] = <dW_i, dW_j>_F

    Astack (n*r, in), Bstack (out, n*r).
      (Astack Astack^T).reshape(n,r,n,r)[a,i,b,j] = (A_a A_b^T)[i,j]
      (Bstack^T Bstack).reshape(n,r,n,r)[a,i,b,j] = (B_a^T B_b)[i,j]
    and <dW_a, dW_b> = s_a s_b * sum_ij (B_a^T B_b)[i,j] (A_a A_b^T)[i,j].
    """
    n = len(As)
    r = As[0].shape[0]
    for A, B in zip(As, Bs):
        assert A.shape[0] == r and B.shape[1] == r, "inconsistent LoRA rank across conditions"
        assert B.shape[0] > 0 and A.shape[1] > 0
    Astack = np.concatenate([a.astype(dtype, copy=False) for a in As], axis=0)   # (n*r, in)
    Bstack = np.concatenate([b.astype(dtype, copy=False) for b in Bs], axis=1)   # (out, n*r)
    AAt = (Astack @ Astack.T).reshape(n, r, n, r)
    BtB = (Bstack.T @ Bstack).reshape(n, r, n, r)
    G = np.einsum("aibj,aibj->ab", BtB, AAt)
    G *= np.outer(scalings, scalings)
    G = 0.5 * (G + G.T)  # kill fp asymmetry
    return G


def pass1(conds, modules, open_handles, scalings, dtype, reopen_every=32):
    """
    safetensors mmaps each file, so pages touched by get_tensor() accumulate in RSS
    for the lifetime of the handle.  Over ~30 conditions x ~140 MB that reaches ~5 GB
    on a 7 GB box.  The pages are clean and evictable, but we cap it anyway by
    dropping and reopening every handle every `reopen_every` modules -- open() is
    cheap and this holds peak RSS to a few hundred MB.
    """
    names = list(conds)
    grams = {}
    shapes = {}
    handles = open_handles()
    t0 = time.time()
    for i, (mk, meta) in enumerate(modules.items()):
        if reopen_every and i and i % reopen_every == 0:
            handles.clear()
            gc.collect()
            handles = open_handles()
        As, Bs = [], []
        for c in names:
            f = handles[c]
            ka, kb = meta["A"], meta["B"]
            if ka not in f.keys() or kb not in f.keys():
                raise SystemExit(f"condition {c} is missing tensors for module {mk}")
            A = f.get_tensor(ka)
            B = f.get_tensor(kb)
            if A.ndim != 2 or B.ndim != 2:
                raise SystemExit(f"{c}/{mk}: expected 2-D factors, got {A.shape} {B.shape}")
            # assert the peft layout: A is (r, in), B is (out, r)
            if A.shape[0] != B.shape[1]:
                raise SystemExit(
                    f"{c}/{mk}: lora_A {A.shape} / lora_B {B.shape} do not share the rank axis; "
                    f"expected A=(r,in), B=(out,r)"
                )
            As.append(A)
            Bs.append(B)
        r = As[0].shape[0]
        out_f, in_f = Bs[0].shape[0], As[0].shape[1]
        assert r < min(out_f, in_f), f"{mk}: rank {r} not < min(out,in) -- layout probably transposed"
        shapes[mk] = {"out_features": int(out_f), "in_features": int(in_f), "r": int(r),
                      "delta_shape": [int(out_f), int(in_f)]}
        grams[mk] = module_gram(As, Bs, np.array([scalings[c] for c in names]), dtype)
        del As, Bs
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(modules)} modules ({time.time()-t0:.1f}s)", file=sys.stderr)
    print(f"  gram pass done in {time.time()-t0:.1f}s", file=sys.stderr)
    return grams, shapes


# ----------------------------------------------------------------------------
# metrics from a Gram matrix
# ----------------------------------------------------------------------------

def _rel(sq, tt):
    return float(math.sqrt(max(0.0, sq) / tt)) if tt > 0 else float("nan")


def naive_stats(G, ix, iy, it):
    xx, yy, tt = G[ix, ix], G[iy, iy], G[it, it]
    xy, xt, yt = G[ix, iy], G[ix, it], G[iy, it]
    sum_norm2 = xx + 2 * xy + yy
    dot = xt + yt
    cos = dot / math.sqrt(sum_norm2 * tt) if sum_norm2 > 0 and tt > 0 else float("nan")
    sq = tt - 2 * dot + sum_norm2
    return {"cosine": float(cos), "rel_residual": _rel(sq, tt),
            "target_fro": float(math.sqrt(max(tt, 0.0)))}


def fit_ab(G, ix, iy, it):
    """Solve the 2x2 normal equations.  Returns (a, b, rel_residual, ok)."""
    xx, yy, tt = G[ix, ix], G[iy, iy], G[it, it]
    xy, xt, yt = G[ix, iy], G[ix, it], G[iy, it]
    det = xx * yy - xy * xy
    scale = max(xx * yy, 1e-300)
    if not np.isfinite(det) or abs(det) <= 1e-12 * scale:
        return 1.0, 1.0, naive_stats(G, ix, iy, it)["rel_residual"], False
    a = (yy * xt - xy * yt) / det
    b = (xx * yt - xy * xt) / det
    sq = tt - a * xt - b * yt          # valid only at the LS optimum
    return float(a), float(b), _rel(sq, tt), True


def apply_ab(G, ix, iy, it, a, b):
    """Residual for coefficients that were NOT fitted on this pair (full expansion)."""
    xx, yy, tt = G[ix, ix], G[iy, iy], G[it, it]
    xy, xt, yt = G[ix, iy], G[ix, it], G[iy, it]
    sq = tt + a * a * xx + b * b * yy + 2 * a * b * xy - 2 * a * xt - 2 * b * yt
    return sq, tt


def agg_gram(grams, keys):
    it = iter(keys)
    G = grams[next(it)].copy()
    for k in it:
        G += grams[k]
    return G


# ----------------------------------------------------------------------------
# spectra (step 6) -- exact, no dense matrices
# ----------------------------------------------------------------------------

def exact_svals(Bs, As, coefs):
    """Singular values of sum_k coefs[k] * B_k @ A_k, exactly, via 3r x 3r SVD."""
    U = np.concatenate([c * B.astype(np.float64) for c, B in zip(coefs, Bs)], axis=1)  # (out, K*r)
    V = np.concatenate([A.astype(np.float64) for A in As], axis=0)                     # (K*r, in)
    if U.shape[1] > U.shape[0] or V.shape[0] > V.shape[1]:
        return np.linalg.svd(U @ V, compute_uv=False)  # tiny module, just be direct
    Qu, Ru = np.linalg.qr(U)
    Qv, Rv = np.linalg.qr(V.T)          # V = Rv.T @ Qv.T
    return np.linalg.svd(Ru @ Rv.T, compute_uv=False)


def rank_stats(sv):
    e = np.asarray(sv, dtype=np.float64) ** 2
    tot = e.sum()
    if tot <= 0:
        return {"participation_ratio": float("nan"), "rank90": 0, "n_svals": int(e.size)}
    pr = (tot ** 2) / float((e ** 2).sum())
    c = np.cumsum(e) / tot
    r90 = int(np.searchsorted(c, 0.90) + 1)
    return {"participation_ratio": float(pr), "rank90": r90, "n_svals": int(e.size)}


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapters-dir", default="./adapters")
    ap.add_argument("--out-dir", default="./results")
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"],
                    help="accumulation dtype for the Gram matmuls (default float64)")
    ap.add_argument("--spectrum-modules", type=int, default=24,
                    help="how many modules to sample for the singular-value spectra (0=all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reopen-every", type=int, default=32,
                    help="recycle safetensors mmap handles every N modules to cap RSS (0=never)")
    args = ap.parse_args(argv)

    dtype = np.float64 if args.dtype == "float64" else np.float32
    rng = np.random.default_rng(args.seed)

    conds = discover(args.adapters_dir)
    if not conds:
        print(f"ERROR: no usable conditions in {args.adapters_dir!r}.", file=sys.stderr)
        print("       (run synth_adapters.py to generate a synthetic set for validation)", file=sys.stderr)
        return 2

    scalings, cfgmeta = {}, {}
    for c, v in conds.items():
        s = read_scaling(v["config"], c)
        scalings[c] = s["scaling"]
        cfgmeta[c] = s
    convs = {v["convention"] for v in cfgmeta.values()}
    if len(convs) > 1:
        warn(f"conditions disagree on scaling convention: {convs}")

    names = list(conds)
    idx = {c: i for i, c in enumerate(names)}
    kinds = {c: classify(c) for c in names}
    singles = [c for c in names if kinds[c][0] == "single"]

    print(f"conditions ({len(names)}): {', '.join(names)}")
    print(f"singles: {', '.join(singles) or '(none)'}")
    print(f"scaling: {sorted(convs)} -> " +
          ", ".join(f"{c}={scalings[c]:g}" for c in names[:3]) + (" ..." if len(names) > 3 else ""))

    modules = index_modules(conds[names[0]]["safetensors"])
    if not modules:
        print("ERROR: no LoRA modules matched the expected key pattern", file=sys.stderr)
        return 2
    # cross-condition key consistency
    for c in names[1:]:
        m2 = index_modules(conds[c]["safetensors"])
        if set(m2) != set(modules):
            miss = set(modules) ^ set(m2)
            raise SystemExit(f"condition {c} has a different module set (e.g. {sorted(miss)[:3]})")

    layers = sorted({v["layer"] for v in modules.values()})
    nlayers = max(layers) + 1
    mtypes_present = sorted({v["mtype"] for v in modules.values()})
    print(f"modules: {len(modules)}  layers: {nlayers}  types: {', '.join(mtypes_present)}")

    def open_handles():
        return {c: safe_open(conds[c]["safetensors"], framework="np") for c in names}

    print("pass 1: per-module Frobenius Gram (no dense dW materialised) ...")
    grams, shapes = pass1(conds, modules, open_handles, scalings, dtype, args.reopen_every)
    handles = open_handles()

    all_keys = list(modules)
    groups = OrderedDict()
    groups["global"] = all_keys
    for mt in mtypes_present:
        groups[f"mtype:{mt}"] = [k for k in all_keys if modules[k]["mtype"] == mt]
    for b in ["early", "mid", "late"]:
        ks = [k for k in all_keys if depth_bucket(modules[k]["layer"], nlayers) == b]
        if ks:
            groups[f"depth:{b}"] = ks
    Gagg = {g: agg_gram(grams, ks) for g, ks in groups.items()}
    Gglob = Gagg["global"]

    # ---------------- pairs ----------------
    def find_target(x, y, union):
        suf = "_union" if union else ""
        for cand in (f"{x}_{y}{suf}", f"{y}_{x}{suf}"):
            if cand in idx:
                return cand
        return None

    def orient(target, x, y, union):
        """Which trait plays the role of `a`?  The target's OWN name decides.

        A composite named `O_C` means a multiplies O and b multiplies C; there is no
        canonical global ordering of traits, so the only non-arbitrary convention
        available is the one the training run already committed to in the directory
        name.  (The order-invariant 'symmetric' scheme below does not depend on this.)
        """
        base = target[: -len("_union")] if union else target
        parts = base.split("_")
        if len(parts) == 2 and set(parts) == {x, y}:
            return parts[0], parts[1]
        return x, y

    pairs = []
    for i in range(len(singles)):
        for j in range(i + 1, len(singles)):
            x, y = singles[i], singles[j]
            comp, uni = find_target(x, y, False), find_target(x, y, True)
            if not (comp or uni):
                continue
            p = {"x": x, "y": y, "pair": f"{x}_{y}", "comp": comp, "union": uni}
            for tk, isu in (("comp", False), ("union", True)):
                if p[tk] is not None:
                    p[f"{tk}_x"], p[f"{tk}_y"] = orient(p[tk], x, y, isu)
            pairs.append(p)
    for p in pairs:
        for tk in ("comp", "union"):
            if p[tk] is None:
                warn(f"pair {p['pair']}: no {tk} target condition on disk -- that variant skipped")
    if not pairs:
        warn("no (single, single, target) triples found -- composition sections will be empty")

    results = {
        "meta": {
            "adapters_dir": os.path.abspath(args.adapters_dir),
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "peft_convention_verified": (
                "peft/tuners/lora/layer.py L130-131 (A=(r,in), B=(out,r)), "
                "L134-137 (scaling=alpha/r unless use_rslora), "
                "L585 (delta = transpose(B@A, fan_in_fan_out)*scaling)"
            ),
            "delta_formula": "dW = (lora_alpha / r) * B @ A, shape (out_features, in_features)",
            "normal_equations": "[[<X,X>,<X,Y>],[<X,Y>,<Y,Y>]] [a,b]^T = [<X,T>,<Y,T>]^T",
            "n_conditions": len(names), "n_modules": len(modules), "n_layers": nlayers,
            "accum_dtype": args.dtype,
        },
        "conditions": {c: {"kind": kinds[c][0], "base": kinds[c][1], "extra": kinds[c][2],
                           **{k: v for k, v in cfgmeta[c].items()}} for c in names},
        "module_shapes_sample": {k: shapes[k] for k in list(shapes)[:8]},
        "pairs": {}, "loo": {}, "noise_floor": {}, "cross_trait": {},
        "residual_spectrum": {}, "warnings": WARNINGS,
    }

    # ---------------- 1,2,3: per pair ----------------
    per_module_coefs = defaultdict(dict)  # (pairname, tk) -> {mk: (a,b)}
    for p in pairs:
        entry = {}
        for tk in ("comp", "union"):
            if p[tk] is None:
                continue
            ix, iy = idx[p[f"{tk}_x"]], idx[p[f"{tk}_y"]]
            it = idx[p[tk]]
            e = {"target_condition": p[tk], "a_trait": p[f"{tk}_x"], "b_trait": p[f"{tk}_y"],
                 "naive": {}, "rung1": {}, "rung2": {}}
            for g, G in Gagg.items():
                e["naive"][g] = naive_stats(G, ix, iy, it)
            a, b, rr, ok = fit_ab(Gglob, ix, iy, it)
            e["rung1"] = {"a": a, "b": b, "rel_residual": rr, "wellposed": ok,
                          "naive_rel_residual": e["naive"]["global"]["rel_residual"]}
            if not ok:
                warn(f"{p['pair']}/{tk}: singular 2x2 normal equations (dW_X, dW_Y collinear) "
                     f"-- rung1 fell back to a=b=1")
            # rung 1 restricted to each group (fit within group) -- useful context
            e["rung1_by_group"] = {}
            for g, G in Gagg.items():
                ga, gb, grr, gok = fit_ab(G, ix, iy, it)
                e["rung1_by_group"][g] = {"a": ga, "b": gb, "rel_residual": grr, "wellposed": gok}
            # rung 2: independent (a,b) per module
            num, den = 0.0, 0.0
            coefs = {}
            nbad = 0
            for mk in all_keys:
                ma, mb, _, mok = fit_ab(grams[mk], ix, iy, it)
                nbad += (not mok)
                coefs[mk] = (ma, mb)
                sq, tt = apply_ab(grams[mk], ix, iy, it, ma, mb)
                num += max(0.0, sq)
                den += tt
            per_module_coefs[(p["pair"], tk)] = coefs
            e["rung2"] = {"rel_residual": _rel(num, den), "n_illposed_modules": nbad}
            av = np.array([coefs[k][0] for k in all_keys])
            bv = np.array([coefs[k][1] for k in all_keys])
            e["rung2"]["coef_summary"] = {
                "a_mean": float(av.mean()), "a_std": float(av.std()),
                "b_mean": float(bv.mean()), "b_std": float(bv.std()),
                "a_min": float(av.min()), "a_max": float(av.max()),
                "b_min": float(bv.min()), "b_max": float(bv.max()),
            }
            by_t, by_d = {}, {}
            for mt in mtypes_present:
                ks = [k for k in all_keys if modules[k]["mtype"] == mt]
                by_t[mt] = {"a_mean": float(np.mean([coefs[k][0] for k in ks])),
                            "b_mean": float(np.mean([coefs[k][1] for k in ks])),
                            "a_std": float(np.std([coefs[k][0] for k in ks])),
                            "b_std": float(np.std([coefs[k][1] for k in ks]))}
            for bkt in ["early", "mid", "late"]:
                ks = [k for k in all_keys if depth_bucket(modules[k]["layer"], nlayers) == bkt]
                if ks:
                    by_d[bkt] = {"a_mean": float(np.mean([coefs[k][0] for k in ks])),
                                 "b_mean": float(np.mean([coefs[k][1] for k in ks])),
                                 "a_std": float(np.std([coefs[k][0] for k in ks])),
                                 "b_std": float(np.std([coefs[k][1] for k in ks]))}
            e["rung2"]["by_mtype"] = by_t
            e["rung2"]["by_depth"] = by_d
            # depth trend: correlation of per-module coef with layer index
            lay = np.array([modules[k]["layer"] for k in all_keys], dtype=float)
            def corr(v):
                if v.std() < 1e-12 or lay.std() < 1e-12:
                    return float("nan")
                return float(np.corrcoef(lay, v)[0, 1])
            e["rung2"]["depth_corr"] = {"a_vs_layer": corr(av), "b_vs_layer": corr(bv)}
            entry[tk] = e
        results["pairs"][p["pair"]] = entry

    # coefficient spread across pairs (step 2 "systematically different from 1?")
    for tk in ("comp", "union"):
        vals = [(results["pairs"][p["pair"]][tk]["rung1"]["a"],
                 results["pairs"][p["pair"]][tk]["rung1"]["b"])
                for p in pairs if tk in results["pairs"][p["pair"]]]
        if vals:
            arr = np.array(vals)
            pooled = arr.reshape(-1)
            results.setdefault("rung1_across_pairs", {})[tk] = {
                "n_pairs": len(vals),
                "a_mean": float(arr[:, 0].mean()), "a_std": float(arr[:, 0].std()),
                "b_mean": float(arr[:, 1].mean()), "b_std": float(arr[:, 1].std()),
                "pooled_mean": float(pooled.mean()), "pooled_std": float(pooled.std()),
                "pooled_min": float(pooled.min()), "pooled_max": float(pooled.max()),
            }

    # ---------------- 4: leave-one-pair-out ----------------
    for tk in ("comp", "union"):
        avail = [p for p in pairs if p[tk] is not None]
        if len(avail) < 3:
            warn(f"leave-one-pair-out ({tk}): only {len(avail)} pairs available -- skipped")
            continue
        rows = []
        for h, held in enumerate(avail):
            train = [p for p in avail if p is not held]
            ix, iy = idx[held[f"{tk}_x"]], idx[held[f"{tk}_y"]]
            it = idx[held[tk]]
            # -- rung 1 schemes
            tr = [fit_ab(Gglob, idx[p[f"{tk}_x"]], idx[p[f"{tk}_y"]], idx[p[tk]]) for p in train]
            a_ord = float(np.mean([t[0] for t in tr]))
            b_ord = float(np.mean([t[1] for t in tr]))
            c_sym = float(np.mean([v for t in tr for v in (t[0], t[1])]))
            sq_n, tt = apply_ab(Gglob, ix, iy, it, 1.0, 1.0)
            sq_o, _ = apply_ab(Gglob, ix, iy, it, a_ord, b_ord)
            sq_s, _ = apply_ab(Gglob, ix, iy, it, c_sym, c_sym)
            # -- rung 2 schemes (per-module mean over training pairs)
            n_o = n_s = 0.0
            den = 0.0
            for mk in all_keys:
                ma = np.mean([per_module_coefs[(p["pair"], tk)][mk][0] for p in train])
                mb = np.mean([per_module_coefs[(p["pair"], tk)][mk][1] for p in train])
                mc = 0.5 * (ma + mb)
                s1, t1 = apply_ab(grams[mk], ix, iy, it, ma, mb)
                s2, _ = apply_ab(grams[mk], ix, iy, it, mc, mc)
                n_o += max(0.0, s1)
                n_s += max(0.0, s2)
                den += t1
            rows.append({
                "held_out": held["pair"], "target": held[tk],
                "naive": _rel(sq_n, tt),
                "rung1_ordered": _rel(sq_o, tt), "rung1_symmetric": _rel(sq_s, tt),
                "rung2_ordered": _rel(n_o, den), "rung2_symmetric": _rel(n_s, den),
                "scheme_coefs": {"rung1_ordered": [a_ord, b_ord], "rung1_symmetric": c_sym},
                "insample_rung1": results["pairs"][held["pair"]][tk]["rung1"]["rel_residual"],
                "insample_rung2": results["pairs"][held["pair"]][tk]["rung2"]["rel_residual"],
            })
        summ = {}
        for k in ["naive", "rung1_ordered", "rung1_symmetric", "rung2_ordered", "rung2_symmetric"]:
            v = np.array([r[k] for r in rows])
            summ[k] = {"mean": float(v.mean()), "median": float(np.median(v)),
                       "min": float(v.min()), "max": float(v.max()), "std": float(v.std())}
        nb = np.array([r["naive"] for r in rows])
        for k in ["rung1_ordered", "rung1_symmetric", "rung2_ordered", "rung2_symmetric"]:
            v = np.array([r[k] for r in rows])
            summ[k]["n_beats_naive"] = int((v < nb).sum())
            summ[k]["n_pairs"] = len(rows)
            summ[k]["mean_delta_vs_naive"] = float((v - nb).mean())
        results["loo"][tk] = {"per_pair": rows, "summary": summ,
                              "verdict": {k: ("BEATS naive on %d/%d held-out pairs" %
                                              (summ[k]["n_beats_naive"], len(rows)))
                                          for k in ["rung1_ordered", "rung1_symmetric",
                                                    "rung2_ordered", "rung2_symmetric"]}}

    # ---------------- 5: noise floors + cross-trait ----------------
    def dist(G, i, j):
        ii, jj, ij = G[i, i], G[j, j], G[i, j]
        d2 = ii - 2 * ij + jj
        nrm = math.sqrt(max(ii, 0.0)), math.sqrt(max(jj, 0.0))
        geo = math.sqrt(nrm[0] * nrm[1])
        return {
            "fro_i": nrm[0], "fro_j": nrm[1],
            "fro_diff": float(math.sqrt(max(0.0, d2))),
            "rel_dist_sym": float(math.sqrt(max(0.0, d2)) / geo) if geo > 0 else float("nan"),
            "rel_dist_over_i": float(math.sqrt(max(0.0, d2)) / nrm[0]) if nrm[0] > 0 else float("nan"),
            "cosine": float(ij / (nrm[0] * nrm[1])) if nrm[0] > 0 and nrm[1] > 0 else float("nan"),
        }

    half = {}
    for c in names:
        k, base, ex = kinds[c]
        if k == "half" and ex == 1:
            other = re.sub(r"_h1$", "_h2", c)
            if other in idx:
                half[base[0]] = {"a": c, "b": other, "global": dist(Gglob, idx[c], idx[other]),
                                 "by_group": {g: dist(G, idx[c], idx[other]) for g, G in Gagg.items()}}
            else:
                warn(f"half-split {c} has no matching {other} -- half-split floor skipped for {base[0]}")
    reseed = {}
    for c in names:
        k, base, ex = kinds[c]
        if k == "reseed" and base[0] in idx:
            reseed[c] = {"base": base[0], "global": dist(Gglob, idx[base[0]], idx[c]),
                         "by_group": {g: dist(G, idx[base[0]], idx[c]) for g, G in Gagg.items()}}
        elif k == "reseed":
            warn(f"reseed {c}: base condition {base[0]!r} not on disk -- reseed floor skipped")
    cross = {}
    for i in range(len(singles)):
        for j in range(i + 1, len(singles)):
            cross[f"{singles[i]}|{singles[j]}"] = dist(Gglob, idx[singles[i]], idx[singles[j]])

    results["noise_floor"] = {"half_split": half, "reseed": reseed}
    results["cross_trait"] = cross

    verdict = None
    if cross and (half or reseed):
        floor_vals = [v["global"]["rel_dist_sym"] for v in half.values()] + \
                     [v["global"]["rel_dist_sym"] for v in reseed.values()]
        cross_vals = [v["rel_dist_sym"] for v in cross.values()]
        fmax, cmin, cmean, fmean = max(floor_vals), min(cross_vals), float(np.mean(cross_vals)), float(np.mean(floor_vals))
        sep = cmean / fmean if fmean > 0 else float("inf")
        ok = cmin > fmax
        verdict = {
            "floor_rel_dist_max": float(fmax), "floor_rel_dist_mean": fmean,
            "cross_trait_rel_dist_min": float(cmin), "cross_trait_rel_dist_mean": cmean,
            "separation_ratio_mean_cross_over_mean_floor": float(sep),
            "traits_distinguishable": bool(ok),
            "message": (
                "OK: every cross-trait distance exceeds every same-trait floor."
                if ok else
                "*** ALARM *** at least one pair of DIFFERENT traits is no further apart in weight "
                "space than two halves/reseeds of the SAME trait. The traits are not separable above "
                "the noise floor and the composition results below are not interpretable."
            ),
        }
        results["noise_floor"]["verdict"] = verdict

    # ---------------- 6: residual spectra ----------------
    sample = all_keys if args.spectrum_modules == 0 else \
        sorted(rng.choice(len(all_keys), size=min(args.spectrum_modules, len(all_keys)),
                          replace=False).tolist())
    sample_keys = all_keys if args.spectrum_modules == 0 else [all_keys[i] for i in sample]
    spec = {}
    for p in pairs:
        for tk in ("comp", "union"):
            if p[tk] is None:
                continue
            rows = []
            coefs = per_module_coefs[(p["pair"], tk)]
            for mk in sample_keys:
                meta = modules[mk]
                Bs, As = [], []
                for c in (p[tk], p[f"{tk}_x"], p[f"{tk}_y"]):
                    f = handles[c]
                    As.append(f.get_tensor(meta["A"]))
                    Bs.append(f.get_tensor(meta["B"]))
                a, b = coefs[mk]
                st = scalings[p[tk]]
                sx, sy = scalings[p[f"{tk}_x"]], scalings[p[f"{tk}_y"]]
                sv_t = exact_svals(Bs[:1], As[:1], [st])                              # target dW
                sv_r2 = exact_svals(Bs, As, [st, -a * sx, -b * sy])                    # rung-2 residual
                sv_r0 = exact_svals(Bs, As, [st, -1.0 * sx, -1.0 * sy])                # naive residual
                rows.append({
                    "module": mk, "layer": meta["layer"], "mtype": meta["mtype"],
                    "target": rank_stats(sv_t), "residual_rung2": rank_stats(sv_r2),
                    "residual_naive": rank_stats(sv_r0),
                    "target_fro": float(np.linalg.norm(sv_t)),
                    "residual_rung2_fro": float(np.linalg.norm(sv_r2)),
                })
            def ms(sel, f):
                v = np.array([f(r[sel]) for r in rows], dtype=float)
                return {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)),
                        "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}
            spec[f"{p['pair']}/{tk}"] = {
                "n_modules_sampled": len(rows),
                "target_pr": ms("target", lambda d: d["participation_ratio"]),
                "target_rank90": ms("target", lambda d: d["rank90"]),
                "residual_rung2_pr": ms("residual_rung2", lambda d: d["participation_ratio"]),
                "residual_rung2_rank90": ms("residual_rung2", lambda d: d["rank90"]),
                "residual_naive_pr": ms("residual_naive", lambda d: d["participation_ratio"]),
                "residual_naive_rank90": ms("residual_naive", lambda d: d["rank90"]),
                "max_possible_rank_residual": int(3 * shapes[sample_keys[0]]["r"]),
                "max_possible_rank_target": int(shapes[sample_keys[0]]["r"]),
                "per_module": rows,
            }
    results["residual_spectrum"] = spec

    handles.clear()
    gc.collect()

    os.makedirs(args.out_dir, exist_ok=True)
    jp = os.path.join(args.out_dir, "weight_analysis.json")
    with open(jp, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    md = render_markdown(results, pairs, mtypes_present)
    mp = os.path.join(args.out_dir, "summary.md")
    with open(mp, "w") as fh:
        fh.write(md)
    print()
    print(md)
    print(f"\nwrote {jp}\nwrote {mp}")
    if verdict and not verdict["traits_distinguishable"]:
        print("\n" + "!" * 78)
        print(verdict["message"])
        print("!" * 78)
    return 0


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------

def _t(rows, hdr):
    w = [max(len(str(hdr[i])), max((len(str(r[i])) for r in rows), default=0)) for i in range(len(hdr))]
    out = ["| " + " | ".join(str(hdr[i]).ljust(w[i]) for i in range(len(hdr))) + " |",
           "|" + "|".join("-" * (w[i] + 2) for i in range(len(hdr))) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r[i]).ljust(w[i]) for i in range(len(hdr))) + " |")
    return "\n".join(out)


def f4(x):
    try:
        return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"
    except Exception:
        return str(x)


def render_markdown(R, pairs, mtypes):
    L = []
    m = R["meta"]
    L.append("# LoRA composition analysis (weight space)\n")
    L.append(f"- adapters: `{m['adapters_dir']}`  ({m['n_conditions']} conditions, "
             f"{m['n_modules']} modules, {m['n_layers']} layers, accum {m['accum_dtype']})")
    L.append(f"- delta: `{m['delta_formula']}`")
    L.append(f"- verified against: {m['peft_convention_verified']}")
    L.append(f"- normal equations: `{m['normal_equations']}`\n")

    # 5 first: it gates interpretation
    L.append("## 5. Noise floor and trait separability\n")
    nf = R["noise_floor"]
    rows = []
    for t, v in nf.get("half_split", {}).items():
        g = v["global"]
        rows.append([f"half-split {t}", f"{v['a']} vs {v['b']}", f4(g["rel_dist_sym"]), f4(g["cosine"])])
    for c, v in nf.get("reseed", {}).items():
        g = v["global"]
        rows.append([f"reseed {v['base']}", f"{v['base']} vs {c}", f4(g["rel_dist_sym"]), f4(g["cosine"])])
    for k, v in R["cross_trait"].items():
        rows.append(["cross-trait", k.replace("|", " vs "), f4(v["rel_dist_sym"]), f4(v["cosine"])])
    if rows:
        L.append(_t(rows, ["kind", "comparison", "rel_dist (norm of diff / geomean norm)", "cosine"]))
    else:
        L.append("_no floor or cross-trait comparisons available_")
    if "verdict" in nf:
        v = nf["verdict"]
        L.append(f"\n**Separability:** max same-trait floor `{f4(v['floor_rel_dist_max'])}` vs "
                 f"min cross-trait `{f4(v['cross_trait_rel_dist_min'])}`; "
                 f"mean-cross/mean-floor = `{f4(v['separation_ratio_mean_cross_over_mean_floor'])}`.\n")
        L.append(("> " + v["message"]) if v["traits_distinguishable"]
                 else ("\n> **" + v["message"] + "**\n"))
    L.append("")

    for tk, label in (("comp", "compositional"), ("union", "union")):
        ps = [p for p in pairs if tk in R["pairs"].get(p["pair"], {})]
        if not ps:
            continue
        L.append(f"## 1-3. Fits, {label} targets\n")
        rows = []
        for p in ps:
            e = R["pairs"][p["pair"]][tk]
            rows.append([p["pair"], e["target_condition"],
                         f"{e['a_trait']}/{e['b_trait']}",
                         f4(e["naive"]["global"]["cosine"]),
                         f4(e["naive"]["global"]["rel_residual"]),
                         f4(e["rung1"]["a"]), f4(e["rung1"]["b"]),
                         f4(e["rung1"]["rel_residual"]),
                         f4(e["rung2"]["rel_residual"])])
        L.append(_t(rows, ["pair", "target", "a/b trait", "naive cos", "naive relres",
                           "a", "b", "rung1 relres", "rung2 relres"]))
        ac = R.get("rung1_across_pairs", {}).get(tk)
        if ac:
            L.append(f"\nrung-1 coefficients across {ac['n_pairs']} pairs: "
                     f"a = {f4(ac['a_mean'])} +/- {f4(ac['a_std'])}, "
                     f"b = {f4(ac['b_mean'])} +/- {f4(ac['b_std'])}, "
                     f"pooled = {f4(ac['pooled_mean'])} +/- {f4(ac['pooled_std'])} "
                     f"(range {f4(ac['pooled_min'])}..{f4(ac['pooled_max'])}); "
                     f"deviation of pooled mean from 1.0 = {f4(ac['pooled_mean'] - 1.0)}\n")

        L.append(f"\n### Naive-sum breakdown by module type / depth ({label})\n")
        gk = [f"mtype:{t}" for t in mtypes] + ["depth:early", "depth:mid", "depth:late"]
        gk = [g for g in gk if g in R["pairs"][ps[0]["pair"]][tk]["naive"]]
        rows = []
        for p in ps:
            e = R["pairs"][p["pair"]][tk]["naive"]
            rows.append([p["pair"]] + [f4(e[g]["rel_residual"]) for g in gk])
        L.append(_t(rows, ["pair (relres)"] + [g.split(":")[1] for g in gk]))
        rows = []
        for p in ps:
            e = R["pairs"][p["pair"]][tk]["naive"]
            rows.append([p["pair"]] + [f4(e[g]["cosine"]) for g in gk])
        L.append("")
        L.append(_t(rows, ["pair (cosine)"] + [g.split(":")[1] for g in gk]))

        L.append(f"\n### Rung-2 per-module coefficient structure ({label})\n")
        rows = []
        for p in ps:
            e = R["pairs"][p["pair"]][tk]["rung2"]
            cs = e["coef_summary"]
            rows.append([p["pair"], f4(cs["a_mean"]), f4(cs["a_std"]), f4(cs["b_mean"]),
                         f4(cs["b_std"]),
                         f4(e["depth_corr"]["a_vs_layer"]), f4(e["depth_corr"]["b_vs_layer"]),
                         f4(e["by_depth"].get("early", {}).get("a_mean")),
                         f4(e["by_depth"].get("mid", {}).get("a_mean")),
                         f4(e["by_depth"].get("late", {}).get("a_mean"))])
        L.append(_t(rows, ["pair", "a mean", "a sd", "b mean", "b sd",
                           "corr(a,layer)", "corr(b,layer)", "a early", "a mid", "a late"]))
        rows = []
        for p in ps:
            e = R["pairs"][p["pair"]][tk]["rung2"]["by_mtype"]
            rows.append([p["pair"]] + [f4(e[t]["a_mean"]) for t in mtypes if t in e])
        L.append("")
        L.append(_t(rows, ["pair (rung2 a by mtype)"] + [t for t in mtypes]))
        L.append("")

    L.append("## 4. Leave-one-pair-out generalisation (the key result)\n")
    if not R["loo"]:
        L.append("_not enough pairs on disk to run leave-one-pair-out_\n")
    for tk, lab in (("comp", "compositional"), ("union", "union")):
        if tk not in R["loo"]:
            continue
        d = R["loo"][tk]
        L.append(f"### {lab} targets\n")
        rows = [[r["held_out"], f4(r["naive"]), f4(r["rung1_ordered"]), f4(r["rung1_symmetric"]),
                 f4(r["rung2_ordered"]), f4(r["rung2_symmetric"]),
                 f4(r["insample_rung1"]), f4(r["insample_rung2"])] for r in d["per_pair"]]
        L.append(_t(rows, ["held-out", "naive", "r1 ord", "r1 sym", "r2 ord", "r2 sym",
                           "(r1 in-sample)", "(r2 in-sample)"]))
        L.append("")
        rows = []
        for k in ["naive", "rung1_ordered", "rung1_symmetric", "rung2_ordered", "rung2_symmetric"]:
            s = d["summary"][k]
            rows.append([k, f4(s["mean"]), f4(s["median"]), f4(s["min"]), f4(s["max"]),
                         "-" if k == "naive" else f"{s['n_beats_naive']}/{s['n_pairs']}",
                         "-" if k == "naive" else f4(s["mean_delta_vs_naive"])])
        L.append(_t(rows, ["scheme", "mean relres", "median", "min", "max",
                           "beats naive", "mean delta vs naive"]))
        L.append("\n(ordered = mean of per-pair (a,b) in alphabetical trait order; symmetric = one "
                 "coefficient, the mean of all fitted a and b, applied to both terms. Only the "
                 "held-out columns count.)\n")

    L.append("## 6. Residual spectrum\n")
    if not R["residual_spectrum"]:
        L.append("_no targets available_\n")
    else:
        rows = []
        for k, v in R["residual_spectrum"].items():
            rows.append([k, v["n_modules_sampled"],
                         f4(v["target_pr"]["mean"]), f4(v["target_rank90"]["mean"]),
                         f4(v["residual_naive_pr"]["mean"]), f4(v["residual_naive_rank90"]["mean"]),
                         f4(v["residual_rung2_pr"]["mean"]), f4(v["residual_rung2_rank90"]["mean"]),
                         f"{v['max_possible_rank_target']}/{v['max_possible_rank_residual']}"])
        L.append(_t(rows, ["pair/target", "n mods", "target PR", "target r90",
                           "naiveR PR", "naiveR r90", "rung2R PR", "rung2R r90",
                           "maxrank T/R"]))
        L.append("\n(PR = participation ratio (sum s^2)^2 / sum s^4; r90 = #svals for 90% of the "
                 "squared-Frobenius energy. Residual rank is capped at 3r by construction; a residual "
                 "PR near that cap means the leftover is spread over all available directions.)\n")

    if R["warnings"]:
        L.append("## Warnings\n")
        for w in R["warnings"]:
            L.append(f"- {w}")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
