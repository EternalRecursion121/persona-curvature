#!/usr/bin/env python
"""
pooling_check.py -- is the "update is orthogonal to every trait direction" null a
POOLING ARTEFACT?

The original constraint / measurement used ONE global scalar: all 252 LoRA modules
concatenated into a single vector, one inner product, one cosine.  If the trait is
carried by a handful of modules and absent in the other ~250, that pooled number is
diluted towards zero and the constraint is applied in the wrong place.  This script
recomputes the same quantity PER MODULE and asks whether the trait-aligned energy is
concentrated or uniform, against a rank-matched, norm-matched RANDOM control (because
with rank-16 factors the cosine between two random deltas is NOT ~0).

Directions
    U        = dW(plain)                       the update under test
    V_syc    = dW(syc_pure)                    separately-trained pure sycophancy
    V_oracle = dW(plain) - dW(neutral)         same 600 problems/answers, register only
    C_syc    = random, rank 16, per-module ||.||_F matched to V_syc
    C_oracle = random, rank 32, per-module ||.||_F matched to V_oracle
(V_oracle has rank <= 2r = 32, hence the two separate rank-matched controls.)

Machinery is reused from weight_analysis.py: no dense dW is ever materialised.  For
dW_i = s_i B_i A_i,
    <dW_1, dW_2>_F = s1 s2 * sum_ij (B_1^T B_2)[i,j] (A_1 A_2^T)[i,j]
generalised here to per-vector ranks so the rank-32 oracle/control live in the same
Gram as the rank-16 ones.  Everything downstream is a quadratic form c_X^T G c_Y.

Outputs: results/pooling_check.json, results/pooling_check.md
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time

import numpy as np
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weight_analysis import index_modules, read_scaling, depth_bucket, module_gram, _t, f4  # noqa: E402

REAL = ["plain", "neutral", "syc_pure"]


# ---------------------------------------------------------------------------
# generalised factored Frobenius Gram (variable rank per vector)
# ---------------------------------------------------------------------------

def general_gram(factors, dtype=np.float64):
    """
    factors: list of (A, B, s) with A (r_k, in_f), B (out_f, r_k), scalar s.
    returns G with G[i,j] = <s_i B_i A_i, s_j B_j A_j>_F, exactly.

    Astack (R, in), Bstack (out, R) with R = sum_k r_k; one BLAS call each for
    Astack Astack^T and Bstack^T Bstack, then block sums of their elementwise product.
    """
    n = len(factors)
    ranks = [f[0].shape[0] for f in factors]
    offs = np.cumsum([0] + ranks)
    in_f = factors[0][0].shape[1]
    out_f = factors[0][1].shape[0]
    for A, B, _ in factors:
        assert A.ndim == 2 and B.ndim == 2, "factors must be 2-D"
        assert A.shape[1] == in_f, f"in_features mismatch {A.shape} vs {in_f}"
        assert B.shape[0] == out_f, f"out_features mismatch {B.shape} vs {out_f}"
        assert A.shape[0] == B.shape[1], f"rank axis mismatch A={A.shape} B={B.shape}"
    Astack = np.concatenate([A.astype(dtype, copy=False) for A, _, _ in factors], axis=0)
    Bstack = np.concatenate([B.astype(dtype, copy=False) for _, B, _ in factors], axis=1)
    M = (Bstack.T @ Bstack) * (Astack @ Astack.T)        # (R, R)
    G = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            v = float(M[offs[i]:offs[i + 1], offs[j]:offs[j + 1]].sum())
            G[i, j] = G[j, i] = v
    s = np.array([f[2] for f in factors], dtype=np.float64)
    G *= np.outer(s, s)
    return 0.5 * (G + G.T)


def quad(G, cx, cy):
    return float(cx @ G @ cy)


# ---------------------------------------------------------------------------
# sanity check: factored vs dense, on tiny matrices
# ---------------------------------------------------------------------------

def sanity_check(verbose=True):
    lines = []
    rng = np.random.default_rng(12345)
    ok = True

    # (a) heterogeneous ranks, factored vs dense
    out_f, in_f = 13, 9
    specs = [(4, 1.7), (6, -0.5), (3, 2.0), (5, 0.31)]
    facs, dense = [], []
    for r, s in specs:
        A = rng.standard_normal((r, in_f))
        B = rng.standard_normal((out_f, r))
        facs.append((A, B, s))
        dense.append(s * (B @ A))
    G = general_gram(facs)
    D = np.array([[float(np.sum(x * y)) for y in dense] for x in dense])
    err = float(np.max(np.abs(G - D)))
    rel = err / float(np.max(np.abs(D)))
    ok &= rel < 1e-10
    lines.append(f"[check 1] general_gram vs dense, ranks {[r for r,_ in specs]}, "
                 f"shape ({out_f},{in_f}): max|abs err| = {err:.3e}, rel = {rel:.3e}  "
                 f"-> {'PASS' if rel < 1e-10 else 'FAIL'}")

    # (b) agreement with weight_analysis.module_gram on equal ranks
    r = 5
    As = [rng.standard_normal((r, in_f)) for _ in range(3)]
    Bs = [rng.standard_normal((out_f, r)) for _ in range(3)]
    sc = np.array([2.0, 2.0, 2.0])
    G1 = module_gram(As, Bs, sc)
    G2 = general_gram([(As[i], Bs[i], 2.0) for i in range(3)])
    e2 = float(np.max(np.abs(G1 - G2)))
    r2 = e2 / float(np.max(np.abs(G1)))
    ok &= r2 < 1e-10
    lines.append(f"[check 2] general_gram vs weight_analysis.module_gram (equal rank {r}): "
                 f"max|abs err| = {e2:.3e}, rel = {r2:.3e}  -> {'PASS' if r2 < 1e-10 else 'FAIL'}")

    # (c) the actual downstream quantity: cos(U, V_oracle) with U=x0, V=x0-x1, dense vs quadratic form
    U = dense[0]
    V = dense[0] - dense[1]
    cos_dense = float(np.sum(U * V) / (np.linalg.norm(U) * np.linalg.norm(V)))
    cU = np.array([1.0, 0, 0, 0])
    cV = np.array([1.0, -1.0, 0, 0])
    cos_fac = quad(G, cU, cV) / math.sqrt(quad(G, cU, cU) * quad(G, cV, cV))
    e3 = abs(cos_dense - cos_fac)
    ok &= e3 < 1e-12
    lines.append(f"[check 3] cos(U, U-N): dense = {cos_dense:.15f}, factored = {cos_fac:.15f}, "
                 f"|diff| = {e3:.3e}  -> {'PASS' if e3 < 1e-12 else 'FAIL'}")

    # (d) pooled-over-modules cosine: concatenating modules == summing Grams
    mods = []
    for (o, i) in [(7, 5), (11, 4), (6, 9)]:
        f = []
        for r_, s_ in [(3, 2.0), (2, 2.0)]:
            f.append((rng.standard_normal((r_, i)), rng.standard_normal((o, r_)), s_))
        mods.append(f)
    Gs = [general_gram(f) for f in mods]
    Gsum = sum(Gs)
    du = np.concatenate([(f[0][2] * (f[0][1] @ f[0][0])).ravel() for f in mods])
    dv = np.concatenate([(f[1][2] * (f[1][1] @ f[1][0])).ravel() for f in mods])
    cos_cat = float(du @ dv / (np.linalg.norm(du) * np.linalg.norm(dv)))
    e0, e1 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    cos_sum = quad(Gsum, e0, e1) / math.sqrt(quad(Gsum, e0, e0) * quad(Gsum, e1, e1))
    e4 = abs(cos_cat - cos_sum)
    ok &= e4 < 1e-12
    lines.append(f"[check 4] global pooled cosine over 3 modules: concat-dense = {cos_cat:.15f}, "
                 f"sum-of-Grams = {cos_sum:.15f}, |diff| = {e4:.3e}  "
                 f"-> {'PASS' if e4 < 1e-12 else 'FAIL'}")

    lines.append(f"[sanity] ALL CHECKS {'PASSED' if ok else 'FAILED'}")
    if verbose:
        for l in lines:
            print(l)
    if not ok:
        raise SystemExit("sanity check FAILED -- refusing to report numbers")
    return lines


# ---------------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------------

def dist_stats(v):
    v = np.asarray(v, dtype=float)
    return {
        "n": int(v.size), "min": float(v.min()), "p10": float(np.percentile(v, 10)),
        "median": float(np.median(v)), "mean": float(v.mean()),
        "p90": float(np.percentile(v, 90)), "p99": float(np.percentile(v, 99)),
        "max": float(v.max()), "std": float(v.std()),
    }


def topk_shares(E, ks=(5, 10, 25)):
    E = np.asarray(E, dtype=float)
    n = E.size
    tot = float(E.sum())
    srt = np.sort(E)[::-1]
    out = {}
    for k in ks:
        share = float(srt[:k].sum() / tot) if tot > 0 else float("nan")
        uni = k / n
        out[f"top{k}"] = {"share": share, "uniform": uni,
                          "ratio_vs_uniform": share / uni if uni > 0 else float("nan")}
    pr = (tot ** 2) / float((E ** 2).sum()) if tot > 0 else float("nan")
    out["participation_ratio"] = pr           # effective number of modules; n == uniform
    out["effective_frac_of_modules"] = pr / n
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--adapters-dir", default=os.path.join(here, "drift", "adapters"))
    ap.add_argument("--out-dir", default=os.path.join(here, "results"))
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--reopen-every", type=int, default=32)
    ap.add_argument("--expect-modules", type=int, default=252)
    args = ap.parse_args(argv)

    print("=" * 78)
    print("SANITY CHECK (factored inner product vs dense reference)")
    print("=" * 78)
    sanity_lines = sanity_check()
    print()

    # ---- adapters, scaling ----
    warnings = []
    cfg = {}
    for c in REAL:
        d = os.path.join(args.adapters_dir, c)
        st = os.path.join(d, "adapter_model.safetensors")
        cf = os.path.join(d, "adapter_config.json")
        if not (os.path.exists(st) and os.path.exists(cf)):
            raise SystemExit(f"missing adapter files for {c!r} in {d}")
        cfg[c] = read_scaling(cf, c)
        cfg[c]["safetensors"] = st
    scal = {c: cfg[c]["scaling"] for c in REAL}
    if len({round(v, 12) for v in scal.values()}) != 1:
        warnings.append(f"conditions disagree on scaling: {scal}")
    print("scaling read from adapter_config.json (not assumed):")
    for c in REAL:
        v = cfg[c]
        print(f"  {c:9s} r={v['r']} alpha={v['lora_alpha']:.0f} rslora={v['use_rslora']} "
              f"-> dW = {v['scaling']:g} * B@A   [{v['convention']}]  base={v['base_model']}")

    # ---- module index + cross-adapter consistency ----
    modules = index_modules(cfg[REAL[0]]["safetensors"])
    keys = list(modules)
    missing_report = {}
    for c in REAL[1:]:
        m2 = index_modules(cfg[c]["safetensors"])
        miss = sorted(set(keys) - set(m2))
        extra = sorted(set(m2) - set(keys))
        if miss or extra:
            missing_report[c] = {"missing": miss, "extra": extra}
    if missing_report:
        raise SystemExit(f"module sets differ across adapters: {missing_report}")
    n_mod = len(keys)
    layers = sorted({modules[k]["layer"] for k in keys})
    nlayers = max(layers) + 1
    mtypes = sorted({modules[k]["mtype"] for k in keys})
    print(f"\nmodules: {n_mod} (expected {args.expect_modules}) | layers: {nlayers} | "
          f"types: {', '.join(mtypes)}")
    if n_mod != args.expect_modules:
        warnings.append(f"found {n_mod} modules, expected {args.expect_modules}")
    print(f"module set identical across all {len(REAL)} adapters: YES (0 missing, 0 extra)")

    # ---- basis, coefficient vectors ----
    # index: 0 plain, 1 neutral, 2 syc_pure, 3 ctrl32 (oracle-matched), 4 ctrl16 (syc-matched)
    NB = 5
    cU = np.zeros(NB); cU[0] = 1.0
    cVs = {
        "V_syc":     np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
        "V_oracle":  np.array([1.0, -1.0, 0.0, 0.0, 0.0]),
        "C_syc":     np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
        "C_oracle":  np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
    }
    MATCH = {"C_syc": "V_syc", "C_oracle": "V_oracle"}
    DIRS = list(cVs)

    per_mod = {d: {"cos": [], "ef": [], "vnorm": [], "ip": []} for d in DIRS}
    unorm = []
    # section F diagnostics: U vs neutral, and neutral vs the oracle direction
    aux = {"nnorm": [], "un_ip": [], "cos_UN": [], "cos_N_oracle": [], "cos_syc_N": []}
    cN = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    shapes = {}
    rng_master = np.random.default_rng(args.seed)
    ctrl_seeds = rng_master.integers(0, 2**63 - 1, size=n_mod)

    def open_handles():
        return {c: safe_open(cfg[c]["safetensors"], framework="np") for c in REAL}

    handles = open_handles()
    t0 = time.time()
    print("\npass: per-module Gram (no dense dW materialised) ...")
    for i, mk in enumerate(keys):
        if args.reopen_every and i and i % args.reopen_every == 0:
            handles.clear(); gc.collect(); handles = open_handles()
        meta = modules[mk]
        facs = []
        for c in REAL:
            f = handles[c]
            A = f.get_tensor(meta["A"]); B = f.get_tensor(meta["B"])
            assert A.ndim == 2 and B.ndim == 2, f"{c}/{mk}: expected 2-D factors"
            assert A.shape[0] == B.shape[1], f"{c}/{mk}: A{A.shape} B{B.shape} rank axis mismatch"
            facs.append((A, B, scal[c]))
        r = facs[0][0].shape[0]
        in_f = facs[0][0].shape[1]
        out_f = facs[0][1].shape[0]
        for c, (A, B, _) in zip(REAL, facs):
            assert A.shape == (r, in_f) and B.shape == (out_f, r), \
                f"{c}/{mk}: shape {A.shape}/{B.shape} != ({r},{in_f})/({out_f},{r})"
        assert r < min(out_f, in_f), f"{mk}: rank {r} not < min(out,in) -- layout suspect"
        shapes[mk] = {"out": int(out_f), "in": int(in_f), "r": int(r)}

        crng = np.random.default_rng(int(ctrl_seeds[i]))
        facs.append((crng.standard_normal((2 * r, in_f)), crng.standard_normal((out_f, 2 * r)), 1.0))
        facs.append((crng.standard_normal((r, in_f)),     crng.standard_normal((out_f, r)),     1.0))

        G = general_gram(facs)

        # rescale each control so ||C_m||_F == ||V_m||_F for its matched real direction
        sc = np.ones(NB)
        for cname, vname in MATCH.items():
            ic = int(np.argmax(cVs[cname]))
            nv = math.sqrt(max(quad(G, cVs[vname], cVs[vname]), 0.0))
            nc = math.sqrt(max(G[ic, ic], 0.0))
            sc[ic] = (nv / nc) if nc > 0 else 0.0
        G = G * np.outer(sc, sc)

        uu = quad(G, cU, cU)
        unorm.append(math.sqrt(max(uu, 0.0)))
        for d in DIRS:
            cv = cVs[d]
            vv = quad(G, cv, cv)
            ip = quad(G, cU, cv)
            cos = ip / math.sqrt(uu * vv) if uu > 0 and vv > 0 else float("nan")
            per_mod[d]["cos"].append(cos)
            per_mod[d]["ef"].append(cos * cos)
            per_mod[d]["vnorm"].append(math.sqrt(max(vv, 0.0)))
            per_mod[d]["ip"].append(ip)
        nn = quad(G, cN, cN)
        un = quad(G, cU, cN)
        vo = quad(G, cVs["V_oracle"], cVs["V_oracle"])
        vs = quad(G, cVs["V_syc"], cVs["V_syc"])
        aux["nnorm"].append(math.sqrt(max(nn, 0.0)))
        aux["un_ip"].append(un)
        aux["cos_UN"].append(un / math.sqrt(uu * nn) if uu > 0 and nn > 0 else float("nan"))
        aux["cos_N_oracle"].append(
            quad(G, cN, cVs["V_oracle"]) / math.sqrt(nn * vo) if nn > 0 and vo > 0 else float("nan"))
        aux["cos_syc_N"].append(
            quad(G, cVs["V_syc"], cN) / math.sqrt(vs * nn) if vs > 0 and nn > 0 else float("nan"))
        del facs, G
        if (i + 1) % 60 == 0:
            print(f"  ... {i+1}/{n_mod} ({time.time()-t0:.1f}s)", file=sys.stderr)
    handles.clear(); gc.collect()
    print(f"  done in {time.time()-t0:.1f}s")

    unorm = np.array(unorm)
    aux = {k: np.array(v, dtype=float) for k, v in aux.items()}
    for d in DIRS:
        for k in per_mod[d]:
            per_mod[d][k] = np.array(per_mod[d][k], dtype=float)

    U2 = unorm ** 2
    results = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "adapters_dir": os.path.abspath(args.adapters_dir),
            "seed": args.seed,
            "n_modules": n_mod, "n_layers": nlayers, "module_types": mtypes,
            "scaling_per_condition": {c: cfg[c]["scaling"] for c in REAL},
            "scaling_convention": {c: cfg[c]["convention"] for c in REAL},
            "lora_r": {c: cfg[c]["r"] for c in REAL},
            "lora_alpha": {c: cfg[c]["lora_alpha"] for c in REAL},
            "base_model": {c: cfg[c]["base_model"] for c in REAL},
            "delta_formula": "dW = (lora_alpha/r) * B @ A",
            "definitions": {
                "U": "dW(plain)", "V_syc": "dW(syc_pure)",
                "V_oracle": "dW(plain) - dW(neutral)",
                "C_syc": "random rank-16 factors, per-module ||.||_F matched to V_syc",
                "C_oracle": "random rank-32 factors, per-module ||.||_F matched to V_oracle",
                "energy_fraction": "e_m = cos(U_m,V_m)^2",
                "aligned_energy": "E_m = <U_m,V_m>^2 / ||V_m||^2 = e_m * ||U_m||_F^2",
            },
            "module_set_identical_across_adapters": True,
            "missing_modules": missing_report,
        },
        "sanity_check": sanity_lines,
        "warnings": warnings,
    }

    # ---- A: global pooled ----
    A_sec = {}
    for d in DIRS:
        ip = float(per_mod[d]["ip"].sum())
        vv = float((per_mod[d]["vnorm"] ** 2).sum())
        uu = float(U2.sum())
        cos = ip / math.sqrt(uu * vv)
        A_sec[d] = {"pooled_cosine": cos, "pooled_energy_fraction": cos * cos,
                    "pooled_energy_fraction_pct": 100 * cos * cos,
                    "U_fro": math.sqrt(uu), "V_fro": math.sqrt(vv)}
    # what a PER-MODULE constraint would have removed instead
    for d in DIRS:
        E = per_mod[d]["ip"] ** 2 / np.maximum(per_mod[d]["vnorm"] ** 2, 1e-300)
        A_sec[d]["per_module_removable_fraction"] = float(E.sum() / U2.sum())
        A_sec[d]["per_module_removable_pct"] = float(100 * E.sum() / U2.sum())
        A_sec[d]["gain_per_module_over_pooled"] = float(
            (E.sum() / U2.sum()) / (A_sec[d]["pooled_energy_fraction"]))
    results["A_global_pooled"] = A_sec

    # ---- B: distributions ----
    B_sec = {}
    for d in DIRS:
        ef = per_mod[d]["ef"]
        order = np.argsort(ef)[::-1]
        B_sec[d] = {
            "energy_fraction": dist_stats(ef),
            "abs_cosine": dist_stats(np.abs(per_mod[d]["cos"])),
            "cosine_signed": dist_stats(per_mod[d]["cos"]),
            "top20": [{"rank": int(j + 1), "module": keys[m], "layer": modules[keys[m]]["layer"],
                       "mtype": modules[keys[m]]["mtype"],
                       "cos": float(per_mod[d]["cos"][m]),
                       "energy_fraction": float(ef[m]),
                       "U_fro": float(unorm[m]), "V_fro": float(per_mod[d]["vnorm"][m])}
                      for j, m in enumerate(order[:20])],
        }
    results["B_distribution"] = B_sec
    results["U_fro_stats"] = dist_stats(unorm)

    # ---- C: concentration ----
    C_sec = {"note": "E_m = aligned energy = <U_m,V_m>^2/||V_m||^2. Compare each real "
                     "direction to its rank-matched random control, NOT to k/252: module "
                     "sizes are heterogeneous, so even a null direction concentrates.",
             "size_baseline_||U_m||^2": topk_shares(U2)}
    for d in DIRS:
        E = per_mod[d]["ip"] ** 2 / np.maximum(per_mod[d]["vnorm"] ** 2, 1e-300)
        C_sec[d] = topk_shares(E)
    for real, ctrl in [("V_syc", "C_syc"), ("V_oracle", "C_oracle")]:
        C_sec[f"{real}_vs_control"] = {
            k: {"real": C_sec[real][k]["share"], "control": C_sec[ctrl][k]["share"],
                "real_over_control": C_sec[real][k]["share"] / C_sec[ctrl][k]["share"]}
            for k in ["top5", "top10", "top25"]
        }
        C_sec[f"{real}_vs_control"]["participation_ratio"] = {
            "real": C_sec[real]["participation_ratio"],
            "control": C_sec[ctrl]["participation_ratio"],
            "uniform": float(n_mod),
        }
    results["C_concentration"] = C_sec

    # ---- D: breakdowns ----
    mt_of = np.array([modules[k]["mtype"] for k in keys])
    ly_of = np.array([modules[k]["layer"] for k in keys])
    dp_of = np.array([depth_bucket(modules[k]["layer"], nlayers) for k in keys])
    D_sec = {"by_mtype": {}, "by_depth": {}, "by_layer": {}, "depth_corr": {}}
    for d in DIRS:
        ef = per_mod[d]["ef"]
        E = per_mod[d]["ip"] ** 2 / np.maximum(per_mod[d]["vnorm"] ** 2, 1e-300)
        Etot = float(E.sum())
        D_sec["by_mtype"][d] = {
            t: {"n": int((mt_of == t).sum()),
                "mean_energy_fraction": float(ef[mt_of == t].mean()),
                "median_energy_fraction": float(np.median(ef[mt_of == t])),
                "max_energy_fraction": float(ef[mt_of == t].max()),
                "share_of_total_aligned_energy": float(E[mt_of == t].sum() / Etot)}
            for t in mtypes}
        D_sec["by_depth"][d] = {
            b: {"n": int((dp_of == b).sum()),
                "mean_energy_fraction": float(ef[dp_of == b].mean()),
                "median_energy_fraction": float(np.median(ef[dp_of == b])),
                "share_of_total_aligned_energy": float(E[dp_of == b].sum() / Etot)}
            for b in ["early", "mid", "late"] if (dp_of == b).any()}
        D_sec["by_layer"][d] = {
            str(L): {"mean_energy_fraction": float(ef[ly_of == L].mean()),
                     "share_of_total_aligned_energy": float(E[ly_of == L].sum() / Etot)}
            for L in layers}
        lay = ly_of.astype(float)
        D_sec["depth_corr"][d] = float(np.corrcoef(lay, ef)[0, 1]) if ef.std() > 0 else float("nan")
    results["D_breakdown"] = D_sec

    # ---- E: control (already in B/C, summarised explicitly) ----
    results["E_random_control"] = {
        "seed": args.seed,
        "construction": ("per module, A ~ N(0,1) of shape (rank, in), B ~ N(0,1) of shape "
                         "(out, rank), then the whole delta rescaled so ||C_m||_F == ||V_m||_F "
                         "for the matched real direction; rank 16 for C_syc, 32 for C_oracle "
                         "(V_oracle = plain - neutral has rank <= 32)"),
        "C_syc": {"energy_fraction": B_sec["C_syc"]["energy_fraction"],
                  "pooled": A_sec["C_syc"], "concentration": C_sec["C_syc"]},
        "C_oracle": {"energy_fraction": B_sec["C_oracle"]["energy_fraction"],
                     "pooled": A_sec["C_oracle"], "concentration": C_sec["C_oracle"]},
    }

    # ---- F: is V_oracle's alignment with U an arithmetic artefact? ----
    # V_oracle = U - N contains U by construction. Given only ||U||, ||N||, <U,N>,
    #   cos(U, U-N) = (uu - un) / sqrt(uu * (uu - 2un + nn))
    # is FULLY DETERMINED -- it carries no information beyond the U/N pair. In
    # particular if U _|_ N and ||U||=||N|| it is exactly 1/sqrt(2) (e = 50%),
    # for any two unrelated updates whatsoever.
    uu_g, nn_g = float(U2.sum()), float((aux["nnorm"] ** 2).sum())
    un_g = float(aux["un_ip"].sum())
    cos_UN_g = un_g / math.sqrt(uu_g * nn_g)
    ratio = math.sqrt(nn_g / uu_g)
    pred = (1.0 - cos_UN_g * ratio) / math.sqrt(1.0 - 2 * cos_UN_g * ratio + ratio ** 2)
    F_sec = {
        "why": ("V_oracle = dW(plain) - dW(neutral) contains U = dW(plain) as a term, so "
                "cos(U, V_oracle) is an algebraic function of ||U||, ||N||, <U,N> alone. "
                "For two ARBITRARY near-orthogonal equal-norm updates it is ~1/sqrt(2), "
                "i.e. an energy fraction of ~50%, with no trait structure required."),
        "cos_U_neutral_pooled": cos_UN_g,
        "norm_ratio_N_over_U": ratio,
        "predicted_cos_U_oracle_from_UN_alone": pred,
        "measured_cos_U_oracle_pooled": A_sec["V_oracle"]["pooled_cosine"],
        "identity_residual": abs(pred - A_sec["V_oracle"]["pooled_cosine"]),
        "orthogonal_equalnorm_reference": {"cos": 1 / math.sqrt(2), "energy_fraction": 0.5},
        "cos_U_neutral_per_module": dist_stats(aux["cos_UN"]),
        "cos_neutral_oracle_per_module": dist_stats(aux["cos_N_oracle"]),
        "cos_syc_neutral_per_module": dist_stats(aux["cos_syc_N"]),
        "excess_over_orthogonal_reference": {
            "V_oracle_pooled_energy_fraction": A_sec["V_oracle"]["pooled_energy_fraction"],
            "minus_0.5_baseline": A_sec["V_oracle"]["pooled_energy_fraction"] - 0.5,
        },
    }
    results["F_oracle_artefact_check"] = F_sec

    # ---- per-module dump ----
    results["per_module"] = [
        {"module": keys[i], "layer": int(modules[keys[i]]["layer"]),
         "mtype": modules[keys[i]]["mtype"],
         "out": shapes[keys[i]]["out"], "in": shapes[keys[i]]["in"], "r": shapes[keys[i]]["r"],
         "U_fro": float(unorm[i]),
         **{f"{d}_{q}": float(per_mod[d][q][i]) for d in DIRS for q in ("cos", "ef", "vnorm")}}
        for i in range(n_mod)
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    jp = os.path.join(args.out_dir, "pooling_check.json")
    with open(jp, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    md = render_md(results, keys, modules, mtypes, layers)
    mp = os.path.join(args.out_dir, "pooling_check.md")
    with open(mp, "w") as fh:
        fh.write(md)
    print()
    print(md)
    print(f"\nwrote {jp}\nwrote {mp}")
    return 0


# ---------------------------------------------------------------------------

def pct(x):
    try:
        return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100*x:.3f}%"
    except Exception:
        return str(x)


def render_md(R, keys, modules, mtypes, layers):
    L = []
    m = R["meta"]
    L.append("# Pooling check: is the orthogonality null a pooling artefact?\n")
    L.append(f"- adapters: `{m['adapters_dir']}`; {m['n_modules']} modules, {m['n_layers']} layers")
    L.append(f"- `{m['delta_formula']}`; scaling read from adapter_config.json = "
             + ", ".join(f"{c}={v:g}" for c, v in m["scaling_per_condition"].items())
             + f" ({list(m['scaling_convention'].values())[0]}, r=16, alpha=32)")
    L.append(f"- U = {m['definitions']['U']}; V_syc = {m['definitions']['V_syc']}; "
             f"V_oracle = {m['definitions']['V_oracle']}")
    L.append(f"- random control seed: `{m['seed']}`")
    L.append(f"- module set identical across all adapters: "
             f"{'YES' if m['module_set_identical_across_adapters'] else 'NO'}\n")

    L.append("## Sanity check (factored inner product vs dense)\n")
    L.append("```")
    L.extend(R["sanity_check"])
    L.append("```\n")

    L.append("## A. Global pooled energy fraction (what the original constraint used)\n")
    rows = []
    for d in ["V_syc", "V_oracle", "C_syc", "C_oracle"]:
        a = R["A_global_pooled"][d]
        rows.append([d, f4(a["pooled_cosine"]), pct(a["pooled_energy_fraction"]),
                     pct(a["per_module_removable_fraction"]),
                     f4(a["gain_per_module_over_pooled"])])
    L.append(_t(rows, ["direction", "pooled cos", "pooled energy frac",
                       "per-module removable frac", "x gain (per-mod/pooled)"]))
    L.append("\n`per-module removable frac` = sum_m <U_m,V_m>^2/||V_m||^2 / sum_m ||U_m||^2: the "
             "energy a PER-MODULE orthogonality constraint would strip, versus the pooled column "
             "which is what a single global scalar strips.\n")

    L.append("## B. Distribution of per-module energy fractions e_m = cos(U_m,V_m)^2\n")
    rows = []
    for d in ["V_syc", "V_oracle", "C_syc", "C_oracle"]:
        s = R["B_distribution"][d]["energy_fraction"]
        rows.append([d, pct(s["min"]), pct(s["median"]), pct(s["mean"]), pct(s["p90"]),
                     pct(s["p99"]), pct(s["max"])])
    L.append(_t(rows, ["direction", "min", "median", "mean", "p90", "p99", "max"]))
    for d in ["V_syc", "V_oracle"]:
        L.append(f"\n### top-20 modules by energy fraction, {d}\n")
        rows = [[t["rank"], t["module"], t["mtype"], t["layer"], f4(t["cos"]),
                 pct(t["energy_fraction"]), f4(t["U_fro"]), f4(t["V_fro"])]
                for t in R["B_distribution"][d]["top20"]]
        L.append(_t(rows, ["#", "module", "type", "layer", "cos", "energy frac",
                           "||U_m||", "||V_m||"]))
    L.append("")

    L.append("## C. THE DECIDING NUMBER: concentration of trait-aligned energy\n")
    C = R["C_concentration"]
    rows = []
    for d in ["V_syc", "C_syc", "V_oracle", "C_oracle", "size_baseline_||U_m||^2"]:
        c = C[d]
        rows.append([d, pct(c["top5"]["share"]), pct(c["top10"]["share"]),
                     pct(c["top25"]["share"]), f4(c["top5"]["ratio_vs_uniform"]),
                     f4(c["top10"]["ratio_vs_uniform"]), f4(c["top25"]["ratio_vs_uniform"]),
                     f4(c["participation_ratio"])])
    L.append(_t(rows, ["direction", "top5", "top10", "top25", "x uni(5)", "x uni(10)",
                       "x uni(25)", "eff. #modules (PR)"]))
    L.append(f"\nuniform prediction: top5 = {pct(5/R['meta']['n_modules'])}, "
             f"top10 = {pct(10/R['meta']['n_modules'])}, top25 = {pct(25/R['meta']['n_modules'])}; "
             f"uniform PR = {R['meta']['n_modules']}.\n")
    rows = []
    for real in ["V_syc", "V_oracle"]:
        v = C[f"{real}_vs_control"]
        rows.append([real] + [f4(v[k]["real_over_control"]) for k in ["top5", "top10", "top25"]]
                    + [f4(v["participation_ratio"]["real"]),
                       f4(v["participation_ratio"]["control"])])
    L.append(_t(rows, ["direction", "top5 real/ctrl", "top10 real/ctrl", "top25 real/ctrl",
                       "PR real", "PR control"]))
    L.append("")

    L.append("## D. Breakdown by module type and depth (mean per-module energy fraction)\n")
    rows = []
    for d in ["V_syc", "V_oracle", "C_syc", "C_oracle"]:
        b = R["D_breakdown"]["by_mtype"][d]
        rows.append([d] + [pct(b[t]["mean_energy_fraction"]) for t in mtypes])
    L.append(_t(rows, ["direction"] + mtypes))
    L.append("\nshare of total aligned energy by module type:\n")
    rows = []
    for d in ["V_syc", "V_oracle", "C_syc", "C_oracle"]:
        b = R["D_breakdown"]["by_mtype"][d]
        rows.append([d] + [pct(b[t]["share_of_total_aligned_energy"]) for t in mtypes])
    L.append(_t(rows, ["direction"] + mtypes))
    L.append("")
    rows = []
    for d in ["V_syc", "V_oracle", "C_syc", "C_oracle"]:
        b = R["D_breakdown"]["by_depth"][d]
        rows.append([d] + [pct(b[k]["mean_energy_fraction"]) for k in b]
                    + [pct(b[k]["share_of_total_aligned_energy"]) for k in b]
                    + [f4(R["D_breakdown"]["depth_corr"][d])])
        hdr = ["direction"] + [f"mean {k}" for k in b] + [f"share {k}" for k in b] + ["corr(e,layer)"]
    L.append(_t(rows, hdr))
    L.append("\nper-layer mean energy fraction:\n")
    rows = []
    for d in ["V_syc", "V_oracle", "C_oracle"]:
        bl = R["D_breakdown"]["by_layer"][d]
        rows.append([d] + [f"{100*bl[str(x)]['mean_energy_fraction']:.2f}" for x in layers])
    L.append(_t(rows, ["dir (mean e_m %)"] + [str(x) for x in layers]))
    L.append("")

    L.append("## E. Random control\n")
    e = R["E_random_control"]
    L.append(f"seed `{e['seed']}`. {e['construction']}\n")
    rows = []
    for d in ["C_syc", "C_oracle"]:
        s = e[d]["energy_fraction"]
        rows.append([d, pct(s["min"]), pct(s["median"]), pct(s["mean"]), pct(s["p90"]),
                     pct(s["p99"]), pct(s["max"]), pct(e[d]["pooled"]["pooled_energy_fraction"])])
    L.append(_t(rows, ["control", "min", "median", "mean", "p90", "p99", "max", "pooled"]))
    L.append("\nThis is what 'no alignment' looks like at rank 16/32 in these shapes. Every real "
             "number above must be read against this row, not against zero.\n")

    F = R["F_oracle_artefact_check"]
    L.append("## F. Is V_oracle's alignment with U an arithmetic artefact?\n")
    L.append(F["why"] + "\n")
    rows = [
        ["cos(U, N) pooled", f4(F["cos_U_neutral_pooled"])],
        ["||N|| / ||U|| pooled", f4(F["norm_ratio_N_over_U"])],
        ["cos(U, U-N) PREDICTED from those two alone", f4(F["predicted_cos_U_oracle_from_UN_alone"])],
        ["cos(U, U-N) MEASURED", f4(F["measured_cos_U_oracle_pooled"])],
        ["|predicted - measured| (must be ~0: it is an identity)", f"{F['identity_residual']:.2e}"],
        ["reference: U _|_ N, equal norms -> cos", f4(F["orthogonal_equalnorm_reference"]["cos"])],
        ["reference energy fraction", pct(F["orthogonal_equalnorm_reference"]["energy_fraction"])],
        ["measured V_oracle pooled energy fraction",
         pct(F["excess_over_orthogonal_reference"]["V_oracle_pooled_energy_fraction"])],
        ["excess over the 50% no-information baseline",
         pct(F["excess_over_orthogonal_reference"]["minus_0.5_baseline"])],
    ]
    L.append(_t(rows, ["quantity", "value"]))
    L.append("")
    rows = []
    for k, lab in [("cos_U_neutral_per_module", "cos(U_m, N_m)"),
                   ("cos_neutral_oracle_per_module", "cos(N_m, V_oracle_m)"),
                   ("cos_syc_neutral_per_module", "cos(V_syc_m, N_m)")]:
        s = F[k]
        rows.append([lab, f4(s["min"]), f4(s["median"]), f4(s["mean"]), f4(s["max"])])
    L.append(_t(rows, ["quantity", "min", "median", "mean", "max"]))
    L.append("\nIf `cos(N_m, V_oracle_m)` is roughly the negative mirror of `cos(U_m, V_oracle_m)`, "
             "the oracle direction does not single out `plain` at all -- it is equally (anti-)aligned "
             "with `neutral`, which is what a difference vector between two unrelated updates does.\n")

    if R["warnings"]:
        L.append("## Warnings\n")
        for w in R["warnings"]:
            L.append(f"- {w}")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
