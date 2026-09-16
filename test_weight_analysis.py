#!/usr/bin/env python
"""
test_weight_analysis.py -- numerical validation of weight_analysis.py.

Five independent checks:
  A. peft cross-check   -- instantiate a REAL peft lora Linear (no base model loaded)
                           and compare peft's own get_delta_weight() to (alpha/r) B @ A.
  B. Gram cross-check   -- the factorised module_gram() vs dense <dW_i, dW_j>_F.
  C. least-squares      -- the closed-form 2x2 solve vs np.linalg.lstsq on dense
                           flattened deltas; and apply_ab()'s residual expansion vs
                           an explicitly formed dense residual.
  D. spectrum           -- exact_svals() vs np.linalg.svd on the dense residual.
  E. end-to-end recovery-- run weight_analysis.py on synthetic adapters whose true
                           (a, b) are known and assert they come back.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weight_analysis as wa  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------- A
def test_peft_convention():
    print("\nA. peft cross-check: peft's own get_delta_weight() vs (alpha/r) B @ A")
    import torch
    import torch.nn as nn
    from peft.tuners.lora.layer import Linear as LoraLinear
    import peft

    in_f, out_f, r, alpha = 37, 53, 8, 32
    torch.manual_seed(0)
    base = nn.Linear(in_f, out_f, bias=False)
    lay = LoraLinear(base, adapter_name="default", r=r, lora_alpha=alpha, lora_dropout=0.0)
    with torch.no_grad():   # lora_B inits to zero; make it non-trivial
        lay.lora_B["default"].weight.copy_(torch.randn(out_f, r))
        lay.lora_A["default"].weight.copy_(torch.randn(r, in_f))

    A = lay.lora_A["default"].weight.detach().numpy().astype(np.float64)
    B = lay.lora_B["default"].weight.detach().numpy().astype(np.float64)
    check(f"peft {peft.__version__}: lora_A.weight is (r, in_features)", A.shape == (r, in_f), str(A.shape))
    check("peft: lora_B.weight is (out_features, r)", B.shape == (out_f, r), str(B.shape))
    check("peft: scaling == lora_alpha / r", abs(lay.scaling["default"] - alpha / r) < 1e-12,
          f"peft says {lay.scaling['default']}, alpha/r = {alpha/r}")

    mine = (alpha / r) * (B @ A)
    theirs = lay.get_delta_weight("default").detach().numpy().astype(np.float64)
    check("dW shape == (out_features, in_features)", mine.shape == (out_f, in_f), str(mine.shape))
    err = np.abs(mine - theirs).max() / (np.abs(theirs).max() + 1e-30)
    check("(alpha/r) B @ A == peft.get_delta_weight()", err < 1e-6, f"max rel err {err:.3e}")

    # and confirm the merge path uses the same thing
    before = base.weight.detach().numpy().copy().astype(np.float64)
    lay.merge()
    after = lay.base_layer.weight.detach().numpy().astype(np.float64)
    err2 = np.abs((after - before) - theirs).max() / (np.abs(theirs).max() + 1e-30)
    check("merge() adds exactly that delta to W", err2 < 1e-6, f"max rel err {err2:.3e}")


# ---------------------------------------------------------------- B
def test_gram():
    print("\nB. Gram cross-check: factorised r x r contraction vs dense Frobenius products")
    rng = np.random.default_rng(7)
    n, r, in_f, out_f = 6, 8, 91, 67
    As = [rng.standard_normal((r, in_f)).astype(np.float32) for _ in range(n)]
    Bs = [rng.standard_normal((out_f, r)).astype(np.float32) for _ in range(n)]
    s = rng.uniform(0.5, 3.0, size=n)

    G = wa.module_gram(As, Bs, s)
    dense = [s[k] * (Bs[k].astype(np.float64) @ As[k].astype(np.float64)) for k in range(n)]
    D = np.array([[float((dense[i] * dense[j]).sum()) for j in range(n)] for i in range(n)])
    err = np.abs(G - D).max() / np.abs(D).max()
    check("module_gram == dense <dW_i, dW_j>_F", err < 1e-9, f"max rel err {err:.3e}")
    check("Gram is symmetric", np.abs(G - G.T).max() < 1e-9)
    check("dW shape is (out, in)", dense[0].shape == (out_f, in_f), str(dense[0].shape))

    # additivity over modules: <concat(dW1,dW2), concat(dW1',dW2')> == sum of per-module Grams.
    # Use a genuinely independent second module (differently shaped, as in a real model).
    As2 = [rng.standard_normal((r, 44)).astype(np.float32) for _ in range(n)]
    Bs2 = [rng.standard_normal((31, r)).astype(np.float32) for _ in range(n)]
    G2 = wa.module_gram(As2, Bs2, s)
    dense2 = [s[k] * (Bs2[k].astype(np.float64) @ As2[k].astype(np.float64)) for k in range(n)]
    cat = [np.concatenate([dense[k].ravel(), dense2[k].ravel()]) for k in range(n)]
    D2 = np.array([[float(cat[i] @ cat[j]) for j in range(n)] for i in range(n)])
    err = np.abs((G + G2) - D2).max() / np.abs(D2).max()
    check("Gram of concatenated modules == sum of per-module Grams", err < 1e-9, f"{err:.3e}")


# ---------------------------------------------------------------- C
def test_least_squares():
    print("\nC. least squares: closed-form 2x2 vs np.linalg.lstsq on dense vectors")
    rng = np.random.default_rng(11)
    for trial in range(4):
        out_f, in_f = 40, 55
        X = rng.standard_normal((out_f, in_f))
        Y = rng.standard_normal((out_f, in_f))
        a0, b0 = rng.uniform(-2, 2), rng.uniform(-2, 2)
        T = a0 * X + b0 * Y + 0.1 * rng.standard_normal((out_f, in_f))

        mats = [X, Y, T]
        G = np.array([[float((mats[i] * mats[j]).sum()) for j in range(3)] for i in range(3)])
        a, b, rr, ok = wa.fit_ab(G, 0, 1, 2)

        M = np.stack([X.ravel(), Y.ravel()], axis=1)
        ref, *_ = np.linalg.lstsq(M, T.ravel(), rcond=None)
        e = max(abs(a - ref[0]), abs(b - ref[1])) / max(1.0, abs(ref[0]), abs(ref[1]))
        check(f"trial {trial}: (a,b) == lstsq", e < 1e-8,
              f"mine=({a:.6f},{b:.6f}) lstsq=({ref[0]:.6f},{ref[1]:.6f}) err={e:.2e}")

        R = T - a * X - b * Y
        rr_dense = np.linalg.norm(R) / np.linalg.norm(T)
        check(f"trial {trial}: fitted rel-residual matches dense", abs(rr - rr_dense) < 1e-8,
              f"{rr:.8f} vs {rr_dense:.8f}")
        check(f"trial {trial}: residual orthogonal to span(X,Y)",
              abs(float((R * X).sum())) / np.linalg.norm(R) / np.linalg.norm(X) < 1e-10)

        # arbitrary (not-fitted) coefficients -> apply_ab full expansion
        ca, cb = 0.31, -1.7
        sq, tt = wa.apply_ab(G, 0, 1, 2, ca, cb)
        rd = np.linalg.norm(T - ca * X - cb * Y) ** 2
        check(f"trial {trial}: apply_ab expansion == dense ||T-aX-bY||^2",
              abs(sq - rd) / rd < 1e-10, f"{sq:.6f} vs {rd:.6f}")
        check(f"trial {trial}: naive_stats == dense",
              abs(wa.naive_stats(G, 0, 1, 2)["rel_residual"]
                  - np.linalg.norm(T - X - Y) / np.linalg.norm(T)) < 1e-10)
        check(f"trial {trial}: fitted residual <= naive residual", rr <=
              np.linalg.norm(T - X - Y) / np.linalg.norm(T) + 1e-12)

    # degenerate: X and Y collinear -> normal equations singular -> flagged, not NaN
    X = rng.standard_normal((10, 10))
    T = rng.standard_normal((10, 10))
    mats = [X, 2.5 * X, T]
    G = np.array([[float((mats[i] * mats[j]).sum()) for j in range(3)] for i in range(3)])
    a, b, rr, ok = wa.fit_ab(G, 0, 1, 2)
    check("collinear regressors flagged as ill-posed (no NaN, falls back to a=b=1)",
          ok is False and a == 1.0 and b == 1.0 and np.isfinite(rr), f"ok={ok} a={a} b={b} rr={rr:.4f}")
    # zero-norm target must not divide by zero
    G0 = np.zeros((3, 3)); G0[0, 0] = G0[1, 1] = 1.0
    s = wa.naive_stats(G0, 0, 1, 2)
    check("zero target -> nan, not ZeroDivisionError", np.isnan(s["rel_residual"]))


# ---------------------------------------------------------------- D
def test_spectrum():
    print("\nD. spectrum: exact_svals() via thin QR vs dense SVD")
    rng = np.random.default_rng(13)
    r, out_f, in_f = 8, 60, 70
    Bs = [rng.standard_normal((out_f, r)) for _ in range(3)]
    As = [rng.standard_normal((r, in_f)) for _ in range(3)]
    coefs = [2.0, -0.7, -1.3]
    sv = wa.exact_svals(Bs, As, coefs)
    dense = sum(c * (B @ A) for c, B, A in zip(coefs, Bs, As))
    ref = np.linalg.svd(dense, compute_uv=False)
    k = min(len(sv), 3 * r)
    err = np.abs(np.sort(sv)[::-1][:k] - ref[:k]).max() / ref[0]
    check("singular values match dense SVD", err < 1e-9, f"max rel err {err:.3e}")
    check("rank <= 3r", (ref[3 * r:] < 1e-9 * ref[0]).all() if len(ref) > 3 * r else True)
    st = wa.rank_stats(ref)
    e2 = ref ** 2
    pr = (e2.sum() ** 2) / (e2 ** 2).sum()
    check("participation ratio", abs(st["participation_ratio"] - pr) < 1e-9)
    c = np.cumsum(np.sort(e2)[::-1]) / e2.sum()
    check("rank90", st["rank90"] == int(np.searchsorted(c, 0.90) + 1),
          f"{st['rank90']} vs {int(np.searchsorted(c, 0.90) + 1)}")
    # fallback branch (out < 3r)
    Bs2 = [b[:20] for b in Bs]
    sv2 = np.sort(wa.exact_svals(Bs2, As, coefs))[::-1]
    ref2 = np.linalg.svd(sum(c * (B @ A) for c, B, A in zip(coefs, Bs2, As)), compute_uv=False)
    check("fallback branch (out < 3r) matches dense",
          np.abs(sv2[:len(ref2)] - ref2).max() / ref2[0] < 1e-9)


# ---------------------------------------------------------------- E
def test_end_to_end(keep=None):
    print("\nE. end-to-end recovery on synthetic adapters with known ground truth")
    tmp = keep or tempfile.mkdtemp(prefix="synth_wa_")
    adir = os.path.join(tmp, "adapters_synth")
    odir = os.path.join(tmp, "results")
    for extra, label in ((["--noise", "0.0"], "noise-free"), (["--noise", "0.02"], "eps=0.02")):
        subprocess.run([PY, os.path.join(HERE, "synth_adapters.py"), "--out", adir] + extra,
                       check=True, capture_output=True)
        p = subprocess.run([PY, os.path.join(HERE, "weight_analysis.py"),
                            "--adapters-dir", adir, "--out-dir", odir, "--spectrum-modules", "12"],
                           capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout[-4000:]); print(p.stderr[-4000:])
            check(f"{label}: weight_analysis.py ran", False)
            return tmp, None
        R = json.load(open(os.path.join(odir, "weight_analysis.json")))
        gt = json.load(open(os.path.join(adir, "GROUND_TRUTH.json")))
        tol = 2e-6 if label == "noise-free" else 0.02

        for tk, key in (("comp", "comp"), ("union", "union")):
            ta, tb = gt[key]["a"], gt[key]["b"]
            ea = [abs(R["pairs"][k][tk]["rung1"]["a"] - ta) for k in R["pairs"] if tk in R["pairs"][k]]
            eb = [abs(R["pairs"][k][tk]["rung1"]["b"] - tb) for k in R["pairs"] if tk in R["pairs"][k]]
            got = R["rung1_across_pairs"][tk]
            check(f"{label}/{tk}: rung1 a -> {ta}", max(ea) < tol,
                  f"a = {got['a_mean']:.6f} +/- {got['a_std']:.2e}  (max |err| {max(ea):.2e})")
            check(f"{label}/{tk}: rung1 b -> {tb}", max(eb) < tol,
                  f"b = {got['b_mean']:.6f} +/- {got['b_std']:.2e}  (max |err| {max(eb):.2e})")
            r2 = [R["pairs"][k][tk]["rung2"] for k in R["pairs"] if tk in R["pairs"][k]]
            check(f"{label}/{tk}: rung2 a -> {ta} per-module",
                  max(abs(d["coef_summary"]["a_mean"] - ta) for d in r2) < max(tol, 0.03),
                  f"a_mean {np.mean([d['coef_summary']['a_mean'] for d in r2]):.6f} "
                  f"sd {np.mean([d['coef_summary']['a_std'] for d in r2]):.2e}")
            rr1 = [R["pairs"][k][tk]["rung1"]["rel_residual"] for k in R["pairs"] if tk in R["pairs"][k]]
            rrn = [R["pairs"][k][tk]["naive"]["global"]["rel_residual"] for k in R["pairs"] if tk in R["pairs"][k]]
            if label == "noise-free":
                check(f"{label}/{tk}: rung1 residual ~ 0", max(rr1) < 1e-5, f"max {max(rr1):.3e}")
            check(f"{label}/{tk}: rung1 beats naive in-sample",
                  all(x < y for x, y in zip(rr1, rrn)),
                  f"rung1 {np.mean(rr1):.4f} vs naive {np.mean(rrn):.4f}")
            lo = R["loo"][tk]["summary"]
            check(f"{label}/{tk}: LOO rung1_ordered beats naive on all pairs",
                  lo["rung1_ordered"]["n_beats_naive"] == lo["rung1_ordered"]["n_pairs"],
                  f"{lo['rung1_ordered']['n_beats_naive']}/{lo['rung1_ordered']['n_pairs']}, "
                  f"held-out mean {lo['rung1_ordered']['mean']:.5f} vs naive {lo['naive']['mean']:.5f}")
            sc = R["loo"][tk]["per_pair"][0]["scheme_coefs"]["rung1_ordered"]
            check(f"{label}/{tk}: LOO scheme coefs -> ({ta},{tb})",
                  abs(sc[0] - ta) < max(tol, 1e-3) and abs(sc[1] - tb) < max(tol, 1e-3),
                  f"({sc[0]:.6f}, {sc[1]:.6f})")

        # noise floor sanity: half-splits/reseeds close, different traits far
        v = R["noise_floor"]["verdict"]
        check(f"{label}: synthetic traits are separable (floor logic fires correctly)",
              v["traits_distinguishable"] is True,
              f"floor max {v['floor_rel_dist_max']:.4f} < cross min {v['cross_trait_rel_dist_min']:.4f}, "
              f"ratio {v['separation_ratio_mean_cross_over_mean_floor']:.2f}x")
        cts = [x["cosine"] for x in R["cross_trait"].values()]
        check(f"{label}: independent traits are near-orthogonal", max(abs(c) for c in cts) < 0.1,
              f"max |cos| {max(abs(c) for c in cts):.4f}")

    # -- negative control for the alarm path: make traits indistinguishable
    print("\n  negative control: traits deliberately swamped by noise -> alarm must fire")
    adir2 = os.path.join(tmp, "adapters_flat")
    subprocess.run([PY, os.path.join(HERE, "synth_adapters.py"), "--out", adir2,
                    "--layers", "4", "--half-noise", "40.0", "--seed-noise", "40.0"],
                   check=True, capture_output=True)
    p = subprocess.run([PY, os.path.join(HERE, "weight_analysis.py"), "--adapters-dir", adir2,
                        "--out-dir", os.path.join(tmp, "results_flat"), "--spectrum-modules", "4"],
                       capture_output=True, text=True)
    R2 = json.load(open(os.path.join(tmp, "results_flat", "weight_analysis.json")))
    v2 = R2["noise_floor"]["verdict"]
    check("alarm fires when floor exceeds cross-trait distance",
          v2["traits_distinguishable"] is False and "ALARM" in v2["message"],
          f"floor max {v2['floor_rel_dist_max']:.3f} vs cross min {v2['cross_trait_rel_dist_min']:.3f}")
    check("alarm is printed loudly to stdout", "ALARM" in p.stdout)

    # -- depth-trend control: rung-2 must see structure rung-1 cannot
    print("\n  depth-trend control: per-module coefficients must track the injected trend")
    adir3 = os.path.join(tmp, "adapters_trend")
    subprocess.run([PY, os.path.join(HERE, "synth_adapters.py"), "--out", adir3,
                    "--layers", "12", "--noise", "0.0", "--depth-trend", "0.4"],
                   check=True, capture_output=True)
    subprocess.run([PY, os.path.join(HERE, "weight_analysis.py"), "--adapters-dir", adir3,
                    "--out-dir", os.path.join(tmp, "results_trend"), "--spectrum-modules", "6"],
                   check=True, capture_output=True)
    R3 = json.load(open(os.path.join(tmp, "results_trend", "weight_analysis.json")))
    k0 = sorted(R3["pairs"])[0]
    e = R3["pairs"][k0]["comp"]["rung2"]
    corr = e["depth_corr"]["a_vs_layer"]
    early, late = e["by_depth"]["early"]["a_mean"], e["by_depth"]["late"]["a_mean"]
    check("rung2 recovers injected depth trend in a", corr > 0.99,
          f"corr(a,layer)={corr:.5f}, a early={early:.4f} late={late:.4f} "
          f"(injected 0.7-0.4 .. 0.7+0.4)")
    check("rung2 residual << rung1 residual when true a varies with depth",
          e["rel_residual"] < 1e-5 < R3["pairs"][k0]["comp"]["rung1"]["rel_residual"],
          f"rung2 {e['rel_residual']:.2e} vs rung1 {R3['pairs'][k0]['comp']['rung1']['rel_residual']:.4f}")
    check("rung2 LOO beats rung1 LOO under depth structure",
          R3["loo"]["comp"]["summary"]["rung2_ordered"]["mean"] <
          R3["loo"]["comp"]["summary"]["rung1_ordered"]["mean"],
          f"r2 {R3['loo']['comp']['summary']['rung2_ordered']['mean']:.2e} vs "
          f"r1 {R3['loo']['comp']['summary']['rung1_ordered']['mean']:.4f}")
    return tmp, odir


if __name__ == "__main__":
    keep = sys.argv[1] if len(sys.argv) > 1 else None
    test_peft_convention()
    test_gram()
    test_least_squares()
    test_spectrum()
    tmp, odir = test_end_to_end(keep)
    print("\n" + "=" * 70)
    if FAILS:
        print(f"FAILED {len(FAILS)}: " + "; ".join(FAILS))
        sys.exit(1)
    print("ALL CHECKS PASSED")
    print(f"synthetic run artefacts: {odir}")
    sys.exit(0)
