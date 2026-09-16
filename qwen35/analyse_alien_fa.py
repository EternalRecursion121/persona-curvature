#!/usr/bin/env python3
"""Where does the trait lexicon NOT go -- on the FACTOR chart?

The factor-chart twin of analyse_alien.py.  Same question, same estimator, same
null; the only thing that changes is the chart the question is asked in.

analyse_alien.py works in the top-k principal subspace of the CENTRED sketch
matrix.  Decision 2026-09-08 (Samuel) makes the factor analysis the primary
frame, so here the chart is fa_chart.FAChart: the orthonormal Gram-Schmidt basis
of the span of the five oblimin factor directions, in the fixed order
Warmth, Competence, FearfulWithdrawal, Arousal, Imagination.  Inner products come
from the exact Gram results/gram_sweep.npz, never from sketches.

    gap(k) = max_{u in S^{k-1}}  min_i  arccos |<u, a_i>|

with a_i the unit-normalised chart coordinates of adapter i in the first k basis
vectors (FAChart.trait_coords[:, :k]).  Adapters are LINES, not points, because
steering takes both signs.  deepest_hole is imported unchanged from
analyse_alien.py so the two charts are compared with one solver.

Null, exactly as the PC version defines it: 134 uniformly random unit vectors in
the same k, 24 draws, same restart budget.

Two differences from the PC construction are definitional and are reported
rather than papered over:
  * the factor chart is built on RAW adapters, not on adapters centred on the
    zoo mean.  The five basis rows are contrasts (they sum to zero over the 134),
    so every chart coordinate is a contrast, but the 134 points themselves are
    not centred in the chart.  `offset` in the output records how big that is.
  * the chart has exactly five dimensions, so the sweep stops at k=5 instead of
    running to k=16.

Outputs analysis/alien_fa.json, analysis/direction_gaps_fa.json and
analysis/alien_v_fa.npy.  Never touches the PC-chart outputs.
"""
import json
import os

import numpy as np

from analyse_alien import deepest_hole
from fa_chart import FAChart

Q = os.path.dirname(os.path.abspath(__file__))
F5 = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]


def trait_meta():
    meta = {}
    for f in ("traits_primary.json", "traits_secondary.json"):
        p = f"{Q}/{f}"
        if os.path.exists(p):
            for r in json.load(open(p)):
                meta[r["trait"].lower().replace(" ", "_").replace("-", "_")] = (r["factor"], r["keyed"])
    return meta


def main():
    ch = FAChart()
    names = ch.names
    n = len(names)
    meta = trait_meta()
    G = ch.G
    norms = ch.norms
    rng = np.random.default_rng(0)

    T = ch.trait_coords                      # 134 x 5, raw adapters in the chart
    print(f"{n} adapters, factor chart k=5 "
          f"(order {', '.join(ch.factor_names)})")
    print(f"mean chart length {ch.trait_chart_len.mean():.4f} against mean adapter "
          f"norm {norms.mean():.4f} -- the chart sees "
          f"{np.mean(ch.trait_chart_len / norms) * 100:.1f}% of an adapter\n")

    out = {"n": n, "traits": names, "factor_order": ch.factor_names, "k_sweep": {}}

    # how far off-centre the 134 sit in this chart (the PC chart centres, this does not)
    m = T.mean(0)
    Un = T / np.linalg.norm(T, axis=1, keepdims=True)
    out["offset"] = {
        "chart_mean_len": float(np.linalg.norm(m)),
        "trait_chart_len_mean": float(ch.trait_chart_len.mean()),
        "ratio": float(np.linalg.norm(m) / ch.trait_chart_len.mean()),
        "mean_cos_with_chart_mean": float((Un @ (m / np.linalg.norm(m))).mean())}

    # Benchmarks for reading any angle on this chart, computed the SAME way as
    # everything else here -- exact Gram, uncentred, adapters as lines.  The
    # figures on analysis/trait_angles.json are centred and sketched, so they are
    # not interchangeable with these and must not be quoted beside them.
    C = np.abs(G / np.outer(norms, norms))
    iu = np.triu_indices(n, 1)
    off = C[iu]
    dpair = np.degrees(np.arccos(np.clip(off, 0, 1)))
    nn = np.degrees(np.arccos(np.clip((C - np.eye(n) * 2).max(1), 0, 1)))
    i_, j_ = iu[0][int(np.argmax(off))], iu[1][int(np.argmax(off))]
    out["benchmarks_exact_uncentred"] = {
        "pair_median": float(np.median(dpair)), "pair_p5": float(np.percentile(dpair, 5)),
        "nn_median": float(np.median(nn)), "nn_min": float(nn.min()),
        "closest": [names[i_], names[j_], float(dpair.min())]}
    print(f"benchmarks (exact, uncentred): two adapters at random "
          f"{np.median(dpair):.1f}d, a trait to its nearest neighbour "
          f"{np.median(nn):.1f}d, closest pair {names[i_]}/{names[j_]} "
          f"{dpair.min():.1f}d\n")

    print(f"{'k':>3s} {'gap(obs)':>9s} {'gap(null)':>10s} {'null sd':>8s} {'z':>6s}"
          f"  nearest trait to the hole")
    for k in (2, 3, 4, 5):
        A = T[:, :k] / np.linalg.norm(T[:, :k], axis=1, keepdims=True)
        u, gap, mins = deepest_hole(A, np.random.default_rng(100 + k),
                                    n_start=2000, iters=1200)
        nulls = []
        for r in range(24):
            R = rng.normal(size=(n, k))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            nulls.append(deepest_hole(R, np.random.default_rng(500 + r),
                                      n_start=120, iters=400)[1])
        nm, ns = float(np.mean(nulls)), float(np.std(nulls))
        near = int(np.argmax(np.abs(A @ u)))
        z = (gap - nm) / max(ns, 1e-9)
        print(f"{k:>3d} {gap:>8.1f}d {nm:>9.1f}d {ns:>7.2f} {z:>+6.1f}  "
              f"{names[near]} ({np.degrees(np.arccos(np.abs(A[near] @ u))):.1f}d)")
        out["k_sweep"][k] = {"gap_deg": gap, "null_mean_deg": nm, "null_sd_deg": ns,
                             "z": float(z), "nearest": names[near], "u": u.tolist()}

    # ---- the alien direction we will actually steer, at k = 5 ---------------
    k = 5
    A = T / np.linalg.norm(T, axis=1, keepdims=True)
    u, gap, mins = deepest_hole(A, np.random.default_rng(105), n_start=2000, iters=1200)
    if out["k_sweep"][5]["gap_deg"] > gap:
        u = np.array(out["k_sweep"][5]["u"])
        gap = out["k_sweep"][5]["gap_deg"]
        print("  (adopting the sweep's k=5 solution, which was deeper)")
    out["k_sweep"][5]["gap_deg"] = gap
    out["k_sweep"][5]["u"] = u.tolist()

    C = ch.direction_from_chart(u)           # 134 coefficients, unit norm in G
    print(f"\nALIEN_FA at k=5: {gap:.1f} degrees from the nearest trait line in the chart")
    print("  factor-basis loadings: "
          + "  ".join(f"{ch.factor_names[i][3:]} {u[i]:+.3f}" for i in range(5)))
    print(f"  coefficients sum to {C.sum():+.2e}, G-norm {ch.norm(C):.4f}")
    order = np.argsort(np.abs(A @ u))[::-1]
    print("  closest traits in the chart:")
    for i in order[:8]:
        s = float(A[i] @ u)
        f, kd = meta.get(names[i], ("?", "?"))
        print(f"    {names[i]:22s} {np.degrees(np.arccos(abs(s))):5.1f}d  "
              f"sign {'+' if s > 0 else '-'}  ({f}{kd})")

    # ---- full-space angle, exact, from G ------------------------------------
    # cos(v, a_i) = c^T G e_i / (|c|_G |a_i|).  No sketch anywhere.
    def full_cos(c):
        c = np.asarray(c, dtype=float)
        return (G @ c) / (np.sqrt(c @ G @ c) * norms)

    fc = np.abs(full_cos(C))
    jf = int(np.argmax(fc))
    full_gap = float(np.degrees(np.arccos(min(fc[jf], 1.0))))
    print(f"\n  full adapter space (exact Gram): nearest of the 134 is {full_gap:.1f} "
          f"degrees ({names[jf]})")
    print(f"  the direction keeps {np.linalg.norm(ch.coords(C)):.4f} of its unit norm "
          f"inside the chart (it is a chart direction, so 1.0 by construction)")

    # ---- named axes and factors in the chart --------------------------------
    axes = {}
    for f in F5:
        p = np.array([1.0 if meta.get(t) == (f, "+") else 0.0 for t in names])
        mI = np.array([1.0 if meta.get(t) == (f, "-") else 0.0 for t in names])
        v = p / p.sum() - mI / mI.sum()
        axes[f"axis_{f}"] = v
    axes["mean_assistant_axis"] = np.ones(n) / n
    for i, f in enumerate(ch.factor_names):
        axes[f] = ch.factor_coef[i]
    AL_PC = json.load(open(f"{Q}/analysis/alien.json"))["alien_k5"]
    assert [str(x) for x in AL_PC["traits"]] == names, "adapter order differs from the PC run"
    axes["alien_k5"] = np.array(AL_PC["coeffs"])
    axes["alien_fa"] = C

    print("\nCHART COORDINATES (unit-normalised direction; |chart| = cosine with the chart)")
    print(f"{'direction':24s} " + " ".join(f"{s[3:8] if s.startswith('FA_') else s[:5]:>8s}"
                                           for s in ch.factor_names) + "   |chart|")
    chart = {}
    for nmk, v in axes.items():
        gn = ch.norm(v)
        x = ch.coords(v / gn)
        chart[nmk] = {"coords": x.tolist(), "chart_len": float(np.linalg.norm(x)),
                      "coef_sum": float(np.asarray(v).sum() / gn)}
        print(f"{nmk:24s} " + " ".join(f"{y:>+8.3f}" for y in x)
              + f"   {np.linalg.norm(x):.4f}")
    out["chart_directions"] = chart
    out["factor_chart_coords"] = ch.factor_chart_coords.tolist()

    # ---- the comparison the whole exercise turns on -------------------------
    c_pc = np.array(AL_PC["coeffs"])
    cos_two = float(c_pc @ G @ C / (ch.norm(c_pc) * ch.norm(C)))
    out["vs_pc"] = {
        "pc_gap_deg_pc_chart": AL_PC["gap_deg"],
        "fa_gap_deg_fa_chart": gap,
        "cos_alien_fa_alien_k5": cos_two,
        "deg_alien_fa_alien_k5": float(np.degrees(np.arccos(np.clip(abs(cos_two), 0, 1)))),
        "pc_alien_chart_len_in_fa_chart": float(np.linalg.norm(ch.coords(c_pc / ch.norm(c_pc)))),
        "pc_alien_gap_in_fa_chart_deg": None}
    # where does the PC alien direction sit as a hole in the FA chart?
    x_pc = ch.coords(c_pc / ch.norm(c_pc))
    x_pc = x_pc / np.linalg.norm(x_pc)
    d = np.abs(A @ x_pc)
    out["vs_pc"]["pc_alien_gap_in_fa_chart_deg"] = float(
        np.degrees(np.arccos(min(d.max(), 1.0))))
    out["vs_pc"]["pc_alien_nearest_in_fa_chart"] = names[int(np.argmax(d))]
    print(f"\nthe PC alien direction, measured in THIS chart: "
          f"{out['vs_pc']['pc_alien_gap_in_fa_chart_deg']:.1f} degrees from "
          f"{out['vs_pc']['pc_alien_nearest_in_fa_chart']}; it lies "
          f"{out['vs_pc']['pc_alien_chart_len_in_fa_chart']:.4f} inside the factor span, "
          f"and the two alien directions are "
          f"{out['vs_pc']['deg_alien_fa_alien_k5']:.1f} degrees apart")

    out["alien_fa"] = {
        "gap_deg": gap, "u": u.tolist(), "coeffs": C.tolist(), "traits": names,
        "chart_coords": ch.coords(C).tolist(),
        "chart_len": float(np.linalg.norm(ch.coords(C))),
        "trait_chart_len_mean": float(ch.trait_chart_len.mean()),
        "coef_sum": float(C.sum()), "g_norm": float(ch.norm(C)),
        "full_gap_deg": full_gap, "full_nearest": names[jf],
        "nearest": [{"trait": names[i],
                     "deg": float(np.degrees(np.arccos(abs(A[i] @ u)))),
                     "sign": int(np.sign(A[i] @ u))} for i in order[:12]]}
    out["trait_coords"] = {names[i]: T[i].tolist() for i in range(n)}
    out["trait_chart_len"] = {names[i]: float(ch.trait_chart_len[i]) for i in range(n)}
    out["factor"] = {t: meta.get(t, ("?", "?"))[0] for t in names}
    out["keyed"] = {t: meta.get(t, ("?", "?"))[1] for t in names}
    json.dump(out, open(f"{Q}/analysis/alien_fa.json", "w"), indent=1)

    # ---- direction_gaps_fa.json --------------------------------------------
    # Same table as analyse_gaps.py, but the angles are exact (Gram), not sketched,
    # and they are computed in the full adapter space with each adapter a line.
    rows = {}
    print(f"\n{'direction':24s} {'nearest trait':>16s} {'angle':>7s}   next two")
    for nmk, v in axes.items():
        c = np.abs(full_cos(v))
        od = np.argsort(c)[::-1]
        deg = float(np.degrees(np.arccos(min(c[od[0]], 1.0))))
        rows[nmk] = {"deg": deg, "nearest": names[od[0]],
                     "next": [names[i] for i in od[1:3]],
                     "next_deg": [float(np.degrees(np.arccos(min(c[i], 1.0)))) for i in od[1:3]],
                     "chart_coords": chart[nmk]["coords"],
                     "chart_len": chart[nmk]["chart_len"]}
        print(f"{nmk:24s} {names[od[0]]:>16s} {deg:>6.1f}d   "
              + ", ".join(f"{names[i]} ({np.degrees(np.arccos(min(c[i], 1.0))):.0f}d)"
                          for i in od[1:3]))
    rows["_note"] = ("angles in the full adapter space from the exact Gram "
                     "results/gram_sweep.npz, adapters uncentred, each a line; "
                     "the PC twin analysis/direction_gaps.json measures the same "
                     "thing in the centred 253,952-dimensional sketch space")
    rows["alien_fa_factor_loadings"] = {ch.factor_names[i]: float(u[i]) for i in range(5)}
    json.dump(rows, open(f"{Q}/analysis/direction_gaps_fa.json", "w"), indent=1)

    # ---- the sketch-space image of the direction, for parity with alien_v_k5 -
    X = np.stack([np.load(f"{Q}/analysis/sketches/stage1_k32/{t}.npz",
                          allow_pickle=True)["sketch"].astype(np.float64) for t in names])
    v_sketch = X.T @ C
    v_sketch /= np.linalg.norm(v_sketch)
    np.save(f"{Q}/analysis/alien_v_fa.npy", v_sketch)

    print("\nwrote analysis/alien_fa.json, analysis/direction_gaps_fa.json, "
          "analysis/alien_v_fa.npy")


if __name__ == "__main__":
    main()
