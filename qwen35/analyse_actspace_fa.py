#!/usr/bin/env python3
"""The factor chart in activation space.

analyse_actspace.py answers the PC version of this question: fit the 134x5
activation PC scores to the 134x5 weight PC scores by Procrustes and report the
R^2.  That number is basis-free and it is also frame-free -- it says the two
clouds have the same shape, not that weight-Warmth is activation-Warmth.

Since 2026-09-08 the factor chart (fa_chart.py) is the project's primary frame,
so here is the same comparison in it, plus the sharper question the PC version
cannot ask.

Method
------
A factor direction is a weighted merge of the 134 stage-one adapters,
v_f = sum_i c_fi a_i, with the coefficients c_f from phase10_runs/steer_spec2_7a.json
(the ones actually steered in phase 10).  The 134 traits also have activation-space
persona vectors x_i -- the trait-centred mean residual-stream shift at layer 16,
saved by analyse_actspace.py as results/gram_actspace_<window>_L16.npz with
G = Xc Xc^T.  Applying the SAME coefficients to the activation vectors gives the
activation-space image of each factor, u_f = sum_i c_fi x_i, and everything else
follows fa_chart.FAChart exactly: Gram-Schmidt the five images in the activation
Gram, in the same fixed order, and read off each trait's coordinates.

No Modal, no GPU, no regeneration: this reads two saved Gram matrices.

Reported
--------
* Procrustes R^2 between the 134x5 weight-chart and activation-chart coordinates,
  with a 500-draw label-shuffle null, beside the PC number from
  analysis/actspace_geometry.json#windows.resp.primary.procrustes_r2.
* Per-factor Pearson r between matching columns, with NO rotation.  This is the
  sharper question: Procrustes is free to rotate Warmth into Competence, a
  correlation is not.  Pearson is shift-invariant, so the one asymmetry in the
  setup -- the weight chart is built on the raw Gram, the activation chart on the
  trait-centred one -- does not bite it.  A centred-weight variant is reported
  alongside anyway, so the asymmetry is visible rather than argued about.
* Pairwise cosines of the five activation images, the analogue of
  FAChart.factor_chart_coords, so a near-collinear Gram-Schmidt would show up
  rather than hide inside a good R^2.
"""
import json
import os

import numpy as np
from scipy.stats import pearsonr

from fa_chart import FACTOR_ORDER, FAChart

Q = os.path.dirname(os.path.abspath(__file__))
PRIMARY_WINDOW = "resp"
LAYER = 16
NULLS = 500
rng = np.random.default_rng(0)
TITLES = ["Warmth", "Competence", "Fearful withdrawal", "Arousal", "Imagination"]


def gs_chart(C, G):
    """fa_chart.FAChart's Gram-Schmidt, on any Gram over the same 134 traits.

    Returns (basis 5xN, trait coords Nx5, the images' pairwise cosines 5x5).
    """
    B = []
    for k in range(len(C)):
        v = C[k].copy()
        for b in B:
            v = v - (b @ G @ v) * b
        q = v @ G @ v
        if q <= 0:
            raise ValueError(f"factor {FACTOR_ORDER[k]} is degenerate in this Gram (q={q:.3e})")
        B.append(v / np.sqrt(q))
    B = np.array(B)
    assert np.allclose(B @ G @ B.T, np.eye(len(C)), atol=1e-6), "basis is not orthonormal"
    img = C @ G @ C.T                                   # 5x5 inner products of the images
    d = np.sqrt(np.diag(img))
    return B, (B @ G).T, img / np.outer(d, d)


def procrustes_r2(A, B):
    """Fraction of B's variance explained by an orthogonal rotation of A (plus scale).

    Byte-identical to analyse_actspace.procrustes_r2, so the FA number and the PC
    number are the same statistic.
    """
    A = A - A.mean(0); B = B - B.mean(0)
    U, s, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    scale = s.sum() / (A ** 2).sum()
    resid = ((B - scale * A @ R) ** 2).sum()
    return float(1 - resid / (B ** 2).sum())


def double_centre(G):
    n = len(G)
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ G @ H


def main():
    ch = FAChart()
    out = {"layer": LAYER, "primary_window": PRIMARY_WINDOW,
           "factor_order": FACTOR_ORDER, "factor_titles": TITLES, "windows": {}}

    # the PC number this is the analogue of; quoted, never recomputed
    pg = f"{Q}/analysis/actspace_geometry.json"
    PC = json.load(open(pg))["windows"] if os.path.exists(pg) else {}

    for window in ("resp", "prompt"):
        gp = f"{Q}/results/gram_actspace_{window}_L{LAYER}.npz"
        if not os.path.exists(gp):
            print(f"skip {window}: {os.path.relpath(gp, Q)} not present")
            continue
        z = np.load(gp, allow_pickle=True)
        Ga = np.array(z["G"], dtype=float)
        traits = [str(t) for t in z["names"]]
        assert sorted(traits) == sorted(ch.names), "trait sets differ"
        w2a = [ch.names.index(t) for t in traits]        # gram_sweep order -> actspace order

        # the same five coefficient rows, in the activation Gram's trait order
        C = ch.factor_coef[:, w2a]
        Ba, Xa, cos_a = gs_chart(C, Ga)                  # activation chart
        Xw = ch.trait_coords[w2a]                        # weight chart, fa_chart's own
        cos_w = (ch.factor_chart_coords /
                 np.linalg.norm(ch.factor_chart_coords, axis=0))
        cos_w = cos_w.T @ cos_w

        # the matched variant: weight chart on the double-centred weight Gram, so
        # both sides are trait-centred
        Gwc = double_centre(ch.G)[np.ix_(w2a, w2a)]
        _, Xwc, cos_wc = gs_chart(C, Gwc)

        r2 = procrustes_r2(Xa, Xw)
        r2c = procrustes_r2(Xa, Xwc)
        null = [procrustes_r2(Xa[rng.permutation(len(traits))], Xw) for _ in range(NULLS)]
        per_factor = [float(pearsonr(Xa[:, k], Xw[:, k]).statistic) for k in range(5)]
        per_factor_c = [float(pearsonr(Xa[:, k], Xwc[:, k]).statistic) for k in range(5)]
        # a null for the per-factor correlations, same shuffle
        pf_null = np.array([[abs(pearsonr(Xa[rng.permutation(len(traits)), k], Xw[:, k]).statistic)
                             for k in range(5)] for _ in range(NULLS)])

        pcr2 = PC.get(window, {}).get("primary", {}).get("procrustes_r2")
        rec = {"traits": traits,
               "procrustes_r2_fa": r2,
               "procrustes_r2_fa_centred_weights": r2c,
               "procrustes_null_mean": float(np.mean(null)),
               "procrustes_null_95pct": float(np.percentile(null, 95)),
               "procrustes_r2_pc": pcr2,
               "per_factor_r": per_factor,
               "per_factor_r_centred_weights": per_factor_c,
               "per_factor_null_95pct": pf_null.mean(0).tolist(),
               "activation_image_cosines": cos_a.round(4).tolist(),
               "weight_factor_cosines": cos_w.round(4).tolist(),
               "activation_chart_len_mean": float(np.linalg.norm(Xa, axis=1).mean()),
               "activation_norm_mean": float(np.sqrt(np.diag(Ga)).mean()),
               "activation_chart_frac_of_norm_mean":
                   float(np.mean(np.linalg.norm(Xa, axis=1) / np.sqrt(np.diag(Ga)))),
               "weight_chart_frac_of_norm_mean":
                   float(ch.summary()["chart_captures_frac_of_norm_mean"]),
               "gram_path": os.path.relpath(gp, Q)}
        out["windows"][window] = rec

        print("=" * 78)
        print(f"WINDOW {window}, layer {LAYER}   ({os.path.relpath(gp, Q)})")
        print("=" * 78)
        print(f"  Procrustes R^2, 134x5 FACTOR-CHART coordinates: {r2:.3f}")
        print(f"    label-shuffle null: mean {np.mean(null):.3f}, 95th pct "
              f"{np.percentile(null, 95):.3f}  ({NULLS} draws)")
        print(f"    same statistic on PC scores (analyse_actspace.py): "
              f"{pcr2:.3f}" if pcr2 is not None else "    PC number not available")
        print(f"    with the weight Gram double-centred too: {r2c:.3f}")
        print("  per-factor Pearson r, matching columns, NO rotation:")
        for k in range(5):
            print(f"    {TITLES[k]:24s} r {per_factor[k]:+.3f}   "
                  f"(centred weights {per_factor_c[k]:+.3f}; "
                  f"shuffle |r| mean {pf_null.mean(0)[k]:.3f})")
        print("  pairwise cosines of the five activation images:")
        for k in range(5):
            print("    " + " ".join(f"{v:+.3f}" for v in cos_a[k]) + f"   {TITLES[k]}")
        print("  the same five in weight space:")
        for k in range(5):
            print("    " + " ".join(f"{v:+.3f}" for v in cos_w[k]) + f"   {TITLES[k]}")
        print(f"  chart captures, activations {rec['activation_chart_frac_of_norm_mean']*100:.1f}%"
              f" of a persona vector's norm; weights "
              f"{rec['weight_chart_frac_of_norm_mean']*100:.1f}% of an adapter's")

    json.dump(out, open(f"{Q}/analysis/actspace_geometry_fa.json", "w"), indent=1)
    print("\nwrote analysis/actspace_geometry_fa.json")


if __name__ == "__main__":
    main()
