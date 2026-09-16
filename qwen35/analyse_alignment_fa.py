#!/usr/bin/env python3
"""Where do the alignment-relevant traits sit on the FACTOR chart?

The factor-chart twin of analyse_alignment.py, on the `aligncommon` arm -- the
four adapters (sycophantic, obsequious, power_seeking, corrigible) retrained on
the zoo's exact shared prompt pool, which is the arm whose angles compare
adapters that answered the same questions.

The gate comes first, unchanged: every angle here is a statement about adapters
sharing one 64-dimensional input window, which holds only if the new adapters
got the zoo's LoRA-A.  The zoo's own internal A drift is 0.0146.

The chart is fa_chart.FAChart.  Placing an external adapter needs the exact
cross-Gram column <a_i, b> over the 134; that did not exist, so it was computed
with the project's own cross_gram_full_on_modal.py into
results/cross_gram_full_root_x_pc-qwen35-adapters_data_alignment_common.npz
(and the 4x4 self-Gram into
results/cross_gram_full_data_alignment_common_x_data_alignment_common.npz).
Every angle below is exact; no sketch is used.

The four preregistered predictions (PREREG_alignment.md) are restated in the
factor frame where the PC frame's objects do not exist:

  1. sycophancy loads on warmth/agreeableness, not on the submissiveness
     component.  PC form: cos with axis_Agreeableness beats cos with PC4 and is
     positive.  FA form: FA_Warmth is the largest of the five factor cosines and
     is positive.  Both are reported.
  2. the synonym pair sycophantic/obsequious sits under 54 degrees (the zoo's
     closest pair, composed/imperturbable).
  3. power-seeking is over 60 degrees from all 134.
  4. corrigible is under 60.

Predictions 2-4 are chart-independent -- they are full-space angles -- so they
are recomputed here from the exact Gram rather than from sketches, and any
difference from the PC run is a sketch-versus-exact difference, not a chart one.
"""
import json
import os

import numpy as np

from analyse_alignment import gate
from fa_chart import FAChart

Q = os.path.dirname(os.path.abspath(__file__))
NEW = ["sycophantic", "obsequious", "power_seeking", "corrigible"]
F5 = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]
XG = f"{Q}/results/cross_gram_full_root_x_pc-qwen35-adapters_data_alignment_common.npz"
SG = f"{Q}/results/cross_gram_full_data_alignment_common_x_data_alignment_common.npz"


def main():
    gate("align_common_files")
    ch = FAChart()
    names = ch.names
    meta = {}
    for f in ("traits_primary.json", "traits_secondary.json"):
        p = f"{Q}/{f}"
        if os.path.exists(p):
            for r in json.load(open(p)):
                meta[r["trait"].lower().replace(" ", "_").replace("-", "_")] = (r["factor"], r["keyed"])

    AL = json.load(open(f"{Q}/analysis/alien_fa.json"))["alien_fa"]
    c_alien = np.array(AL["coeffs"])
    u = np.array(AL["u"])
    T = ch.trait_coords
    A = T / np.linalg.norm(T, axis=1, keepdims=True)

    z = np.load(XG, allow_pickle=True)
    assert [str(x) for x in z["names_a"]] == names, "cross-Gram row order differs"
    nb = [str(x) for x in z["names_b"]]
    X = np.array(z["X"], dtype=float)
    norm_b = np.array(z["norms_b"], dtype=float)
    col = {t: X[:, nb.index(t)] for t in NEW}
    nrm = {t: float(norm_b[nb.index(t)]) for t in NEW}

    zs = np.load(SG, allow_pickle=True)
    sn = [str(x) for x in zs["names_a"]]
    P = np.array(zs["X"], dtype=float)
    pn = np.array(zs["norms_a"], dtype=float)

    deg = lambda c: float(np.degrees(np.arccos(np.clip(abs(c), 0, 1))))

    # named directions as coefficient vectors over the 134
    D = {}
    for f in F5:
        p = np.array([1.0 if meta.get(t) == (f, "+") else 0.0 for t in names])
        m = np.array([1.0 if meta.get(t) == (f, "-") else 0.0 for t in names])
        D[f"axis_{f}"] = p / p.sum() - m / m.sum()
    for i, f in enumerate(ch.factor_names):
        D[f] = ch.factor_coef[i]
    D["personality_axis"] = np.ones(len(names)) / len(names)
    D["alien_fa"] = c_alien

    print(f"\nreference angles from the existing zoo (exact Gram): the widest hole on "
          f"the factor chart is {AL['gap_deg']:.1f} degrees in-chart and "
          f"{AL['full_gap_deg']:.1f} in the full space;\nthe zoo's closest pair is 54.0 "
          f"and two adapters at random are 83.3 apart (analysis/trait_angles.json)\n")

    print("ANGLE TO THE NEAREST OF THE 134, each treated as a line (exact Gram)")
    print(f"{'trait':16s} {'nearest':>16s} {'angle':>7s}   next two")
    res = {}
    for t in NEW:
        c = np.abs(col[t] / (ch.norms * nrm[t]))
        od = np.argsort(c)[::-1]
        res[t] = {"nearest": names[od[0]], "deg": deg(c[od[0]]),
                  "next": [(names[i], deg(c[i])) for i in od[1:3]]}
        print(f"{t:16s} {names[od[0]]:>16s} {res[t]['deg']:>6.1f}d   "
              + ", ".join(f"{n} ({d:.0f}d)" for n, d in res[t]["next"]))

    print("\nCOSINE WITH NAMED DIRECTIONS (full adapter space, exact Gram)")
    cols = ["axis_Agreeableness", "FA_Warmth", "FA_Competence", "FA_FearfulWithdrawal",
            "FA_Arousal", "FA_Imagination", "personality_axis", "alien_fa"]
    print(f"{'trait':16s} " + " ".join(f"{c[:13]:>14s}" for c in cols))
    for t in NEW:
        cs = {}
        for k in D:
            cs[k] = float(col[t] @ D[k] / (ch.norm(D[k]) * nrm[t]))
        res[t]["cos"] = cs
        print(f"{t:16s} " + " ".join(f"{cs[c]:>+14.3f}" for c in cols))

    print("\nFACTOR CHART: orthonormal basis coordinates of the unit adapter, "
          "then oblique factor coordinates")
    print(f"{'trait':16s} " + " ".join(f"{s[3:8]:>8s}" for s in ch.factor_names)
          + "   |chart|  x/trait   to u   to alien_fa (full)")
    Fm = ch.factor_chart_coords                      # column f = factor f in basis coords
    for t in NEW:
        x = ch.coords_external(col[t])
        xu = x / nrm[t]                              # unit-normalised adapter
        a = x / np.linalg.norm(x)
        c_u = float(a @ u)
        c_v = res[t]["cos"]["alien_fa"]
        ob = np.linalg.solve(Fm, xu)
        res[t]["chart"] = xu.tolist()
        res[t]["chart_oblique"] = ob.tolist()
        res[t]["chart_len"] = float(np.linalg.norm(x))
        res[t]["frac_in_chart"] = float(np.linalg.norm(x) / nrm[t])
        res[t]["chart_len_frac_of_trait_mean"] = float(
            np.linalg.norm(x) / ch.trait_chart_len.mean())
        res[t]["deg_to_u_chart"] = deg(c_u)
        res[t]["sign_u"] = int(np.sign(c_u))
        res[t]["deg_to_alien_fa_full"] = deg(c_v)
        print(f"{t:16s} " + " ".join(f"{y:>+8.3f}" for y in xu)
              + f"   {res[t]['frac_in_chart']:.4f}  "
              f"{res[t]['chart_len_frac_of_trait_mean']:.2f}x "
              f"{deg(c_u):>6.1f}d {deg(c_v):>10.1f}d")
        print(f"{'  oblique':16s} " + " ".join(f"{y:>+8.3f}" for y in ob))

    # Attribution check.  The PC page's angles are computed on adapters CENTRED on
    # the zoo mean, in the sketch space; the factor chart is defined on raw
    # adapters and the Gram is exact.  Recomputing the nearest-of-134 angle on
    # exactly-centred vectors separates the two causes of any difference: what is
    # left after centring is sketch error (about 0.3 degrees), the rest is centring.
    G = ch.G
    gm = G.mean(1)
    gmm = float(G.mean())
    cen = {}
    for t in NEW:
        cb = col[t] - gm - float(col[t].mean()) + gmm
        na = np.sqrt(np.diag(G) - 2 * gm + gmm)
        nbn = float(np.sqrt(nrm[t] ** 2 - 2 * col[t].mean() + gmm))
        c = np.abs(cb / (na * nbn))
        j = int(np.argmax(c))
        cen[t] = {"deg": deg(c[j]), "nearest": names[j]}
        print(f"  centred-exact check  {t:16s} {cen[t]['deg']:6.1f}d ({cen[t]['nearest']})")
    res["_centred_exact"] = cen

    pair = deg(P[sn.index("sycophantic"), sn.index("obsequious")]
               / (pn[sn.index("sycophantic")] * pn[sn.index("obsequious")]))
    print(f"\nsycophantic / obsequious: {pair:.1f} degrees apart, exact "
          f"(the sketch figure on the PC page is 61.9; zoo's closest pair is 54.0)")
    res["_pair_deg"] = pair

    # chance level for landing that close to u, same analytic form as the hole page
    def p_within(theta_deg):
        c = np.cos(np.radians(theta_deg))
        return ((1 - c) - (1 - c ** 3) / 3) / (2 / 3)

    best = min(res[t]["deg_to_u_chart"] for t in NEW)
    res["_p_one"] = p_within(best)
    res["_p_any_of_four"] = 1 - (1 - p_within(best)) ** 4
    res["_hole_nearest_existing_deg"] = AL["gap_deg"]
    res["_hole_nearest_existing_full_deg"] = AL["full_gap_deg"]
    res["_trait_chart_len_mean"] = float(ch.trait_chart_len.mean())
    print(f"\nnearest of the four to the hole in the chart: {best:.1f} degrees; chance "
          f"for one random 5-d direction {p_within(best)*100:.1f}%, for at least one of "
          f"four {res['_p_any_of_four']*100:.1f}%")

    print("\nPREDICTIONS (factor frame)")
    s = res["sycophantic"]["cos"]
    fac = {f: s[f] for f in ch.factor_names}
    top = max(fac, key=lambda k: fac[k])
    p1_fa = top == "FA_Warmth" and fac["FA_Warmth"] > 0
    p1_pc = s["axis_Agreeableness"] > 0
    print(f"  1 sycophancy on warmth/agreeableness    : {'HELD' if p1_fa else 'FAILED'} "
          f"(largest factor cosine {top} {fac[top]:+.3f}; axis_Agreeableness "
          f"{s['axis_Agreeableness']:+.3f})")
    print(f"  2 synonym pair under 54 degrees         : "
          f"{'HELD' if pair < 54 else 'FAILED'} ({pair:.1f})")
    print(f"  3 power-seeking over 60 from all 134    : "
          f"{'HELD' if res['power_seeking']['deg'] > 60 else 'FAILED'} "
          f"({res['power_seeking']['deg']:.1f})")
    print(f"  4 corrigible under 60                   : "
          f"{'HELD' if res['corrigible']['deg'] < 60 else 'FAILED'} "
          f"({res['corrigible']['deg']:.1f})")
    res["_predictions"] = {"1_sycophancy_on_warmth_fa": bool(p1_fa),
                           "1_sycophancy_agreeableness_positive": bool(p1_pc),
                           "1_top_factor": top,
                           "2_pair_under_54": bool(pair < 54),
                           "3_power_seeking_over_60": bool(res["power_seeking"]["deg"] > 60),
                           "4_corrigible_under_60": bool(res["corrigible"]["deg"] < 60)}
    json.dump(res, open(f"{Q}/analysis/alignment_geometry_fa.json", "w"), indent=1)
    print("\nwrote analysis/alignment_geometry_fa.json")


if __name__ == "__main__":
    main()
