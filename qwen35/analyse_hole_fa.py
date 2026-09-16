#!/usr/bin/env python3
"""Do the externally suggested names for the hole land in the hole -- on the FACTOR chart?

The factor-chart twin of analyse_hole.py.  Same three adapters (cavalier, blase,
insouciant, trained on the zoo's shared prompt pool at the matched objective),
same gate, same questions, same verdict language.  What changes is the chart:
fa_chart.FAChart, the orthonormal basis of the span of the five oblimin factor
directions, and analysis/alien_fa.json#alien_fa for the hole itself.

One thing had to be built rather than read.  analyse_hole.py works in the
253,952-dimensional sketch space, so it could place a new adapter from its
sketch alone.  The factor chart is defined on the EXACT Gram
(results/gram_sweep.npz) and placing an external adapter b needs the exact
cross-Gram column <a_i, b> over the 134.  That did not exist, so it was computed
with the project's own cross_gram_full_on_modal.py against the zoo root on
pc-qwen35-sweep, giving
results/cross_gram_full_root_x_pc-qwen35-adapters_data_hole_common.npz.
Every angle below is exact; no sketch is used anywhere.  (As a check, the
cosines from that file agree with the sketch cosines to a mean of 0.005.)

  1. GATE  do they share the zoo's LoRA-A window (drift ~0.015, not ~1.0)?
  2. in the five-factor chart, the angle from each new adapter to the hole u,
     against the 54.8 degrees the nearest existing adapter manages.
  3. in the full adapter space, the angle to the hole direction and to the
     nearest of the 134, both from the exact Gram.
  4. the three pairwise angles.
  5. the analytic chance level for one 5-d direction landing that close.
"""
import json
import os

import numpy as np

from analyse_hole import gate
from fa_chart import FAChart

Q = os.path.dirname(os.path.abspath(__file__))
NEW = ["cavalier", "blase", "insouciant"]
XG = f"{Q}/results/cross_gram_full_root_x_pc-qwen35-adapters_data_hole_common.npz"
K = 5


def main():
    gate()
    ch = FAChart()
    names = ch.names
    AL = json.load(open(f"{Q}/analysis/alien_fa.json"))["alien_fa"]
    assert AL["traits"] == names, "adapter order differs from the chart"
    c_alien = np.array(AL["coeffs"])                       # unit norm in G
    u = np.array(AL["u"])                                  # unit 5-vector in the chart
    T = ch.trait_coords
    A = T / np.linalg.norm(T, axis=1, keepdims=True)

    z = np.load(XG, allow_pickle=True)
    assert [str(x) for x in z["names_a"]] == names, "cross-Gram row order differs"
    nb = [str(x) for x in z["names_b"]]
    X = np.array(z["X"], dtype=float)                      # 134 x len(nb)
    norm_b = np.array(z["norms_b"], dtype=float)
    col = {t: X[:, nb.index(t)] for t in NEW}
    nrm = {t: float(norm_b[nb.index(t)]) for t in NEW}

    deg = lambda c: float(np.degrees(np.arccos(np.clip(abs(c), 0, 1))))
    nearest_existing = min(deg(A[i] @ u) for i in range(len(names)))
    full_nearest = min(deg(v) for v in (ch.G @ c_alien) / (ch.norm(c_alien) * ch.norms))
    print(f"\nHOLE at k={K} on the factor chart: nearest existing adapter line is "
          f"{nearest_existing:.1f} degrees from u ({AL['nearest'][0]['trait']}); "
          f"in the full adapter space {full_nearest:.1f} ({AL['full_nearest']})")
    print(f"a trait adapter's mean chart length is {ch.trait_chart_len.mean():.4f} "
          f"and it keeps {np.mean(ch.trait_chart_len / ch.norms):.2f} of its norm "
          f"inside the chart")

    out = {}
    print(f"\n{'trait':12s} {'to u (chart)':>13s} {'in-chart':>9s} {'chart len':>10s} "
          f"{'x/trait':>8s} {'to v (full)':>12s} {'nearest of 134':>22s}")
    for t in NEW:
        x = ch.coords_external(col[t])                     # 5-vector
        frac = float(np.linalg.norm(x) / nrm[t])           # cosine with the chart
        a = x / np.linalg.norm(x)
        c_u = float(a @ u)
        c_v = float(col[t] @ c_alien / (ch.norm(c_alien) * nrm[t]))
        cs = np.abs(col[t] / (ch.norms * nrm[t]))
        j = int(np.argmax(cs))
        out[t] = {"deg_to_u_chart": deg(c_u), "sign_u": int(np.sign(c_u)),
                  "frac_in_chart": frac,
                  "chart_len": float(np.linalg.norm(x)),
                  "chart_len_frac_of_trait_mean":
                      float(np.linalg.norm(x) / ch.trait_chart_len.mean()),
                  "deg_to_v_full": deg(c_v),
                  "nearest": names[j], "deg_nearest": deg(cs[j]),
                  "chart_coords": x.tolist(), "chart_unit": a.tolist(),
                  "norm": nrm[t]}
        print(f"{t:12s} {deg(c_u):>11.1f}d{'+' if c_u > 0 else '-'} {frac:>8.2f} "
              f"{np.linalg.norm(x):>10.4f} {out[t]['chart_len_frac_of_trait_mean']:>7.2f}x "
              f"{deg(c_v):>11.1f}d {names[j]:>14s} {deg(cs[j]):>5.1f}d")

    # Attribution check: the PC page centres on the zoo mean and works in the
    # sketch space; the factor chart is raw and the Gram is exact.  Recomputing
    # the nearest-of-134 angle on exactly-centred vectors separates the two.
    G = ch.G
    gm = G.mean(1)
    gmm = float(G.mean())
    na = np.sqrt(np.diag(G) - 2 * gm + gmm)
    cen = {}
    for t in NEW:
        cb = col[t] - gm - float(col[t].mean()) + gmm
        nbn = float(np.sqrt(nrm[t] ** 2 - 2 * col[t].mean() + gmm))
        cc = np.abs(cb / (na * nbn))
        j = int(np.argmax(cc))
        cen[t] = {"deg": deg(cc[j]), "nearest": names[j]}
        print(f"  centred-exact check  {t:12s} {cen[t]['deg']:6.1f}d ({cen[t]['nearest']})")
    out["_centred_exact"] = cen

    print("\nPAIRWISE, full space (sycophantic/obsequious was 61.9; zoo's closest pair 54.0)")
    PW = f"{Q}/results/cross_gram_full_data_hole_common_x_data_hole_common.npz"
    zp = np.load(PW, allow_pickle=True) if os.path.exists(PW) else None
    pw = {}
    if zp is not None:
        pn = [str(x) for x in zp["names_a"]]
        P = np.array(zp["X"], dtype=float)
        pnorm = np.array(zp["norms_a"], dtype=float)
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = NEW[i], NEW[j]
                d = deg(P[pn.index(a), pn.index(b)] / (pnorm[pn.index(a)] * pnorm[pn.index(b)]))
                pw[f"{a}/{b}"] = d
                print(f"  {a:11s} {b:11s} {d:5.1f}d")
    else:
        # no exact hole x hole Gram was computed; the PC run's sketch pairwise
        # angles stand unchanged, and are quoted rather than recomputed
        pw = json.load(open(f"{Q}/analysis/hole_geometry.json"))["_pairwise"]
        for k_, v in pw.items():
            print(f"  {k_:26s} {v:5.1f}d   (from analysis/hole_geometry.json, sketch space)")
    out["_pairwise"] = pw
    out["_pairwise_source"] = ("exact cross-Gram" if zp is not None
                               else "analysis/hole_geometry.json#_pairwise (sketch space)")
    out["_hole_nearest_existing_deg"] = nearest_existing
    out["_hole_nearest_existing_full_deg"] = full_nearest
    out["_trait_chart_len_mean"] = float(ch.trait_chart_len.mean())

    # Chance level, exactly as the PC version defines it: for a direction drawn
    # uniformly in K dims, |cos| to a fixed line has density proportional to
    # (1 - c^2)^((K-3)/2); at K=5 that is (1 - c^2), so
    #   P(|cos| > c) = [(1 - c) - (1 - c^3)/3] / (2/3).
    def p_within(theta_deg):
        c = np.cos(np.radians(theta_deg))
        return ((1 - c) - (1 - c ** 3) / 3) / (2 / 3)

    print("\nVERDICT")
    best = min(out[t]["deg_to_u_chart"] for t in NEW)
    p1 = p_within(best)
    print(f"  chance that ONE random 5-d direction falls within {best:.1f} degrees of u: "
          f"{p1*100:.1f}%; that at least one of THREE does: {(1 - (1 - p1) ** 3) * 100:.1f}%")
    out["_p_one"] = p1
    out["_p_any_of_three"] = 1 - (1 - p1) ** 3
    best_full = min(out[t]["deg_to_v_full"] for t in NEW)
    if best < nearest_existing - 10 and full_nearest > best_full:
        v = "the hole has a name"
    elif best < nearest_existing - 10:
        v = "suggestive in the chart, unconfirmed in full"
    elif best < nearest_existing:
        v = "marginal"
    else:
        v = "the names miss"
    out["_verdict"] = v
    print(f"  {v}: closest new adapter is {best:.1f} degrees from u against "
          f"{nearest_existing:.1f} for anything that existed; in the full space the "
          f"best new adapter is {best_full:.1f} against {full_nearest:.1f} for the "
          f"nearest of the 134")
    json.dump(out, open(f"{Q}/analysis/hole_geometry_fa.json", "w"), indent=1)
    print("\nwrote analysis/hole_geometry_fa.json")


if __name__ == "__main__":
    main()
