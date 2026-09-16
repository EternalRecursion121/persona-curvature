#!/usr/bin/env python3
"""Dose-response: does moving along a weight-space direction move judged behaviour?

For each direction we have judged Big Five scores at each alpha. Two questions:
  MONOTONICITY -- does the judged score move consistently with alpha? Spearman
    rho over the alpha grid, which does not assume linearity.
  SELECTIVITY  -- does it move the factor it is supposed to, or all five? A
    direction that shifts every factor is changing fluency or verbosity, not
    personality. This is the same damage-vs-content test used on the adapters.

The alpha=0 row is the same model for every direction, so any difference between
directions at alpha=0 is judge noise and sets the floor.
"""
import json, sys, collections
import numpy as np

Q = "/home/vibe12/projects/persona-curvature/qwen35"
F = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]
SHORT = {f: f[:5] for f in F}


def load(path):
    J = json.load(open(path))
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in J["records"]:
        for f in F:
            v = r["scores"].get(f)
            if isinstance(v, (int, float)):
                agg[(r["trait"], r["condition"])][f].append(v)
    return agg


def alpha_of(cond):
    s = cond[1:].replace("m", "-").replace("_", ".")
    return float(s)


def main(path):
    agg = load(path)
    dirs = sorted({k[0] for k in agg})
    print(f"{len(dirs)} directions\n")
    for d in dirs:
        conds = sorted([k[1] for k in agg if k[0] == d], key=alpha_of)
        al = np.array([alpha_of(c) for c in conds])
        M = np.array([[np.mean(agg[(d, c)][f]) if agg[(d, c)].get(f) else np.nan
                       for f in F] for c in conds])
        base = M[np.argmin(np.abs(al))]
        rho = []
        for k in range(5):
            v = M[:, k]
            ok = ~np.isnan(v)
            r = np.corrcoef(np.argsort(np.argsort(al[ok])),
                            np.argsort(np.argsort(v[ok])))[0, 1] if ok.sum() > 3 else np.nan
            rho.append(r)
        rng = M.max(0) - M.min(0)
        best = int(np.nanargmax(np.abs(rho)))
        print(f"{d:24s} strongest: {F[best]:18s} rho={rho[best]:+.3f}  range={rng[best]:.2f}")
        print(f"{'':24s} " + "  ".join(f"{SHORT[f]} {rho[i]:+.2f}/{rng[i]:.1f}" for i, f in enumerate(F)))
    print("\n(rho = Spearman of judged score vs alpha; range = max-min judged score over the alpha grid)")
    print("A direction that encodes a trait shows ONE large rho. Uniform rho across all five = fluency effect.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else f"{Q}/phase10_runs/judged_steer.json")
