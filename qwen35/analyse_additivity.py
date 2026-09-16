#!/usr/bin/env python3
"""Is the personality chart flat between the points we sampled?

Every geometry result so far validates the chart AT the 134 adapters, or at the
five axes built from them. The span is a linear object fitted to a convenience
sample, and nothing yet says a point in its INTERIOR behaves the way its
coordinates claim.

Additivity is the sharpest form of the question. If steering along axis A moves
the judged profile by dA, and axis B by dB, then a matched-norm combination of
the two should move it by dA + dB -- but only if the map from weights to
behaviour is linear over that region. The residual

    r = d(A+B) - (dA + dB)

is the deviation from flatness, measured in behaviour rather than inferred from
geometry. Its size relative to the effects themselves is the number that decides
whether "manifold" is doing any work beyond "subspace".

Two things keep this honest. The mixtures are renormalised to unit coefficient
norm, so they are not simply larger perturbations than the singles; and the
comparison is restricted to |alpha| <= 2, because at |alpha| = 4 the model loops
verbatim on most prompts and additivity of wreckage is not a finding.

The null: judge noise alone produces a non-zero residual. It is estimated from
the alpha = 0 rows, which are the same untouched model under every direction, so
their spread is pure measurement error.
"""
import collections
import json
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
F = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]


def load(path):
    """(direction, alpha) -> mean judged vector, and the per-prompt spread at alpha 0."""
    J = json.load(open(path))
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in J["records"]:
        a = float(r["condition"][1:].replace("m", "-").replace("_", "."))
        for f in F:
            v = r["scores"].get(f)
            if isinstance(v, (int, float)):
                agg[(r["trait"], a)][f].append(v)
    return {k: np.array([np.mean(s[f]) if s.get(f) else np.nan for f in F])
            for k, s in agg.items()}


def main():
    singles = load(f"{Q}/phase10_runs/judged_steerfix.json")
    mixes = load(f"{Q}/phase10_runs/judged_mix.json")
    spec = json.load(open(f"{Q}/phase10_runs/steer_spec_mix.json"))

    # alpha = 0 is the untouched model under every direction; its spread across
    # directions is judge noise with no signal in it at all.
    z = [v for (d, a), v in list(singles.items()) + list(mixes.items()) if a == 0.0]
    noise = np.nanstd(np.stack(z), 0)
    print(f"judge noise at alpha=0, per scale: " + "  ".join(f"{F[i][:4]} {noise[i]:.3f}"
                                                             for i in range(5)))
    print(f"  norm of that noise vector: {np.linalg.norm(noise):.3f}\n")

    rows = []
    print(f"{'mixture':20s} {'alpha':>6s} {'|observed|':>11s} {'|predicted|':>12s} "
          f"{'|residual|':>11s} {'resid/effect':>13s}")
    for job in spec["jobs"]:
        nm, w = job["name"], job["weights"]
        for a in (-2.0, -1.0, 1.0, 2.0):
            obs = mixes.get((nm, a))
            base = mixes.get((nm, 0.0))
            if obs is None or base is None:
                continue
            d_obs = obs - base
            # prediction: the same weighted sum of the single-axis effects, with
            # the mixture's own renormalisation applied so the scales match
            pred = np.zeros(5)
            ok = True
            for f, v in w.items():
                s1, s0 = singles.get((f"axis_{f}", a)), singles.get((f"axis_{f}", 0.0))
                if s1 is None or s0 is None:
                    ok = False
                    break
                pred += v * (s1 - s0)
            if not ok:
                continue
            pred /= max(np.linalg.norm(list(w.values())), 1e-9)
            r = d_obs - pred
            frac = np.linalg.norm(r) / max(np.linalg.norm(pred), 1e-9)
            rows.append({"mix": nm, "alpha": a, "obs": d_obs.tolist(),
                         "pred": pred.tolist(), "resid": r.tolist(),
                         "resid_over_effect": float(frac)})
            print(f"{nm:20s} {a:>+6.1f} {np.linalg.norm(d_obs):>11.3f} "
                  f"{np.linalg.norm(pred):>12.3f} {np.linalg.norm(r):>11.3f} {frac:>13.2f}")

    if rows:
        fr = np.array([r["resid_over_effect"] for r in rows])
        rn = np.array([np.linalg.norm(r["resid"]) for r in rows])
        print(f"\nresidual / predicted effect: median {np.median(fr):.2f}  "
              f"mean {fr.mean():.2f}  range {fr.min():.2f}-{fr.max():.2f}")
        print(f"residual norm vs judge noise: median {np.median(rn):.3f} "
              f"against noise {np.linalg.norm(noise):.3f}  "
              f"-> ratio {np.median(rn)/max(np.linalg.norm(noise),1e-9):.2f}")
        print("\nreading: a ratio near 1 means the deviation from additivity is "
              "indistinguishable\nfrom judge noise and the chart is flat over its "
              "interior. Well above 1 means\nthe interior is curved and coordinates do "
              "not compose.")
        json.dump({"rows": rows, "noise": noise.tolist(),
                   "median_resid_over_effect": float(np.median(fr)),
                   "median_resid_norm": float(np.median(rn)),
                   "noise_norm": float(np.linalg.norm(noise))},
                  open(f"{Q}/analysis/additivity.json", "w"), indent=1)
        print(f"\nwrote analysis/additivity.json")


if __name__ == "__main__":
    main()
