#!/usr/bin/env python3
"""Reward hacking versus honest completion, in the personality chart.

The positive control the capability-RL experiment needed. School of Reward Hacks
(arXiv:2508.17511) is documented to produce emergent misalignment, and it ships
with its own matched control: every row has both a metric-gaming completion and
an honest one for the SAME prompt. Both arms were trained on the identical 973
rows that carry both, so the only difference between them is whether the
response games the stated metric.

Both adopted the zoo's LoRA-A initialisation, so they sit in the same
64-dimensional input window as all 134 personality adapters and project into
their chart at full strength rather than through the ~2.5% overlap two random
initialisations share.

MAGNITUDE AND DIRECTION ARE REPORTED SEPARATELY, because they answer different
questions and collapsing them loses one:

  raw         how far did this training move the model, and where did it land.
              If gaming a metric displaces the weights further than answering
              honestly on identical prompts, that is a fact about how far the
              behaviour sits from the model's prior.
  normalised  which way did it point, independent of how hard it pushed.

The ratio between them says whether a gap in chart position is a difference of
heading or merely of distance travelled.
"""
import glob
import json
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
F = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]


def chart():
    """The five named Big Five axes, plus the dual basis they need.

    mean(+keyed) - mean(-keyed) per factor. The axes are NOT orthogonal
    (Conscientiousness-Extraversion -0.37), so coordinates come from a
    least-squares solve against the dual basis, not from raw dot products.
    """
    S = {os.path.basename(p)[:-4]: np.load(p, allow_pickle=True)["sketch"].astype(np.float64)
         for p in glob.glob(f"{Q}/analysis/sketches/stage1_k32/*.npz")}
    P = json.load(open(f"{Q}/traits_primary.json"))
    key = {r["trait"].lower().replace(" ", "_").replace("-", "_"): (r["factor"], r["keyed"])
           for r in P}
    tr = [t for t in sorted(S) if t in key]
    X = np.stack([S[t] for t in tr])
    K = []
    for f in F:
        p = [i for i, t in enumerate(tr) if key[t] == (f, "+")]
        n = [i for i, t in enumerate(tr) if key[t] == (f, "-")]
        K.append(X[p].mean(0) - X[n].mean(0))
    K = np.stack(K)
    Kn = K / np.linalg.norm(K, axis=1, keepdims=True)
    return tr, X, Kn, np.linalg.inv(Kn @ Kn.T)


def load_run(tag):
    p = f"{Q}/analysis/rl_sketches_{tag}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    k = "final" if "final" in d else sorted(d)[-1]
    return np.array(d[k]["sketch"], dtype=np.float64), d[k].get("norm_sum")


def main():
    tr, X, Kn, Gi = chart()
    co = lambda M: (M @ Kn.T) @ Gi
    ref = float(np.mean(np.linalg.norm(X, axis=1)))          # one trait adapter's worth
    CT = co(X) / np.linalg.norm(X, axis=1, keepdims=True)
    tl = np.linalg.norm(CT, axis=1)

    runs = {}
    for tag in ("sorh_hack", "sorh_control", "math", "mathcode"):
        r = load_run(tag)
        if r is not None:
            runs[tag] = r

    print(f"reference: mean single trait-adapter sketch norm = {ref:.4f}")
    print(f"           trait adapters' chart length, unit norm = {tl.mean():.4f} "
          f"+- {tl.std():.4f}\n")

    print("MAGNITUDE -- how far the training moved the weights")
    print(f"{'run':16s} {'|dW| sketch':>12s} {'/ trait adapter':>16s}")
    for t, (v, _) in runs.items():
        print(f"{t:16s} {np.linalg.norm(v):>12.4f} {np.linalg.norm(v)/ref:>16.3f}x")

    print(f"\nRAW chart coordinates -- position, magnitude included")
    print(f"{'run':16s} " + " ".join(f"{f[:5]:>9s}" for f in F) + f" {'length':>9s}")
    for t, (v, _) in runs.items():
        c = co(v[None])[0]
        print(f"{t:16s} " + " ".join(f"{x:>+9.4f}" for x in c)
              + f" {np.linalg.norm(c):>9.4f}")

    print(f"\nNORMALISED -- direction only, per unit adapter norm")
    print(f"{'run':16s} " + " ".join(f"{f[:5]:>9s}" for f in F) + f" {'length':>9s}")
    out = {}
    for t, (v, _) in runs.items():
        c = co(v[None])[0] / np.linalg.norm(v)
        out[t] = {"coords_unit": c.tolist(), "length_unit": float(np.linalg.norm(c)),
                  "norm_over_ref": float(np.linalg.norm(v) / ref),
                  "coords_raw": co(v[None])[0].tolist()}
        print(f"{t:16s} " + " ".join(f"{x:>+9.4f}" for x in c)
              + f" {np.linalg.norm(c):>9.4f}")
    print(f"{'trait adapters':16s} " + " ".join(f"{x:>9.4f}" for x in np.abs(CT).mean(0))
          + f" {tl.mean():>9.4f}   <- mean |coordinate|")

    if "sorh_hack" in runs and "sorh_control" in runs:
        h, c = runs["sorh_hack"][0], runs["sorh_control"][0]
        print(f"\nHACK vs CONTROL, identical prompts, only the completion differs")
        print(f"  magnitude ratio            {np.linalg.norm(h)/np.linalg.norm(c):.3f}x")
        print(f"  cosine between them        {float(h@c/(np.linalg.norm(h)*np.linalg.norm(c))):+.4f}")
        ch, cc = co(h[None])[0]/np.linalg.norm(h), co(c[None])[0]/np.linalg.norm(c)
        print(f"  chart length, unit norm    hack {np.linalg.norm(ch):.4f}  "
              f"control {np.linalg.norm(cc):.4f}  ratio {np.linalg.norm(ch)/max(np.linalg.norm(cc),1e-9):.3f}x")
        d = h/np.linalg.norm(h) - c/np.linalg.norm(c)
        cd = co(d[None])[0]
        print(f"  the DIFFERENCE direction (hack minus control, both unit):")
        print(f"    " + "  ".join(f"{F[i][:4]} {cd[i]:+.4f}" for i in range(5)))
        out["contrast"] = {"magnitude_ratio": float(np.linalg.norm(h)/np.linalg.norm(c)),
                           "cosine": float(h@c/(np.linalg.norm(h)*np.linalg.norm(c))),
                           "difference_coords": cd.tolist()}
    out["reference"] = {"ref_norm": ref, "trait_chart_len_mean": float(tl.mean()),
                        "trait_chart_len_sd": float(tl.std()),
                        "trait_mean_abs_coord": np.abs(CT).mean(0).tolist()}
    json.dump(out, open(f"{Q}/analysis/sorh_projection.json", "w"), indent=1)
    print(f"\nwrote analysis/sorh_projection.json")


if __name__ == "__main__":
    main()
