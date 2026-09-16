#!/usr/bin/env python3
"""Did the optimised data actually train the model where it was aimed?

The search that produced the data optimises a first-order quantity: the very
first gradient step, taken at LoRA initialisation. Whether that survives real
training is a separate question, and the only way to answer it is to train.

Four adapters, all on 64 preference pairs balanced across the same eight
questions, all for the same 16 optimizer steps. Three were selected to point at
a direction; the fourth was drawn at random from the same candidate pool.

WHY THE CONTROL IS SUBTRACTED
-----------------------------
Every arm shares an enormous common-mode signal. The rejected half of each pair
is the base model's own greedy answer, so "stop producing your own default" is
an easy, strong gradient that all four arms receive equally -- the DPO margins
saturate above 30 nats within ten steps and accuracy hits 1.0. Comparing an arm
against zero would mostly measure that shared component.

The random-selection arm is the same data, the same prompts, the same number of
steps and the same saturation, differing only in that its responses were not
chosen for anything. Subtracting it removes the common mode and leaves what
selection contributed. Both matrices are reported, because the raw one is what
a reader would compute first and it is honest to show why it does not answer
the question.

PRE-REGISTERED: success is the control-subtracted matrix having its largest
entry on the diagonal for each arm -- each arm closest to the direction it was
selected for, and closer to it than the other arms are.
"""
import glob
import json
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
F5 = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]
ARMS = [("opt_alien", "alien_k5"), ("opt_agree", "axis_Agreeableness"), ("opt_pc4", "PC4")]


def targets():
    AL = json.load(open(f"{Q}/analysis/alien.json"))
    names = AL["alien_k5"]["traits"]
    meta = {}
    for f in ("traits_primary.json", "traits_secondary.json"):
        p = f"{Q}/{f}"
        if os.path.exists(p):
            for r in json.load(open(p)):
                meta[r["trait"].lower().replace(" ", "_").replace("-", "_")] = (r["factor"], r["keyed"])
    X = np.stack([np.load(f"{Q}/analysis/sketches/stage1_k32/{t}.npz",
                          allow_pickle=True)["sketch"].astype(np.float64) for t in names])
    Xc = X - X.mean(0)
    w, V = np.linalg.eigh(Xc @ Xc.T)
    o = np.argsort(w)[::-1]
    w, V = np.maximum(w[o], 1e-12), V[:, o]
    T = {}
    p = [i for i, t in enumerate(names) if meta[t] == ("Agreeableness", "+")]
    m = [i for i, t in enumerate(names) if meta[t] == ("Agreeableness", "-")]
    v = X[p].mean(0) - X[m].mean(0)
    T["axis_Agreeableness"] = v / np.linalg.norm(v)
    v = Xc.T @ (V[:, 3] / np.sqrt(w[3]))
    T["PC4"] = v / np.linalg.norm(v)
    T["alien_k5"] = np.load(f"{Q}/analysis/alien_v_k5.npy")
    return T


def main():
    T = targets()
    S = {}
    for a, _ in ARMS + [("opt_random", None)]:
        p = f"{Q}/analysis/sketches/optimised_k32/{a}.npz"
        if not os.path.exists(p):
            raise SystemExit(f"missing sketch {p} -- run sketch_adapters.py --source optimised")
        S[a] = np.load(p, allow_pickle=True)["sketch"].astype(np.float64)
    ctrl = S["opt_random"]
    tn = [t for _, t in ARMS]

    def cos(u, v):
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))

    print("RAW cosine, each adapter against each target direction")
    print(f"{'adapter':14s} " + " ".join(f"{t[:16]:>17s}" for t in tn))
    raw = {}
    for a, _ in ARMS + [("opt_random", None)]:
        raw[a] = [cos(S[a], T[t]) for t in tn]
        print(f"{a:14s} " + " ".join(f"{x:>+17.4f}" for x in raw[a]))
    print("\n  the four rows are near-identical: that is the shared "
          "'unlearn your own default' component, not selection.\n")

    print("CONTROL-SUBTRACTED, (adapter - random arm) against each target")
    print(f"{'adapter':14s} " + " ".join(f"{t[:16]:>17s}" for t in tn) + "   own target")
    sub, ok = {}, 0
    for a, own in ARMS:
        d = S[a] - ctrl
        sub[a] = [cos(d, T[t]) for t in tn]
        best = tn[int(np.argmax(sub[a]))]
        hit = best == own
        ok += hit
        print(f"{a:14s} " + " ".join(f"{x:>+17.4f}" for x in sub[a])
              + f"   {'HIT' if hit else 'miss (' + best + ')'}")
    print(f"\ndiagonal dominance: {ok}/3 arms closest to the direction they were "
          f"selected for")

    # how big is the selected component next to the shared one?
    frac = {a: float(np.linalg.norm(S[a] - ctrl) / np.linalg.norm(S[a])) for a, _ in ARMS}
    print("\nselection's share of each adapter's update (||arm - random|| / ||arm||):")
    for a, _ in ARMS:
        print(f"  {a:14s} {frac[a]*100:5.1f}%")

    json.dump({"raw": raw, "sub": sub, "targets": tn, "hits": ok,
               "frac": frac}, open(f"{Q}/analysis/verify.json", "w"), indent=1)
    print("\nwrote analysis/verify.json")


if __name__ == "__main__":
    main()
