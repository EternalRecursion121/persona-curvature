#!/usr/bin/env python3
"""Do the externally suggested names for the hole land in the hole?

analyse_alien.py found the direction in the top-5 PC subspace that is furthest
from every adapter line (52.5 degrees at k=5; the 68.9-degree figure on the page
is the same construction in the full sketch space).  Reviewing the page, an external reviewer
suggested it might simply be "cavalier / blase / insouciant" -- a disposition the
Big Five lexicon lacks a word for.  Three adapters were trained on those words,
on the zoo's shared prompt pool at the matched objective.  This script asks:

  1. GATE  do they share the zoo's LoRA-A window (drift ~0.015, not ~1.0)?
  2. in the k=5 PC subspace, the angle from each new adapter to the hole u,
     against the 52.5 degrees the nearest existing adapter manages.  If a new
     adapter is much closer than 52.5, the hole had a name.
  3. in the full sketch space, the angle to v_sketch (the hole as a direction
     over raw deltas) and to the nearest of the 134.
  4. the three pairwise angles -- a second within-batch synonym floor to set
     beside sycophantic/obsequious (61.9).

Same coordinates as analyse_alien.py: centre on the ZOO mean, PCs from the zoo's
Gram only (the new adapters do not move the basis), rows normalised after
projection exactly as A was.
"""
import glob
import json
import os
import sys

import numpy as np
from safetensors import safe_open

Q = os.path.dirname(os.path.abspath(__file__))
NEW = ["cavalier", "blase", "insouciant"]
FILES = f"{Q}/hole_files"
K = 5


def gate():
    z = f"{FILES}/_zoo_bold.safetensors"
    with safe_open(z, framework="np") as f:
        keys = [k for k in f.keys() if ".lora_A." in k]
        A0 = {k: f.get_tensor(k).astype(np.float64) for k in keys}
    print("GATE -- LoRA-A drift against the zoo's A_0 (zoo's own figure 0.0146)")
    worst = 0.0
    for t in NEW:
        with safe_open(f"{FILES}/{t}.safetensors", framework="np") as f:
            d = np.mean([np.linalg.norm(f.get_tensor(k).astype(np.float64) - A0[k])
                         / np.linalg.norm(A0[k]) for k in keys])
        worst = max(worst, d)
        print(f"  {t:12s} {d:.4f}   {'same window' if d < 0.2 else 'DIFFERENT WINDOW'}")
    if worst >= 0.2:
        sys.exit("GATE FAILED")


def main():
    gate()
    meta = {}
    for f in ("traits_primary.json", "traits_secondary.json"):
        for r in json.load(open(f"{Q}/{f}")):
            meta[r["trait"].lower().replace(" ", "_").replace("-", "_")] = (r["factor"], r["keyed"])
    AL = json.load(open(f"{Q}/analysis/alien.json"))["alien_k5"]
    names = AL["traits"]
    X = np.stack([np.load(f"{Q}/analysis/sketches/stage1_k32/{t}.npz",
                          allow_pickle=True)["sketch"].astype(np.float64) for t in names])
    mu = X.mean(0)
    Xc = X - mu
    w, V = np.linalg.eigh(Xc @ Xc.T)
    o = np.argsort(w)[::-1]
    w, V = np.maximum(w[o], 1e-12), V[:, o]
    u = np.array(AL["u"])
    # PC_j as a unit vector in sketch space; the projection of any centred
    # vector onto the top-k PCs is then y_j = <x, PC_j>.  For the zoo's own rows
    # this reproduces Z = V sqrt(w) exactly.
    P = (Xc.T @ (V[:, :K] / np.sqrt(w[:K])))           # (D, K), unit columns
    Z = Xc @ P
    assert np.allclose(Z, V[:, :K] * np.sqrt(w[:K]), atol=1e-6)
    A = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    v_sketch = P @ u
    v_sketch /= np.linalg.norm(v_sketch)
    U = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)

    S = {t: np.load(f"{Q}/analysis/sketches/hole_k32/{t}.npz",
                    allow_pickle=True)["sketch"].astype(np.float64) - mu for t in NEW}
    deg = lambda c: float(np.degrees(np.arccos(np.clip(abs(c), 0, 1))))
    nearest_existing = min(deg(A[i] @ u) for i in range(len(names)))
    print(f"\nHOLE at k={K}: nearest existing adapter line is {nearest_existing:.1f} degrees "
          f"from u ({AL['nearest'][0]['trait']})")

    out = {}
    print(f"\n{'trait':12s} {'to u (k=5)':>11s} {'in-plane':>9s} {'to v (full)':>12s} {'nearest of 134':>22s}")
    for t in NEW:
        x = S[t]
        z = x @ P
        frac = float(np.linalg.norm(z) / np.linalg.norm(x))     # how much of it lives in the top-5
        a = z / np.linalg.norm(z)
        c_u = float(a @ u)
        xn = x / np.linalg.norm(x)
        c_v = float(xn @ v_sketch)
        cs = np.abs(U @ xn)
        j = int(np.argmax(cs))
        out[t] = {"deg_to_u_k5": deg(c_u), "sign_u": int(np.sign(c_u)),
                  "frac_in_top5": frac, "deg_to_v_full": deg(c_v),
                  "nearest": names[j], "deg_nearest": deg(cs[j]),
                  "pc_coords": a.tolist()}
        print(f"{t:12s} {deg(c_u):>9.1f}d{'+' if c_u > 0 else '-'} {frac:>8.2f} {deg(c_v):>11.1f}d "
              f"{names[j]:>14s} {deg(cs[j]):>5.1f}d")
    print(f"\n  (a zoo adapter keeps {np.mean(np.linalg.norm(Z, axis=1) / np.linalg.norm(Xc, axis=1)):.2f} "
          f"of its norm in the top-5 subspace, for comparison with the in-plane column)")

    print("\nPAIRWISE, full space (sycophantic/obsequious was 61.9; zoo's closest pair 54.0)")
    pw = {}
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = NEW[i], NEW[j]
            d = deg((S[a] / np.linalg.norm(S[a])) @ (S[b] / np.linalg.norm(S[b])))
            pw[f"{a}/{b}"] = d
            print(f"  {a:11s} {b:11s} {d:5.1f}d")
    out["_pairwise"] = pw
    out["_hole_nearest_existing_deg"] = nearest_existing

    # How surprising is a new adapter within theta of u?  For a direction drawn
    # uniformly in K dims, |cos| to a fixed line has density proportional to
    # (1 - c^2)^((K-3)/2); at K=5 that is (1 - c^2), so
    #   P(|cos| > c) = [(1 - c) - (1 - c^3)/3] / (2/3).
    # This is the chance level for ONE new adapter; three were trained.
    def p_within(theta_deg):
        c = np.cos(np.radians(theta_deg))
        return ((1 - c) - (1 - c ** 3) / 3) / (2 / 3)
    full_nearest = min(deg(U[i] @ v_sketch) for i in range(len(names)))
    out["_hole_nearest_existing_full_deg"] = full_nearest
    print(f"\n  full space: nearest existing adapter to v is {full_nearest:.1f} degrees")

    print("\nVERDICT")
    best = min(out[t]["deg_to_u_k5"] for t in NEW)
    p1 = p_within(best)
    print(f"  chance that ONE random 5-d direction falls within {best:.1f} degrees of u: "
          f"{p1*100:.1f}%; that at least one of THREE does: {(1 - (1 - p1) ** 3) * 100:.1f}%")
    out["_p_one"] = p1; out["_p_any_of_three"] = 1 - (1 - p1) ** 3
    if best < nearest_existing - 10 and full_nearest > min(out[t]["deg_to_v_full"] for t in NEW):
        print(f"  the hole has a name: closest new adapter is {best:.1f} degrees from u, "
              f"against {nearest_existing:.1f} for anything that existed, and it is also "
              f"the closest thing to v in the full space")
    elif best < nearest_existing - 10:
        print(f"  suggestive in the plane, unconfirmed in full: {best:.1f} degrees from u "
              f"against {nearest_existing:.1f} for any existing adapter, but in the full "
              f"sketch space the nearest existing adapter ({full_nearest:.1f}) still beats "
              f"the new one; and with three candidates the chance level above is not small")
    elif best < nearest_existing:
        print(f"  marginal: {best:.1f} vs {nearest_existing:.1f} -- closer than any existing "
              f"adapter, but not by enough to call it")
    else:
        print(f"  the names miss: closest new adapter is {best:.1f} degrees from u; the "
              f"existing zoo already had one at {nearest_existing:.1f}")
    json.dump(out, open(f"{Q}/analysis/hole_geometry.json", "w"), indent=1)
    print("\nwrote analysis/hole_geometry.json")


if __name__ == "__main__":
    main()
