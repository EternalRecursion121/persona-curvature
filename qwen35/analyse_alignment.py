#!/usr/bin/env python3
"""Where do the alignment-relevant traits sit relative to the Big Five lexicon?

Four adapters trained on the zoo's recipe, seed and prompt pool: Sycophantic,
Obsequious, Power-seeking, Corrigible. None of them is a Big Five adjective and
none of the 29 Honesty-Humility markers appears anywhere in the 140.

Predictions are in PREREG_alignment.md, written before any of this existed.

THE GATE COMES FIRST
--------------------
Every angle below is a statement about adapters sharing one 64-dimensional input
window, which holds only if the new adapters got the zoo's LoRA-A. The zoo's own
internal A drift is 0.0146. If these come back near 1.0 the seed did not take,
they occupy a near-orthogonal subspace, and nothing downstream means anything --
a failure that would read as a finding, which is the kind worth gating on.
"""
import glob
import json
import os
import sys

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
NEW = ["sycophantic", "obsequious", "power_seeking", "corrigible"]
# which sketch arm to read.  `alignment` is the first run (497 prompts, 53 of
# them unseen by any zoo adapter); `aligncommon` is the retrain on the zoo's
# exact shared pool.  Set PC_ALIGN_ARM=aligncommon for the matched numbers.
ARM = os.environ.get("PC_ALIGN_ARM", "alignment")
F5 = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]
ZOO_DRIFT = 0.0146


def gate(files_dir):
    """A-matrix drift of each new adapter against the zoo's shared A_0."""
    from safetensors import safe_open
    z = f"{Q}/{files_dir}/_zoo_bold.safetensors"
    if not os.path.exists(z):
        print(f"  gate skipped: no zoo reference at {z}")
        return None
    with safe_open(z, framework="np") as f:
        keys = [k for k in f.keys() if ".lora_A." in k]
        A0 = {k: f.get_tensor(k).astype(np.float64) for k in keys}
    out = {}
    for t in NEW:
        p = f"{Q}/{files_dir}/{t}.safetensors"
        if not os.path.exists(p):
            continue
        with safe_open(p, framework="np") as f:
            d = [np.linalg.norm(f.get_tensor(k).astype(np.float64) - A0[k]) /
                 np.linalg.norm(A0[k]) for k in keys if k in f.keys()]
        out[t] = float(np.mean(d))
    print("GATE -- LoRA-A drift against the zoo's A_0 "
          f"(zoo's own internal figure {ZOO_DRIFT:.4f})")
    for t, v in out.items():
        verdict = "same window" if v < 0.2 else "DIFFERENT WINDOW -- angles meaningless"
        print(f"  {t:16s} {v:.4f}   {verdict}")
    if out and max(out.values()) >= 0.2:
        sys.exit("GATE FAILED: at least one adapter does not share the zoo's A.")
    return out


def main():
    meta = {}
    for f in ("traits_primary.json", "traits_secondary.json"):
        p = f"{Q}/{f}"
        if os.path.exists(p):
            for r in json.load(open(p)):
                meta[r["trait"].lower().replace(" ", "_").replace("-", "_")] = (r["factor"], r["keyed"])
    zoo = sorted(t for t in
                 (os.path.basename(p)[:-4] for p in glob.glob(f"{Q}/analysis/sketches/stage1_k32/*.npz"))
                 if t in meta)
    X = np.stack([np.load(f"{Q}/analysis/sketches/stage1_k32/{t}.npz",
                          allow_pickle=True)["sketch"].astype(np.float64) for t in zoo])
    Xc = X - X.mean(0)
    U = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    w, V = np.linalg.eigh(Xc @ Xc.T)
    o = np.argsort(w)[::-1]
    w, V = np.maximum(w[o], 1e-12), V[:, o]
    unit = lambda v: v / np.linalg.norm(v)

    D = {f"PC{j+1}": unit(Xc.T @ (V[:, j] / np.sqrt(w[j]))) for j in range(6)}
    K = []
    for f in F5:
        p = [i for i, t in enumerate(zoo) if meta[t] == (f, "+")]
        m = [i for i, t in enumerate(zoo) if meta[t] == (f, "-")]
        k = X[p].mean(0) - X[m].mean(0)
        D[f"axis_{f}"] = unit(k)
        K.append(unit(k))
    K = np.stack(K)
    Gi = np.linalg.inv(K @ K.T)
    D["personality_axis"] = unit(X.mean(0))

    S = {}
    for t in NEW:
        p = f"{Q}/analysis/sketches/{ARM}_k32/{t}.npz"
        if not os.path.exists(p):
            sys.exit(f"missing sketch {p} -- run sketch_adapters.py --source {ARM}")
        S[t] = np.load(p, allow_pickle=True)["sketch"].astype(np.float64)

    print(f"\nreference angles from the existing zoo: named axes 47-53 deg to their "
          f"nearest adjective,\nclosest pair (composed/imperturbable) 54.0, median trait "
          f"to its nearest 65.6,\ntwo at random 83.3, widest hole 68.9\n")

    print("ANGLE TO THE NEAREST OF THE 134, each treated as a line")
    print(f"{'trait':16s} {'nearest':>16s} {'angle':>7s}   next two")
    res = {}
    for t in NEW:
        v = unit(S[t] - X.mean(0))
        c = np.abs(U @ v)
        od = np.argsort(c)[::-1]
        deg = float(np.degrees(np.arccos(min(c[od[0]], 1.0))))
        res[t] = {"nearest": zoo[od[0]], "deg": deg,
                  "next": [(zoo[i], float(np.degrees(np.arccos(min(c[i], 1.0))))) for i in od[1:3]]}
        print(f"{t:16s} {zoo[od[0]]:>16s} {deg:>6.1f}d   "
              + ", ".join(f"{n} ({d:.0f}d)" for n, d in res[t]["next"]))

    print("\nCOSINE WITH NAMED DIRECTIONS")
    cols = ["axis_Agreeableness", "PC4", "axis_Extraversion", "axis_Conscientiousness",
            "personality_axis"]
    print(f"{'trait':16s} " + " ".join(f"{c[:13]:>14s}" for c in cols))
    for t in NEW:
        v = unit(S[t] - X.mean(0))
        print(f"{t:16s} " + " ".join(f"{float(v @ D[c]):>+14.3f}" for c in cols))
        res[t]["cos"] = {c: float(v @ D[c]) for c in cols}

    print("\nBIG FIVE CHART COORDINATES (unit norm)")
    print(f"{'trait':16s} " + " ".join(f"{f[:5]:>9s}" for f in F5))
    for t in NEW:
        v = unit(S[t] - X.mean(0))
        co = (v @ K.T) @ Gi
        res[t]["chart"] = co.tolist()
        print(f"{t:16s} " + " ".join(f"{x:>+9.3f}" for x in co))

    a = unit(S["sycophantic"] - X.mean(0))
    b = unit(S["obsequious"] - X.mean(0))
    pair = float(np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1))))
    print(f"\nsycophantic / obsequious: {pair:.1f} degrees apart "
          f"(zoo's closest pair is 54.0)")

    print("\nPREDICTIONS")
    p1 = res["sycophantic"]["cos"]["axis_Agreeableness"] > res["sycophantic"]["cos"]["PC4"] \
        and res["sycophantic"]["cos"]["axis_Agreeableness"] > 0
    print(f"  1 sycophancy on Agreeableness not PC4 : {'HELD' if p1 else 'FAILED'}")
    print(f"  2 synonym pair under 54 degrees        : "
          f"{'HELD' if pair < 54 else 'FAILED'} ({pair:.1f})")
    print(f"  3 power-seeking over 60 from all 134   : "
          f"{'HELD' if res['power_seeking']['deg'] > 60 else 'FAILED'} "
          f"({res['power_seeking']['deg']:.1f})")
    print(f"  4 corrigible under 60                  : "
          f"{'HELD' if res['corrigible']['deg'] < 60 else 'FAILED'} "
          f"({res['corrigible']['deg']:.1f})")
    res["_pair_deg"] = pair
    out = f"{Q}/analysis/alignment_geometry{'' if ARM == 'alignment' else '_' + ARM}.json"
    json.dump(res, open(out, "w"), indent=1)
    print(f"\nwrote {os.path.relpath(out, Q)}")


if __name__ == "__main__":
    if "--gate" in sys.argv:
        gate(sys.argv[sys.argv.index("--gate") + 1])
    else:
        main()
