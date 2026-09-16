#!/usr/bin/env python3
"""Where do the ten Big Five FACTOR adapters sit in the zoo's geometry?

The zoo's 134 adapters are per ADJECTIVE. A factor only exists there as an
average over marker adjectives, or as a merged steering direction. These ten are
the factor itself: one amplifier and one suppressor per OCEAN factor, trained as
stage-one DPO adapters on the zoo's shared prompt pool at the matched objective,
from Persona Cartography's own Figure 2 constitutions.

That makes them a direct test of something the zoo could only assume: does an
adapter trained ON a factor land on the direction the zoo RECOVERED for that
factor? Four questions, all in the same coordinates analyse_alignment.py and
analyse_hole.py use -- centre on the zoo mean, treat every adapter as a line:

  0. GATE   do they share the zoo's LoRA-A window (drift ~0.015, not ~1.0)?
  1. cosine with the five named Big Five AXIS directions, mean(+keyed) minus
     mean(-keyed) over the zoo's markers, read from phase10_runs/steer_spec.json
     so the direction is byte-identical to the one that was steered and judged.
  2. cosine with the five FA FACTOR directions (FA_Warmth, FA_Competence,
     FA_Arousal, FA_FearfulWithdrawal, FA_Imagination) from
     phase10_runs/steer_spec2_7a.json -- the factors the zoo recovered without
     being told to look for Big Five.
  3. nearest zoo adapters, and the within-pair angle between each factor's own
     high and low adapter (mirror-image training data, so the angle between them
     is this batch's own noise floor for "the same dimension, opposite poles").

Every coefficient dict in both spec files sums to zero, so a direction built
from raw sketches equals one built from mean-centred sketches; checked below.

usage:  python analyse_bigfive.py
out:    analysis/bigfive_adapters_geometry.json
"""
import glob
import json
import os
import sys

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
ARM = "bigfive"
FILES = f"{Q}/bigfive_files"
F5 = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability",
      "Intellect"]
FA = ["FA_Warmth", "FA_Competence", "FA_Arousal", "FA_FearfulWithdrawal",
      "FA_Imagination"]
ZOO_DRIFT = 0.0146

deg = lambda c: float(np.degrees(np.arccos(np.clip(abs(c), 0.0, 1.0))))
sdeg = lambda c: float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
unit = lambda v: v / np.linalg.norm(v)


def gate(traits):
    """A-matrix drift against the zoo's shared A_0, as analyse_hole.py does."""
    from safetensors import safe_open
    z = f"{FILES}/_zoo_bold.safetensors"
    if not os.path.exists(z):
        sys.exit(f"no zoo reference at {z} -- copy it from hole_files/")
    with safe_open(z, framework="np") as f:
        keys = [k for k in f.keys() if ".lora_A." in k]
        A0 = {k: f.get_tensor(k).astype(np.float64) for k in keys}
    print(f"GATE -- LoRA-A drift against the zoo's A_0 (zoo's own figure {ZOO_DRIFT:.4f})")
    out = {}
    for t in traits:
        p = f"{FILES}/{t}.safetensors"
        if not os.path.exists(p):
            sys.exit(f"missing {p} -- run sketch_adapters.py --keep-files {FILES}")
        with safe_open(p, framework="np") as f:
            out[t] = float(np.mean([
                np.linalg.norm(f.get_tensor(k).astype(np.float64) - A0[k])
                / np.linalg.norm(A0[k]) for k in keys]))
        print(f"  {t:28s} {out[t]:.4f}   "
              f"{'same window' if out[t] < 0.2 else 'DIFFERENT WINDOW'}")
    if max(out.values()) >= 0.2:
        sys.exit("GATE FAILED: at least one adapter does not share the zoo's A.")
    return out


def main():
    bf = json.load(open(f"{Q}/traits_bigfive.json"))
    traits = [r["trait"] for r in bf]
    info = {r["trait"]: r for r in bf}
    gate_drift = gate(traits)

    # ---- the zoo ----------------------------------------------------------
    meta = {}
    for f in ("traits_primary.json", "traits_secondary.json"):
        for r in json.load(open(f"{Q}/{f}")):
            meta[r["trait"].lower().replace(" ", "_").replace("-", "_")] = \
                (r["factor"].replace(" ", ""), r["keyed"])
    zoo = sorted(t for t in (os.path.basename(p)[:-4] for p in
                             glob.glob(f"{Q}/analysis/sketches/stage1_k32/*.npz"))
                 if t in meta)
    X = np.stack([np.load(f"{Q}/analysis/sketches/stage1_k32/{t}.npz",
                          allow_pickle=True)["sketch"].astype(np.float64)
                  for t in zoo])
    mu = X.mean(0)
    Xc = X - mu
    U = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    pos = {t: i for i, t in enumerate(zoo)}
    print(f"\nzoo: {len(zoo)} sketched stage-one adapters, dim {X.shape[1]}")

    # ---- directions, built from the steering specs ------------------------
    D, prov = {}, {}
    for spec, want in (("steer_spec.json", [f"axis_{f}" for f in F5]),
                       ("steer_spec2_7a.json", FA)):
        jobs = {j["name"]: j for j in
                json.load(open(f"{Q}/phase10_runs/{spec}"))["jobs"]}
        for name in want:
            if name not in jobs:
                sys.exit(f"{spec} has no direction {name!r}")
            coef = jobs[name]["coef"]
            s = sum(coef.values())
            if abs(s) > 1e-9:
                sys.exit(f"{name}: coefficients sum to {s}, not 0 -- raw and "
                         f"centred sketches would disagree")
            missing = [t for t in coef if t not in pos]
            if missing:
                sys.exit(f"{name}: {len(missing)} coefficient traits not sketched, "
                         f"e.g. {missing[:3]}")
            c = np.zeros(len(zoo))
            for t, v in coef.items():
                c[pos[t]] = v
            D[name] = unit(c @ X)
            prov[name] = {"spec": f"phase10_runs/{spec}", "n_coef": len(coef)}

    # cross-check: the axes rebuilt from keying must equal the spec's axes
    for f in F5:
        p = [i for i, t in enumerate(zoo) if meta[t] == (f, "+")]
        m = [i for i, t in enumerate(zoo) if meta[t] == (f, "-")]
        k = unit(X[p].mean(0) - X[m].mean(0))
        c = float(k @ D[f"axis_{f}"])
        print(f"  cross-check axis_{f:20s} spec vs keying cos {c:+.6f} "
              f"({len(p)}+/{len(m)}-)")
        if c < 0.999:
            sys.exit(f"axis_{f}: spec direction disagrees with the keying")

    # ---- the ten new adapters ---------------------------------------------
    S = {}
    for t in traits:
        p = f"{Q}/analysis/sketches/{ARM}_k32/{t}.npz"
        if not os.path.exists(p):
            sys.exit(f"missing sketch {p} -- run sketch_adapters.py --source {ARM}")
        S[t] = np.load(p, allow_pickle=True)["sketch"].astype(np.float64)
    V = {t: unit(S[t] - mu) for t in traits}

    res = {"_meta": {
        "arm": ARM, "n_zoo": len(zoo), "sketch_dim": int(X.shape[1]),
        "sketch": f"analysis/sketches/{ARM}_k32", "gate_zoo_drift": ZOO_DRIFT,
        "directions": prov,
        "convention": ("sketches centred on the zoo mean and unit-normalised; "
                       "cosines are SIGNED against the named directions and "
                       "UNSIGNED (line vs line) against individual adapters, "
                       "as analyse_alignment.py does"),
    }, "gate_drift": gate_drift, "traits": {}}

    print("\nCOSINE WITH THE FIVE NAMED BIG FIVE AXES (steer_spec.json)")
    print(f"{'trait':28s} " + " ".join(f"{f[:9]:>10s}" for f in F5) + "   own")
    for t in traits:
        cs = {f: float(V[t] @ D[f"axis_{f}"]) for f in F5}
        own = info[t]["zoo_factor"]
        res["traits"][t] = {"cos_axis": cs, "zoo_factor": own,
                            "zoo_pole": info[t]["zoo_pole"],
                            "bigfive_factor": info[t]["bigfive_factor"],
                            "pole": info[t]["pole"]}
        print(f"{t:28s} " + " ".join(f"{cs[f]:>+10.3f}" for f in F5)
              + f"   {own}{info[t]['zoo_pole']}")

    print("\nCOSINE WITH THE FIVE RECOVERED FA FACTORS (steer_spec2_7a.json)")
    print(f"{'trait':28s} " + " ".join(f"{f[3:12]:>10s}" for f in FA))
    for t in traits:
        cs = {f: float(V[t] @ D[f]) for f in FA}
        res["traits"][t]["cos_fa"] = cs
        print(f"{t:28s} " + " ".join(f"{cs[f]:>+10.3f}" for f in FA))

    print("\nNEAREST ZOO ADAPTERS (line vs line)")
    print(f"{'trait':28s} {'nearest':>18s} {'angle':>7s}   next two")
    for t in traits:
        c = np.abs(U @ V[t])
        od = np.argsort(c)[::-1]
        res["traits"][t]["nearest"] = [
            {"trait": zoo[i], "deg": deg(c[i]),
             "factor": meta[zoo[i]][0], "keyed": meta[zoo[i]][1]} for i in od[:5]]
        print(f"{t:28s} {zoo[od[0]]:>18s} {deg(c[od[0]]):>6.1f}d   "
              + ", ".join(f"{zoo[i]} ({deg(c[i]):.0f}d)" for i in od[1:3]))

    print("\nHIGH vs LOW OF THE SAME FACTOR (mirror-image training data)")
    pairs = {}
    for r in bf:
        if r["pole"] != "high":
            continue
        a = r["trait"]
        b = a[:-5] + "_low"
        signed = float(V[a] @ V[b])
        pairs[f"{a}/{b}"] = {"cos": signed, "deg_signed": sdeg(signed),
                             "deg_line": deg(signed)}
        print(f"  {a:28s} {b:28s} cos {signed:+.3f}  "
              f"{sdeg(signed):5.1f}d signed / {deg(signed):5.1f}d as lines")
    res["high_low_pairs"] = pairs

    print("\nALL-PAIRS AMONG THE TEN (line vs line, degrees)")
    M = np.zeros((10, 10))
    for i, a in enumerate(traits):
        for j, b in enumerate(traits):
            M[i, j] = deg(V[a] @ V[b]) if i != j else 0.0
    res["pairwise_deg"] = {a: {b: M[i, j] for j, b in enumerate(traits)}
                           for i, a in enumerate(traits)}
    off = M[np.triu_indices(10, 1)]
    print(f"  min {off.min():.1f}d  median {np.median(off):.1f}d  max {off.max():.1f}d")

    # zoo reference angles, recomputed here so the comparison is like for like
    Cz = np.abs(U @ U.T)
    np.fill_diagonal(Cz, -1.0)
    nn = np.array([deg(Cz[i].max()) for i in range(len(zoo))])
    iu = np.triu_indices(len(zoo), 1)
    res["_meta"]["zoo_reference"] = {
        "median_nearest_neighbour_deg": float(np.median(nn)),
        "closest_pair_deg": float(np.min([deg(c) for c in np.abs(U @ U.T)[iu]])),
        "median_pair_deg": float(np.median([deg(c) for c in np.abs(U @ U.T)[iu]])),
    }
    print(f"\nzoo reference: median adapter-to-nearest "
          f"{res['_meta']['zoo_reference']['median_nearest_neighbour_deg']:.1f}d, "
          f"closest pair {res['_meta']['zoo_reference']['closest_pair_deg']:.1f}d, "
          f"median pair {res['_meta']['zoo_reference']['median_pair_deg']:.1f}d")

    out = f"{Q}/analysis/bigfive_adapters_geometry.json"
    json.dump(res, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
