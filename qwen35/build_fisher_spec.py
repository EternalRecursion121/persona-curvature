#!/usr/bin/env python3
"""Enumerate every direction whose Fisher norm the curvature run measures.

A direction is a coefficient dict over adapters.  fisher.py builds the dense
delta exactly as steer_fix.py does (sum_i c_i * (lora_alpha/r) * B_i @ A_i) and
divides by its Frobenius norm, so the coefficients here need no normalisation --
only the *direction* in coefficient space matters.

Families
--------
  fa        the five recovered factors and PC4-6      (steer_spec2_7a.json)
  pc        PC1-3                                     (steer_spec.json)
  axis      the five Big Five axes + the grand mean   (steer_spec.json)
  alien     the factor-chart alien direction + 2 controls (alien_spec_fa.json)
  stage2    the stage-two grand mean + balanced control
  single    ten single stage-one adapters, two per factor
  random    20 seeded Gaussian merges of the 134 stage-one adapters (the null band)
  sphere    the 72 factor-sphere points (loop rates known at alpha 1.5)

ALPHAS.  The named families get six alphas so the quadratic fit has a lever arm
of 4x; the two large families get the four the task specifies.  Fisher norms are
reported per REF-unit alpha at a single ref (0.8078003190997738), and each
direction records the ref of the spec it was published at -- they are NOT all
the same: steer_spec.json, alien_spec_fa.json and sphere_sweep_spec_fa.json use
0.8102592902648793.
"""
import json
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(Q, "phase10_runs")
REF = 0.8078003190997738          # the ref used for THIS measurement, everywhere

SINGLES = ["agreeable", "rude", "organized", "careless", "anxious",
           "relaxed", "extraverted", "quiet", "imaginative", "unimaginative"]
N_RANDOM = 20
RANDOM_SEED0 = 41000

ALPHAS_6 = [-0.5, -0.25, -0.125, 0.125, 0.25, 0.5]
ALPHAS_4 = [-0.5, -0.25, 0.25, 0.5]


def spec(name):
    return json.load(open(os.path.join(P, name)))


def main():
    out = []
    s7a = spec("steer_spec2_7a.json")
    s1 = spec("steer_spec.json")
    afa = spec("alien_spec_fa.json")
    s2m = spec("steer_spec_s2mean.json")
    s2b = spec("steer_spec_s2balanced.json")
    sph = spec("sphere_sweep_spec_fa.json")

    def add(name, family, coef, source, pub_ref, alphas, extra=None):
        d = {"name": name, "family": family, "source": source, "coef": coef,
             "published_ref": pub_ref, "alphas": alphas}
        if extra:
            d.update(extra)
        out.append(d)

    for j in s7a["jobs"]:
        fam = "fa" if j["name"].startswith("FA_") else "pc"
        add(j["name"], fam, j["coef"], j["source"], j["ref"], ALPHAS_6)
    for j in s1["jobs"]:
        fam = "pc" if j["name"].startswith("PC") else "axis"
        add(j["name"], fam, j["coef"], j["source"], j["ref"], ALPHAS_6)
    for j in afa["jobs"]:
        add(j["name"], "alien", j["coef"], j["source"], j["ref"], ALPHAS_6)
    j = s2m["jobs"][0]
    assert j["name"] == "S2_mean", j["name"]
    add(j["name"], "stage2", j["coef"], j["source"], j["ref"], ALPHAS_6)
    j = s2b["jobs"][0]
    add(j["name"], "stage2", j["coef"], j["source"], j["ref"], ALPHAS_6)

    # the 134 stage-one adapters, taken from the only spec that spans all of them
    traits134 = sorted(s7a["jobs"][0]["coef"])
    assert len(traits134) == 134, len(traits134)
    for t in SINGLES:
        assert t in traits134, t
        add(f"single_{t}", "single", {t: 1.0}, "stage1", None, ALPHAS_6)

    for k in range(N_RANDOM):
        seed = RANDOM_SEED0 + k
        c = np.random.default_rng(seed).standard_normal(134)
        add(f"random_{k:02d}", "random", {t: float(v) for t, v in zip(traits134, c)},
            "stage1", None, ALPHAS_4, {"seed": seed})

    basis = sph["basis"]
    for p in sph["points"]:
        c = {}
        for w, b in zip(p["u"], basis):
            for t, v in b["coef"].items():
                c[t] = c.get(t, 0.0) + w * v
        add(f"sphere_{p['name']}", "sphere", c, "stage1", sph["ref"], ALPHAS_4,
            {"u": p["u"], "basis_names": [b["name"] for b in basis]})

    # the fixed text: the base's alpha-0 greedy responses as published.  NOTE the
    # audit in this run: the seven-alpha walk in steer_fix.py adds and subtracts
    # bf16 increments, so the stored "0.0" texts are NOT bit-identical to each
    # other and therefore not exactly the base's output.  PC1's are taken as the
    # canonical fixed text; identical tokens for every direction is what the
    # measurement needs, and the same run re-generates the true base greedy text
    # as a diagnostic.
    R = json.load(open(os.path.join(P, "steer_results_fix.json")))
    pc1 = [r for r in R if r["name"] == "PC1"][0]
    S = {"ref": REF, "max_resp_tokens": 192, "traits134": traits134,
         "prompts": pc1["prompts"], "texts": pc1["generations"]["0.0"],
         "text_source": "phase10_runs/steer_results_fix.json PC1 generations 0.0",
         "directions": out}
    path = os.path.join(P, "fisher_spec.json")
    with open(path, "w") as f:
        json.dump(S, f)
    fams = {}
    for d in out:
        fams[d["family"]] = fams.get(d["family"], 0) + 1
    npass = sum(len(d["alphas"]) for d in out)
    print(f"wrote {path}: {len(out)} directions, {npass} (direction,alpha) passes")
    print(" families:", fams)
    print(f" prompts {len(S['prompts'])}, texts {len(S['texts'])}")


if __name__ == "__main__":
    main()
