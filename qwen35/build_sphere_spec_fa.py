#!/usr/bin/env python3
"""Sample the personality sphere -- on the FACTOR chart.

This is the 2026-09-08 redo of build_sphere_spec.py. The design is unchanged:
72 Fibonacci-lattice directions on the unit sphere of a three-dimensional
subspace, each steered at alpha = +1.5 on the same eight prompts, so that a
reader can click anywhere and read what the model becomes there, and so that
"every direction gives a coherent persona" can be tested rather than assumed.

WHAT CHANGED
------------
The subspace. The 2026-09-01 sphere used the top three principal components of
the double-centred Gram. Samuel's decision of 2026-09-08 makes the factor
analysis the primary frame, so this version samples the span of the first three
vectors of the shared factor chart (fa_chart.FAChart): the Gram-Schmidt
orthonormalisation, in the G inner product, of the five oblimin factor
directions taken in descending order of oblimin sum of squared loadings. The
first three are Warmth, Competence and Fearful withdrawal, so the sphere is the
unit sphere of the space those three factors span.

Nothing else moves: alpha 1.5, the same eight prompts, the same reference norm
(phase10_runs/steer_spec.json ref = 0.8102592902648793, what the PC sphere used,
NOT the 0.80780 of steer_spec2_7a) so the steering magnitude is identical and
the two spheres are comparable point for point.

LANDMARKS
---------
Everything a reader might navigate by, projected into the same 3-space and
normalised: the five oblique factor directions themselves (which do NOT sit on
the basis axes, because the factors are oblique and only the first three are in
the span at all), the five Big Five axes, the grand mean direction
(mean_assistant_axis), and for continuity the three old PC directions. All of
them come from coefficient dicts that were actually steered -- steer_spec.json
for the axes, the mean and the PCs, steer_spec2_7a.json (via FAChart) for the
factors -- so a landmark on this sphere is a direction the study has generated
from, not a construction.

Outputs
  phase10_runs/sphere_spec_fa.json        steer_fix-shaped job list (one job per point)
  phase10_runs/sphere_sweep_spec_fa.json  sphere_sweep.py input (3 basis deltas + 72 u)
  analysis/sphere_layout_fa.json          geometry for the page
"""
import json
import os

import numpy as np

from fa_chart import FAChart, SPHERE_FACTORS

Q = os.path.dirname(os.path.abspath(__file__))
N = 72
ALPHA = 1.5
PROMPT_IDX = [0, 1, 5, 9, 11, 14, 15, 22]
TAG = "fa"

S = json.load(open(f"{Q}/phase10_runs/steer_spec.json"))
REF, PROMPTS = S["ref"], S["jobs"][0]["prompts"]
SJOBS = {j["name"]: j["coef"] for j in S["jobs"]}

ch = FAChart()
names = ch.names                     # 134 slugs, Gram order
B3 = ch.basis[:3]                    # 3 x 134, B G B^T = I

meta = {}
for f in ("traits_primary.json", "traits_secondary.json"):
    p = f"{Q}/{f}"
    if os.path.exists(p):
        for r in json.load(open(p)):
            meta[r["trait"].lower().replace(" ", "_").replace("-", "_")] = (r["factor"], r["keyed"])


def fibonacci_sphere(n):
    """Near-uniform points on S^2 -- identical lattice to the PC sphere, so the
    two runs differ only in which subspace the lattice is laid on."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(np.maximum(1 - z * z, 0))
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(th), r * np.sin(th), z], 1)


U = fibonacci_sphere(N)
jobs = [{"name": f"S{i:03d}", "u": u.tolist()} for i, u in enumerate(U)]


def chart3(coef):
    """Unit 3-vector: where a coefficient dict points inside the top-3 chart."""
    x = ch.coords(coef)[:3]
    return x, float(np.linalg.norm(x))


land, land_len = {}, {}
for k, f in enumerate(ch.factor_names):
    x = ch.factor_chart_coords[:3, k]
    land[f"factor_{f}"] = (x / np.linalg.norm(x)).tolist()
    land_len[f"factor_{f}"] = float(np.linalg.norm(x) / ch.norm(ch.factor_coef[k]))
for n in ("axis_Extraversion", "axis_Agreeableness", "axis_Conscientiousness",
          "axis_EmotionalStability", "axis_Intellect", "mean_assistant_axis",
          "PC1", "PC2", "PC3"):
    x, r = chart3(SJOBS[n])
    land[n] = (x / r).tolist()
    land_len[n] = float(r / ch.norm([SJOBS[n].get(t, 0.0) for t in names]))

# every trait's own position on the same sphere, for the backdrop cloud
T3 = ch.trait_coords[:, :3]
rmax = float(np.linalg.norm(T3, axis=1).max())
tr = {t: {"u": (T3[k] / np.linalg.norm(T3[k])).tolist(),
          "r": float(np.linalg.norm(T3[k]) / rmax),
          "cos": float(np.linalg.norm(T3[k]) / ch.norms[k]),
          "factor": meta[t][0], "keyed": meta[t][1]}
      for k, t in enumerate(names)}

# --- how much of the zoo does this 3-space see? -----------------------------
# Defined exactly as the PC sphere's var3 was, so the two numbers mean the same
# thing: the share of the total centred adapter variance (trace of the
# double-centred Gram) that projects onto each unit basis vector.
G = ch.G
n134 = len(names)
J = np.eye(n134) - np.ones((n134, n134)) / n134
Gc = J @ G @ J
comp = (J @ G @ B3.T)                       # 134 x 3, row i = <a_i - abar, b_k>
var3 = (comp ** 2).sum(0) / np.trace(Gc)

# Two like-for-like reference numbers, because the PC sphere's own var3
# (analysis/sphere_layout.json#var3, summing to 28.9%) was computed from the
# eigenvalues of the k=32 SKETCH Gram, not the exact one. Same definition,
# different object, so it is not directly comparable to the line above.
ev = np.sort(np.linalg.eigvalsh(Gc))[::-1]
var3_pc_exact_gram = (ev[:3] / ev.sum()).tolist()
PC3 = np.array([[SJOBS[n].get(t, 0.0) for t in names] for n in ("PC1", "PC2", "PC3")])
Bp = []                                     # the PC sphere's own 3-space, orthonormalised in G
for k in range(3):
    v = PC3[k].copy()
    for b in Bp:
        v = v - (b @ G @ v) * b
    Bp.append(v / np.sqrt(v @ G @ v))
var3_pc_steered = ((J @ G @ np.array(Bp).T) ** 2).sum(0) / np.trace(Gc)

spec = {"ref": REF, "n": n134, "alpha": ALPHA, "prompt_idx": PROMPT_IDX, "chart": "fa",
        "jobs": [{"name": j["name"], "source": "stage1", "alphas": [ALPHA], "ref": REF,
                  "prompts": [PROMPTS[i] for i in PROMPT_IDX],
                  "coef": {names[k]: float(x)
                           for k, x in enumerate(np.array(j["u"]) @ B3)}}
                 for j in jobs]}
json.dump(spec, open(f"{Q}/phase10_runs/sphere_spec_fa.json", "w"))

json.dump({"basis": [{"name": f"B{k+1}_{SPHERE_FACTORS[k]}",
                      "coef": {names[i]: float(v) for i, v in enumerate(B3[k])}}
                     for k in range(3)],
           "points": jobs, "prompts": [PROMPTS[i] for i in PROMPT_IDX],
           "alpha": ALPHA, "ref": REF, "tag": TAG},
          open(f"{Q}/phase10_runs/sphere_sweep_spec_fa.json", "w"))

json.dump({"chart": "fa", "points": jobs, "landmarks": land, "landmark_cos": land_len,
           "traits": tr, "alpha": ALPHA, "ref": REF,
           "prompts": [PROMPTS[i] for i in PROMPT_IDX],
           "axis_names": SPHERE_FACTORS, "factor_order": ch.factor_names,
           "var3": var3.tolist(),
           "var3_pc_exact_gram": var3_pc_exact_gram,
           "var3_pc_steered_dirs": var3_pc_steered.tolist(),
           "ss_loadings_oblimin": json.load(
               open(f"{Q}/results/fa_qwen35.json"))["solutions"]["centred_k5"]["ss_loadings"]["oblimin"],
           "factor_pairwise_cosines": ch.summary()["factor_pairwise_cosines"]},
          open(f"{Q}/analysis/sphere_layout_fa.json", "w"))

Tm = np.stack([tr[t]["u"] for t in names])
d = np.degrees(np.arccos(np.clip(np.abs(U @ Tm.T).max(1), 0, 1)))
print(f"{N} sampled directions at alpha={ALPHA}, {len(PROMPT_IDX)} prompts each "
      f"= {N*len(PROMPT_IDX)} generations")
print(f"sphere axes: {', '.join(SPHERE_FACTORS)}   ref={REF}")
print(f"top-3 factor subspace carries {var3.sum()*100:.1f}% of the centred variance "
      f"({', '.join(f'{v*100:.1f}%' for v in var3)})")
print(f"  reference, exact centred Gram: top-3 eigenvalues {sum(var3_pc_exact_gram)*100:.1f}%, "
      f"the three steered PC directions {var3_pc_steered.sum()*100:.1f}%")
print(f"angle from each sample to the NEAREST trait line: "
      f"min {d.min():.1f}d  median {np.median(d):.1f}d  max {d.max():.1f}d")
print("landmark cosine with the 3-space: "
      + ", ".join(f"{k} {v:.2f}" for k, v in land_len.items()))
print("wrote phase10_runs/sphere_spec_fa.json, phase10_runs/sphere_sweep_spec_fa.json, "
      "analysis/sphere_layout_fa.json")
