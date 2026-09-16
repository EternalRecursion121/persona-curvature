#!/usr/bin/env python3
"""Coordinates for the factor-first views: analysis/viz_fa.json.

Decision 2026-09-08 (Samuel): the factor analysis is the primary frame and the
principal-component chart is secondary.  build_viz_data.py stays as it is -- it
still writes analysis/viz.json, which is where the PC scores, the fidelity table
and the coverage sweep live.  This file writes the factor-chart companion.

Everything geometric here comes from fa_chart.FAChart, so the map, the sphere,
the hole and the direction cards cannot drift apart: one basis, one order,
Warmth, Competence, Fearful withdrawal, Arousal, Imagination, Gram-Schmidt in
the exact Gram inner product.

Row order is analysis/viz.json's trait order, asserted, so that viz_fa["coords"][i]
lines up with viz["traits"][i], viz["factor"][i] and viz["keyed"][i] and the page's
javascript can index both arrays with the same i.

Two distinct five-vectors per trait, and they are NOT the same object:

  coords     where the adapter sits in the orthonormal factor chart -- an inner
             product with a unit basis vector, in the units the Gram is in.
  loadings   the oblimin pattern loadings of the k=5 centred PAF solution,
             results/fa_qwen35.json#per_trait.<Trait>.oblimin_loadings_centred_k5,
             which is what the factor solution says the trait is made of.

The chart is built from the factor directions as merges of adapters; the loadings
are the solution's own coefficients.  They agree in sign and rank but not in scale,
and the loadings are the number to quote when the question is "which factor is this
trait".
"""
import json
import os

import numpy as np

from fa_chart import FACTOR_ORDER, FAChart

Q = os.path.dirname(os.path.abspath(__file__))
F5 = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]
# The names the page prints, in FACTOR_ORDER.  FACTOR_ORDER carries the spec's
# job names; these are the titles used in build_blog_page.py's FAS table.
FACTOR_TITLES = ["Warmth", "Competence", "Fearful withdrawal", "Arousal", "Imagination"]
# The solution's own names for the same five, in the same order (sum of squared
# oblimin loadings descending), from analysis/fa_summary.json#centred_k5.factors.
SOLUTION_NAMES = ["Warmth / prosociality", "Competence", "Fearful withdrawal",
                  "Arousal / activation", "Imagination"]

ch = FAChart()
viz = json.load(open(f"{Q}/analysis/viz.json"))
R = json.load(open(f"{Q}/results/fa_qwen35.json"))
FS = json.load(open(f"{Q}/analysis/fa_summary.json"))["centred_k5"]

names = viz["traits"]
assert sorted(names) == sorted(ch.names), "viz.json and gram_sweep.npz disagree on the trait set"
ordr = [ch.names.index(t) for t in names]                 # gram order -> viz order

# --- the solution's factor order is the loading-column order, asserted --------
ss = R["solutions"]["centred_k5"]["ss_loadings"]["oblimin"]
L = np.array(R["solutions"]["centred_k5"]["loadings"]["oblimin"])          # 134 x 5
assert L.shape == (134, 5)
assert np.allclose((L ** 2).sum(0), ss, atol=1e-3), "ss_loadings not the column sums"
assert list(np.argsort(ss)[::-1]) == [0, 1, 2, 3, 4], "oblimin columns not in descending SS order"
for i, f in enumerate(sorted(FS["factors"], key=lambda x: -x["ss"])):
    assert f["name"] == SOLUTION_NAMES[i], f'factor {i} is {f["name"]!r}, expected {SOLUTION_NAMES[i]!r}'

# --- per-trait rows ----------------------------------------------------------
slug_of = dict(zip(R["trait_order"], R["trait_slug"]))
disp_of = {s: d for d, s in slug_of.items()}

coords = ch.trait_coords[ordr]                                            # 134 x 5
chart_len = ch.trait_chart_len[ordr]
norms = ch.norms[ordr]

loadings, communality, uniqueness, assign = [], [], [], []
for t in names:
    rec = R["per_trait"][disp_of[t]]
    lo = rec["oblimin_loadings_centred_k5"]
    loadings.append(lo)
    communality.append(rec["communality_centred_k5"])
    uniqueness.append(rec["uniqueness_centred_k5"])
    k = max(range(5), key=lambda i: abs(lo[i]))
    assign.append({"idx": k, "factor": FACTOR_ORDER[k], "title": FACTOR_TITLES[k],
                   "loading": lo[k], "sign": "+" if lo[k] >= 0 else "-"})
loadings = np.array(loadings)

# --- the five factor directions in their own chart ---------------------------
# column f of factor_chart_coords is where oblique factor f sits; transposed here
# so factor_coords[f] is a row, matching coords[i].
factor_coords = ch.factor_chart_coords.T                                  # 5 x 5
cosff = ch.factor_chart_coords / np.linalg.norm(ch.factor_chart_coords, axis=0)
factor_cos = (cosff.T @ cosff)

# --- 2-D map layouts ---------------------------------------------------------
# Primary: the first two chart axes, Warmth against Competence.  These are the
# first two Gram-Schmidt basis vectors, so axis 0 IS the Warmth direction and
# axis 1 is the part of Competence orthogonal to it -- see fa_chart.py.
# Secondary: PC1 x PC2 from viz.json#scores, for comparison.
# There is no UMAP layout to keep: analysis/umap_test.json holds kNN accuracies
# only (factor(5-way), keying(2-way)), no coordinates.  Recorded, not invented.
umap_path = f"{Q}/analysis/umap_test.json"
umap = json.load(open(umap_path)) if os.path.exists(umap_path) else {}
umap_layout = umap.get("layout") or umap.get("coords") or umap.get("embedding")
S = np.array(viz["scores"])
map2d = {"primary": {"basis": "factor-chart", "axes": [FACTOR_TITLES[0], FACTOR_TITLES[1]],
                     "xy": coords[:, :2].tolist()},
         "pc": {"basis": "pca", "axes": ["PC1", "PC2"], "xy": S[:, :2].tolist()},
         "umap": ({"basis": "umap", "xy": umap_layout} if umap_layout else None),
         "umap_note": ("analysis/umap_test.json holds kNN accuracies only "
                       "(factor(5-way), keying(2-way)); no saved layout exists"
                       if not umap_layout else "")}

# --- the alien / hole direction, in the factor chart -------------------------
# Whatever coefficients analysis/alien.json currently holds; that direction is
# being redefined onto this chart by another analysis, and this file only plots
# what the file says.
special = {}
AL = json.load(open(f"{Q}/analysis/alien.json"))
for key in ("alien_k5",):
    if key in AL and "traits" in AL[key] and "coeffs" in AL[key]:
        c = dict(zip(AL[key]["traits"], AL[key]["coeffs"]))
        u = ch.coords(c)
        special[key] = {"u_fa": (u / (np.linalg.norm(u) or 1.0)).tolist(),
                        "chart_len": float(np.linalg.norm(u))}

out = {
    "basis": "fa_chart.FAChart",
    "factor_order": FACTOR_ORDER,
    "factor_titles": FACTOR_TITLES,
    "solution_names": SOLUTION_NAMES,
    "big_five_scale": dict(zip(FACTOR_ORDER, ["Agreeableness", "Conscientiousness",
                                              "EmotionalStability", "Extraversion", "Intellect"])),
    "ss_loadings_oblimin": ss,
    "traits": names,
    "factor": viz["factor"],
    "keyed": viz["keyed"],
    "coords": coords.tolist(),
    "loadings": loadings.tolist(),
    "assignment": assign,
    "chart_len": chart_len.tolist(),
    "adapter_norm": norms.tolist(),
    "chart_frac_of_norm": (chart_len / norms).tolist(),
    "communality": communality,
    "uniqueness": uniqueness,
    "factor_coords": factor_coords.tolist(),
    "factor_pairwise_cosines": factor_cos.round(4).tolist(),
    "summary": ch.summary(),
    "map2d": map2d,
    "special": special,
    "factors": F5,
}
json.dump(out, open(f"{Q}/analysis/viz_fa.json", "w"))

# --- what it says ------------------------------------------------------------
cnt = {}
for a in assign:
    cnt[a["title"]] = cnt.get(a["title"], 0) + 1
print(f"wrote analysis/viz_fa.json  ({os.path.getsize(f'{Q}/analysis/viz_fa.json')/1e6:.2f} MB)")
print(f"factor order: {', '.join(FACTOR_TITLES)}")
print(f"SS loadings (oblimin): {' '.join(f'{x:.2f}' for x in ss)}")
print("assignment by largest |loading|: " + ", ".join(f"{k} {v}" for k, v in cnt.items()))
print(f"chart length mean {chart_len.mean():.4f}, adapter norm mean {norms.mean():.4f}, "
      f"chart captures {np.mean(chart_len / norms)*100:.1f}% of an adapter's norm")
for t in ("agreeable", "rude", "fearful", "anxious", "imaginative"):
    if t in names:
        i = names.index(t)
        print(f"  {t:12s} coords " + " ".join(f"{x:+.3f}" for x in coords[i]) +
              "   loadings " + " ".join(f"{x:+.3f}" for x in loadings[i]) +
              f"   -> {assign[i]['title']} {assign[i]['sign']}")
if map2d["umap"] is None:
    print("no UMAP layout kept: " + map2d["umap_note"])
