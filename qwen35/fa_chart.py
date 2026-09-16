#!/usr/bin/env python3
"""The factor chart: one shared convention for every analysis and figure that
places adapters or directions in the space of the five factor-analytic factors.

Decision 2026-09-08 (Samuel): the factor analysis is the primary frame; the
principal-component chart (k=5 eigenvectors of the double-centred Gram) becomes
secondary.  Everything that previously used the PC chart -- the sphere sweep,
the hole / alien direction, the map, the direction cards, the activation-space
comparison -- should use this module so they agree on the subspace.

Definitions
-----------
* Factor directions.  Each of the five oblimin PAF factors (analyse_fa_qwen35.py,
  results/fa_qwen35.json, centred_k5) is a direction in adapter space defined as a
  weighted merge of the 134 stage-one adapters, v_f = sum_i c_fi a_i.  The
  coefficient dicts are the ones steered in phase 10, phase10_runs/steer_spec2_7a.json
  (names FA_Warmth, FA_Competence, FA_FearfulWithdrawal, FA_Arousal, FA_Imagination),
  in descending order of oblimin sum of squared loadings (10.76, 8.42, 7.02, 6.81, 5.80).
* Inner products come from the exact Gram G = results/gram_sweep.npz, never from
  coordinates: <sum_i c_i a_i, sum_j d_j a_j> = c^T G d.
* Chart basis.  The factors are oblique, so the chart uses an orthonormal basis of
  their 5-dimensional span obtained by Gram-Schmidt in the order above (Warmth first).
  ORDER MATTERS and is fixed here.  Basis vectors are stored as coefficient rows B
  (5 x 134), with B G B^T = I.
* Chart coordinates of any adapter or direction given by coefficients c over the
  134 adapters: x = B G c  (a 5-vector of inner products with the unit basis vectors).
  For an external adapter b (a hole word, an alignment adapter, a second seed) with
  cross-Gram column X[:, b] = <a_i, b>: x = B X[:, b].  Cosine with the chart:
  |x| / |b|.  "Chart length" = |x|; a trait adapter's mean chart length is the
  reference for "how much of an adapter the chart sees".
* Top-3 sphere.  The sphere sweep samples the unit sphere of the span of the first
  three basis vectors (Warmth, Competence, Fearful withdrawal), i.e. the top three
  factors by sum of squared loadings.

Usage:  from fa_chart import FAChart;  ch = FAChart();  ch.coords(c) / ch.coords_external(col)
        ch.basis (5 x 134), ch.factor_names, ch.factor_coef (5 x 134, raw oblique directions)
        ch.factor_chart_coords (5 x 5, where each oblique factor direction sits in the chart)
        ch.trait_coords (134 x 5, every stage-one adapter), ch.names (134 slugs, Gram order)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FACTOR_ORDER = ["FA_Warmth", "FA_Competence", "FA_FearfulWithdrawal", "FA_Arousal", "FA_Imagination"]
SPHERE_FACTORS = FACTOR_ORDER[:3]


class FAChart:
    def __init__(self, gram=f"{HERE}/results/gram_sweep.npz", spec=f"{HERE}/phase10_runs/steer_spec2_7a.json"):
        z = np.load(gram, allow_pickle=True)
        self.G = np.array(z["G"], dtype=float)
        self.names = [str(x) for x in z["names"]]
        self.norms = np.sqrt(np.diag(self.G))
        jobs = {j["name"]: j["coef"] for j in json.load(open(spec))["jobs"]}
        self.factor_names = FACTOR_ORDER
        C = np.array([[jobs[f].get(t, 0.0) for t in self.names] for f in FACTOR_ORDER])   # 5 x 134
        self.factor_coef = C
        # Gram-Schmidt in the G inner product
        B = []
        for k in range(5):
            v = C[k].copy()
            for b in B:
                v = v - (b @ self.G @ v) * b
            v = v / np.sqrt(v @ self.G @ v)
            B.append(v)
        self.basis = np.array(B)                                  # 5 x 134, B G B^T = I
        assert np.allclose(self.basis @ self.G @ self.basis.T, np.eye(5), atol=1e-8)
        self.factor_chart_coords = self.basis @ self.G @ C.T      # 5 (basis) x 5 (factor) -> column f is factor f
        self.trait_coords = (self.basis @ self.G).T               # 134 x 5, row i = adapter i
        self.trait_chart_len = np.linalg.norm(self.trait_coords, axis=1)

    def coords(self, c):
        """Chart coordinates of a merge with coefficients c (dict or 134-vector in Gram order)."""
        if isinstance(c, dict):
            c = np.array([c.get(t, 0.0) for t in self.names])
        return self.basis @ self.G @ np.asarray(c, dtype=float)

    def coords_external(self, col):
        """Chart coordinates of an external adapter from its cross-Gram column <a_i, b> (134, Gram order)."""
        return self.basis @ np.asarray(col, dtype=float)

    def direction_from_chart(self, x):
        """Coefficients over the 134 adapters of the unit direction at chart coordinates x (5-vector)."""
        x = np.asarray(x, dtype=float); x = x / np.linalg.norm(x)
        return x @ self.basis                                     # 134 coefficients; unit norm in G

    def norm(self, c):
        c = np.asarray(c, dtype=float); return float(np.sqrt(c @ self.G @ c))

    def summary(self):
        cos_ff = self.factor_chart_coords / np.linalg.norm(self.factor_chart_coords, axis=0)
        return {"factor_order": self.factor_names, "sphere_factors": SPHERE_FACTORS,
                "factor_pairwise_cosines": (cos_ff.T @ cos_ff).round(4).tolist(),
                "trait_chart_len_mean": float(self.trait_chart_len.mean()),
                "trait_norm_mean": float(self.norms.mean()),
                "chart_captures_frac_of_norm_mean": float(np.mean(self.trait_chart_len / self.norms))}


if __name__ == "__main__":
    ch = FAChart()
    s = ch.summary()
    json.dump(s, open(f"{HERE}/analysis/fa_chart_summary.json", "w"), indent=1)
    print(json.dumps(s, indent=1))
