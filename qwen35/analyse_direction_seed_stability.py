#!/usr/bin/env python3
"""Does a trait's coordinate on each named direction replicate across LoRA seeds?

Directions are the weighted merges steered in phase 10: PC1-6, the five PAF factors
(FA_*), the five Big Five axes and the grand mean, with coefficients c over the
seed-0 adapters taken from phase10_runs/steer_spec.json and steer_spec2_7a.json.
For a direction v = sum_i c_i a_i (a_i the seed-0 stage-1 deltas), the cosine of any
adapter b with v is  <b, v> / (|b||v|) = sum_i c_i X[i, b] / (|b| sqrt(c^T G c)),
which needs only Grams.  Seed-0 coordinates come from results/gram_sweep.npz, seed-1
coordinates (40 matched-objective adapters) from the exact 134 x 40 cross-Gram
results/cross_gram_full_root_x_data_null_seedpaired_s40_matched.npz.

Reported per direction: Pearson r over the 40 traits between seed-0 and seed-1
coordinates (with a permutation p), and a rank-based version; plus the same for the
40 traits' coordinates against a random direction of the same construction.
Output: analysis/direction_seed_stability.json.
"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = f"{HERE}/results"
rng = np.random.default_rng(0)

z = np.load(f"{R}/gram_sweep.npz", allow_pickle=True); G = np.array(z["G"]); n0 = [str(x) for x in z["names"]]
x = np.load(f"{R}/cross_gram_full_root_x_data_null_seedpaired_s40_matched.npz", allow_pickle=True)
X = np.array(x["X"]); na = [str(t) for t in x["names_a"]]; nb = [str(t) for t in x["names_b"]]; Nb = np.array(x["norms_b"])
assert na == n0 or set(na) == set(n0)
ia = [na.index(t) for t in n0]; X = X[ia, :]                      # rows in gram_sweep order
norm0 = np.sqrt(np.diag(G))

dirs = {}
for spec in ("steer_spec.json", "steer_spec2_7a.json"):
    for j in json.load(open(f"{HERE}/phase10_runs/{spec}"))["jobs"]:
        dirs[j["name"]] = j["coef"]

def coords(coef):
    c = np.array([coef.get(t, 0.0) for t in n0])
    vn = np.sqrt(c @ G @ c)
    q0 = (G @ c) / (norm0 * vn)                  # seed-0 adapters vs v (134)
    q1 = (X.T @ c) / (Nb * vn)                   # seed-1 adapters vs v (40)
    i0 = [n0.index(t) for t in nb]
    return q0[i0], q1

def perm_p(a, b, r, n=20000):
    cnt = 0
    for _ in range(n):
        if abs(np.corrcoef(a, rng.permutation(b))[0, 1]) >= abs(r): cnt += 1
    return (cnt + 1) / (n + 1)

out = {"n_seed1_traits": len(nb), "traits": nb, "directions": {}}
from scipy.stats import spearmanr
for name, coef in dirs.items():
    q0, q1 = coords(coef)
    r = float(np.corrcoef(q0, q1)[0, 1]); rho = float(spearmanr(q0, q1).statistic)
    slope = float(np.polyfit(q0, q1, 1)[0])
    sign_agree = float(np.mean(np.sign(q0) == np.sign(q1)))
    out["directions"][name] = {"pearson": r, "spearman": rho, "perm_p": perm_p(q0, q1, r, 5000),
                               "slope_seed1_on_seed0": slope, "sign_agreement": sign_agree,
                               "seed0_sd": float(q0.std()), "seed1_sd": float(q1.std()),
                               "seed0": dict(zip(nb, np.round(q0, 4).tolist())), "seed1": dict(zip(nb, np.round(q1, 4).tolist()))}
# random directions of matched construction: random coefficients over the 134 adapters
rs = []
for k in range(200):
    c = {t: float(v) for t, v in zip(n0, rng.standard_normal(len(n0)))}
    q0, q1 = coords(c); rs.append(float(np.corrcoef(q0, q1)[0, 1]))
out["random_direction_pearson"] = {"mean": float(np.mean(rs)), "sd": float(np.std(rs)), "p95": float(np.percentile(rs, 95)), "n": 200,
                                   "note": "random Gaussian coefficients over the 134 seed-0 adapters; the shared LoRA-A makes even random merges partly reproducible"}
json.dump(out, open(f"{HERE}/analysis/direction_seed_stability.json", "w"), indent=1)
print(f"{len(nb)} seed-1 traits; random-direction r {out['random_direction_pearson']['mean']:.3f} +- {out['random_direction_pearson']['sd']:.3f} (p95 {out['random_direction_pearson']['p95']:.3f})")
for name, d in sorted(out["directions"].items(), key=lambda kv: -kv[1]["pearson"]):
    print(f"  {name:26s} r {d['pearson']:+.3f}  rho {d['spearman']:+.3f}  p {d['perm_p']:.4f}  slope {d['slope_seed1_on_seed0']:.3f}  sign agree {d['sign_agreement']:.2f}  sd0 {d['seed0_sd']:.3f} sd1 {d['seed1_sd']:.3f}")
