#!/usr/bin/env python3
"""Does a dataset's first-order score forecast the behaviour of the model trained on it?

134 datasets (one per zoo trait), each trained into an adapter that was judged blind on the
Big Five. From analysis/nxn_scores.json (40 pairs per trait scored against all 134 unit adapter
directions) the score of a dataset along any merge direction sum_i c_i a_i follows by linearity:
score(data, sum c_i a_i) = sum_i c_i |a_i| score(data, a_i / |a_i|).  Directions: the five Big
Five axes and the five factors (phase10_runs/steer_spec*.json).  Leave-one-out: when scoring
trait t's data along a direction, adapter t is removed from the merge (its coefficient set to
zero), so the dataset is never scored against the adapter trained on it.
Behaviour: judged_100.json stage1 minus base, per dimension, mean over 24 prompts.
Output: analysis/data_forecast.json
"""
import json, os
import numpy as np
from scipy.stats import pearsonr, spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
DIMS = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]
AXIS = {d: f"axis_{d}" for d in DIMS}
FA_FOR = {"Extraversion": "FA_Arousal", "Agreeableness": "FA_Warmth", "Conscientiousness": "FA_Competence",
          "EmotionalStability": "FA_FearfulWithdrawal", "Intellect": "FA_Imagination"}
rng = np.random.default_rng(0)

S = json.load(open(f"{HERE}/analysis/nxn_scores.json"))
names = [n.replace("trait_", "") for n in S["names"]]; idx = {n: i for i, n in enumerate(names)}
norms = np.array(S["bu_norm"], dtype=float)           # target norms used to unit-normalise
# per-trait mean pair score vector over its 40 pairs
acc = {}; cnt = {}
for r in S["scores"]:
    p = np.array(r["chosen"]) - np.array(r["rejected"])
    acc[r["trait"]] = acc.get(r["trait"], 0) + p; cnt[r["trait"]] = cnt.get(r["trait"], 0) + 1
P = np.array([acc[t] / cnt[t] for t in names])          # 134 datasets x 134 unit directions
z = np.load(f"{HERE}/results/gram_sweep.npz", allow_pickle=True)
gn = [str(x) for x in z["names"]]; anorm = np.sqrt(np.diag(np.array(z["G"])))
anorm = np.array([anorm[gn.index(t)] for t in names])
coefs = {}
for f in ("steer_spec.json", "steer_spec2_7a.json"):
    for j in json.load(open(f"{HERE}/phase10_runs/{f}"))["jobs"]:
        coefs[j["name"]] = np.array([j["coef"].get(t, 0.0) for t in names])

def score_along(c, loo=True):
    """dataset scores along merge c; LOO zeroes c_t for dataset t."""
    w = c * anorm                                        # coefficient x adapter norm
    out = np.zeros(len(names))
    for t in range(len(names)):
        wt = w.copy()
        if loo: wt[t] = 0.0
        out[t] = P[t] @ wt / np.linalg.norm(wt)
    return out

# judged behaviour: stage1 minus base per trait and dimension
J = json.load(open(f"{HERE}/phase10_runs/judged_100.json"))["records"]
def prof(cond, trait=None):
    rs = [r for r in J if r["condition"] == cond and (trait is None or r["trait"] == trait) and r.get("scores")]
    return {d: np.mean([r["scores"][d] for r in rs if r["scores"].get(d) is not None]) for d in DIMS} if rs else None
base = prof("base")
shift = {}
for t in names:
    p = prof("stage1", t)
    if p: shift[t] = {d: p[d] - base[d] for d in DIMS}
have = [t for t in names if t in shift]; hi = [idx[t] for t in have]
prim = {r["trait"].lower().replace(" ", "_").replace("-", "_"): r for r in json.load(open(f"{HERE}/traits_primary.json"))}
def keyed(t, D):
    r = prim.get(t); 
    if not r or r["factor"].replace(" ", "") != D: return 0.0
    return 1.0 if r["keyed"] == "+" else -1.0

def perm_p(x, y, r, n=10000):
    return float((np.sum([abs(pearsonr(x, rng.permutation(y))[0]) >= abs(r) for _ in range(n)]) + 1) / (n + 1))

out = {"n_datasets_with_behaviour": len(have), "directions": {}}
for D in DIMS:
    y = np.array([shift[t][D] for t in have])
    lab = np.array([keyed(t, D) for t in have])
    row = {"label_baseline_pearson": float(pearsonr(lab, y)[0])}
    for kind, dname in (("axis", AXIS[D]), ("factor", FA_FOR[D])):
        for loo in (True, False):
            x = score_along(coefs[dname], loo)[hi]
            r = float(pearsonr(x, y)[0]); rho = float(spearmanr(x, y).statistic)
            # partial on the keying label: residualise both on the label
            def resid(v): 
                A = np.c_[np.ones_like(lab), lab]; b = np.linalg.lstsq(A, v, rcond=None)[0]; return v - A @ b
            rp = float(pearsonr(resid(x), resid(y))[0])
            # within the 20 markers of this factor only
            m = np.array([keyed(t, D) != 0 for t in have])
            rw = float(pearsonr(x[m], y[m])[0]) if m.sum() > 3 else None
            # held-out lexicon traits only (no label at all)
            lex = np.array([t not in prim for t in have])
            rl = float(pearsonr(x[lex], y[lex])[0]) if lex.sum() > 3 else None
            row[f"{kind}_{dname}_{'loo' if loo else 'full'}"] = {
                "pearson": r, "spearman": rho, "perm_p": perm_p(x, y, r, 5000) if loo else None,
                "partial_on_label": rp, "within_factor_markers_n": int(m.sum()), "within_factor_markers_pearson": rw,
                "lexicon_only_n": int(lex.sum()), "lexicon_only_pearson": rl}
    out["directions"][D] = row
    a = row[f"axis_{AXIS[D]}_loo"]; f = row[f"factor_{FA_FOR[D]}_loo"]
    print(f"{D:18s} label r {row['label_baseline_pearson']:+.3f} | axis LOO r {a['pearson']:+.3f} (p {a['perm_p']:.4f}) partial {a['partial_on_label']:+.3f} within-factor {a['within_factor_markers_pearson']:+.3f} lexicon n={a['lexicon_only_n']} | factor LOO r {f['pearson']:+.3f} partial {f['partial_on_label']:+.3f}")
# cross-dimension specificity: score along axis D vs shift on D' (matrix)
M = np.zeros((5, 5))
for i, D in enumerate(DIMS):
    x = score_along(coefs[AXIS[D]], True)[hi]
    for k, D2 in enumerate(DIMS):
        M[i, k] = pearsonr(x, np.array([shift[t][D2] for t in have]))[0]
out["axis_score_vs_shift_matrix_rows_axis_cols_dim"] = M.round(3).tolist()
print("axis(row) x judged dim(col):\n", M.round(2))
json.dump(out, open(f"{HERE}/analysis/data_forecast.json", "w"), indent=1); print("wrote analysis/data_forecast.json")
