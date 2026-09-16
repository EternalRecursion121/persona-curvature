"""Assemble one data blob: a page's worth of analysis for each of the 100 traits.

Quantitative comes from the completed PCA / FA / Gram; qualitative comes from the
DPO pairs the adapter was actually trained on. Nothing is recomputed here except
neighbour rankings, which are read straight off the Gram.
"""
import json, os, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.dirname(HERE)                      # sweep100/
R = os.path.join(S, "results")

traits = json.load(open(os.path.join(S, "traits.json")))
pca = json.load(open(os.path.join(R, "pca.json")))
fa = json.load(open(os.path.join(R, "fa.json")))
G = np.load(os.path.join(R, "gram.npy"))
_gn = json.load(open(os.path.join(R, "gram_names.json")))
names = _gn["names"] if isinstance(_gn, dict) else _gn

# adapter dir name <- trait word
def slug(t): return t.lower().replace("-", "_")
idx = {n: i for i, n in enumerate(names)}

# --- cosine matrix, and the mean-removed version (traits differ from each other) ---
n = len(names)
d = np.sqrt(np.diag(G))
C = G / np.outer(d, d)
one = np.ones((n, n)) / n
Gc = G - one @ G - G @ one + one @ G @ one     # double-centred
dc = np.sqrt(np.clip(np.diag(Gc), 1e-12, None))
Cc = Gc / np.outer(dc, dc)

# --- PCA loadings per trait ---
pc_load = {}
allld = pca.get("all_loadings") or {}
for word, v in allld.items():          # keyed by trait WORD -> [PC1..PCk]
    sl = word.lower().replace("-", "_")
    if isinstance(v, list):
        pc_load[sl] = {f"PC{i+1}": float(x) for i, x in enumerate(v)}

# --- FA loadings: prefer the oblimin k=5 solution ---
fa_load, fa_meta = {}, {}
sols = fa.get("solutions", {})
sol = sols.get("centred_k5") or next(iter(sols.values()), {})
order = fa.get("trait_order") or names
_L = sol.get("loadings", {})
_comm = sol.get("communalities") or []
for rot in ("oblimin", "varimax"):
    pat = _L.get(rot)
    if pat:
        arr = np.array(pat)
        for i, t in enumerate(order):
            if i < arr.shape[0]:
                fa_load.setdefault(str(t).lower().replace("-","_"), {})[rot] = [round(float(x), 3) for x in arr[i]]
        fa_meta[rot] = {"n_factors": int(arr.shape[1])}

# communality is per-trait in FA trait order; uniqueness = 1 - communality
uniq = {}
for i, t in enumerate(order):
    if i < len(_comm):
        uniq[str(t).lower().replace("-","_")] = round(1.0 - float(_comm[i]), 3)

# --- DPO pairs actually trained on ---
def pairs_for(slug_name, k=3):
    p = os.path.join(S, "data_bal", f"{slug_name}.jsonl")
    if not os.path.exists(p): return []
    rows = [json.loads(l) for l in open(p)]
    # pick pairs where chosen and rejected differ most in length -- the clearest contrast
    rows.sort(key=lambda r: -abs(len(r["chosen"]) - len(r["rejected"])))
    out = []
    for r in rows[:k]:
        out.append({"prompt": r["prompt"],
                    "chosen": r["chosen"][:900],
                    "rejected": r["rejected"][:900]})
    return out

RESEEDS = {"anxious", "creative", "organized", "shy", "warm"}
noise = pca.get("noise_floor", {})

out = {"meta": {
        "n_traits": len(traits),
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "pairs_per_trait": 214,
        "reseed_cos": 0.855, "between_trait_cos": 0.099, "ratio": 2.41,
        "fa": fa_meta,
       }, "traits": {}}

by_word = {t["trait"]: t for t in traits}
opp = {}   # same factor, opposite pole -> the canonical partner is just "any"; we rank them
for t in traits:
    sl = slug(t["trait"])
    if sl not in idx: continue
    i = idx[sl]
    row = Cc[i].copy(); row[i] = -9
    near = np.argsort(-row)[:6]
    far = np.argsort(row)[:6]
    same_fac_opp = [(names[j], round(float(Cc[i, j]), 3)) for j in range(n)
                    if names[j] != sl
                    and by_word.get(next((x["trait"] for x in traits if slug(x["trait"]) == names[j]), ""), {}).get("factor") == t["factor"]
                    and by_word.get(next((x["trait"] for x in traits if slug(x["trait"]) == names[j]), ""), {}).get("keyed") != t["keyed"]]
    same_fac_opp.sort(key=lambda x: x[1])
    out["traits"][sl] = {
        "trait": t["trait"], "factor": t["factor"], "keyed": t["keyed"],
        "norm": round(float(d[i]), 4),
        "pc": {f"PC{i}": round(float(pc_load.get(sl, {}).get(f"PC{i}", 0.0)), 3) for i in range(1, 5)},
        "fa": fa_load.get(sl, {}),
        "uniqueness": uniq.get(sl),
        "nearest": [(names[j], round(float(Cc[i, j]), 3)) for j in near],
        "furthest": [(names[j], round(float(Cc[i, j]), 3)) for j in far],
        "opposite_pole": same_fac_opp[:5],
        "is_reseed_control": sl in RESEEDS,
        "pairs": pairs_for(sl),
    }

json.dump(out, open(os.path.join(HERE, "traits_data.json"), "w"))
print(f"wrote traits_data.json: {len(out['traits'])} traits, "
      f"{os.path.getsize(os.path.join(HERE,'traits_data.json'))/1e6:.1f} MB")
missing = [t['trait'] for t in traits if slug(t['trait']) not in out['traits']]
print("missing:", missing or "none")
ex = out["traits"]["bold"]
print("sample bold: pc keys", list(ex["pc"].keys()), "| nearest", ex["nearest"][:3],
      "| pairs", len(ex["pairs"]))
