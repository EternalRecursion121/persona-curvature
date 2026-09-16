"""One page's worth of analysis per COMPONENT: the PCA components and the
oblimin factors. Quantitative from pca.json / fa.json; qualitative from the
steering generations, paired at every dose with the matched random control.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.dirname(HERE)
R = os.path.join(S, "results")

pca = json.load(open(os.path.join(R, "pca.json")))
fa = json.load(open(os.path.join(R, "fa.json")))
steer = json.load(open(os.path.join(R, "steer.json")))
traits = json.load(open(os.path.join(S, "traits.json")))
FAC = {t["trait"]: (t["factor"], t["keyed"]) for t in traits}

# ---- steering, indexed by (direction, alpha) on the misalignment probe set ----
mis = {}
for e in steer["by_condition"]:
    if e.get("set") != "misalignment":
        continue
    mis[(e["direction"], float(e["alpha"]))] = {
        "pole": e["pole"],
        "alignment": e["mean"]["alignment"],
        "coherence": e["mean"]["coherence"],
        "lucid": e.get("misaligned_rate", 0.0),
        "truncated": e.get("frac_truncated"),
        "chars": e.get("mean_chars"),
    }
qual = steer.get("qualitative", {}) or {}
worst = steer.get("worst_case", {}) or {}

def gens_for(direction, alpha, limit=3):
    """Verbatim generations at this dose, preferring the worst-case block."""
    key = f"{direction}_a{alpha:+.1f}"
    out = []
    for src in (worst, qual):
        blk = src.get(key)
        if not blk:
            continue
        for g in (blk.get("gens") or [])[:limit]:
            out.append({"prompt": g.get("prompt", ""),
                        "response": (g.get("response") or "")[:900],
                        "alignment": g.get("alignment"), "coherence": g.get("coherence")})
        if out:
            break
    return out

def steer_block(direction):
    alphas = sorted({a for (d, a) in mis if d == direction})
    rows = []
    for a in alphas:
        m = mis[(direction, a)]
        c = mis.get(("random", a))
        rows.append({"alpha": a, "pole": m["pole"],
                     "alignment": m["alignment"], "coherence": m["coherence"],
                     "lucid": m["lucid"], "truncated": m["truncated"],
                     "ctrl_alignment": c["alignment"] if c else None,
                     "ctrl_coherence": c["coherence"] if c else None,
                     "ctrl_lucid": c["lucid"] if c else None,
                     "gens": gens_for(direction, a),
                     "ctrl_gens": gens_for("random", a, 1)})
    return rows

comps = []

# ---- PCA components ----
pcvf = pca.get("pc_vs_factor_abscos")
for c in pca["components"]:
    k = int(c["pc"])
    row = None
    _m = pcvf.get("matrix") if isinstance(pcvf, dict) else pcvf
    if isinstance(_m, list) and k - 1 < len(_m):
        row = _m[k - 1]
    comps.append({
        "id": f"pc{k}", "kind": "PCA component", "title": f"PC{k}",
        "headline": f"{c['var_pct']:.1f}% of between-trait variance",
        "var_pct": c["var_pct"],
        "vs_factors": row,
        "top_positive": c.get("top_positive", [])[:10],
        "top_negative": c.get("top_negative", [])[:10],
        "steer": steer_block(f"pc{k}") if any(d == f"pc{k}" for (d, _) in mis) else [],
    })

# ---- oblimin factors ----
sol = fa["solutions"].get("centred_k5", {})
L = (sol.get("loadings") or {}).get("oblimin")
cong = sol.get("congruence_oblimin")
order = fa.get("trait_order") or [t["trait"] for t in traits]
ss = sol.get("ss_loadings", {})
ss_ob = ss.get("oblimin") if isinstance(ss, dict) else None
if L:
    A = np.array(L)
    for j in range(A.shape[1]):
        col = A[:, j]
        rank = np.argsort(-np.abs(col))
        pos = [{"trait": order[i], "loading": float(col[i]),
                "factor": FAC.get(order[i], ("", ""))[0], "keyed": FAC.get(order[i], ("", ""))[1]}
               for i in rank if col[i] > 0][:10]
        neg = [{"trait": order[i], "loading": float(col[i]),
                "factor": FAC.get(order[i], ("", ""))[0], "keyed": FAC.get(order[i], ("", ""))[1]}
               for i in rank if col[i] < 0][:10]
        cr = None
        if isinstance(cong, list) and j < len(cong):
            cr = cong[j]
        elif isinstance(cong, dict):
            cr = cong.get(f"F{j+1}")
        comps.append({
            "id": f"fa{j+1}", "kind": "oblimin factor", "title": f"Factor {j+1}",
            "headline": (f"SS loading {ss_ob[j]:.2f}" if ss_ob and j < len(ss_ob) else "oblimin, k=5"),
            "vs_factors": cr, "top_positive": pos, "top_negative": neg,
            "steer": (steer_block(f"fa{j+1}")
                      if any(d == f"fa{j+1}" for (d, _) in mis) else []),
        })

DESC = json.load(open(os.path.join(HERE, "trait_descriptions.json")))
def _slug(t): return str(t).lower().replace("-", "_")
for c in comps:
    for lst in (c["top_positive"], c["top_negative"]):
        for t in lst:
            t["desc"] = DESC.get(_slug(t["trait"]))

SUM = json.load(open(os.path.join(HERE, "summaries.json")))
for c in comps:
    c["summary"] = SUM.get(c["id"])

ANA = json.load(open(os.path.join(HERE, "analysis.json")))
for c in comps:
    c["analysis"] = ANA.get(c["id"])
    c["steer_analysis"] = ANA.get(c["id"] + "_steer")

out = {"meta": {
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "reseed_cos": 0.855, "between_trait_cos": 0.099,
        "eval_baseline": 0.447,
        "goldberg": ["E", "A", "C", "ES", "I"],
       }, "components": comps}
json.dump(out, open(os.path.join(HERE, "components_data.json"), "w"))
print(f"wrote components_data.json: {len(comps)} components "
      f"({os.path.getsize(os.path.join(HERE,'components_data.json'))/1e6:.2f} MB)")
for c in comps:
    ng = sum(len(r["gens"]) for r in c["steer"])
    print(f"  {c['id']:5s} {c['kind']:16s} +{len(c['top_positive'])}/-{len(c['top_negative'])} traits, "
          f"{len(c['steer'])} doses, {ng} gens, vs_factors={'yes' if c['vs_factors'] else 'NO'}")
