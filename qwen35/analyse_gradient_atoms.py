#!/usr/bin/env python3
"""Assemble analysis/gradient_atoms.json from the three gradient-atoms stages.

Stage 1 (GPU, gradient_atoms_on_modal.py::extract) produced the EKFAC-projected
per-completion gradients and a count sketch of the raw ones.
Stage 2 (CPU, ::atoms) did the sparse dictionary learning, coherence and label
purity on all three arms.
Stage 3 (CPU, ::weightspace) projected the named weight-space directions through
the identical EKFAC map and unprojected the atoms back, exactly.

This script does the arithmetic that needs files that live only on the host --
the judged evaluations, the Fisher norms, the exact Gram -- and writes one JSON
whose keys the wiki page cites.  It computes nothing on the GPU and rereads
nothing that the earlier stages already decided.

usage:  analyse_gradient_atoms.py [--tag zoo] [--out analysis/gradient_atoms.json]
"""
import argparse
import collections
import json
import os

import numpy as np
from scipy import stats

Q = os.path.dirname(os.path.abspath(__file__))
F5 = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability",
      "Intellect"]
# The Big Five scale each published direction is supposed to move.  For the five
# factor directions this is build_blog_page.py's FAS table; for axis_* and
# identity_* the name is the scale.  PC1-PC6 and mean_assistant_axis have no a
# priori own scale, so the empirical one recorded in
# analysis/steerfix_replication.json#<dir>.named is used and flagged.
OWN = {"FA_Warmth": "Agreeableness", "FA_Competence": "Conscientiousness",
       "FA_FearfulWithdrawal": "EmotionalStability", "FA_Arousal": "Extraversion",
       "FA_Imagination": "Intellect"}
for f in F5:
    OWN[f"axis_{f}"] = f
    OWN[f"identity_{f}"] = f


def judged_amplification():
    """Own-scale change from alpha 0 to alpha +2, per published direction."""
    rep = json.load(open(f"{Q}/analysis/steerfix_replication.json"))
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in ("judged_steerfix.json", "judged_steerfix23.json"):
        for r in json.load(open(f"{Q}/phase10_runs/{p}"))["records"]:
            for f in F5:
                v = r["scores"].get(f)
                if isinstance(v, (int, float)):
                    agg[(r["trait"], r["condition"])][f].append(v)
    out = {}
    for d in sorted({k[0] for k in agg}):
        scale = OWN.get(d) or (rep.get(d) or {}).get("named")
        if scale is None:
            continue
        a2 = agg.get((d, "a2_0"), {}).get(scale)
        a0 = agg.get((d, "a0_0"), {}).get(scale)
        if not a2 or not a0:
            continue
        out[d] = {"own_scale": scale,
                  "own_scale_source": ("a priori" if d in OWN else
                                       "analysis/steerfix_replication.json#named"),
                  "mean_a2": float(np.mean(a2)), "mean_a0": float(np.mean(a0)),
                  "amplification_alpha2": float(np.mean(a2) - np.mean(a0)),
                  "n_a2": len(a2), "n_a0": len(a0)}
    return out


def trait_amplification():
    """Stage-one adapter minus base on the trait's own Big Five factor."""
    fa = json.load(open(f"{Q}/results/fa_qwen35.json"))
    tf = fa["trait_factor"]
    slug = fa["trait_slug"]
    fmap = {}
    for i, s in enumerate(slug):
        f = tf[i] if isinstance(tf, list) else tf.get(s)
        fmap[s] = f
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in json.load(open(f"{Q}/phase10_runs/judged_100.json"))["records"]:
        for f in F5:
            v = r["scores"].get(f)
            if isinstance(v, (int, float)):
                agg[(r["trait"], r["condition"])][f].append(v)
    out = {}
    for t in sorted({k[0] for k in agg}):
        f = fmap.get(t)
        if f not in F5:
            continue
        s1 = agg.get((t, "stage1"), {}).get(f)
        b = agg.get((t, "base"), {}).get(f)
        if not s1 or not b:
            continue
        out[t] = {"factor": f, "stage1": float(np.mean(s1)), "base": float(np.mean(b)),
                  "own_factor_shift": float(np.mean(s1) - np.mean(b)), "n": len(s1)}
    return out


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return {"n": int(m.sum()), "rho": None, "p": None}
    r = stats.spearmanr(x[m], y[m])
    return {"n": int(m.sum()), "rho": float(r.statistic), "p": float(r.pvalue)}


def partial_spearman(x, y, z):
    """Spearman of x and y with z partialled out, on ranks."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 5:
        return {"n": int(m.sum()), "rho": None, "p": None}
    rx, ry, rz = (stats.rankdata(v[m]) for v in (x, y, z))
    def resid(a):
        A = np.vstack([np.ones_like(rz), rz]).T
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    r = stats.pearsonr(resid(rx), resid(ry))
    return {"n": int(m.sum()), "rho": float(r.statistic), "p": float(r.pvalue)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="zoo")
    ap.add_argument("--out", default="analysis/gradient_atoms.json")
    a = ap.parse_args()

    EX = json.load(open(f"{Q}/analysis/gradient_atoms_extract_{a.tag}.json"))
    AT = json.load(open(f"{Q}/analysis/gradient_atoms_atoms_{a.tag}.json"))
    WS = json.load(open(f"{Q}/analysis/gradient_atoms_weightspace.json"))
    Z = np.load(f"{Q}/results/gradient_atoms/{a.tag}_atoms.npz", allow_pickle=False)
    D = Z["atoms"].astype(np.float64)
    K, Dd = D.shape
    Dn = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-30)[:, None][:, 0]

    names = WS["names"]
    P = np.array(WS["target_proj"], dtype=np.float64)
    Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-30)
    CS = Dn @ Pn.T                                       # (K, T) atom-space cosine
    WC = np.array(WS["atom_target_cos"], dtype=np.float64)   # weight-space cosine

    prim = [c for c in AT["configs"] if c.get("arm") == "real"
            and c["K"] == K and "atom_rows" in c]
    prim = prim[0] if prim else [c for c in AT["configs"] if c.get("arm") == "real"][0]
    coh = np.array([r["coherence"] if r["coherence"] is not None else np.nan
                    for r in prim["atom_rows"]])

    # ---- null band for "nearest atom" in atom space -------------------------
    rng = np.random.default_rng(11)
    G = rng.normal(size=(1000, Dd))
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    nullmax = np.abs(G @ Dn.T).max(1)
    band = {"n_draws": 1000, "mean_max_abs_cos": float(nullmax.mean()),
            "p95": float(np.percentile(nullmax, 95)),
            "max": float(nullmax.max()),
            "note": "max |cos| of a random unit vector in the EKFAC projection "
                    "against the K atoms; the reference for every atom-space cosine"}

    ni = {n: i for i, n in enumerate(names)}
    traits = [n for n in names if n.startswith("trait_")]
    tref = np.array([np.abs(CS[:, ni[n]]).max() for n in traits])
    band["single_adapter_max_abs_cos"] = {
        "n": len(traits), "mean": float(tref.mean()), "min": float(tref.min()),
        "max": float(tref.max()), "median": float(np.median(tref))}

    def entry(n):
        j = ni[n]
        o = np.argsort(-np.abs(CS[:, j]))[:5]
        ow = np.argsort(-np.abs(WC[:, j]))[:5]
        return {"atom_space_max_abs_cos": float(np.abs(CS[:, j]).max()),
                "atom_space_z_vs_random": float(
                    (np.abs(CS[:, j]).max() - nullmax.mean())
                    / max(nullmax.std(), 1e-12)),
                "top_atoms": [{"atom": int(t), "cos": float(CS[t, j]),
                               "coherence": (None if not np.isfinite(coh[t])
                                             else float(coh[t]))} for t in o],
                "weight_space_max_abs_cos": float(np.abs(WC[:, j]).max()),
                "weight_space_top_atoms": [
                    {"atom": int(t), "cos": float(WC[t, j]),
                     "coherence": (None if not np.isfinite(coh[t])
                                   else float(coh[t]))} for t in ow],
                "target_norm": WS["target_norm"][j]}

    named = [n for n in names if not n.startswith("trait_")
             and not n.startswith("rand_merge_")]
    rand = [n for n in names if n.startswith("rand_merge_")]
    directions = {n: entry(n) for n in named}

    def dist(sel, M):
        v = np.array([np.abs(M[:, ni[n]]).max() for n in sel])
        return {"n": len(sel), "mean": float(v.mean()), "sd": float(v.std()),
                "min": float(v.min()), "max": float(v.max()),
                "median": float(np.median(v))}

    # THE NULL THAT MATTERS.  A random unit vector in the 6,944-dimensional
    # projection is a hopeless null: the atoms and every zoo direction live in the
    # same data-adapted subspace, so anything built out of zoo adapters scores far
    # above it.  The 20 Gaussian merges of the 134 adapters are the band this
    # project uses elsewhere, and they are the comparison the numbers below carry.
    randband = {"atom_space": dist(rand, CS), "weight_space": dist(rand, WC),
                "note": "20 Gaussian merges of the 134 stage-one adapters, the "
                        "same band construction the reward-hacks data scoring used"}
    singleband = {"atom_space": dist(traits, CS), "weight_space": dist(traits, WC)}
    for n in named:
        for sp, M, bd in (("atom_space", CS, randband["atom_space"]),
                          ("weight_space", WC, randband["weight_space"])):
            v = float(np.abs(M[:, ni[n]]).max())
            directions[n][f"{sp}_z_vs_random_merge"] = float(
                (v - bd["mean"]) / max(bd["sd"], 1e-12))
            directions[n][f"{sp}_above_widest_random_merge"] = bool(v > bd["max"])

    # ---- coherence restricted to atoms that actually have a cluster ---------
    # An atom whose code is nonzero on two documents can have a large "mean
    # pairwise cosine over its top-20" computed from those two, which is not what
    # the statistic is for.  Every coherence count is therefore reported twice:
    # over all atoms, as the paper does, and over atoms with at least 20 active
    # documents.
    def coh_stats(cfg):
        v = [(r["coherence"], r["n_active"]) for r in cfg.get("atom_rows", [])
             if r["coherence"] is not None]
        a = np.array([x for x, _ in v])
        n = np.array([m for _, m in v])
        keep = a[n >= 20]
        return {"n_atoms": len(v), "n_with_20_active": int((n >= 20).sum()),
                "all": {"gt_0_5": int((a > 0.5).sum()), "gt_0_1": int((a > 0.1).sum()),
                        "max": float(a.max()), "mean": float(a.mean()),
                        "median": float(np.median(a))},
                "n_active_ge_20": {
                    "gt_0_5": int((keep > 0.5).sum()),
                    "gt_0_1": int((keep > 0.1).sum()),
                    "max": (float(keep.max()) if len(keep) else None),
                    "mean": (float(keep.mean()) if len(keep) else None),
                    "median": (float(np.median(keep)) if len(keep) else None)}}

    cohstats = {c["arm"]: coh_stats(c) for c in AT["configs"] if "atom_rows" in c}

    # ---- the ten most coherent atoms, named by what they are nearest -------
    nact = np.array([r["n_active"] for r in prim["atom_rows"]])
    order = np.argsort(-np.nan_to_num(np.where(nact >= 20, coh, np.nan),
                                      nan=-1))[:15]
    topatoms = []
    for t in order:
        r = prim["atom_rows"][int(t)]
        j = np.argsort(-np.abs(WC[int(t)]))[:3]
        topatoms.append({
            "atom": int(t), "coherence": r["coherence"],
            "n_active": r["n_active"],
            "trait_counts": dict(collections.Counter(r["top_traits"]).most_common()),
            "factor_counts": dict(collections.Counter(r["top_factors"]).most_common()),
            "keyed_counts": dict(collections.Counter(r["top_keyed"]).most_common()),
            "neg_trait_counts": dict(collections.Counter(
                r.get("top_neg_traits") or []).most_common()),
            "weight_space_nearest": [{"target": names[int(x)],
                                      "cos": float(WC[int(t), int(x)])} for x in j]})

    # ---- the two poles of an atom -------------------------------------------
    # Sparse coding is equivariant to flipping a document's sign together with its
    # code, and swapping a DPO pair negates its gradient almost exactly (the SFT
    # term is the only part that does not flip, and it is about 2% of the
    # gradient), so the shuffled arm CANNOT destroy the dictionary.  What it moves
    # is which pole of an atom a pair sits on.  For the real arm the question the
    # project's own bipolarity warning asks is whether an atom's two poles are the
    # two keyings of one factor.
    def poles(cfg):
        out = {"n_atoms": 0, "same_majority_trait": 0, "same_majority_factor": 0,
               "same_factor_opposite_keying": 0}
        for r in cfg.get("atom_rows", []):
            pt, nt = r["top_traits"], r.get("top_neg_traits") or []
            pf, nf = r["top_factors"], r.get("top_neg_factors") or []
            pk, nk = r["top_keyed"], r.get("top_neg_keyed") or []
            if len(pt) < 2 or len(nt) < 2:
                continue
            out["n_atoms"] += 1
            mt = collections.Counter(pt).most_common(1)[0][0]
            mtn = collections.Counter(nt).most_common(1)[0][0]
            mf = collections.Counter(pf).most_common(1)[0][0]
            mfn = collections.Counter(nf).most_common(1)[0][0]
            mk = collections.Counter(pk).most_common(1)[0][0]
            mkn = collections.Counter(nk).most_common(1)[0][0]
            out["same_majority_trait"] += int(mt == mtn)
            out["same_majority_factor"] += int(mf == mfn)
            out["same_factor_opposite_keying"] += int(mf == mfn and mk != mkn)
        n = max(out["n_atoms"], 1)
        for k in ("same_majority_trait", "same_majority_factor",
                  "same_factor_opposite_keying"):
            out[k + "_frac"] = out[k] / n
        return out

    pole = {c["arm"]: poles(c) for c in AT["configs"] if "atom_rows" in c}

    # ---- G4 -----------------------------------------------------------------
    amp = judged_amplification()
    g4d = []
    for d, rec in sorted(amp.items()):
        if d not in ni:
            continue
        e = directions[d]
        g4d.append({"direction": d, **rec,
                    "nearest_atom": e["top_atoms"][0]["atom"],
                    "nearest_atom_cos": e["top_atoms"][0]["cos"],
                    "nearest_atom_coherence": e["top_atoms"][0]["coherence"],
                    "mean_top5_coherence": float(np.nanmean(
                        [t["coherence"] for t in e["top_atoms"]
                         if t["coherence"] is not None])),
                    "weight_space_nearest_coherence":
                        e["weight_space_top_atoms"][0]["coherence"]})
    g4 = {"per_direction": g4d,
          "coherence_vs_amplification": spearman(
              [r["nearest_atom_coherence"] for r in g4d],
              [r["amplification_alpha2"] for r in g4d]),
          "top5_coherence_vs_amplification": spearman(
              [r["mean_top5_coherence"] for r in g4d],
              [r["amplification_alpha2"] for r in g4d]),
          "coherence_vs_abs_amplification": spearman(
              [r["nearest_atom_coherence"] for r in g4d],
              [abs(r["amplification_alpha2"]) for r in g4d])}

    # per-trait coherence against the trait's own judged shift and its norms
    tc = AT.get("per_trait_coherence", {})
    ta = trait_amplification()
    gz = np.load(f"{Q}/results/gram_sweep.npz")
    gnames = [str(x) for x in gz["names"]]
    fro = {n: float(gz["norms"][i]) for i, n in enumerate(gnames)}
    fn = json.load(open(f"{Q}/analysis/fisher_norms.json"))
    fdir = fn.get("directions", {})
    fsingle = {k[len("single_"):]: v for k, v in fdir.items()
               if k.startswith("single_")}
    rows = []
    for t, c in sorted(tc.items()):
        if t.startswith("_"):
            continue
        f = fsingle.get(t)
        rows.append({"trait": t, "coherence": c,
                     "own_factor_shift": (ta.get(t) or {}).get("own_factor_shift"),
                     "frobenius_norm": fro.get(t),
                     "fisher_F": (f.get("F_ref") if isinstance(f, dict) else f)})
    g4t = {"n_traits": len(rows), "rows": rows,
           "within_trait_mean": tc.get("_within_trait_mean"),
           "cross_trait_mean": tc.get("_cross_trait_mean"),
           "cross_trait_sd": tc.get("_cross_trait_sd"),
           "coherence_vs_own_factor_shift": spearman(
               [r["coherence"] for r in rows],
               [r["own_factor_shift"] for r in rows]),
           "coherence_vs_frobenius": spearman(
               [r["coherence"] for r in rows], [r["frobenius_norm"] for r in rows]),
           "coherence_vs_fisher_F": spearman(
               [r["coherence"] for r in rows], [r["fisher_F"] for r in rows]),
           "coherence_vs_own_factor_shift_partial_frobenius": partial_spearman(
               [r["coherence"] for r in rows],
               [r["own_factor_shift"] for r in rows],
               [r["frobenius_norm"] for r in rows])}

    out = {
        "what": "Gradient atoms (Rosser, arXiv:2603.14665v2) on the zoo's own DPO "
                "preference gradients, with both null corpora, and coherence "
                "against steerability.  Experiments G2 and G4 of "
                "wiki/pages/history/paper-reading-2026-09-09.md.",
        "extraction": EX,
        "atoms": {k: v for k, v in AT.items() if k != "per_trait_coherence"},
        "per_trait_coherence": tc,
        "atom_space": {"K": int(K), "dim": int(Dd), "null_band": band,
                       "random_merge_band": randband,
                       "single_adapter_band": singleband,
                       "directions": directions,
                       "top_atoms": topatoms},
        "poles": pole,
        "coherence_by_arm": cohstats,
        "g4_directions": g4,
        "g4_traits": g4t,
        "weightspace_seconds": WS.get("seconds"),
    }
    with open(f"{Q}/{a.out}", "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}")
    print(f"atoms K={K} dim={Dd}; null max|cos| mean {band['mean_max_abs_cos']:.4f}")
    for d in ("FA_Warmth", "FA_Competence", "FA_FearfulWithdrawal", "FA_Arousal",
              "FA_Imagination"):
        if d in directions:
            e = directions[d]
            print(f"  {d}: atom-space max|cos| {e['atom_space_max_abs_cos']:.4f} "
                  f"(z {e['atom_space_z_vs_random']:.1f}), weight-space "
                  f"{e['weight_space_max_abs_cos']:.4f}")
    print("G4 directions:", g4["coherence_vs_amplification"])
    print("G4 traits:", g4t["coherence_vs_own_factor_shift"])


if __name__ == "__main__":
    main()
