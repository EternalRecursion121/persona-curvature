"""PCA over a 100-trait LoRA sweep (Goldberg Big-Five markers).

Each adapter's effective delta dW = (alpha/r) * B @ A, concatenated over all
target modules, is ONE data point in a very high dimensional space. We never
materialise dW: everything needed (PCA, cosines, distances, factor directions)
is a function of the pairwise Frobenius Gram matrix, which is computed from the
FACTORED form, per module, with two BLAS calls:

    <dW_i, dW_j> = s_i s_j * sum_m sum( (B_i^T B_j) * (A_i A_j^T) )

Machinery follows ../pca_over_loras.py, but STREAMS one module at a time across
all adapters (105 adapters x 115MB fp32 does not fit in 7GB RAM).

Usage:  ~/cartovenv/bin/python analyse_pca.py
Outputs: results/gram.npy, results/pca.json, results/pca.md
"""
import itertools
import json
import os
import sys
import time

import numpy as np
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
ADIR = os.path.join(HERE, "adapters")
RDIR = os.path.join(HERE, "results")
FACTORS = ["Extraversion", "Agreeableness", "Conscientiousness",
           "EmotionalStability", "Intellect"]
FSHORT = {"Extraversion": "E", "Agreeableness": "A", "Conscientiousness": "C",
          "EmotionalStability": "ES", "Intellect": "I"}
NPERM = 10000
SEED = 0


# ---------------------------------------------------------------- verification

def sanity_check_gram():
    """Factored Gram vs dense reference on tiny random A/B pairs."""
    rng = np.random.default_rng(1234)
    out, inn, r = 7, 5, 3
    A1, B1 = rng.normal(size=(r, inn)), rng.normal(size=(out, r))
    A2, B2 = rng.normal(size=(r, inn)), rng.normal(size=(out, r))
    s1, s2 = 2.0, 2.0
    dense = [s1 * B1 @ A1, s2 * B2 @ A2]
    Gd = np.array([[float((dense[i] * dense[j]).sum()) for j in range(2)]
                   for i in range(2)])
    # factored, exactly as in gram_streaming
    Bst = np.concatenate([B1, B2], axis=1)
    Ast = np.concatenate([A1, A2], axis=0)
    M = (Bst.T @ Bst) * (Ast @ Ast.T)
    blk = M.reshape(2, r, 2, r).sum(axis=(1, 3))
    Gf = blk * np.outer([s1, s2], [s1, s2])
    err = float(np.abs(Gd - Gf).max() / np.abs(Gd).max())
    return {"dense": Gd.tolist(), "factored": Gf.tolist(),
            "max_rel_err": err, "ok": bool(err < 1e-12)}


def anova_f(vals, labels):
    """One-way ANOVA F statistic. labels: integer group ids."""
    vals = np.asarray(vals, dtype=np.float64)
    labels = np.asarray(labels)
    groups = np.unique(labels)
    k, n = len(groups), len(vals)
    gm = vals.mean()
    ssb = sum(len(vals[labels == g]) * (vals[labels == g].mean() - gm) ** 2
              for g in groups)
    ssw = sum(((vals[labels == g] - vals[labels == g].mean()) ** 2).sum()
              for g in groups)
    if ssw <= 0:
        return float("inf"), ssb, ssw
    return float((ssb / (k - 1)) / (ssw / (n - k))), float(ssb), float(ssw)


def sanity_check_anova():
    """Hand-checked example: groups [1,2,3],[4,5,6],[7,8,9].
    grand mean 5; SSB = 3*((2-5)^2+(5-5)^2+(8-5)^2) = 54; SSW = 2+2+2 = 6;
    F = (54/2)/(6/6) = 27.0 exactly."""
    v = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    lab = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    f, ssb, ssw = anova_f(v, lab)
    return {"F": f, "SSB": ssb, "SSW": ssw, "expected_F": 27.0,
            "ok": bool(abs(f - 27.0) < 1e-12)}


def perm_p(vals, labels, nperm=NPERM, seed=SEED):
    """Permutation p-value for one-way ANOVA F by shuffling group labels."""
    obs = anova_f(vals, labels)[0]
    rng = np.random.default_rng(seed)
    lab = np.asarray(labels).copy()
    ge = 0
    for _ in range(nperm):
        rng.shuffle(lab)
        if anova_f(vals, lab)[0] >= obs:
            ge += 1
    return obs, (ge + 1) / (nperm + 1)


# ------------------------------------------------------------------- gram

def adapter_meta(name):
    cfg = json.load(open(f"{ADIR}/{name}/adapter_config.json"))
    assert not cfg.get("use_rslora"), f"{name}: rslora"
    assert not cfg.get("use_dora"), f"{name}: dora"
    assert not cfg.get("rank_pattern") and not cfg.get("alpha_pattern"), name
    return cfg["r"], cfg["lora_alpha"], cfg["lora_alpha"] / cfg["r"]


def gram_streaming(names, log=sys.stderr):
    """Full Gram over `names`, streaming one module at a time."""
    n = len(names)
    scales, rs = [], []
    for a in names:
        r, al, s = adapter_meta(a)
        rs.append(r)
        scales.append(s)
    assert len(set(rs)) == 1, f"mixed ranks: {set(rs)}"
    assert len(set(scales)) == 1, f"mixed scales: {set(scales)}"
    r = rs[0]
    sc = np.array(scales)

    # guard against half-downloaded adapters: all files must be the same size
    sizes = {a: os.path.getsize(f"{ADIR}/{a}/adapter_model.safetensors")
             for a in names}
    med = np.median(list(sizes.values()))
    bad = {a: s for a, s in sizes.items() if abs(s - med) > 0.001 * med}
    assert not bad, f"adapter files with anomalous size (incomplete fetch?): {bad}"

    handles = {a: safe_open(f"{ADIR}/{a}/adapter_model.safetensors",
                            framework="numpy") for a in names}
    keysets = [set(k.replace(".lora_A.weight", "")
                   for k in h.keys() if "lora_A" in k) for h in handles.values()]
    mods = sorted(set.intersection(*keysets))
    extra = set.union(*keysets) - set(mods)
    assert not extra, f"modules not shared by all adapters: {sorted(extra)[:5]}"

    G = np.zeros((n, n))
    t0 = time.time()
    for mi, m in enumerate(mods):
        ka, kb = m + ".lora_A.weight", m + ".lora_B.weight"
        A0 = handles[names[0]].get_tensor(ka)
        B0 = handles[names[0]].get_tensor(kb)
        assert A0.shape[0] == r and B0.shape[1] == r, (m, A0.shape, B0.shape)
        Ast = np.empty((n * r, A0.shape[1]), dtype=np.float64)
        Bst = np.empty((B0.shape[0], n * r), dtype=np.float64)
        for i, a in enumerate(names):
            Ast[i * r:(i + 1) * r] = handles[a].get_tensor(ka)
            Bst[:, i * r:(i + 1) * r] = handles[a].get_tensor(kb)
        M = (Bst.T @ Bst) * (Ast @ Ast.T)
        G += M.reshape(n, r, n, r).sum(axis=(1, 3)) * np.outer(sc, sc)
        del Ast, Bst, M
        if mi % 25 == 0 or mi == len(mods) - 1:
            print(f"  module {mi+1}/{len(mods)}  {time.time()-t0:.0f}s",
                  file=log, flush=True)
    G = 0.5 * (G + G.T)
    return G, mods, float(sc[0]), r


# ------------------------------------------------------------------ helpers

def cos_from_G(G):
    d = np.sqrt(np.diag(G))
    return G / np.outer(d, d)


def dist_from_G(G, i, j):
    return float(np.sqrt(max(G[i, i] + G[j, j] - 2 * G[i, j], 0.0)))


def main():
    os.makedirs(RDIR, exist_ok=True)
    out = {}

    # ---- verification -----------------------------------------------------
    sc_gram = sanity_check_gram()
    sc_anova = sanity_check_anova()
    assert sc_gram["ok"], sc_gram
    assert sc_anova["ok"], sc_anova
    out["verification"] = {"gram_factored_vs_dense": sc_gram,
                           "anova_hand_check": sc_anova}
    print("sanity: gram rel_err %.2e, anova F=%.6f" %
          (sc_gram["max_rel_err"], sc_anova["F"]), file=sys.stderr)

    # ---- trait -> adapter mapping ----------------------------------------
    traits = json.load(open(f"{HERE}/traits.json"))
    assert len(traits) == 100, len(traits)

    def adir_name(t):
        return t.lower().replace("-", "_")

    names, missing = [], []
    for t in traits:
        d = adir_name(t["trait"])
        (names if os.path.isdir(f"{ADIR}/{d}") else missing).append(d)
    assert not missing, f"MISSING adapter dirs for traits: {missing}"
    assert len(set(names)) == 100, "duplicate adapter dir names"

    reseeds = [d for d in sorted(os.listdir(ADIR)) if d.endswith("__s1")]
    for rd in reseeds:
        assert rd[:-4] in names, f"reseed {rd} has no base trait"
    allnames = names + reseeds
    out["coverage"] = {"n_traits": len(names), "n_reseeds": len(reseeds),
                       "missing": missing, "reseeds": reseeds}

    # ---- gram -------------------------------------------------------------
    gpath = f"{RDIR}/gram.npy"
    npath = f"{RDIR}/gram_names.json"
    if os.path.exists(gpath) and json.load(open(npath))["names"] == allnames:
        Gall = np.load(gpath)
        meta = json.load(open(npath))
        mods, scale, r = meta["modules"], meta["scale"], meta["r"]
        print("loaded cached gram", file=sys.stderr)
    else:
        Gall, mods, scale, r = gram_streaming(allnames)
        np.save(gpath, Gall)
        json.dump({"names": allnames, "modules": mods, "scale": scale, "r": r},
                  open(npath, "w"))
    out["setup"] = {"n_modules": len(mods), "r": r, "scale_alpha_over_r": scale,
                    "dW_formula": f"dW = {scale:g} * B @ A per module",
                    "base_model": json.load(
                        open(f"{ADIR}/{allnames[0]}/adapter_config.json"))[
                        "base_model_name_or_path"]}

    IDX = {a: i for i, a in enumerate(allnames)}
    ti = np.array([IDX[a] for a in names])
    G = Gall[np.ix_(ti, ti)]          # 100 x 100 traits only
    Call = cos_from_G(Gall)
    norms = np.sqrt(np.diag(G))
    factor = np.array([t["factor"] for t in traits])
    keyed = np.array([t["keyed"] for t in traits])
    label = np.array([t["trait"] for t in traits])
    fid = np.array([FACTORS.index(f) for f in factor])

    # ---- 6. noise floor ---------------------------------------------------
    n = 100
    off = ~np.eye(n, dtype=bool)
    cosT = cos_from_G(G)
    dmat = np.sqrt(np.maximum(np.diag(G)[:, None] + np.diag(G)[None, :]
                              - 2 * G, 0.0))
    mean_between_d = float(dmat[off].mean())
    mean_norm = float(norms.mean())
    nf = []
    for rd in reseeds:
        base = rd[:-4]
        i, j = IDX[base], IDX[rd]
        d = dist_from_G(Gall, i, j)
        nb = 0.5 * (np.sqrt(Gall[i, i]) + np.sqrt(Gall[j, j]))
        nf.append({"trait": base, "reseed_dist": d, "reseed_dist_over_norm": d / nb,
                   "reseed_cos": float(Call[i, j]),
                   "reseed_dist_over_mean_between_trait_dist": d / mean_between_d})
    out["noise_floor"] = {
        "per_reseed": nf,
        "mean_reseed_dist": float(np.mean([x["reseed_dist"] for x in nf])),
        "mean_reseed_cos": float(np.mean([x["reseed_cos"] for x in nf])),
        "mean_reseed_dist_over_norm": float(
            np.mean([x["reseed_dist_over_norm"] for x in nf])),
        "mean_between_trait_dist": mean_between_d,
        "mean_between_trait_cos": float(cosT[off].mean()),
        "mean_dW_norm": mean_norm,
        "ratio_between_over_reseed": mean_between_d / float(
            np.mean([x["reseed_dist"] for x in nf])),
    }

    # ---- 2. PCA -----------------------------------------------------------
    H = np.eye(n) - np.ones((n, n)) / n
    Gc = H @ G @ H
    Gc = 0.5 * (Gc + Gc.T)
    lu, Uu = np.linalg.eigh(G)
    lu, Uu = np.clip(lu[::-1], 0, None), Uu[:, ::-1]
    lc, Uc = np.linalg.eigh(Gc)
    lc, Uc = np.clip(lc[::-1], 0, None), Uc[:, ::-1]

    # sign convention: largest-|score| trait is positive
    S = Uc * np.sqrt(lc)[None, :]
    for k in range(S.shape[1]):
        if S[np.argmax(np.abs(S[:, k])), k] < 0:
            S[:, k] *= -1
            Uc[:, k] *= -1

    e = np.ones(n) / n
    mean_norm2 = float(e @ G @ e)
    trG = float(np.trace(G))
    out["pca"] = {
        "uncentered_var_pct": [float(100 * x / lu.sum()) for x in lu[:10]],
        "centered_var_pct": [float(100 * x / lc.sum()) for x in lc[:10]],
        "centered_cum_pct": [float(100 * np.cumsum(lc)[i] / lc.sum())
                             for i in range(10)],
        "shared_mean_share_of_uncentered_energy_pct":
            100 * n * mean_norm2 / trG,
        "cos_uncenteredPC1_vs_mean_direction": float(
            abs(Uu[:, 0] @ (G @ e)) / (np.sqrt(lu[0]) * np.sqrt(mean_norm2))),
        "sign_convention": "each PC oriented so the trait with largest |score| is positive",
    }

    # ---- 3. what each component represents --------------------------------
    comps = []
    for k in range(6):
        o = np.argsort(S[:, k])
        neg = [{"trait": label[i], "loading": float(S[i, k]),
                "factor": factor[i], "keyed": keyed[i]} for i in o[:10]]
        pos = [{"trait": label[i], "loading": float(S[i, k]),
                "factor": factor[i], "keyed": keyed[i]} for i in o[::-1][:10]]
        comps.append({"pc": k + 1, "var_pct": float(100 * lc[k] / lc.sum()),
                      "top_positive": pos, "top_negative": neg})
    out["components"] = comps
    out["all_loadings"] = {label[i]: [float(S[i, k]) for k in range(6)]
                           for i in range(n)}

    # ---- 4. ground truth: Big Five ----------------------------------------
    # 4a. factor directions f_F = mean(+ keyed) - mean(- keyed), as coeff vectors
    Cf = np.zeros((5, n))
    for a, F in enumerate(FACTORS):
        p = (factor == F) & (keyed == "+")
        m = (factor == F) & (keyed == "-")
        Cf[a, p] = 1.0 / p.sum()
        Cf[a, m] = -1.0 / m.sum()
    Gff = Cf @ G @ Cf.T                      # inner products of factor dirs
    fn = np.sqrt(np.diag(Gff))
    Cff = Gff / np.outer(fn, fn)
    out["factor_directions"] = {
        "n_plus_minus": {F: [int(((factor == F) & (keyed == "+")).sum()),
                             int(((factor == F) & (keyed == "-")).sum())]
                         for F in FACTORS},
        "norms": {FACTORS[a]: float(fn[a]) for a in range(5)},
        "norm_over_mean_trait_norm": {FACTORS[a]: float(fn[a] / mean_norm)
                                      for a in range(5)},
        "pairwise_cos": Cff.tolist(),
        "labels": FACTORS,
    }

    # 4b. |cos| between each centered PC direction and each factor direction
    HGC = H @ G @ Cf.T                       # (n,5): x_i . f_a
    PCF = np.zeros((6, 5))
    for k in range(6):
        for a in range(5):
            PCF[k, a] = abs(Uc[:, k] @ HGC[:, a]) / (np.sqrt(lc[k]) * fn[a])
    out["pc_vs_factor_abscos"] = {"matrix": PCF.tolist(), "labels": FACTORS}

    # also: |cos| with the grand-mean direction (the "shared" component)
    out["pc_vs_mean_abscos"] = [
        float(abs(Uc[:, k] @ (H @ G @ e)) / (np.sqrt(lc[k]) * np.sqrt(mean_norm2)))
        for k in range(6)]

    # 4c. ANOVA of loadings by factor membership, permutation p
    anovas = []
    for k in range(6):
        f, p = perm_p(S[:, k], fid)
        gm = {FACTORS[a]: float(S[fid == a, k].mean()) for a in range(5)}
        anovas.append({"pc": k + 1, "F": f, "p_perm": p, "df": [4, n - 5],
                       "group_means": gm})
    # The raw-loading ANOVA is confounded by bipolarity: a PC that IS a factor
    # splits that factor's + and - markers to opposite signs, so the group mean
    # is ~0 and F is small. Two polarity-aware variants actually answer the
    # question "does factor membership explain this PC?".
    pole = np.where(keyed == "+", 1.0, -1.0)
    anovas_abs, anovas_signed = [], []
    for k in range(6):
        f, p = perm_p(np.abs(S[:, k]), fid)
        anovas_abs.append({"pc": k + 1, "F": f, "p_perm": p,
                           "group_means": {FACTORS[a]: float(np.abs(S[fid == a, k]).mean())
                                           for a in range(5)}})
        v = S[:, k] * pole      # + if the PC agrees with the factor's keyed axis
        f, p = perm_p(v, fid)
        anovas_signed.append({"pc": k + 1, "F": f, "p_perm": p,
                              "group_means": {FACTORS[a]: float(v[fid == a].mean())
                                              for a in range(5)}})

    # bonus: does keyed sign explain loadings? (2-group ANOVA)
    kid = (keyed == "+").astype(int)
    anovas_keyed = []
    for k in range(6):
        f, p = perm_p(S[:, k], kid)
        anovas_keyed.append({"pc": k + 1, "F": f, "p_perm": p})
    out["anova_by_factor"] = anovas
    out["anova_by_factor_abs_loading"] = anovas_abs
    out["anova_by_factor_pole_signed_loading"] = anovas_signed
    out["anova_by_keyed_sign"] = anovas_keyed

    # ---- 5. polarity ------------------------------------------------------
    def pair_stats(C):
        same_f_same_p, same_f_opp_p, diff_f = [], [], []
        for i, j in itertools.combinations(range(n), 2):
            v = float(C[i, j])
            if factor[i] == factor[j]:
                (same_f_same_p if keyed[i] == keyed[j] else same_f_opp_p).append(v)
            else:
                diff_f.append(v)
        return {"same_factor_same_pole": {"mean": float(np.mean(same_f_same_p)),
                                          "sd": float(np.std(same_f_same_p)),
                                          "n": len(same_f_same_p)},
                "same_factor_opposite_pole": {"mean": float(np.mean(same_f_opp_p)),
                                              "sd": float(np.std(same_f_opp_p)),
                                              "n": len(same_f_opp_p)},
                "different_factor": {"mean": float(np.mean(diff_f)),
                                     "sd": float(np.std(diff_f)),
                                     "n": len(diff_f)}}

    Ccen = cos_from_G(Gc + 1e-300 * np.eye(n))
    out["polarity"] = {"raw_cosine": pair_stats(cosT),
                       "mean_removed_cosine": pair_stats(Ccen),
                       "note": "raw = cos(dW_i,dW_j); mean_removed = cos after "
                               "subtracting the grand mean dW over the 100 traits"}
    # per-factor breakdown, raw
    pf = {}
    for F in FACTORS:
        ii = np.where(factor == F)[0]
        ss, oo = [], []
        for i, j in itertools.combinations(ii, 2):
            (ss if keyed[i] == keyed[j] else oo).append(float(cosT[i, j]))
        pf[F] = {"same_pole_mean": float(np.mean(ss)),
                 "opp_pole_mean": float(np.mean(oo)),
                 "same_pole_mean_centered": float(np.mean(
                     [Ccen[i, j] for i, j in itertools.combinations(ii, 2)
                      if keyed[i] == keyed[j]])),
                 "opp_pole_mean_centered": float(np.mean(
                     [Ccen[i, j] for i, j in itertools.combinations(ii, 2)
                      if keyed[i] != keyed[j]]))}
    out["polarity"]["per_factor_raw_and_centered"] = pf

    json.dump(out, open(f"{RDIR}/pca.json", "w"), indent=1)
    write_md(out, S, label, factor, keyed, lc, lu)
    print("wrote results/pca.json and results/pca.md", file=sys.stderr)
    return out


# ------------------------------------------------------------------ markdown

def write_md(o, S, label, factor, keyed, lc, lu):
    L = []
    W = L.append
    W("# PCA over 100 Big-Five trait LoRAs\n")
    W(f"Base model `{o['setup']['base_model']}`, "
      f"{o['setup']['n_modules']} LoRA modules, r={o['setup']['r']}, "
      f"scaling alpha/r = {o['setup']['scale_alpha_over_r']:g} "
      f"(`{o['setup']['dW_formula']}`, confirmed from adapter_config.json; "
      f"use_rslora/use_dora both false).\n")
    W(f"{o['coverage']['n_traits']}/100 traits mapped to adapter directories "
      f"and loaded; missing: {o['coverage']['missing'] or 'none'}. "
      f"{o['coverage']['n_reseeds']} reseed controls "
      f"(`{'`, `'.join(o['coverage']['reseeds'])}`) held out of the PCA and "
      f"used as the noise floor.\n")
    v = o["verification"]
    W("**Verification.** Factored Gram vs dense reference on two tiny random "
      f"A/B pairs: max relative error "
      f"{v['gram_factored_vs_dense']['max_rel_err']:.2e} "
      f"(dense {np.array(v['gram_factored_vs_dense']['dense']).round(4).tolist()}, "
      f"factored {np.array(v['gram_factored_vs_dense']['factored']).round(4).tolist()}). "
      f"ANOVA F hand-check on groups [1,2,3],[4,5,6],[7,8,9]: "
      f"F={v['anova_hand_check']['F']:.6f} vs expected 27.0 "
      f"(SSB {v['anova_hand_check']['SSB']:.1f}, SSW {v['anova_hand_check']['SSW']:.1f}).\n")

    # ---- noise floor
    nf = o["noise_floor"]
    W("\n## 1. Noise floor (read everything below against this)\n")
    W("Same trait, different training seed. This is how far apart two adapters "
      "are when *nothing* differs but the seed.\n")
    W("| trait | dist(trait, reseed) | / mean ||dW|| | cos(trait, reseed) | dist / mean between-trait dist |")
    W("|---|---|---|---|---|")
    for x in nf["per_reseed"]:
        W(f"| {x['trait']} | {x['reseed_dist']:.4f} | "
          f"{x['reseed_dist_over_norm']:.3f} | {x['reseed_cos']:.4f} | "
          f"{x['reseed_dist_over_mean_between_trait_dist']:.3f} |")
    W(f"\n- mean reseed distance **{nf['mean_reseed_dist']:.4f}**, "
      f"mean reseed cosine **{nf['mean_reseed_cos']:.4f}**")
    W(f"- mean between-trait distance **{nf['mean_between_trait_dist']:.4f}**, "
      f"mean between-trait cosine **{nf['mean_between_trait_cos']:.4f}**")
    W(f"- mean ||dW|| = {nf['mean_dW_norm']:.4f}")
    W(f"- **between-trait / reseed distance ratio = {nf['ratio_between_over_reseed']:.3f}**"
      " — the closer to 1.0, the more of the apparent trait geometry is seed noise.\n")

    # ---- spectra
    p = o["pca"]
    W("\n## 2. Variance spectra\n")
    W("| PC | uncentered var % | centered var % | centered cumulative % |")
    W("|---|---|---|---|")
    for i in range(10):
        W(f"| {i+1} | {p['uncentered_var_pct'][i]:.2f} | "
          f"{p['centered_var_pct'][i]:.2f} | {p['centered_cum_pct'][i]:.2f} |")
    W(f"\nThe **uncentered** spectrum includes the grand mean: "
      f"{p['shared_mean_share_of_uncentered_energy_pct']:.1f}% of total "
      f"uncentered energy is the shared mean dW, and uncentered PC1 has "
      f"|cos| = {p['cos_uncenteredPC1_vs_mean_direction']:.4f} with the mean "
      "direction — i.e. uncentered PC1 is essentially \"an adapter was trained "
      "here\", not a personality axis.")
    W("The **centered** spectrum is the one that describes how traits DIFFER; "
      "use it for everything below.\n")

    # ---- headline
    W("\n## 3. What each component is (HEADLINE)\n")
    W(f"Loadings are principal-coordinate scores, score_ik = sqrt(lambda_k) u_ik. "
      f"Sign is arbitrary in PCA; convention here: {p['sign_convention']}.\n")
    for c in o["components"]:
        W(f"\n### PC{c['pc']} — {c['var_pct']:.2f}% of centered variance\n")
        W("| rank | + trait | loading | factor | keyed | | - trait | loading | factor | keyed |")
        W("|---|---|---|---|---|---|---|---|---|---|")
        for i in range(10):
            a, b = c["top_positive"][i], c["top_negative"][i]
            W(f"| {i+1} | {a['trait']} | {a['loading']:+.4f} | "
              f"{FSHORT[a['factor']]} | {a['keyed']} | | {b['trait']} | "
              f"{b['loading']:+.4f} | {FSHORT[b['factor']]} | {b['keyed']} |")
    W("\n(factor codes: E Extraversion, A Agreeableness, C Conscientiousness, "
      "ES EmotionalStability, I Intellect)\n")

    # ---- ground truth
    fd = o["factor_directions"]
    W("\n## 4. Ground truth: does the PCA recover the Big Five?\n")
    W("### 4a. Factor directions and their mutual angles\n")
    W("Factor direction f_F = mean(dW of +keyed markers) - mean(dW of -keyed markers).\n")
    W("| | " + " | ".join(FSHORT[f] for f in FACTORS) + " | ||f|| / mean||dW|| | n(+/-) |")
    W("|---|" + "---|" * 7)
    for a, F in enumerate(FACTORS):
        row = " | ".join(f"{fd['pairwise_cos'][a][b]:+.3f}" for b in range(5))
        W(f"| **{FSHORT[F]}** | {row} | "
          f"{fd['norm_over_mean_trait_norm'][F]:.3f} | "
          f"{fd['n_plus_minus'][F][0]}/{fd['n_plus_minus'][F][1]} |")
    W("")
    W("### 4b. |cos| between centered PCs and factor directions\n")
    W("| PC | " + " | ".join(FSHORT[f] for f in FACTORS) + " | max | (|cos| w/ mean dW) |")
    W("|---|" + "---|" * 7)
    for k in range(6):
        row = o["pc_vs_factor_abscos"]["matrix"][k]
        W(f"| PC{k+1} | " + " | ".join(f"{x:.3f}" for x in row) +
          f" | **{max(row):.3f}** | {o['pc_vs_mean_abscos'][k]:.3f} |")
    W("\n(A PC that *was* a Big Five factor would show |cos| near 1 in one column.)\n")
    W("### 4c. Does factor membership explain the loadings?\n")
    W(f"One-way ANOVA across the 5 factor groups, permutation p from "
      f"{NPERM} label shuffles (p floor = {1/(NPERM+1):.1e}).\n")
    W("| PC | F (df 4,95) | p_perm | " +
      " | ".join(f"mean {FSHORT[f]}" for f in FACTORS) + " |")
    W("|---|---|---|" + "---|" * 5)
    for a in o["anova_by_factor"]:
        W(f"| PC{a['pc']} | {a['F']:.3f} | {a['p_perm']:.4f} | " +
          " | ".join(f"{a['group_means'][f]:+.4f}" for f in FACTORS) + " |")
    W("\n**Caveat: this raw-loading test is confounded by bipolarity.** A PC "
      "that *is* a factor pushes that factor's + and - markers to opposite "
      "signs, so the group mean is ~0 and F is small (see PC3, which has "
      "|cos| 0.95 with Extraversion yet F=0.70). Two polarity-aware variants:\n")
    W("\n(i) ANOVA on **|loading|** — do a factor's markers load *heavily* on "
      "this PC, whatever the sign?\n")
    W("| PC | F (df 4,95) | p_perm | " +
      " | ".join(f"mean|{FSHORT[f]}|" for f in FACTORS) + " |")
    W("|---|---|---|" + "---|" * 5)
    for a in o["anova_by_factor_abs_loading"]:
        W(f"| PC{a['pc']} | {a['F']:.3f} | {a['p_perm']:.4f} | " +
          " | ".join(f"{a['group_means'][f]:.4f}" for f in FACTORS) + " |")
    W("\n(ii) ANOVA on **pole-signed loading** (loading x +1 if keyed '+', -1 "
      "if '-') — does this PC line up with the factor's own +/- axis? A large "
      "positive group mean means that factor's poles are separated along this PC.\n")
    W("| PC | F (df 4,95) | p_perm | " +
      " | ".join(f"mean {FSHORT[f]}" for f in FACTORS) + " |")
    W("|---|---|---|" + "---|" * 5)
    for a in o["anova_by_factor_pole_signed_loading"]:
        W(f"| PC{a['pc']} | {a['F']:.3f} | {a['p_perm']:.4f} | " +
          " | ".join(f"{a['group_means'][f]:+.4f}" for f in FACTORS) + " |")
    W("\nSame test using **keyed sign** (+ vs -) instead of factor:\n")
    W("| PC | F (df 1,98) | p_perm |")
    W("|---|---|---|")
    for a in o["anova_by_keyed_sign"]:
        W(f"| PC{a['pc']} | {a['F']:.3f} | {a['p_perm']:.4f} |")

    # ---- polarity
    W("\n## 5. Polarity test\n")
    W("If a factor is encoded as a *direction*, opposite-poled markers of the "
      "same factor should be ANTI-correlated.\n")
    for keyname, tag in [("raw_cosine", "raw cos(dW_i, dW_j)"),
                         ("mean_removed_cosine", "cos after removing the grand mean dW")]:
        s = o["polarity"][keyname]
        W(f"\n**{tag}**\n")
        W("| pair type | mean cos | sd | n pairs |")
        W("|---|---|---|---|")
        for kk, nm in [("same_factor_same_pole", "same factor, SAME pole"),
                       ("same_factor_opposite_pole", "same factor, OPPOSITE pole"),
                       ("different_factor", "different factor")]:
            W(f"| {nm} | **{s[kk]['mean']:+.4f}** | {s[kk]['sd']:.4f} | {s[kk]['n']} |")
    W("\nPer factor (raw / mean-removed):\n")
    W("| factor | same-pole raw | opp-pole raw | same-pole centered | opp-pole centered |")
    W("|---|---|---|---|---|")
    for F in FACTORS:
        q = o["polarity"]["per_factor_raw_and_centered"][F]
        W(f"| {F} | {q['same_pole_mean']:+.4f} | {q['opp_pole_mean']:+.4f} | "
          f"{q['same_pole_mean_centered']:+.4f} | {q['opp_pole_mean_centered']:+.4f} |")

    # ---- verdict
    M = np.array(o["pc_vs_factor_abscos"]["matrix"])
    pol = o["polarity"]
    nf2 = o["noise_floor"]
    W("\n## 6. Verdict\n")
    best = [(k, int(np.argmax(M[k])), float(M[k].max())) for k in range(6)]
    W("Best-matching factor per PC (|cos|): " + ", ".join(
        f"PC{k+1}->{FSHORT[FACTORS[a]]} {c:.2f}" for k, a, c in best) + ".\n")
    W(f"**The Big Five are partly recovered, and the polarity prediction holds "
      f"cleanly.** Opposite-poled markers of the same factor are genuinely "
      f"ANTI-correlated in weight space (mean cos "
      f"{pol['raw_cosine']['same_factor_opposite_pole']['mean']:+.3f} raw, "
      f"{pol['mean_removed_cosine']['same_factor_opposite_pole']['mean']:+.3f} "
      f"after removing the grand mean) against "
      f"{pol['raw_cosine']['same_factor_same_pole']['mean']:+.3f} / "
      f"{pol['mean_removed_cosine']['same_factor_same_pole']['mean']:+.3f} for "
      f"same-pole pairs and "
      f"{pol['raw_cosine']['different_factor']['mean']:+.3f} / "
      f"{pol['mean_removed_cosine']['different_factor']['mean']:+.3f} across "
      f"factors. So the adapters encode a signed DIRECTION per factor, not "
      f"merely \"this trait was trained\" — that failure mode is ruled out.\n")
    W(f"But the components are not the Big Five one-for-one. Only PC3 is a "
      f"clean factor axis (|cos| {M[2,0]:.2f} with Extraversion, and its top-10 "
      f"lists are pure Extraversion with the poles split). PC2 is mostly "
      f"Agreeableness ({M[1,1]:.2f}) but mixes in the emotionality markers. "
      f"PC1, the largest component at {o['pca']['centered_var_pct'][0]:.1f}%, "
      f"is not any single factor: it loads {M[0,2]:.2f} on Conscientiousness "
      f"and {M[0,4]:.2f} on Intellect simultaneously, and its poles are almost "
      f"perfectly the keyed sign (ANOVA on keyed sign F="
      f"{o['anova_by_keyed_sign'][0]['F']:.0f}, p={o['anova_by_keyed_sign'][0]['p_perm']:.4f}) "
      f"— it is a general desirable/undesirable (evaluative) axis cutting "
      f"across C, I and A, exactly the \"big one\"/social-desirability factor "
      f"that shows up in human Big-Five data too. That is consistent with the "
      f"factor directions themselves being non-orthogonal in weight space: "
      f"C-I cos {o['factor_directions']['pairwise_cos'][2][4]:+.2f}, "
      f"C-ES {o['factor_directions']['pairwise_cos'][2][3]:+.2f}, "
      f"A-I {o['factor_directions']['pairwise_cos'][1][4]:+.2f}. "
      f"The Big Five are NOT orthogonal here, so no rotation-free PCA could "
      f"return them as separate components.\n")
    W(f"Scale caveat: the reseed noise floor is not small. Two adapters for the "
      f"SAME trait differ by {nf2['mean_reseed_dist']:.3f} against a mean "
      f"between-trait distance of {nf2['mean_between_trait_dist']:.3f} — a ratio "
      f"of only {nf2['ratio_between_over_reseed']:.2f}. Trait identity is well "
      f"above seed noise (reseed cos {nf2['mean_reseed_cos']:.2f} vs "
      f"between-trait {nf2['mean_between_trait_cos']:.2f}), but roughly a "
      f"{100/nf2['ratio_between_over_reseed']**2:.0f}% share of squared "
      f"between-trait distance is seed variance, so components below about "
      f"PC4-PC5 ({o['pca']['centered_var_pct'][4]:.1f}% and under) are at or "
      f"near the level where reseed noise could produce comparable structure "
      f"and should not be interpreted without a reseed-based null.\n")

    open(f"{RDIR}/pca.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
