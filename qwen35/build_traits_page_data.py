#!/usr/bin/env python3
"""Assemble site_traits/data.json — everything the 134-trait page shows.

Inputs (all already on disk in results/ unless noted):
    gram_sweep.npz                      134x134 Gram, names, norms
    gram_data_null_{shuffled,permuted}_p100.npz
    gram_data_null_seedpaired_s40.npz   40x40 within-seed-B Gram
    cross_gram_seedpaired.npz           40 paired cross-seed cosines
    cross_gram_full_root_x_data_null_seedpaired_s40.npz   (if present)
    decomposition{,_shuffled,_permuted,_seedB}.json
    runmeta_sweep.json                  margins, losses, train seconds
    ../traits_primary.json ../traits_secondary.json      labels
    trait_descriptions.json             (optional, written by describe pass)
    steer134_judged.json                (optional, merged when steering lands)

Output: site_traits/data.json.  Idempotent; optional inputs missing are
reported and their sections omitted, never silently faked.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
OUT_DIR = os.path.join(HERE, "site_traits")
os.makedirs(OUT_DIR, exist_ok=True)


def load_labels():
    labs = {}
    for fn in ("traits_primary.json", "traits_secondary.json"):
        for row in json.load(open(os.path.join(HERE, fn))):
            key = row["trait"].lower().replace(" ", "_").replace("-", "_")
            labs[key] = {"trait": row["trait"], "factor": row["factor"],
                         "keyed": row["keyed"]}
    return labs


def eig_sorted(G):
    n = G.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Gc = H @ G @ H
    w, V = np.linalg.eigh(Gc)
    idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx], Gc


def main():
    z = np.load(os.path.join(R, "gram_sweep.npz"), allow_pickle=True)
    G, names = np.array(z["G"]), [str(n) for n in z["names"]]
    norms = np.sqrt(np.diag(G))
    C = G / np.outer(norms, norms)
    labs = load_labels()
    missing_lab = [n for n in names if n not in labs]
    if missing_lab:
        raise SystemExit(f"traits without labels: {missing_lab[:10]}")

    # residual view: remove the leading component of the double-centred Gram
    w, V, Gc = eig_sorted(G)
    # leading-component-removed cosine, same construction decompose.py uses:
    # subtract the rank-1 part of the CENTRED Gram from the centred Gram,
    # then re-normalise.  (The common mean is out either way.)
    G1 = np.outer(V[:, 0], V[:, 0]) * w[0]
    Gres = Gc - G1
    dres = np.sqrt(np.clip(np.diag(Gres), 1e-12, None))
    Cres = Gres / np.outer(dres, dres)

    # 2D/3D embedding: classical MDS = principal coordinates of Gc
    wpos = np.clip(w, 0, None)
    coords = V[:, :3] * np.sqrt(wpos[:3])

    runmeta = {}
    p = os.path.join(R, "runmeta_sweep.json")
    if os.path.exists(p):
        runmeta = json.load(open(p))

    descs = {}
    p = os.path.join(R, "trait_descriptions.json")
    if os.path.exists(p):
        descs = json.load(open(p))

    order = np.argsort([
        ("Extraversion", "Agreeableness", "Conscientiousness",
         "EmotionalStability", "Intellect", "Lexicon").index(
             labs[n]["factor"]) * 10
        + (0 if labs[n]["keyed"] == "+" else 5) for n in names],
        kind="stable")

    traits = []
    for i, n in enumerate(names):
        lab = labs[n]
        c_row = C[i].copy()
        c_row[i] = -2
        nearest = np.argsort(c_row)[::-1][:5]
        cres_row = Cres[i].copy()
        cres_row[i] = -2
        nearest_res = np.argsort(cres_row)[::-1][:5]
        far = np.argsort(C[i])[:3]
        m = runmeta.get(n, {})
        traits.append({
            "slug": n, "trait": lab["trait"], "factor": lab["factor"],
            "keyed": lab["keyed"], "norm": round(float(norms[i]), 4),
            "pc": [round(float(coords[i, k]), 4) for k in range(3)],
            "reward_margin": m.get("reward_margin"),
            "loss_last": m.get("loss_last"),
            "train_seconds": m.get("train_seconds"),
            "nearest": [{"slug": names[j],
                         "cos": round(float(C[i, j]), 3)} for j in nearest],
            "nearest_resid": [{"slug": names[j],
                               "cos": round(float(Cres[i, j]), 3)}
                              for j in nearest_res],
            "furthest": [{"slug": names[j],
                          "cos": round(float(C[i, j]), 3)} for j in far],
            "desc": descs.get(n),
        })

    def dec(fn):
        p = os.path.join(R, fn)
        return json.load(open(p)) if os.path.exists(p) else None

    d_real = dec("decomposition.json")
    d_sh = dec("decomposition_shuffled.json")
    d_pm = dec("decomposition_permuted.json")
    d_sB = dec("decomposition_seedB.json")

    # seed-paired story
    seed = None
    p = os.path.join(R, "cross_gram_seedpaired.npz")
    if os.path.exists(p):
        cz = np.load(p, allow_pickle=True)
        seed = {"names": [str(x) for x in cz["names"]],
                "self_cos": [round(float(x), 4) for x in cz["cos"]],
                "norm_a": [round(float(x), 3) for x in cz["norm_a"]],
                "norm_b": [round(float(x), 3) for x in cz["norm_b"]]}
    cross_full = None
    p = os.path.join(R, "cross_gram_full_root_x_data_null_seedpaired_s40.npz")
    if os.path.exists(p):
        xz = np.load(p, allow_pickle=True)
        X = np.array(xz["X"])
        na = [str(x) for x in xz["names_a"]]
        nb = [str(x) for x in xz["names_b"]]
        XC = X / np.outer(np.array(xz["norms_a"]), np.array(xz["norms_b"]))
        # cross-seed factor statistics: same trait / same factor+keying
        # (different trait) / different factor
        ia = {n: i for i, n in enumerate(na)}
        same_t, same_fk, diff_f = [], [], []
        for j, n in enumerate(nb):
            fj, kj = labs[n]["factor"], labs[n]["keyed"]
            for m_, i in ia.items():
                v = float(XC[i, j])
                if m_ == n:
                    same_t.append(v)
                elif labs[m_]["factor"] == fj and labs[m_]["factor"] != "Lexicon":
                    (same_fk if labs[m_]["keyed"] == kj else diff_f).append(v)
        def stats(xs):
            xs = np.array(xs)
            return {"n": len(xs), "mean": round(float(xs.mean()), 4),
                    "median": round(float(np.median(xs)), 4),
                    "sd": round(float(xs.std()), 4)}
        # POST-HOC (not preregistered): twin identification — for each seed-B
        # adapter, the rank of its own trait's seed-0 adapter among all 134
        # cross-seed cosines.  Magnitude and identity are different claims.
        ranks = []
        for j, n in enumerate(nb):
            col = XC[:, j]
            ranks.append(int((col > col[ia[n]]).sum()) + 1)
        cross_full = {"same_trait": stats(same_t),
                      "same_factor_same_keyed_diff_trait": stats(same_fk),
                      "same_factor_opp_keyed": stats(diff_f),
                      "overall": stats(XC.flatten().tolist()),
                      "twin_rank": {"ranks": ranks,
                                    "top1": sum(r == 1 for r in ranks),
                                    "n": len(ranks),
                                    "posthoc": True}}

    # spectra for the scree comparison
    def spectrum_of(npz_name):
        p = os.path.join(R, npz_name)
        if not os.path.exists(p):
            return None
        zz = np.load(p, allow_pickle=True)
        Gx = np.array(zz["G"])
        wx, _, _ = eig_sorted(Gx)
        wp = np.clip(wx, 0, None)
        tot = wp.sum()
        pr = float(wp.sum() ** 2 / (wp ** 2).sum())
        return {"share": [round(float(x / tot), 5) for x in wp[:20]],
                "participation_ratio": round(pr, 2), "n": Gx.shape[0]}

    spectra = {"real": spectrum_of("gram_sweep.npz"),
               "shuffled": spectrum_of("gram_data_null_shuffled_p100.npz"),
               "permuted": spectrum_of("gram_data_null_permuted_p100.npz"),
               "seedB": spectrum_of("gram_data_null_seedpaired_s40.npz")}

    # heatmap (factor-ordered), 3dp to keep the blob small
    heat = {
        "order": [names[i] for i in order],
        "cos": [[round(float(C[i, j]), 3) for j in order] for i in order],
        "cos_resid": [[round(float(Cres[i, j]), 3) for j in order]
                      for i in order],
    }

    steer = None
    p = os.path.join(R, "steer134_judged.json")
    if os.path.exists(p):
        j = json.load(open(p))
        per = {}
        for row in j.get("trait_curves", []):
            doses = []
            for d_ in row["doses"]:
                e, c = d_.get("expression") or {}, d_.get("coherence") or {}
                ct = d_.get("ctrl_expression_at_same_alpha") or {}
                doses.append({
                    "alpha": d_["alpha"],
                    "expression": e.get("mean"), "n_expr": e.get("n"),
                    "expr_ci": e.get("ci95"),
                    "coherence": c.get("mean"), "n_coh": c.get("n"),
                    "ctrl_expression": ct.get("mean"), "n_ctrl": ct.get("n"),
                })
            base = row.get("baseline_expression") or {}
            per[row["trait"]] = {"factor": row.get("factor"),
                                 "keyed": row.get("keyed"),
                                 "baseline_expression": base.get("mean"),
                                 "doses": doses}
        # Verbatim example transcripts, selected by FIXED prompt indices
        # (0 and 4, a subset of the judge's control subsample) at fixed doses
        # -- chosen by rule, never by reading the content, so the examples
        # cannot be cherry-picked.  Scores attach from the judge cache where
        # that unit was judged; absent scores stay absent.
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(
            "js", os.path.join(HERE, "judge_steer134.py"))
        JS = _ilu.module_from_spec(spec)
        spec.loader.exec_module(JS)
        cache = {}
        cp = os.path.join(R, "steer134_judge_cache.jsonl")
        if os.path.exists(cp):
            for line in open(cp):
                try:
                    rec = json.loads(line)
                    cache[rec["k"]] = rec["v"]
                except Exception:
                    pass

        def scored(slug, prompt, response):
            e = cache.get(JS.unit_key("expression", slug, prompt, response))
            c = cache.get(JS.unit_key("coherence", None, prompt, response))
            return {"prompt": prompt,
                    "response": response[:700] + ("…" if len(response) > 700
                                                  else ""),
                    "expression": (e or {}).get("expression"),
                    "coherence": (c or {}).get("coherence")}

        GEN_DIR = os.path.join(R, "steer134_gen")
        EX_IDX = (0, 4)
        EX_DOSES = ("-4.0", "+2.0", "+4.0")

        def gens_of(fn, slug):
            p = os.path.join(GEN_DIR, fn)
            if not os.path.exists(p):
                return None
            rows = json.load(open(p))["responses"]
            return [scored(slug, r["prompt"], r["response"])
                    for r in rows if r["idx"] in EX_IDX]

        for slug, entry in per.items():
            ex = {"baseline": gens_of("base_a+0.0.json", slug), "doses": []}
            for d_ in EX_DOSES:
                g = gens_of(f"trait_{slug}_a{d_}.json", slug)
                if g:
                    ex["doses"].append({"alpha": float(d_), "gens": g})
            ex["control"] = {"alpha": 4.0,
                             "gens": gens_of("random1_a+4.0.json", slug)}
            entry["examples"] = ex

        steer = {"per_trait": per,
                 "coverage": j.get("coverage"),
                 "cost": j.get("cost"),
                 "judge_model": j.get("judge_model"),
                 "by_condition": {k: v for k, v in
                                  (j.get("by_condition") or {}).items()
                                  if not k.startswith("trait_")}}

    # ------------------------------------------------------------ components
    # PC loadings from the double-centred Gram; FA loadings from
    # analyse_fa_qwen35.py when present.  Signs are oriented to MATCH the
    # steering conditions: each pc gen file names its positive pole's top
    # traits, and the loading vector is flipped until those traits load
    # positive -- the page and the transcripts must agree on which way is +.
    GEN_DIR = os.path.join(R, "steer134_gen")

    def judged_curve(bc, prefix):
        curve = []
        for alpha in (-8, -4, -2, -1, 1, 2, 4, 8):
            c = bc.get(f"{prefix}_a{alpha:+.1f}")
            if not c:
                continue
            r1 = bc.get(f"random1_a{alpha:+.1f}") or {}
            r2 = bc.get(f"random2_a{alpha:+.1f}") or {}
            def mean_of(a, b, k):
                vs = [x.get(k, {}).get("mean") for x in (a, b)
                      if x.get(k, {}).get("mean") is not None]
                return sum(vs) / len(vs) if vs else None
            curve.append({
                "alpha": alpha,
                "alignment": (c.get("alignment") or {}).get("mean"),
                "coherence": (c.get("coherence") or {}).get("mean"),
                "ctrl_alignment": mean_of(r1, r2, "alignment"),
                "ctrl_coherence": mean_of(r1, r2, "coherence"),
                "frac_truncated": c.get("frac_truncated"),
                "mean_chars": c.get("mean_chars"),
            })
        return curve

    def comp_examples(prefix):
        try:
            import importlib.util as _ilu2
            spec2 = _ilu2.spec_from_file_location(
                "js2", os.path.join(HERE, "judge_steer134.py"))
            JS2 = _ilu2.module_from_spec(spec2)
            spec2.loader.exec_module(JS2)
        except Exception:
            JS2 = None
        cache2 = {}
        cp2 = os.path.join(R, "steer134_judge_cache.jsonl")
        if os.path.exists(cp2):
            for line in open(cp2):
                try:
                    rec = json.loads(line)
                    cache2[rec["k"]] = rec["v"]
                except Exception:
                    pass

        def scored2(prompt, response):
            a = c = None
            if JS2:
                a = (cache2.get(JS2.unit_key("alignment", None, prompt,
                                             response)) or {}).get("alignment")
                c = (cache2.get(JS2.unit_key("coherence", None, prompt,
                                             response)) or {}).get("coherence")
            return {"prompt": prompt,
                    "response": response[:700] + ("…" if len(response) > 700
                                                  else ""),
                    "alignment": a, "coherence": c}
        out = {"doses": []}
        bp = os.path.join(GEN_DIR, "base_a+0.0.json")
        if os.path.exists(bp):
            rows = json.load(open(bp))["responses"]
            out["baseline"] = [scored2(r["prompt"], r["response"])
                               for r in rows if r["idx"] in (0, 4)]
        for d_ in ("-4.0", "-2.0", "+2.0", "+4.0"):
            p_ = os.path.join(GEN_DIR, f"{prefix}_a{d_}.json")
            if not os.path.exists(p_):
                continue
            j_ = json.load(open(p_))
            out["doses"].append({
                "alpha": float(d_), "pole": j_.get("pole"),
                "gens": [scored2(r["prompt"], r["response"])
                         for r in j_["responses"] if r["idx"] in (0, 4)]})
        cp_ = os.path.join(GEN_DIR, "random1_a+4.0.json")
        if os.path.exists(cp_):
            j_ = json.load(open(cp_))
            out["control"] = {"alpha": 4.0,
                              "gens": [scored2(r["prompt"], r["response"])
                                       for r in j_["responses"]
                                       if r["idx"] in (0, 4)]}
        return out

    def factor_dirs_cos(coeffs):
        # |cos| of a coefficient direction with each Goldberg factor's mean
        # signed direction, all in the Gram metric.
        out = []
        for f in ("Extraversion", "Agreeableness", "Conscientiousness",
                  "EmotionalStability", "Intellect"):
            c2 = np.array([(1.0 if labs[n]["keyed"] == "+" else -1.0)
                           if labs[n]["factor"] == f else 0.0 for n in names])
            c2 -= c2.mean()
            denom = np.sqrt(max(c2 @ Gc @ c2, 1e-12))
            c2 /= denom
            out.append(round(float(abs(coeffs @ Gc @ c2)), 3))
        return out

    analysis = {}
    ap = os.path.join(OUT_DIR, "analysis.json")
    if os.path.exists(ap):
        analysis = json.load(open(ap))

    components = []
    wpos_tot = np.clip(w, 0, None).sum()
    for k in range(3):
        lo = V[:, k] * np.sqrt(max(w[k], 0.0))
        coeffs = V[:, k] / np.sqrt(max(w[k], 1e-12))
        pole_txt = None
        gp = os.path.join(GEN_DIR, f"pc{k + 1}_a+2.0.json")
        if os.path.exists(gp):
            pole_txt = json.load(open(gp)).get("pole", "")
            first = pole_txt.split("(")[-1].split(",")[0].strip().rstrip(")")
            i_first = names.index(first) if first in names else None
            if i_first is not None and lo[i_first] < 0:
                lo, coeffs = -lo, -coeffs
        order_ = np.argsort(lo)
        top_pos = [{"slug": names[i], "loading": round(float(lo[i]), 3),
                    "desc": descs.get(names[i])} for i in order_[::-1][:10]]
        top_neg = [{"slug": names[i], "loading": round(float(lo[i]), 3),
                    "desc": descs.get(names[i])} for i in order_[:10]]
        components.append({
            "id": f"pc{k + 1}", "kind": "principal component",
            "title": f"PC{k + 1}",
            "headline": f"{100 * w[k] / wpos_tot:.1f}% of between-trait variance",
            "pole": pole_txt,
            "vs_factors": factor_dirs_cos(coeffs),
            "top_positive": top_pos, "top_negative": top_neg,
            "curve": judged_curve((steer or {}).get("by_condition", {}),
                                  f"pc{k + 1}") if steer else [],
            "examples": comp_examples(f"pc{k + 1}"),
            "analysis": analysis.get(f"pc{k + 1}"),
            "steer_analysis": analysis.get(f"pc{k + 1}_steer"),
        })

    fap = os.path.join(R, "fa_qwen35.json")
    if os.path.exists(fap):
        fa_all = json.load(open(fap))
        fa_st = fa_all["steering"]
        sol = fa_all["solutions"][fa_st["solution"]]
        L = fa_st["oblimin_loadings"]
        slug_order = fa_st["slug_order"]
        kfa = fa_st["k"]
        fa = {"ss_loadings": [round(float(x), 2) for x in
                              (sol.get("ss_loadings") or {}).get("oblimin",
                                                                 [])],
              "congruence": {f"fa{j + 1}":
                             [round(abs(float(v)), 2) for v in row[:5]]
                             for j, row in
                             enumerate(sol.get("congruence_oblimin") or [])}}
        for j in range(kfa):
            lo = np.array([L[s][j] for s in slug_order])
            cf = {s: L[s][j] for s in slug_order}
            coeffs = np.array([cf.get(n, 0.0) for n in names])
            coeffs -= coeffs.mean()
            denom = np.sqrt(max(coeffs @ Gc @ coeffs, 1e-12))
            coeffs /= denom
            pole_txt = None
            gp = os.path.join(GEN_DIR, f"fa{j + 1}_a+2.0.json")
            if os.path.exists(gp):
                pole_txt = json.load(open(gp)).get("pole", "")
                first = pole_txt.split("(")[-1].split(",")[0].strip().rstrip(")")
                if first in slug_order and cf[first] < 0:
                    lo = -lo
                    coeffs = -coeffs
                    cf = {s: -v for s, v in cf.items()}
            ordr = np.argsort(lo)
            components.append({
                "id": f"fa{j + 1}", "kind": "oblimin factor",
                "title": f"Factor {j + 1}",
                "headline": f"SS loadings "
                            f"{(fa.get('ss_loadings') or [None]*kfa)[j]}",
                "pole": pole_txt,
                "congruence": (fa.get("congruence") or {}).get(f"fa{j + 1}"),
                "top_positive": [{"slug": slug_order[i],
                                  "loading": round(float(lo[i]), 3),
                                  "desc": descs.get(slug_order[i])}
                                 for i in ordr[::-1][:10]],
                "top_negative": [{"slug": slug_order[i],
                                  "loading": round(float(lo[i]), 3),
                                  "desc": descs.get(slug_order[i])}
                                 for i in ordr[:10]],
                "curve": judged_curve((steer or {}).get("by_condition", {}),
                                      f"fa{j + 1}") if steer else [],
                "examples": comp_examples(f"fa{j + 1}"),
                "analysis": analysis.get(f"fa{j + 1}"),
                "steer_analysis": analysis.get(f"fa{j + 1}_steer"),
            })

    # ------------------------------------------------------------------ umap
    umap = None
    up = os.path.join(R, "umap_embeddings.json")
    if os.path.exists(up):
        uj = json.load(open(up))
        umap = {"params": uj["params"], "arms": {}}
        for arm in ("real", "shuffled", "permuted", "seedB"):
            if arm not in uj:
                continue
            names_u = uj[arm]["names"]
            X = np.array(uj[arm]["resid"]["0"])
            # neighbourhood purity vs the arm's own chance level
            D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(D2, np.inf)
            groups = {}
            for n in names_u:
                if labs[n]["factor"] != "Lexicon":
                    groups.setdefault((labs[n]["factor"], labs[n]["keyed"]),
                                      []).append(n)
            hits, chance_num = [], []
            for i, n in enumerate(names_u):
                f, ky = labs[n]["factor"], labs[n]["keyed"]
                if f == "Lexicon":
                    continue
                nn = np.argsort(D2[i])[:10]
                hits.append(np.mean([
                    labs[names_u[j]]["factor"] == f
                    and labs[names_u[j]]["keyed"] == ky for j in nn]))
                chance_num.append((len(groups[(f, ky)]) - 1)
                                  / (len(names_u) - 1))
            umap["arms"][arm] = {
                "points": [{"slug": n,
                            "x": round(float(X[i, 0]), 3),
                            "y": round(float(X[i, 1]), 3),
                            "factor": labs[n]["factor"],
                            "keyed": labs[n]["keyed"]}
                           for i, n in enumerate(names_u)],
                "purity": round(float(np.mean(hits)), 3),
                "chance": round(float(np.mean(chance_num)), 3),
            }

    data = {
        "meta": {
            "base_model": "Qwen/Qwen3.5-4B",
            "n_traits": len(names), "n_modules": int(z["n_modules"]),
            "scale": float(z["scale"]),
            "pairs_per_trait": 445,
            "mean_cos": round(float(C[~np.eye(len(names), dtype=bool)].mean()), 4),
            "built": None,
        },
        "tests": {"real": d_real, "shuffled": d_sh, "permuted": d_pm,
                  "seedB": d_sB},
        "seedpaired": seed,
        "cross_seed_factor": cross_full,
        "spectra": spectra,
        "traits": traits,
        "heatmap": heat,
        "steering": steer,
        "components": components,
        "umap": umap,
    }
    def denan(x):
        # json.dump writes literal NaN by default, which no browser will parse.
        if isinstance(x, dict):
            return {k: denan(v) for k, v in x.items()}
        if isinstance(x, list):
            return [denan(v) for v in x]
        if isinstance(x, float) and (x != x or x in (float("inf"),
                                                     float("-inf"))):
            return None
        return x

    out = os.path.join(OUT_DIR, "data.json")
    with open(out, "w") as f:
        json.dump(denan(data), f, allow_nan=False)
    absent = [k for k, v in (("cross_seed_factor", cross_full),
                             ("steering", steer),
                             ("descriptions", descs or None)) if v is None]
    print(f"wrote {out} ({os.path.getsize(out)//1024} KB); "
          f"sections absent: {absent or 'none'}")


if __name__ == "__main__":
    main()
