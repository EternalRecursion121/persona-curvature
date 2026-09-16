#!/usr/bin/env python3
"""Host side of the column-space test.

Reads results/column_space_{stage1,stage2,crossstage}.npz written by
column_space_on_modal.py and writes analysis/column_space.json.

The question: two adapters trained for the same trait from different random
inits are near-orthogonal in the Frobenius inner product (+0.0181,
analysis/crossseed_arms.json).  A delta is dW = s B A; the Frobenius inner
product is multiplied by the overlap of the two ROW spaces, which is A's random
draw.  B is accumulated from output-side error vectors, which the data and the
base model set.  Do same-trait adapters share COLUMN space?

Everything here is read off the per-pair matrices the Modal job returns:
  col_unw_k<k>   mean squared cosine of principal angles between the top-k
                 column spaces, ||U_i[:, :k]^T U_j[:, :k]||_F^2 / k
  col_wtd_k<k>   the same weighted by adapter i's sigma^2, which equals the
                 fraction of adapter i's delta energy lying in adapter j's
                 top-k column space
  row_unw/row_wtd   the same for the row spaces (a subset of modules)
  top1abs        mean over modules of |cos| between the two top left singular
                 vectors
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KS = [1, 2, 4, 8, 16, 64]
RNG = np.random.default_rng(0)


def load(tag):
    return np.load(os.path.join(HERE, "results", f"column_space_{tag}.npz"),
                   allow_pickle=True)


def module_summary(z, cname, ks=(1, 8, 64)):
    """The per-module class means the container recorded, split by whether the
    module can carry a rank-64 subspace at all.  48 of the 248 modules are the
    linear-attention in_proj_a / in_proj_b, whose output is 32-dimensional: for
    those the k = 8 null is 8/32 = 0.25 and every adapter's column space is a
    large fraction of the whole output space, so a straight average over all 248
    modules is dominated by them.  Also reports the excess over the analytic
    null k / d_out (k / d_in for the row views), which is comparable across
    modules of any width."""
    recs = json.loads(str(z["mod_records"]))
    out = {}
    for k in ks:
        for view in ("col_unw", "col_wtd", "row_unw"):
            key = f"{view}_k{k}_{cname}"
            amb = "d_in" if view.startswith("row") else "d_out"
            for label, sel in (("all_modules", lambda r: True),
                               ("rank64_modules", lambda r: r["rank"] >= 64)):
                v = [r for r in recs if key in r and sel(r)]
                if not v:
                    continue
                out.setdefault(f"k{k}", {}).setdefault(view, {})[label] = {
                    "n_modules": len(v),
                    "mean": float(np.mean([r[key] for r in v])),
                    "mean_analytic_null": float(np.mean([k / r[amb] for r in v])),
                    "mean_excess_over_null": float(np.mean([r[key] - k / r[amb]
                                                            for r in v]))}
    return out


def margins(sim, q_traits, c_traits):
    """min over true pairs, max over the rest, and the gap -- the same three
    numbers analysis/crossseed_arms.json reports for the Frobenius cross-Gram."""
    same = np.zeros_like(sim, dtype=bool)
    for qi, t in enumerate(q_traits):
        same[qi, int(np.where(c_traits == t)[0][0])] = True
    return {"min_same": float(sim[same].min()), "max_off": float(sim[~same].max()),
            "sep": float(sim[same].min() - sim[~same].max()),
            "same_mean": float(sim[same].mean()), "off_mean": float(sim[~same].mean())}


def stat(M, mask):
    v = M[mask]
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std()),
            "min": float(v.min()), "max": float(v.max())}


def classes_of(z):
    ta, tb = z["traits_a"], z["traits_b"]
    ga, gb = z["tags_a"], z["tags_b"]
    same_trait = ta[:, None] == tb[None, :]
    same_tag = ga[:, None] == gb[None, :]
    if bool(z["cross_only"]):
        selfp = np.zeros_like(same_trait)
    else:
        selfp = np.eye(len(ta), dtype=bool)
    out = {"same_trait_cross_seed": same_trait & ~same_tag & ~selfp,
           "diff_trait_cross_seed": ~same_trait & ~same_tag,
           "diff_trait_same_seed": ~same_trait & same_tag}
    # the pooled within-seed class mixes the two sets; keep each on its own too
    for t in sorted(set(ga.tolist())):
        m = (ga[:, None] == t) & (gb[None, :] == t) & ~same_trait
        if m.any():
            out[f"diff_trait_within_{t}"] = m
    return out


def null_summary(z):
    """Analytic k/d and the empirical 200-draw null, over the modules that
    actually contribute at each k (rank >= k)."""
    d_out, d_in, rank = z["d_outs"], z["d_ins"], z["ranks"]
    nr = json.loads(str(z["null_records"]))
    out = {}
    for k in KS:
        ok = rank >= k
        row = {"n_modules": int(ok.sum()),
               "analytic_col_mean_k_over_dout": float((k / d_out[ok]).mean()),
               "analytic_row_mean_k_over_din": float((k / d_in[ok]).mean())}
        emp = [r[f"null_mean_k{k}"] for r in nr if f"null_mean_k{k}" in r]
        if emp:
            row["empirical_col_mean"] = float(np.mean(emp))
            row["empirical_col_sd_across_modules"] = float(np.std(emp))
            row["empirical_n_modules"] = len(emp)
            row["empirical_n_draws_per_module"] = int(nr[0]["n_draws"])
            row["analytic_col_mean_on_the_same_modules"] = float(np.mean(
                [r[f"analytic_col_k{k}"] for r in nr if f"null_mean_k{k}" in r]))
            row["analytic_row_mean_on_the_same_modules"] = float(np.mean(
                [r[f"analytic_row_k{k}"] for r in nr if f"null_mean_k{k}" in r]))
        out[f"k{k}"] = row
    return out


def block(z, tag_a, tag_b):
    """Row/col index arrays for one tag pair."""
    ra = np.where(z["tags_a"] == tag_a)[0]
    cb = np.where(z["tags_b"] == tag_b)[0]
    return ra, cb


def identify(sim, q_traits, c_traits):
    """sim[q, c]: rank each query's candidates by decreasing similarity."""
    ranks, top1 = [], 0
    for qi, t in enumerate(q_traits):
        j = int(np.where(c_traits == t)[0][0])
        order = np.argsort(-sim[qi])
        r = int(np.where(order == j)[0][0]) + 1
        ranks.append(r)
        top1 += (r == 1)
    return {"n": len(q_traits), "top1": int(top1), "mean_rank": float(np.mean(ranks)),
            "worst_rank": int(max(ranks)), "chance_rank": (len(c_traits) + 1) / 2}


def perm_p(M, mask_within, mask_between, labels, obs, n=5000):
    ge = 0
    idx = np.arange(len(labels))
    for _ in range(n):
        RNG.shuffle(idx)
        f = labels[idx]
        sw = f[:, None] == f[None, :]
        off = ~np.eye(len(labels), dtype=bool)
        v = M[sw & off].mean() - M[(~sw) & off].mean()
        ge += v >= obs
    return float((ge + 1) / (n + 1))


def main():
    out = {"what": "column-space (span of B) overlap between LoRA adapters",
           "question": "do LoRA adapters trained for the same trait from "
                       "different random initialisations share COLUMN space "
                       "even though their deltas are near-orthogonal in the "
                       "Frobenius inner product?",
           "produced_by": ["column_space_on_modal.py (Modal, CPU only, "
                           "app pc-qwen35-colspace, zoo-colspace.service, "
                           "log phase10_runs/colspace.log)",
                           "analyse_column_space.py"],
           "definitions": {
               "col_unw_k": "||U_i[:, :k]^T U_j[:, :k]||_F^2 / k, the mean "
                            "squared cosine of the principal angles between the "
                            "two top-k column spaces; averaged over modules",
               "col_wtd_k": "the same weighted by adapter i's sigma^2 over its "
                            "first k directions: the fraction of adapter i's "
                            "top-k delta energy that lies in adapter j's top-k "
                            "column space (at k = 64 that is the whole delta). "
                            "Not symmetric; for a class mean over the square "
                            "matrix both orientations are averaged.",
               "row_unw_k/row_wtd_k": "the same for the row spaces (span of A), "
                                      "measured on a 31-module stride subset",
               "top1abs": "mean over modules of |cos| between the two top left "
                          "singular vectors"},
           "sources": {}}

    # ---------------- stage one -------------------------------------------
    z1 = load("stage1")
    out["sources"]["stage1"] = "results/column_space_stage1.npz"
    cls1 = classes_of(z1)
    out["stage1"] = {
        "n_adapters": int(len(z1["traits"])),
        "n_modules": int(z1["n_modules"]),
        "n_modules_per_k_column": dict(zip([f"k{k}" for k in KS],
                                           z1["col_cnt"].tolist())),
        "n_modules_per_k_row": dict(zip([f"k{k}" for k in KS],
                                        z1["row_cnt"].tolist())),
        "pair_counts": {c: int(m.sum()) for c, m in cls1.items()},
        "null": null_summary(z1),
        "null_note": "the column entries average k / d_out over every module "
                     "with rank >= k; the row entries average k / d_in over the "
                     "same set, which is NOT the 31-module subset the row "
                     "overlaps were measured on. For a like-for-like null use "
                     "by_module_class[...][k][row_unw][...]['mean_analytic_null'].",
        "classes": {},
    }
    for k in KS:
        for view in ["col_unw", "col_wtd", "row_unw", "row_wtd"]:
            key = f"{view}_k{k}"
            if key not in z1:
                continue
            for c, m in cls1.items():
                if m.any():
                    out["stage1"]["classes"].setdefault(c, {})[key] = stat(z1[key], m)

    out["stage1"]["by_module_class"] = {
        "note": "class means computed per module and then averaged, split by "
                "whether the module's output is wide enough for a rank-64 "
                "column space (48 of 248 modules are linear_attn in_proj_a / "
                "in_proj_b with d_out = 32). The rank64_modules row is the "
                "headline; mean_excess_over_null is comparable across widths.",
        "same_trait_cross_set": module_summary(z1, "same_trait_cross_set"),
        "diff_trait_cross_set": module_summary(z1, "diff_trait_cross_set"),
        "diff_trait_same_set": module_summary(z1, "diff_trait_same_set"),
    }
    for k in (1, 8, 64):
        for view in ("col_unw", "col_wtd"):
            for lab in ("all_modules", "rank64_modules"):
                try:
                    a = out["stage1"]["by_module_class"]["same_trait_cross_set"][f"k{k}"][view][lab]["mean"]
                    b = out["stage1"]["by_module_class"]["diff_trait_cross_set"][f"k{k}"][view][lab]["mean"]
                except KeyError:
                    continue
                out["stage1"]["by_module_class"].setdefault("same_minus_diff", {}) \
                    .setdefault(f"k{k}", {}).setdefault(view, {})[lab] = float(a - b)

    # spectra
    out["stage1"]["spectrum"] = {
        "mean_top1_share_of_sum_sigma": float(z1["spec_prof"][:, 0].mean()),
        "mean_top8_share_of_sum_sigma": float(z1["spec_prof"][:, :8].sum(1).mean()),
        "mean_top16_share_of_sum_sigma": float(z1["spec_prof"][:, :16].sum(1).mean()),
    }

    # top-1 left singular vector sharing within the 134 seed-0 stage-one set
    ra, _ = block(z1, "s1s0", "s1s0")
    T = z1["top1abs"][np.ix_(ra, ra)]
    off = ~np.eye(len(ra), dtype=bool)
    out["stage1"]["top1_left_vector_sharing_within_134_seed0"] = {
        "mean_abs_cos": float(T[off].mean()), "sd": float(T[off].std()),
        "note": "mean over 248 modules of |cos| between the two top left "
                "singular vectors, then averaged over the 134x133 pairs"}
    rb, _ = block(z1, "s1s1", "s1s1")
    st = cls1["same_trait_cross_seed"]
    out["stage1"]["top1_left_vector_same_trait_cross_seed"] = {
        "mean_abs_cos": float(z1["top1abs"][st].mean()),
        "mean_signed_cos": float(z1["top1sgn"][st].mean()),
        "n": int(st.sum())}
    dt = cls1["diff_trait_cross_seed"]
    out["stage1"]["top1_left_vector_diff_trait_cross_seed"] = {
        "mean_abs_cos": float(z1["top1abs"][dt].mean()),
        "mean_signed_cos": float(z1["top1sgn"][dt].mean()),
        "n": int(dt.sum())}

    # ---------------- identification --------------------------------------
    qi = np.where(z1["tags_a"] == "s1s1")[0]     # 40 seed-1 queries
    ci = np.where(z1["tags_b"] == "s1s0")[0]     # 134 seed-0 candidates
    qt, ct = z1["traits_a"][qi], z1["traits_b"][ci]
    ident = {}
    for k in (8, 64):
        for view in ("col_wtd", "col_unw"):
            key = f"{view}_k{k}"
            if key in z1:
                ident[key] = identify(z1[key][np.ix_(qi, ci)], qt, ct)
        # symmetrised weighted
        W = z1[f"col_wtd_k{k}"]
        Wsym = 0.5 * (W + W.T)
        ident[f"col_wtd_sym_k{k}"] = identify(Wsym[np.ix_(qi, ci)], qt, ct)
        ident[f"row_unw_k{k}"] = identify(z1[f"row_unw_k{k}"][np.ix_(qi, ci)], qt, ct)
    for k in (8, 64):
        ident[f"col_wtd_k{k}"]["margins"] = margins(
            z1[f"col_wtd_k{k}"][np.ix_(qi, ci)], qt, ct)
        ident[f"col_unw_k{k}"]["margins"] = margins(
            z1[f"col_unw_k{k}"][np.ix_(qi, ci)], qt, ct)
    ident["frobenius_reference"] = {
        "top1": 40, "n": 40, "mean_rank": 1.0, "chance_rank": 67.5,
        "min_same": 0.015933681838395445, "max_off": 0.015842137704656464,
        "sep": 9.154413373898065e-05, "same_mean": 0.01806099632415457,
        "source": "analysis/crossseed_arms.json arms[1] (path "
                  "cross_gram_full_root_x_data_null_seedpaired_s40_matched.npz)"}
    out["stage1"]["identification_40_seed1_queries_vs_134_seed0"] = ident

    # ---------------- per-module profile ----------------------------------
    recs = json.loads(str(z1["mod_records"]))
    key = "col_wtd_k8_same_trait_cross_set"
    keyd = "col_wtd_k8_diff_trait_cross_set"
    for r in recs:
        r["proj"] = r["module"].split(".")[-1]
        r["layer"] = int(r["module"].split(".")[2])
        r["gap"] = r[key] - r[keyd]
    by_proj = {}
    for r in recs:
        by_proj.setdefault(r["proj"], []).append(r)
    out["stage1"]["per_module_profile"] = {
        "statistic": "col_wtd_k8, same-trait cross-seed minus different-trait "
                     "cross-seed, per module",
        "by_projection": {p: {"n_modules": len(v),
                              "d_out": v[0]["d_out"],
                              "same_trait_mean": float(np.mean([x[key] for x in v])),
                              "diff_trait_mean": float(np.mean([x[keyd] for x in v])),
                              "gap_mean": float(np.mean([x["gap"] for x in v]))}
                          for p, v in sorted(by_proj.items())},
        "by_layer_quartile": {},
        "top10_modules_by_gap": [{"module": r["module"], "same": r[key],
                                  "diff": r[keyd], "gap": r["gap"]}
                                 for r in sorted(recs, key=lambda r: -r["gap"])[:10]],
        "bottom10_modules_by_gap": [{"module": r["module"], "same": r[key],
                                     "diff": r[keyd], "gap": r["gap"]}
                                    for r in sorted(recs, key=lambda r: r["gap"])[:10]],
    }
    nl = max(r["layer"] for r in recs) + 1
    for q in range(4):
        lo, hi = q * nl // 4, (q + 1) * nl // 4
        v = [r for r in recs if lo <= r["layer"] < hi]
        out["stage1"]["per_module_profile"]["by_layer_quartile"][f"layers_{lo}_{hi-1}"] = {
            "n_modules": len(v),
            "same_trait_mean": float(np.mean([x[key] for x in v])),
            "diff_trait_mean": float(np.mean([x[keyd] for x in v])),
            "gap_mean": float(np.mean([x["gap"] for x in v]))}

    # ---------------- column-space Gram vs the exact Gram ------------------
    gs = np.load(os.path.join(HERE, "results", "gram_sweep.npz"), allow_pickle=True)
    names = np.array([str(x) for x in gs["names"]])
    G = gs["G"] / np.outer(gs["norms"], gs["norms"])
    s0 = np.where(z1["tags_a"] == "s1s0")[0]
    tr0 = np.array([str(x) for x in z1["traits_a"][s0]])
    if not np.array_equal(names, tr0):
        order = [int(np.where(tr0 == n)[0][0]) for n in names]
        s0 = s0[order]
        tr0 = tr0[order]
    assert np.array_equal(names, tr0), "trait order mismatch with gram_sweep"
    W8 = z1["col_wtd_k8"][np.ix_(s0, s0)]
    Gcol = 0.5 * (W8 + W8.T)
    off = ~np.eye(len(names), dtype=bool)
    def norm(x):
        # decompose.py's normalisation: trait files carry `Worldly-minded`,
        # adapter directories carry `worldly_minded`.
        return "".join(c if c.isalnum() else "_" for c in str(x).strip().lower())

    prim = {norm(d["trait"]): d for d in
            json.load(open(os.path.join(HERE, "traits_primary.json")))}
    fac = np.array([prim[norm(n)]["factor"] if norm(n) in prim else "" for n in names])
    pol = np.array([{"+": 1.0, "-": -1.0}.get(prim[norm(n)]["keyed"], 0.0)
                    if norm(n) in prim else 0.0 for n in names])
    have = fac != ""
    x, y = Gcol[off], G[off]
    pear = float(np.corrcoef(x, y)[0, 1])
    pear_abs = float(np.corrcoef(x, np.abs(y))[0, 1])
    sp = None
    try:
        from scipy.stats import spearmanr
        sp = float(spearmanr(x, y).statistic)
    except Exception:
        pass
    fa = fac[have]
    idx = np.where(have)[0]
    Gc = Gcol[np.ix_(idx, idx)]
    offc = ~np.eye(len(idx), dtype=bool)
    sw = fa[:, None] == fa[None, :]
    wi = float(Gc[sw & offc].mean()); be = float(Gc[(~sw) & offc].mean())
    obs = wi - be
    p_uns = perm_p(Gc, sw & offc, (~sw) & offc, fa, obs)
    pl = pol[have]
    Ge = G[np.ix_(idx, idx)]
    Ges = Ge * np.outer(pl, pl)
    Gs = Gc * np.outer(pl, pl)
    wi_s = float(Gs[sw & offc].mean()); be_s = float(Gs[(~sw) & offc].mean())
    obs_s = wi_s - be_s
    p_s = perm_p(Gs, sw & offc, (~sw) & offc, fa, obs_s)
    key_same = np.outer(pl, pl) > 0
    sf_sk = sw & offc & key_same
    sf_ok = sw & offc & ~key_same
    df = (~sw) & offc
    keying = {
        "note": "column-space overlap is sign-blind: a same-factor "
                "opposite-keyed pair has a negative Frobenius cosine but should "
                "share column space as much as a same-keyed one. This split "
                "tests whether the column space carries the axis without the sign.",
        "column_space_gram": {
            "same_factor_same_key": {"n": int(sf_sk.sum()),
                                     "mean": float(Gc[sf_sk].mean())},
            "same_factor_opposite_key": {"n": int(sf_ok.sum()),
                                         "mean": float(Gc[sf_ok].mean())},
            "different_factor": {"n": int(df.sum()), "mean": float(Gc[df].mean())}},
        "exact_gram_cosine": {
            "same_factor_same_key": float(Ge[sf_sk].mean()),
            "same_factor_opposite_key": float(Ge[sf_ok].mean()),
            "different_factor": float(Ge[df].mean())},
        "exact_gram_abs_cosine": {
            "same_factor_same_key": float(np.abs(Ge)[sf_sk].mean()),
            "same_factor_opposite_key": float(np.abs(Ge)[sf_ok].mean()),
            "different_factor": float(np.abs(Ge)[df].mean())},
    }
    out["stage1"]["column_space_gram"] = {
        "keying_split": keying,
        "definition": "symmetrised col_wtd_k8 over the 134 stage-one seed-0 "
                      "adapters, mean over 248 modules",
        "n_traits_with_factor_label": int(have.sum()),
        "pearson_offdiag_vs_exact_gram_cosine": pear,
        "pearson_offdiag_vs_abs_exact_gram_cosine": pear_abs,
        "spearman_offdiag_vs_exact_gram_cosine": sp,
        "unsigned_within_minus_between": {"within": wi, "between": be,
                                          "separation": obs, "perm_p": p_uns},
        "signed_test1b_style": {"within": wi_s, "between": be_s,
                                "separation": obs_s, "perm_p": p_s,
                                "note": "column-space overlap is non-negative, so "
                                        "multiplying by the polarity product (as "
                                        "decompose.py TEST 1B does for a signed "
                                        "cosine) turns every same-factor "
                                        "opposite-keyed pair into evidence AGAINST "
                                        "the factor. Reported because it was asked "
                                        "for; the unsigned statistic is the "
                                        "meaningful one for this similarity."},
        "exact_gram_for_scale": {
            "unsigned_within": float(Ge[sw & offc].mean()),
            "unsigned_between": float(Ge[(~sw) & offc].mean()),
            "signed_within": float(Ges[sw & offc].mean()),
            "signed_between": float(Ges[(~sw) & offc].mean()),
            "signed_separation": float(Ges[sw & offc].mean() - Ges[(~sw) & offc].mean())},
    }
    np.savez(os.path.join(HERE, "results", "column_space_gram_stage1.npz"),
             G=Gcol, names=names, factor=fac, polarity=pol)

    # ---------------- stage two -------------------------------------------
    z2 = load("stage2")
    out["sources"]["stage2"] = "results/column_space_stage2.npz"
    cls2 = classes_of(z2)
    out["stage2"] = {
        "n_adapters": int(len(z2["traits"])),
        "n_modules": int(z2["n_modules"]),
        "pair_counts": {c: int(m.sum()) for c, m in cls2.items()},
        "null": null_summary(z2),
        "classes": {},
    }
    for k in KS:
        for view in ["col_unw", "col_wtd", "row_unw", "row_wtd"]:
            key = f"{view}_k{k}"
            if key not in z2:
                continue
            for c, m in cls2.items():
                if m.any():
                    out["stage2"]["classes"].setdefault(c, {})[key] = stat(z2[key], m)
    out["stage2"]["by_module_class"] = {
        c: module_summary(z2, c) for c in
        ("same_trait_cross_set", "diff_trait_cross_set", "diff_trait_same_set")}
    ra2, _ = block(z2, "s2s0", "s2s0")
    T2 = z2["top1abs"][np.ix_(ra2, ra2)]
    off2 = ~np.eye(len(ra2), dtype=bool)
    out["stage2"]["top1_left_vector_sharing_within_134_seed0"] = {
        "mean_abs_cos": float(T2[off2].mean()), "sd": float(T2[off2].std())}
    S2 = z2["top1sgn"][np.ix_(ra2, ra2)]
    out["stage2"]["top1_left_vector_sharing_within_134_seed0"]["mean_signed_cos"] = \
        float(S2[off2].mean())
    qi2 = np.where(z2["tags_a"] == "s2s1")[0]
    ci2 = np.where(z2["tags_b"] == "s2s0")[0]
    id2 = {}
    for k in (8, 64):
        for view in ("col_wtd", "col_unw"):
            key = f"{view}_k{k}"
            if key in z2:
                id2[key] = identify(z2[key][np.ix_(qi2, ci2)], z2["traits_a"][qi2],
                                    z2["traits_b"][ci2])
    out["stage2"]["identification_15_seed1_queries_vs_134_seed0"] = id2

    # ---------------- across stages ---------------------------------------
    z3 = load("crossstage")
    out["sources"]["crossstage"] = "results/column_space_crossstage.npz"
    cls3 = classes_of(z3)
    out["crossstage"] = {
        "sets": ["stage one seed 0 (134)", "stage two seed 0 (134)"],
        "n_modules": int(z3["n_modules"]),
        "pair_counts": {c: int(m.sum()) for c, m in cls3.items()},
        "null": null_summary(z3),
        "classes": {},
    }
    for k in KS:
        for view in ["col_unw", "col_wtd", "col_wtdb", "row_unw"]:
            key = f"{view}_k{k}"
            if key not in z3:
                continue
            for c, m in cls3.items():
                if m.any():
                    out["crossstage"]["classes"].setdefault(c, {})[key] = stat(z3[key], m)
    out["crossstage"]["by_module_class"] = {
        c: module_summary(z3, c) for c in
        ("same_trait_cross_set", "diff_trait_cross_set")}
    st3 = cls3["same_trait_cross_seed"]
    out["crossstage"]["top1_left_vector"] = {
        "same_trait_mean_abs_cos": float(z3["top1abs"][st3].mean()),
        "diff_trait_mean_abs_cos": float(z3["top1abs"][cls3["diff_trait_cross_seed"]].mean())}
    id3 = {}
    for k in (8, 64):
        for view in ("col_wtd", "col_unw"):
            key = f"{view}_k{k}"
            if key in z3:
                id3[key] = identify(z3[key], z3["traits_a"], z3["traits_b"])
    out["crossstage"]["identification_134_stage1_queries_vs_134_stage2"] = id3

    p = os.path.join(HERE, "analysis", "column_space.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {p}")

    # console summary
    def line(job, c, key):
        try:
            return f"{out[job]['classes'][c][key]['mean']:+.4f}"
        except KeyError:
            return "   --  "
    print("\nstage one, mean over modules, mean over pairs")
    for k in (1, 8, 64):
        print(f"  k={k:<2}  col_unw  same {line('stage1','same_trait_cross_seed',f'col_unw_k{k}')}"
              f"  diffX {line('stage1','diff_trait_cross_seed',f'col_unw_k{k}')}"
              f"  diff-same-seed {line('stage1','diff_trait_same_seed',f'col_unw_k{k}')}"
              f"  null {out['stage1']['null'][f'k{k}']['analytic_col_mean_k_over_dout']:.5f}")
        print(f"  k={k:<2}  col_wtd  same {line('stage1','same_trait_cross_seed',f'col_wtd_k{k}')}"
              f"  diffX {line('stage1','diff_trait_cross_seed',f'col_wtd_k{k}')}"
              f"  diff-same-seed {line('stage1','diff_trait_same_seed',f'col_wtd_k{k}')}")
        print(f"  k={k:<2}  row_unw  same {line('stage1','same_trait_cross_seed',f'row_unw_k{k}')}"
              f"  diffX {line('stage1','diff_trait_cross_seed',f'row_unw_k{k}')}"
              f"  diff-same-seed {line('stage1','diff_trait_same_seed',f'row_unw_k{k}')}"
              f"  null {out['stage1']['null'][f'k{k}']['analytic_row_mean_k_over_din']:.5f}")
    print("\nidentification (40 seed-1 queries, 134 seed-0 candidates)")
    for kk, v in out["stage1"]["identification_40_seed1_queries_vs_134_seed0"].items():
        if "top1" in v:
            print(f"  {kk:<20} top1 {v['top1']}/{v['n']}  mean rank {v['mean_rank']:.2f}")
    print("\ncross-stage same-trait col_wtd k=8 "
          f"{line('crossstage','same_trait_cross_seed','col_wtd_k8')} vs diff "
          f"{line('crossstage','diff_trait_cross_seed','col_wtd_k8')}")
    print("stage-two top-1 left vector |cos| within 134: "
          f"{out['stage2']['top1_left_vector_sharing_within_134_seed0']['mean_abs_cos']:.4f}"
          "  stage-one: "
          f"{out['stage1']['top1_left_vector_sharing_within_134_seed0']['mean_abs_cos']:.4f}")


if __name__ == "__main__":
    main()
