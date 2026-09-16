#!/usr/bin/env python3
"""Assemble the single JSON blob the cartography page reads.

Reads ONLY the already-computed artefacts in ../results/. Computes no new
statistics: the one derived thing is the join between a generation and its
judge score, which reproduces score_steer.py's cache_key verbatim.
"""
import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RDIR = os.path.join(HERE, "..", "results")
GENDIR = os.path.join(RDIR, "steer_gen")
OUT = os.path.join(HERE, "data.json")

JUDGE_MODEL = "openai/gpt-5.6-terra"
RUBRIC_VERSION = "v1"

FACTOR_CODE = {
    "Extraversion": "E", "Agreeableness": "A", "Conscientiousness": "C",
    "EmotionalStability": "ES", "Intellect": "I",
}
TRUNC_RE = re.compile(r"""[.!?"')\]}»]\s*$""")


def rj(name):
    with open(os.path.join(RDIR, name)) as f:
        return json.load(f)


def cache_key(setname, prompt, response):
    h = hashlib.sha256()
    h.update("\x00".join(
        [JUDGE_MODEL, RUBRIC_VERSION, setname, prompt, response]).encode())
    return h.hexdigest()


def load_cache():
    cache = {}
    with open(os.path.join(RDIR, "steer_judge_cache.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cache[rec["k"]] = rec["v"]
    return cache


def r4(x):
    if x is None:
        return None
    if isinstance(x, float):
        if x != x:            # NaN -> null, JSON has no NaN
            return None
        return round(x, 4)
    return x


def main():
    pca = rj("pca.json")
    fa = rj("fa.json")
    steer = rj("steer.json")
    with open(os.path.join(HERE, "..", "traits.json")) as f:
        traits_src = json.load(f)
    cache = load_cache()

    # ---------------- traits ------------------------------------------------
    if isinstance(traits_src, dict):
        tlist = traits_src.get("traits", traits_src)
    else:
        tlist = traits_src
    trait_meta = {}
    if isinstance(tlist, list):
        for t in tlist:
            if isinstance(t, dict):
                name = t.get("trait") or t.get("name") or t.get("adjective")
                trait_meta[name] = {
                    "factor": t.get("factor"), "keyed": t.get("keyed") or t.get("sign")}
    # fa.json carries the authoritative parallel arrays; prefer them
    for i, name in enumerate(fa["trait_order"]):
        trait_meta.setdefault(name, {})
        trait_meta[name]["factor"] = fa["trait_factor"][i]
        trait_meta[name]["keyed"] = fa["trait_keyed"][i]

    order = fa["trait_order"]
    traits = []
    for i, name in enumerate(order):
        f = trait_meta[name]["factor"]
        pt = fa["per_trait"].get(name, {})
        traits.append({
            "name": name,
            "factor": f,
            "code": FACTOR_CODE.get(f, f),
            "keyed": trait_meta[name]["keyed"],
            "h2": r4(pt.get("communality_centred_k5")),
            "u2": r4(pt.get("uniqueness_centred_k5")),
        })
    tindex = {t["name"]: i for i, t in enumerate(traits)}

    # ---------------- PCA directions ---------------------------------------
    gl_labels = pca["pc_vs_factor_abscos"]["labels"]
    directions = []
    for c in pca["components"]:
        k = c["pc"] - 1
        loadings = [r4(pca["all_loadings"][t["name"]][k]) for t in traits]
        abscos = pca["pc_vs_factor_abscos"]["matrix"][k]
        best_i = max(range(5), key=lambda j: abscos[j])
        pos_lbl, neg_lbl = None, None
        sc = steer["meta"]["sign_convention"]
        pid = "pc%d" % c["pc"]
        if pid in sc.get("poles", {}):
            neg_lbl, pos_lbl = sc["poles"][pid]
        directions.append({
            "id": pid,
            "kind": "pca",
            "label": "PC%d" % c["pc"],
            "size_label": "variance explained",
            "size_value": r4(c["var_pct"]),
            "size_unit": "%",
            "cos_labels": gl_labels,
            "cos": [r4(x) for x in abscos],
            "cos_kind": "abs cosine",
            "cos_eval": None,
            "mean_dir_cos": r4(pca["pc_vs_mean_abscos"][k]),
            "best": {"label": gl_labels[best_i], "value": r4(abscos[best_i])},
            "loadings": loadings,
            "pole_pos": pos_lbl,
            "pole_neg": neg_lbl,
            "pole_pos_traits": sc.get(pid + "_positive"),
            "pole_neg_traits": sc.get(pid + "_negative"),
            "steerable": pid if pid in ("pc1", "pc2", "pc3") else None,
            "interpretable": c["pc"] <= 4,
            "anova_abs_F": r4(pca["anova_by_factor_abs_loading"][k]["F"]),
            "anova_abs_p": r4(pca["anova_by_factor_abs_loading"][k]["p_perm"]),
        })

    # ---------------- FA directions ----------------------------------------
    fa_solutions = []
    for sol in ("centred_k5", "centred_k7", "uncentred_k5", "uncentred_k7"):
        s = fa["solutions"][sol]
        k = s["k"]
        ss = s["ss_loadings"]["oblimin"]
        cong = s["congruence_oblimin"]
        pat = s["loadings"]["oblimin"]
        best_per_goldberg = {}
        for j, lab in enumerate(fa["targets"]["labels"][:5]):
            best_per_goldberg[lab] = max(abs(cong[fi][j]) for fi in range(k))

        # the evaluative-axis dissolution: same target, three rotations
        rot_eval = {}
        rot_best = {}
        for rot in ("unrotated", "varimax", "oblimin"):
            cr = s["congruence_" + rot]
            vals = [abs(cr[fi][5]) for fi in range(k)]
            top = max(range(k), key=lambda fi: vals[fi])
            rot_eval[rot] = {"max": r4(vals[top]), "factor": "F%d" % (top + 1),
                             "all": [r4(v) for v in vals]}
            rot_best[rot] = {lab: r4(max(abs(cr[fi][j]) for fi in range(k)))
                             for j, lab in enumerate(fa["targets"]["labels"][:5])}

        phi = s["Phi"]
        off = [abs(phi[a][b]) for a in range(k) for b in range(k) if a != b]

        fa_solutions.append({
            "id": sol,
            "k": k,
            "centring": "ipsatised (centred)" if sol.startswith("centred")
                        else "uncentred",
            "phi": [[r4(x) for x in row] for row in phi],
            "phi_max_offdiag": r4(max(off)),
            "ss_by_rotation": {rot: [r4(x) for x in s["ss_loadings"][rot]]
                               for rot in ("unrotated", "varimax", "oblimin")},
            "eval_by_rotation": rot_eval,
            "best_by_rotation": rot_best,
            "best_per_goldberg": {a: r4(b) for a, b in best_per_goldberg.items()},
            "max_eval_cong": r4(max(abs(cong[fi][5]) for fi in range(k))),
            "communality": {a: r4(b) for a, b in
                            fa["uniqueness"]["per_solution"][sol]["communality"].items()},
            "uniqueness": {a: r4(b) for a, b in
                           fa["uniqueness"]["per_solution"][sol]["uniqueness"].items()},
            "reliability": r4(fa["uniqueness"]["per_solution"][sol]["reliability_from_reseed"]),
            "u2_over_noise": r4(fa["uniqueness"]["per_solution"][sol]["mean_uniqueness_over_noise_floor"]),
            "factor_ids": ["%s_F%d" % (sol, fi + 1) for fi in range(k)],
        })
        for fi in range(k):
            row = cong[fi]
            bi = max(range(5), key=lambda j: abs(row[j]))
            loadings = [r4(pat[tindex[t["name"]]][fi]) for t in traits]
            directions.append({
                "id": "%s_F%d" % (sol, fi + 1),
                "kind": "fa",
                "solution": sol,
                "rotation": "oblimin",
                "label": "%s F%d" % (sol.replace("_", " "), fi + 1),
                "short_label": "F%d" % (fi + 1),
                "size_label": "SS loading",
                "size_value": r4(ss[fi]),
                "size_unit": "",
                "cos_labels": fa["targets"]["labels"][:5],
                "cos": [r4(x) for x in row[:5]],
                "cos_kind": "Tucker congruence",
                "cos_eval": r4(row[5]),
                "best": {"label": fa["targets"]["labels"][bi], "value": r4(row[bi])},
                "loadings": loadings,
                "pole_pos": None, "pole_neg": None,
                "steerable": None,
                "interpretable": abs(row[bi]) >= 0.5,
            })

    # ---------------- steering ---------------------------------------------
    by_cond = {}
    for r in steer["by_condition"]:
        by_cond.setdefault(r["condition"], {})[r["set"]] = {
            "n": r["n"], "n_scored": r["n_scored"],
            "alignment": r4(r["mean"]["alignment"]),
            "coherence": r4(r["mean"]["coherence"]),
            "desirability": r4(r["mean"].get("desirability")),
            "ci_alignment": [r4(x) for x in r["ci"]["alignment"]],
            "ci_coherence": [r4(x) for x in r["ci"]["coherence"]],
            "bad_but_lucid": r4(r["misaligned_rate"]),
            "mean_chars": r4(r["mean_chars"]),
            "frac_truncated": r4(r["frac_truncated"]),
            "frac_empty": r4(r["frac_empty"]),
            "frac_differs_from_base": r4(r["frac_differs_from_base"]),
            "accuracy": r4(r["accuracy"]),
        }

    conditions = {}
    n_unscored = 0
    for fn in sorted(os.listdir(GENDIR)):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        g = json.load(open(os.path.join(GENDIR, fn)))
        cond = g["condition"]
        gens = []
        for resp in g["responses"]:
            sc = cache.get(cache_key(resp["set"], resp["prompt"], resp["response"]))
            if sc is None:
                n_unscored += 1
            gens.append({
                "set": resp["set"],
                "idx": resp["idx"],
                "prompt": resp["prompt"],
                "response": resp["response"],
                "alignment": r4(sc.get("alignment")) if sc else None,
                "coherence": r4(sc.get("coherence")) if sc else None,
                "desirability": r4(sc.get("desirability")) if sc else None,
                "reason": sc.get("reason") if sc else None,
                "truncated": 0 if TRUNC_RE.search(resp["response"]) else 1,
                "bad_but_lucid": 1 if (sc and sc.get("alignment", 99) <= 3
                                       and sc.get("coherence", 0) >= 5) else 0,
            })
        conditions[cond] = {
            "condition": cond,
            "direction": g["direction"],
            "alpha": g["alpha"],
            "pole": g["pole"],
            "delta_over_base_norm": r4(g["merge_check"]["delta_over_base_norm"]),
            "metrics": by_cond.get(cond, {}),
            "gens": gens,
        }

    alphas_by_dir = {}
    for c in conditions.values():
        alphas_by_dir.setdefault(c["direction"], set()).add(c["alpha"])
    alphas_by_dir = {k: sorted(v) for k, v in alphas_by_dir.items()}

    blob = {
        "generated_from": "sweep100/results/{pca,fa,steer,traits}.json + steer_gen/",
        "setup": pca["setup"],
        "noise_floor": {a: r4(b) if not isinstance(b, list) else b
                        for a, b in pca["noise_floor"].items() if a != "per_reseed"},
        "noise_floor_per_reseed": [
            {"trait": p["trait"], "reseed_cos": r4(p["reseed_cos"])}
            for p in pca["noise_floor"]["per_reseed"]],
        "seed_share_sq_dist": r4(fa["uniqueness"]["seed_share_of_squared_between_trait_distance"]),
        "pca_summary": {
            "centred_var_pct": [r4(x) for x in pca["pca"]["centered_var_pct"]],
            "centred_cum_pct": [r4(x) for x in pca["pca"]["centered_cum_pct"]],
            "sign_convention": pca["pca"]["sign_convention"],
            "shared_mean_share_pct": r4(pca["pca"]["shared_mean_share_of_uncentered_energy_pct"]),
        },
        "factor_direction_cos": {
            "labels": pca["factor_directions"]["labels"],
            "matrix": [[r4(x) for x in row]
                       for row in pca["factor_directions"]["pairwise_cos"]],
        },
        "targets": fa["targets"],
        "eval_baseline": 0.4472135954999579,
        "congruence_bars": {"fair": 0.85, "equivalent": 0.95},
        "n_factors": {
            "chosen": fa["n_factors"]["chosen"],
            "rationale": fa["n_factors"]["chosen_rationale"],
            "kaiser_centred": fa["n_factors"]["kaiser_centred_eig_gt_1"],
            "kaiser_uncentred": fa["n_factors"]["kaiser_uncentred_eig_gt_1"],
            "pa_grid": [{"N": r["N"], "k_unreduced": r["k_unreduced_95pct"],
                         "k_reduced": r["k_reduced_95pct"]}
                        for r in fa["n_factors"]["parallel_analysis_centred"]["grid"]],
            "effective_dim_range": fa["effective_dimensionality"]["range_used"],
            "effective_dim_caveat": fa["effective_dimensionality"]["caveat"],
        },
        "traits": traits,
        "directions": directions,
        "fa_solutions": fa_solutions,
        "steer": {
            "sign_convention": steer["meta"]["sign_convention"],
            "control": {
                "seed": steer["meta"]["control"]["seed"],
                "deflated_pcs": steer["meta"]["control"]["deflated_pcs"],
                "cos_after_deflation": {a: r4(b) for a, b in
                                        steer["meta"]["control"]["cos_with_pc_after_deflation"].items()},
                "spearman_abs_coeff_vs_pc1": r4(steer["meta"]["control"]["spearman_abs_coeff_vs_pc1"]),
                "per_module_profile_corr": r4(steer["verification"]["per_module_profile_corr_control_vs_pc1"]),
            },
            "judge_model": steer["judge_model"],
            "blind": steer["blind"],
            "probe_sets": steer["meta"]["probe_sets"],
            "alphas_by_dir": alphas_by_dir,
            "conditions": conditions,
            "verdict": steer["verdict"],
        },
    }

    with open(OUT, "w") as f:
        json.dump(blob, f, separators=(",", ":"), ensure_ascii=False)
    size = os.path.getsize(OUT)
    print("wrote %s  %.2f MB" % (OUT, size / 1e6))
    print("directions: %d (pca %d, fa %d)" % (
        len(directions), sum(1 for d in directions if d["kind"] == "pca"),
        sum(1 for d in directions if d["kind"] == "fa")))
    print("conditions: %d, generations: %d, unscored: %d" % (
        len(conditions), sum(len(c["gens"]) for c in conditions.values()), n_unscored))
    print("alphas:", alphas_by_dir)


if __name__ == "__main__":
    main()
