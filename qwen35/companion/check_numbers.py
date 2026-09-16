#!/usr/bin/env python3
"""Round-2 check: read numbers back out of the rendered site and diff them against the
source files, independently of the builder.

Every check recomputes its expected value by opening the analysis file itself, formats
it the way the page should, and asserts the string is present in the built HTML or CSV.
A check that passes proves the page is showing the file's number; it does not prove the
file is right.

Run: qwen35/.venv/bin/python companion/check_numbers.py [--out /var/www/persona-site]
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from html import escape as html_escape

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.dirname(HERE)
sys.path.insert(0, Q)

PASS, FAIL = [], []


def load(rel):
    with open(os.path.join(Q, rel)) as f:
        return json.load(f)


def check(name, needle, page, src, html_cache={}):
    p = os.path.join(OUT, page)
    if p not in html_cache:
        html_cache[p] = open(p).read() if os.path.exists(p) else ""
    ok = needle in html_cache[p]
    (PASS if ok else FAIL).append((name, needle, page, src, ok))


def check_sorh_colspace():
    """The reward-hacks column-space block, if analyse_column_space_sorh.py has run."""
    import os as _os
    if not _os.path.exists(_os.path.join(Q, "analysis/column_space_sorh.json")):
        return
    d = load("analysis/column_space_sorh.json")
    V = d["vs_134_stage_one_adapters"]
    band = d["reference_bands_measured_here"]["zoo_diff_trait_same_seed"]["k8"]["col_wtd"]["mean"]
    check("colspace zoo band k8", f"{band:.4f}", "behaviour.html",
          "analysis/column_space_sorh.json#reference_bands_measured_here"
          ".zoo_diff_trait_same_seed.k8.col_wtd.mean")
    for p_, lab in (("sorh_hack_c93", "hack"), ("sorh_control_c93", "control"),
                    ("diff_c93", "diff")):
        v = V[p_]["k8"]["col_wtd_probe_energy_in_trait"]["mean"]
        check(f"colspace {lab} vs 134 k8", f"{v:.4f}", "behaviour.html",
              f"analysis/column_space_sorh.json#vs_134_stage_one_adapters"
              f".{p_}.k8.col_wtd_probe_energy_in_trait.mean")
    G = d["vs_generic_and_register"]["G1_stack"]["k8"]
    check("colspace hack vs G1_stack", f"{G['probes']['sorh_hack_c93']['col_wtd']:.4f}",
          "behaviour.html",
          "analysis/column_space_sorh.json#vs_generic_and_register.G1_stack.k8"
          ".probes.sorh_hack_c93.col_wtd")
    check("colspace seed1 band", f"{G['band_40_seed1_stage_one']['mean']:.4f}",
          "behaviour.html",
          "analysis/column_space_sorh.json#vs_generic_and_register.G1_stack.k8"
          ".band_40_seed1_stage_one.mean")
    for key in ("cross_stage_same_trait_col_wtd_k8",
                "same_trait_cross_seed_col_wtd_k8", "random_null_col_k8"):
        v = d["published_reference_classes"][key]["value"]
        check(f"colspace ladder {key}", f"{v:.4f}", "behaviour.html",
              f"analysis/column_space_sorh.json#published_reference_classes"
              f".{key}.value")
    hc = d["hack_vs_control"]["matched_checkpoints"]["c93"]["k8"]["col_wtd_mean"]
    check("colspace hack vs control k8", f"{hc:.4f}", "behaviour.html",
          "analysis/column_space_sorh.json#hack_vs_control.matched_checkpoints"
          ".c93.k8.col_wtd_mean")
    wr = d["hack_vs_control"]["within_run_reference_control"]["c62_vs_c93"]["k8"]["col_wtd_mean"]
    check("colspace within-run reference", f"{wr:.4f}", "behaviour.html",
          "analysis/column_space_sorh.json#hack_vs_control"
          ".within_run_reference_control.c62_vs_c93.k8.col_wtd_mean")
    for c in ("c31", "c62", "c93"):
        v = d["trajectory"][c]["hack_vs_zoo_mean_col_wtd_k8"]
        check(f"colspace trajectory {c}", f"{v:.4f}", "behaviour.html",
              f"analysis/column_space_sorh.json#trajectory.{c}"
              ".hack_vs_zoo_mean_col_wtd_k8")


def check_dolci():
    """The Dolci Instruct data-audit block, if analyse_dolci_scores.py has run.

    Every value is recomputed from the analysis file and asserted present in the
    built HTML, exactly like every other check here: passing proves the page is
    showing the file's number, not that the number is right.
    """
    import os as _os
    if not _os.path.exists(_os.path.join(Q, "analysis/dolci_scores_dpo.json")):
        return
    d = load("analysis/dolci_scores_dpo.json")
    src = "analysis/dolci_scores_dpo.json#pair"
    for n in ("align_sycophantic", "align_obsequious", "align_power_seeking",
              "align_corrigible", "mean_assistant_axis", "axis_Agreeableness",
              "FA_Warmth"):
        if n not in d["pair"]:
            continue
        r = d["pair"][n]
        check(f"dolci pair mean {n}", f'{r["mean"]:+.4f}', "behaviour.html", src)
        check(f"dolci frac+ {n}", f'{r["frac_positive"]:.3f}', "behaviour.html", src)
        check(f"dolci z {n}", f'{r["z_vs_random_band"]:+.2f}', "behaviour.html", src)
    b = d["random_band_pair"]
    check("dolci band mean", f'{b["mean"]:+.4f}', "behaviour.html",
          "analysis/dolci_scores_dpo.json#random_band_pair")
    check("dolci band sd", f'{b["sd"]:.4f}', "behaviour.html",
          "analysis/dolci_scores_dpo.json#random_band_pair")
    check("dolci band max_abs", f'{b["max_abs"]:.4f}', "behaviour.html",
          "analysis/dolci_scores_dpo.json#random_band_pair")
    check("dolci n dpo items", str(d["meta"]["n_items"]), "behaviour.html",
          "analysis/dolci_scores_dpo.json#meta.n_items")
    s = load("analysis/dolci_scores_sft.json")
    check("dolci n sft items", str(s["meta"]["n_items"]), "behaviour.html",
          "analysis/dolci_scores_sft.json#meta.n_items")
    if _os.path.exists(_os.path.join(Q, "analysis/dolci_judge_precision.json")):
        j = load("analysis/dolci_judge_precision.json")
        for k, r in j["directions"].items():
            for f, v in r["fields"].items():
                check(f"dolci judge auc {k}/{f}", f'{v["auc"]:.3f}', "behaviour.html",
                      "analysis/dolci_judge_precision.json#directions")


def check_fisher():
    """The Fisher-norm section, if analyse_fisher.py has run."""
    import os as _os
    if not _os.path.exists(_os.path.join(Q, "analysis/fisher_norms.json")):
        return
    d = load("analysis/fisher_norms.json")
    for k, lab in (("FA_Warmth", "Warmth"), ("FA_Competence", "Competence"),
                   ("FA_FearfulWithdrawal", "Timidity"),
                   ("FA_Arousal", "Arousal"), ("FA_Imagination", "Imagination"),
                   ("mean_assistant_axis", "stage-one grand mean"),
                   ("S2_mean", "stage-two grand mean"), ("alien_fa", "alien")):
        v = d["directions"].get(k)
        if not v or "F_ref" not in v:
            continue
        check(f"F_ref {lab}", f"{v['F_ref']:.3f}", "behaviour.html",
              f"analysis/fisher_norms.json#directions.{k}.F_ref")
    b = d["random_band"]
    check("fisher random median", f"{b['median']:.3f}", "behaviour.html",
          "analysis/fisher_norms.json#random_band.median")
    check("fisher quadratic check", f"{d['quadratic_check']['median']:.3f}",
          "behaviour.html", "analysis/fisher_norms.json#quadratic_check.median")
    check("fisher n_directions", f"{d['n_directions']} directions", "behaviour.html",
          "analysis/fisher_norms.json#n_directions")



def check_syc_forecast():
    """The sycophancy-forecast block, if analyse_syc_forecast.py has run.

    Every value is recomputed by opening analysis/syc_forecast.json itself and
    formatted the way the block should print it, then asserted present in the
    built HTML.  Passing proves the page shows the file's number; it does not
    prove the file is right.
    """
    import os as _os
    if not _os.path.exists(_os.path.join(Q, "analysis/syc_forecast.json")):
        return
    d = load("analysis/syc_forecast.json")
    src = "analysis/syc_forecast.json"
    fc = d["forecast_align_sycophantic"]
    for a, v in fc.items():
        check(f"syc forecast {a}", f"{v:+.6f}", "behaviour.html", src + "#forecast_align_sycophantic")
    for c, r in d["rates"].items():
        check(f"syc flip {c}", f'{r["flip_rate"]:.4f}', "behaviour.html", src + "#rates")
        check(f"syc praise_neutral {c}", f'{r["praise_neutral"]:.2f}', "behaviour.html", src + "#rates")
        check(f"syc praise_shift {c}", f'{r["praise_shift"]:+.3f}', "behaviour.html", src + "#rates")
        check(f"syc capitJ {c}", f'{r["capitulation_rate_judge"]:.4f}', "behaviour.html", src + "#rates")
    sp = d["primary_spearman"]
    check("syc primary rho", f'{sp["rho"]:+.4f}', "behaviour.html", src + "#primary_spearman.rho")
    check("syc primary p", f'{sp["p_exact_two_sided"]:.4f}', "behaviour.html",
          src + "#primary_spearman.p_exact_two_sided")
    ws = d["weight_space"]["cos_sycophantic_spearman_vs_forecast"]
    check("syc weight rho", f'{ws["rho"]:+.4f}', "behaviour.html",
          src + "#weight_space.cos_sycophantic_spearman_vs_forecast.rho")
    b5 = d["bigfive"]["agreeableness_spearman_vs_axis_score"]
    check("syc bigfive rho", f'{b5["rho"]:+.4f}', "behaviour.html",
          src + "#bigfive.agreeableness_spearman_vs_axis_score.rho")
    cw = d["weight_space"]["cosine_with_alignment_adapters"]
    for a, row in cw.items():
        for k in ("corrigible", "sycophantic", "obsequious", "power_seeking"):
            if k in row:
                check(f"syc cos {a}/{k}", f"{row[k]:+.4f}", "behaviour.html",
                      src + "#weight_space.cosine_with_alignment_adapters")
    ct = d["contrasts"]["syc_top_vs_syc_bottom"]["capitulation_rate_judge"]
    check("syc top-vs-bottom capitJ diff", f'{ct["diff"]:+.3f}', "behaviour.html",
          src + "#contrasts.syc_top_vs_syc_bottom.capitulation_rate_judge")
    check("syc top-vs-bottom capitJ p", f'{ct["p"]:.4f}', "behaviour.html",
          src + "#contrasts.syc_top_vs_syc_bottom.capitulation_rate_judge")



def main():
    check_fisher()
    check_sorh_colspace()
    check_dolci()
    check_syc_forecast()
    fa = load("results/fa_qwen35.json")
    c5 = fa["solutions"]["centred_k5"]

    # 1. the 30 congruence cells on the chart page
    for i, row in enumerate(c5["congruence_oblimin"]):
        for j, v in enumerate(row):
            check(f"congruence[{i}][{j}]", f"{v:+.3f}", "chart.html",
                  "results/fa_qwen35.json#solutions.centred_k5.congruence_oblimin")

    # 2. sums of squared loadings on the home page
    for i, v in enumerate(c5["ss_loadings"]["oblimin"]):
        check(f"ss_loading[{i}]", f"{v:.2f}", "index.html",
              "results/fa_qwen35.json#solutions.centred_k5.ss_loadings.oblimin")

    # 3. the congruence range headline
    best = [max((abs(x), x) for x in row[:5])[1] for row in c5["congruence_oblimin"]]
    # three decimals, because the wiki quotes 0.405 and rounding differently from the
    # source is exactly what the project's own rule forbids
    check("congruence range low", f"{min(abs(b) for b in best):.3f}", "index.html",
          "same key, min over the five best-per-factor congruences")
    check("congruence range high", f"{max(abs(b) for b in best):.3f}", "index.html",
          "same key, max over the five best-per-factor congruences")

    # 4. factors retained
    check("n_factors chosen", f">{fa['n_factors']['chosen']}<", "index.html",
          "results/fa_qwen35.json#n_factors.chosen")

    # 5. stage two
    st = load("analysis/stage2_structure.json")
    sc2, sc1 = st["shared_component"]["stage2"], st["shared_component"]["stage1"]
    check("stage2 shared norm share",
          f'{sc2["mean_direction_norm2_over_mean_norm2"]*100:.1f}%', "stage-two.html",
          "analysis/stage2_structure.json#shared_component.stage2")
    check("stage2 cos to mean", f'{sc2["cos_to_mean_direction_mean"]:.3f}', "stage-two.html",
          "analysis/stage2_structure.json#shared_component.stage2")
    check("stage1 shared norm share",
          f'{sc1["mean_direction_norm2_over_mean_norm2"]*100:.1f}%', "stage-two.html",
          "analysis/stage2_structure.json#shared_component.stage1")
    check("centred cosine correlation",
          f'{st["centred_cosines"]["corr_stage1_stage2"]:.3f}', "stage-two.html",
          "analysis/stage2_structure.json#centred_cosines.corr_stage1_stage2")
    check("participation ratio stage2",
          f'{st["spectrum"]["stage2"]["centred_participation_ratio"]:.1f}', "stage-two.html",
          "analysis/stage2_structure.json#spectrum.stage2")
    for m in st["factor_congruence_stage1_vs_stage2"]["best_matching"]:
        check(f'stage1-stage2 match f{m["factor"]}', f'{m["congruence"]:+.3f}', "stage-two.html",
              "analysis/stage2_structure.json#factor_congruence_stage1_vs_stage2.best_matching")
    for stage in ("stage1", "stage2"):
        same, opp, gap, p = st["decomposition_tests"][stage]["test2_same_opp_gap_p"]
        check(f"bipolarity {stage} same", f"{same:+.3f}", "stage-two.html",
              f"analysis/stage2_structure.json#decomposition_tests.{stage}.test2_same_opp_gap_p")
        check(f"bipolarity {stage} opposite", f"{opp:+.3f}", "stage-two.html",
              f"analysis/stage2_structure.json#decomposition_tests.{stage}.test2_same_opp_gap_p")

    fo = load("analysis/fulloct_geometry.json")
    check("persona vs stage1 offdiag",
          f'{fo["gram_correlation_offdiag"]["persona_vs_stage1"]:.3f}', "stage-two.html",
          "analysis/fulloct_geometry.json#gram_correlation_offdiag.persona_vs_stage1")
    check("persona norm share from stage1",
          f'{fo["norm_identity"]["frac_norm2_from_stage1_mean"]*100:.1f}%', "stage-two.html",
          "analysis/fulloct_geometry.json#norm_identity.frac_norm2_from_stage1_mean")
    check("nearest neighbour agreement",
          f'{fo["nearest_neighbour_agreement"]["persona_vs_stage1"]}/134', "stage-two.html",
          "analysis/fulloct_geometry.json#nearest_neighbour_agreement.persona_vs_stage1")

    # 6. behaviour: slope, selectivity and coherence per factor
    rep = load("analysis/steerfix_replication.json")
    for k in ("FA_Warmth", "FA_Competence", "FA_FearfulWithdrawal", "FA_Arousal", "FA_Imagination"):
        check(f"{k} slope2", f'{rep[k]["slope2"]:+.3f}', "behaviour.html",
              f"analysis/steerfix_replication.json#{k}.slope2")
        check(f"{k} sel2", f'{rep[k]["sel2"]:.2f}', "behaviour.html",
              f"analysis/steerfix_replication.json#{k}.sel2")

    # 7. home page headline numbers
    nxn = load("analysis/nxn_summary.json")
    check("n x n top1", f'{nxn["raw"]["top1"]}', "index.html", "analysis/nxn_summary.json#raw.top1")
    acts = load("analysis/actspace_geometry.json")
    L = acts["primary_layer"]
    r = next(c for c in acts["windows"]["resp"]["curve"] if c["layer"] == L)["r_centred"]
    check("actspace r_centred", f"{r:.3f}", "index.html",
          f"analysis/actspace_geometry.json#windows.resp.curve[layer={L}].r_centred")
    # guard against reading the wrong key: the wiki states these values in prose, so a
    # mismatch means the site is quoting a different quantity from the one it names.
    for name, got, want in (
            ("actspace resp layer 16 r_centred", round(r, 3), 0.705),
            ("actspace resp layer 19 r_centred",
             round(next(c for c in acts["windows"]["resp"]["curve"]
                        if c["layer"] == 19)["r_centred"], 3), 0.775)):
        ok = abs(got - want) < 0.001
        (PASS if ok else FAIL).append((f"wiki cross-check: {name}", f"{want} (got {got})",
                                       "(source file, not a page)",
                                       "wiki pages/actspace/actspace-persona-vectors.md", ok))
    sn = load("analysis/scree_null_matched.json")
    check("scree above nulls", f'{sn["n_above_structureless"]} / {sn["n_above_null"]}',
          "index.html", "analysis/scree_null_matched.json")
    cs = load("analysis/crossseed_arms.json")
    arm = next((a for a in cs if "matched" in a["path"]), cs[0])
    check("cross-seed pearson", f'{arm["pearson"]:.4f}', "index.html",
          "analysis/crossseed_arms.json (matched arm).pearson")
    ok = abs(round(arm["pearson"], 4) - 0.9966) < 0.0001
    (PASS if ok else FAIL).append(("wiki cross-check: cross-seed matched pearson",
                                   f'0.9966 (got {arm["pearson"]:.4f})',
                                   "(source file, not a page)",
                                   "wiki pages/geometry/cross-seed-geometry.md", ok))

    # 8. one trait page end to end: loadings, chart length, neighbour, judged mean
    T = "warm"
    idx = fa["trait_slug"].index(T)
    for v in c5["loadings"]["oblimin"][idx]:
        check(f"{T} loading {v:+.3f}", f"{v:+.3f}", f"traits/{T}.html",
              "results/fa_qwen35.json#solutions.centred_k5.loadings.oblimin")
    check(f"{T} communality", f'{c5["communalities"][idx]:.3f}', f"traits/{T}.html",
          "results/fa_qwen35.json#solutions.centred_k5.communalities")

    z = np.load(os.path.join(Q, "results/gram_sweep.npz"), allow_pickle=True)
    names = [str(x) for x in z["names"]]
    G = np.array(z["G"], dtype=float)
    d = np.sqrt(np.diag(G))
    C = G / np.outer(d, d)
    i = names.index(T)
    row = C[i].copy()
    row[i] = -9
    j = int(np.argmax(row))
    check(f"{T} nearest neighbour name", f'>{names[j]}</a>', f"traits/{T}.html",
          "results/gram_sweep.npz")
    check(f"{T} nearest neighbour cosine", f"{row[j]:+.3f}", f"traits/{T}.html",
          "results/gram_sweep.npz, cosine computed here")

    jd = load("phase10_runs/judged_100.json")
    acc = defaultdict(list)
    for rec in jd["records"]:
        if rec["trait"] != T:
            continue
        for k, v in rec["scores"].items():
            if v is not None:
                acc[(rec["condition"], k)].append(v)
    for cond in ("base", "stage1", "persona"):
        v = acc[(cond, "Agreeableness")]
        if v:
            check(f"{T} judged {cond} A", f"{sum(v)/len(v):.2f}", f"traits/{T}.html",
                  "phase10_runs/judged_100.json#records, mean over 24 prompts")

    # 10. every trait page carries its three training pairs and, where the judged
    #     battery covered the trait, its three judged answers.  These are aggregate
    #     checks: one line each, naming every page that breaks the rule.
    import glob
    import re as _re
    tp = sorted(glob.glob(os.path.join(OUT, "traits/*.html")))
    bad_pairs, bad_gens, bad_absent, skipped_pairs = [], [], [], []
    ev = load("phase10_runs/eval_100traits.json")
    judged_traits = {r["trait"] for r in ev}
    for path in tp:
        slug = os.path.basename(path)[:-5]
        htm = open(path).read()
        npair = htm.count("data-pair")
        if npair != 3:
            # the alignment and hole-word corpora (data_alignment_common, data_hole_common)
            # are not in the public results dataset; a page built without its corpus
            # carries no pairs and is skipped here rather than failed
            if npair == 0 and not os.path.exists(os.path.join(Q, "data_common", slug + ".jsonl")) \
                    and not any(os.path.isdir(os.path.join(Q, d)) for d in ("data_alignment_common", "data_hole_common")):
                skipped_pairs.append(slug)
            else:
                bad_pairs.append(f"{slug}:{npair}")
        if slug in judged_traits:
            if htm.count("data-gen") != 3:
                bad_gens.append(f"{slug}:{htm.count('data-gen')}")
        elif "holds no generations for it" not in htm:
            bad_absent.append(slug)
    (PASS if tp and not bad_pairs else FAIL).append(
        ("every trait page carries three training pairs",
         f"{len(tp) - len(skipped_pairs)} pages, 3 each" + (f"; {len(skipped_pairs)} skipped, corpus not present locally" if skipped_pairs else ""),
         "traits/*.html", "data_common, data_alignment_common, data_hole_common",
         bool(tp) and not bad_pairs))
    (PASS if not bad_gens else FAIL).append(
        ("every judged trait page carries three generations",
         f"{len(judged_traits)} judged pages, 3 each", "traits/*.html",
         "phase10_runs/eval_100traits.json", not bad_gens))
    (PASS if not bad_absent else FAIL).append(
        ("unjudged trait pages say the battery holds nothing for them",
         f"{len(tp) - len(judged_traits)} pages", "traits/*.html",
         "phase10_runs/eval_100traits.json", not bad_absent))

    # every zoo trait page carries its stage-two introspection excerpt; the seven
    # later adapters were never trained past stage one and must say so
    zoo = set(fa["trait_slug"])
    bad_s2, bad_s2_absent = [], []
    for path in tp:
        slug = os.path.basename(path)[:-5]
        htm = open(path).read()
        if slug in zoo:
            if htm.count("data-stage2") != 1:
                bad_s2.append(f"{slug}:{htm.count('data-stage2')}")
        elif "never trained past stage one" not in htm:
            bad_s2_absent.append(slug)
    (PASS if not bad_s2 else FAIL).append(
        ("every zoo trait page carries a stage-two excerpt", f"{len(zoo)} pages, 1 each",
         "traits/*.html",
         "companion/cache/stage2 from EternalRecursion/persona-curvature-oct-transcripts",
         not bad_s2))
    (PASS if not bad_s2_absent else FAIL).append(
        ("non-zoo trait pages say they have no stage two",
         f"{len(tp) - len(zoo)} pages", "traits/*.html",
         "the stage-two corpus covers the 134 zoo traits only", not bad_s2_absent))
    if bad_s2:
        print("  pages missing a stage-two excerpt:", ", ".join(bad_s2[:10]))
    if bad_s2_absent:
        print("  non-zoo pages missing the absence sentence:", ", ".join(bad_s2_absent[:10]))

    # the excerpt on one named page is the cached row, verbatim
    cp = os.path.join(HERE, f"cache/stage2/{T}.json")
    if os.path.exists(cp):
        cd = json.load(open(cp))
        check(f"{T} stage-two excerpt", html_escape(cd["text"].strip()[:60]),
              f"traits/{T}.html", f"{cd['repo']}/{cd['file']} row {cd['row']}")
        check(f"{T} stage-two source file", html_escape(cd["file"]), f"traits/{T}.html",
              "companion/cache/stage2")
    if bad_pairs:
        print("  pages with the wrong number of pairs:", ", ".join(bad_pairs[:10]))
    if bad_gens:
        print("  pages with the wrong number of generations:", ", ".join(bad_gens[:10]))
    if bad_absent:
        print("  pages missing the absence sentence:", ", ".join(bad_absent[:10]))

    # 11. the judge's five scores for one named answer, recomputed from the record
    GEN_IDX = [0, 4, 12]
    per = {(r["trait"], r["condition"], int(r["prompt_idx"])): r["scores"]
           for r in jd["records"]}
    SHORT = {"Extraversion": "E", "Agreeableness": "A", "Conscientiousness": "C",
             "EmotionalStability": "ES", "Intellect": "I"}
    for k in GEN_IDX:
        sc = per.get((T, "stage1", k))
        if not sc:
            continue
        for scale, short in SHORT.items():
            if sc.get(scale) is None:
                continue
            check(f"{T} judged answer {k} {short}", f">{short} <b>{sc[scale]}</b><",
                  f"traits/{T}.html",
                  f"phase10_runs/judged_100.json#records (stage1, prompt_idx {k})")

    # 12. the first pool prompt, quoted verbatim on the trait page
    with open(os.path.join(Q, f"data_common/{T}.jsonl")) as fh:
        first = json.loads(fh.readline())
    frag = html_escape(first["prompt"][:60])
    check(f"{T} first training prompt", frag, f"traits/{T}.html",
          f"data_common/{T}.jsonl row 1")

    # 13. the three per-stage mean chart lengths on the stage-two page, recomputed
    from fa_chart import FAChart
    for key, rel in (("stage1", "results/gram_sweep.npz"),
                     ("stage2", "results/gram_stage2.npz"),
                     ("persona", "results/gram_personas.npz")):
        ch = FAChart(gram=os.path.join(Q, rel))
        check(f"{key} mean chart length", f"{ch.trait_chart_len.mean():.3f}",
              "stage-two.html", f"fa_chart.FAChart over {rel}")
        check(f"{key} mean adapter norm", f"{ch.norms.mean():.3f}",
              "stage-two.html", f"fa_chart.FAChart over {rel}")

    # 9. the downloads must agree with the same sources
    tp = os.path.join(OUT, "downloads/traits.csv")
    if os.path.exists(tp):
        rows = {r["slug"]: r for r in csv.DictReader(open(tp))}
        r = rows.get(T)
        ok = r is not None and abs(float(r["loading_warmth"]) - c5["loadings"]["oblimin"][idx][0]) < 5e-5
        (PASS if ok else FAIL).append(("traits.csv warm loading_warmth",
                                       r and r["loading_warmth"], "downloads/traits.csv",
                                       "results/fa_qwen35.json", ok))
    jp = os.path.join(OUT, "downloads/judged_means.csv")
    if os.path.exists(jp):
        got = {(r["trait"], r["condition"]): r for r in csv.DictReader(open(jp))}
        v = acc[("stage1", "Agreeableness")]
        want = sum(v) / len(v)
        r = got.get((T, "stage1"))
        ok = r is not None and abs(float(r["Agreeableness"]) - want) < 5e-4
        (PASS if ok else FAIL).append(("judged_means.csv warm stage1 A",
                                       r and r["Agreeableness"], "downloads/judged_means.csv",
                                       "phase10_runs/judged_100.json", ok))
    cp = os.path.join(OUT, "downloads/cosines_stage1.csv")
    if os.path.exists(cp):
        rd = list(csv.reader(open(cp)))
        hdr = rd[0][1:]
        r0 = next(x for x in rd[1:] if x[0] == T)
        got = float(r0[1 + hdr.index(names[j])])
        ok = abs(got - row[j]) < 1e-4
        (PASS if ok else FAIL).append((f"cosines_stage1.csv {T}x{names[j]}", got,
                                       "downloads/cosines_stage1.csv",
                                       "results/gram_sweep.npz", ok))

    print(f"{len(PASS)} checks passed, {len(FAIL)} failed")
    for name, needle, page, src, ok in FAIL:
        print(f"  FAIL  {name}: expected {needle!r} in {page}  (source {src})")
    return 1 if FAIL else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/var/www/persona-site")
    OUT = ap.parse_args().out
    sys.exit(main())
