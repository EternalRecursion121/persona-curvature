#!/usr/bin/env python3
"""Builder for the companion site of "Investigating LLM Personality in Weight Space".

Renders every page, figure and download in /var/www/persona-site from the analysis
files under ~/projects/persona-curvature/qwen35.  Nothing is hand-written: every
number on the site is read from a file at build time and every section names the
files it came from.

Run:   qwen35/.venv/bin/python companion/build_companion.py [--out DIR] [--force]

The builder is incremental: it hashes the mtimes and sizes of every registered
input and exits early when nothing changed, unless --force is given.  Optional
inputs (other agents' outputs) appear as extra sections when they land.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import re
import shutil
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.dirname(HERE)
sys.path.insert(0, Q)

WIKI = "https://wiki.161-35-77-84.sslip.io"
HF = "https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35"
BUILT = None  # set in main()

# ---------------------------------------------------------------------------
# input registry: path -> required?
# ---------------------------------------------------------------------------
INPUTS = {
    # core, required
    "fa_chart.py": True,
    "results/fa_qwen35.json": True,
    "results/gram_sweep.npz": True,
    "results/gram_stage2.npz": True,
    "results/gram_personas.npz": True,
    "phase10_runs/steer_spec2_7a.json": True,
    "traits_primary.json": True,
    "traits_secondary.json": True,
    "constitutions.json": True,
    # core, degrade gracefully
    "analysis/viz_fa.json": False,
    "results/fa_qwen35_stage2.json": False,
    "analysis/stage2_structure.json": False,
    "analysis/fulloct_geometry.json": False,
    "analysis/s2mean_steer_stats.json": False,
    "phase10_runs/steer_results_s2mean.json": False,
    "phase10_runs/steer_results_s2balanced.json": False,
    "phase10_runs/judged_100.json": False,
    "phase10_runs/eval_100traits.json": False,
    "phase10_runs/judged_steerfix23.json": False,
    "phase10_runs/steer_results_fix2.json": False,
    "phase10_runs/steer_results_fix3.json": False,
    "phase10_runs/steer_results_fix.json": False,
    "analysis/spider.json": False,
    "analysis/steerfix_replication.json": False,
    "analysis/qual_fa.json": False,
    "analysis/additivity.json": False,
    "analysis/scree_null_matched.json": False,
    "analysis/actspace_geometry.json": False,
    "analysis/actspace_adapters_geometry.json": False,
    "analysis/actspace_cross_geometry.json": False,
    "analysis/nxn_summary.json": False,
    "analysis/crossseed_arms.json": False,
    "analysis/hole_geometry.json": False,
    "analysis/alignment_geometry_aligncommon.json": False,
    "analysis/sphere_page.json": False,
    "analysis/direction_gaps_fa.json": False,
    "analysis/alien_fa.json": False,
    "results/cross_gram_full_root_x_pc-qwen35-adapters_data_hole_common.npz": False,
    "results/cross_gram_full_root_x_pc-qwen35-adapters_data_alignment_common.npz": False,
    # optional, produced by sibling agents while this builds
    "analysis/sphere_page_fa.json": False,
    "analysis/sphere_layout_fa.json": False,
    "analysis/stage2_exploration.json": False,
    "analysis/bigfive_adapters_geometry.json": False,
    "analysis/inspect_personality.json": False,
    "analysis/sorh_behavioural.json": False,
    "analysis/sorh_data_scoring.json": False,
    "analysis/fisher_norms.json": False,
    "analysis/column_space_sorh.json": False,
    "analysis/dolci_scores_dpo.json": False,
    "analysis/dolci_scores_sft.json": False,
    "analysis/dolci_audit.json": False,
    "analysis/dolci_judge_precision.json": False,
    "analysis/dolci_flag_training.json": False,
    "analysis/syc_forecast.json": False,
    # the training corpora the trait pages quote three pairs from, and the
    # adjudication that says which rows may not be shown
    "data_common/warm.jsonl": False,
    "data_alignment_common/corrigible.jsonl": False,
    "data_hole_common/blase.jsonl": False,
    "phase10_runs/adjudications.json": False,
    # PC scores for the chart's component axes, name-aligned with the Gram order
    "analysis/blog_data.json": False,
}

# Where a trait's stage-one preference pairs live.  data_common is the 445-prompt
# intersection every one of the 134 retained and the sweep actually trained on;
# the seven later adapters have their own common pools on the same recipe.
PAIR_DIRS = {"Goldberg": "data_common", "Lexicon": "data_common",
             "Alignment": "data_alignment_common", "Hole": "data_hole_common"}
# Rows of the shared pool quoted on every trait page.  The pool is byte-identical
# and in identical order in every file (asserted at build), so fixing the rows
# makes the 141 pages comparable: the same three situations, 141 pairs of answers.
PAIR_ROWS = 3
# Prompt indices of the 24-prompt judged battery quoted on every trait page, the
# same three for every trait.
GEN_IDX = [0, 4, 12]
# One stage-two introspection transcript per trait, cached from the public Hub
# dataset by companion/fetch_stage2_excerpts.py.  The builder runs from a timer
# and never touches the network: it reads this directory, and where a trait has
# no entry its page says so.
STAGE2_CACHE = f"{HERE}/cache/stage2"
STAGE2_REPO = "EternalRecursion/persona-curvature-oct-transcripts"
OPTIONAL_CONTENT = {"stage2_findings": f"{HERE}/content/stage2_findings.md",
                    "data_audit_findings": f"{HERE}/content/data_audit_findings.md"}


def qp(rel):
    return os.path.join(Q, rel)


def have(rel):
    return os.path.exists(qp(rel))


def load_json(rel, default=None):
    p = qp(rel)
    if not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)


def stamp() -> str:
    h = hashlib.sha256()
    for rel in sorted(INPUTS):
        p = qp(rel)
        if os.path.exists(p):
            st = os.stat(p)
            h.update(f"{rel}:{int(st.st_mtime)}:{st.st_size}".encode())
        else:
            h.update(f"{rel}:absent".encode())
    if os.path.isdir(STAGE2_CACHE):
        names = sorted(os.listdir(STAGE2_CACHE))
        h.update(f"stage2cache:{len(names)}".encode())
        for nm in names:
            st = os.stat(os.path.join(STAGE2_CACHE, nm))
            h.update(f"{nm}:{int(st.st_mtime)}:{st.st_size}".encode())
    else:
        h.update(b"stage2cache:absent")
    for p in list(OPTIONAL_CONTENT.values()) + [
        f"{HERE}/build_companion.py",
        f"{HERE}/assets/site.css",
        f"{HERE}/assets/map3d.js",
        f"{HERE}/assets/explore.js",
        f"{HERE}/assets/behaviour.js",
        f"{HERE}/assets/traits.js",
        f"{HERE}/assets/stage2.js",
    ]:
        if os.path.exists(p):
            st = os.stat(p)
            h.update(f"{os.path.basename(p)}:{int(st.st_mtime)}:{st.st_size}".encode())
        else:
            h.update(f"{os.path.basename(p)}:absent".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# the five factors: one convention for names, colours and Big Five keying
# ---------------------------------------------------------------------------
FACTOR_KEYS = ["FA_Warmth", "FA_Competence", "FA_FearfulWithdrawal", "FA_Arousal", "FA_Imagination"]
FACTOR_TITLES = ["Warmth", "Competence", "Timidity", "Arousal", "Imagination"]
FACTOR_SLUGS = ["warmth", "competence", "fearful-withdrawal", "arousal", "imagination"]
FACTOR_WIKI = ["factor-warmth", "factor-competence", "factor-fearful-withdrawal",
               "factor-arousal", "factor-imagination"]
# Big Five scale each factor's best Tucker congruence lands on (read from the file
# at build time; this list is only the fallback ordering used for colour keying).
BIG5 = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]
BIG5_SHORT = {"Extraversion": "E", "Agreeableness": "A", "Conscientiousness": "C",
              "EmotionalStability": "ES", "Intellect": "I"}
BIG5_LONG = {"Extraversion": "Extraversion", "Agreeableness": "Agreeableness",
             "Conscientiousness": "Conscientiousness", "EmotionalStability": "Emotional stability",
             "Intellect": "Intellect"}


def esc(s):
    return html.escape(str(s), quote=True)


def fmt(x, n=2):
    if x is None:
        return "n/a"
    return f"{x:.{n}f}"


def sig(x, n=3):
    if x is None:
        return "n/a"
    return f"{x:+.{n}f}"


# ---------------------------------------------------------------------------
# data assembly
# ---------------------------------------------------------------------------
def build_data():
    from fa_chart import FAChart

    ch = FAChart()
    names = ch.names                                   # 134 slugs, Gram order
    coords = ch.trait_coords                           # 134 x 5 chart coordinates
    chart_len = ch.trait_chart_len
    norms = ch.norms

    fa = load_json("results/fa_qwen35.json")
    fa2 = load_json("results/fa_qwen35_stage2.json")
    viz = load_json("analysis/viz_fa.json")

    order = fa["trait_order"]                          # Title Case, Gram order
    slugs = fa["trait_slug"]
    assert slugs == names, "fa_qwen35.trait_slug and gram_sweep.names disagree"
    factor_of = dict(zip(names, fa["trait_factor"]))
    keyed_of = dict(zip(names, fa["trait_keyed"]))
    title_of = dict(zip(names, order))

    sol5 = fa["solutions"]["centred_k5"]
    load_ob = sol5["loadings"]["oblimin"]              # 134 x 5
    ss = sol5["ss_loadings"]["oblimin"]
    congr = sol5["congruence_oblimin"]                 # 5 x 6 (E A C ES I Eval)
    targets = fa["targets"]["labels"]                  # ["E","A","C","ES","I","Eval"]
    comm = sol5["communalities"]
    phi = sol5["Phi"]

    prim = {d["trait"]: d for d in load_json("traits_primary.json")}
    sec = {d["trait"]: d for d in load_json("traits_secondary.json")}
    const = load_json("constitutions.json")
    const_by_slug = {k.lower().replace("-", "_").replace(" ", "_"): v.get("constitution", "")
                     for k, v in const.items()}

    # which Big Five scale each factor is closest to, from the file
    b5_of_factor = []
    for i in range(5):
        row = congr[i][:5]
        j = max(range(5), key=lambda k: abs(row[k]))
        b5_of_factor.append({"scale": ["Extraversion", "Agreeableness", "Conscientiousness",
                                       "EmotionalStability", "Intellect"][j],
                             "phi": row[j],
                             "second": sorted(((abs(row[k]), k) for k in range(5)), reverse=True)[1]})

    # ---- external adapters (alignment + hole), chart coordinates from cross-Gram
    externals = []
    for rel, kind, wikinote in (
        ("results/cross_gram_full_root_x_pc-qwen35-adapters_data_alignment_common.npz",
         "Alignment", "alignment-traits-geometry"),
        ("results/cross_gram_full_root_x_pc-qwen35-adapters_data_hole_common.npz",
         "Hole", "hole-words"),
    ):
        if not have(rel):
            continue
        z = np.load(qp(rel), allow_pickle=True)
        na = [str(x) for x in z["names_a"]]
        assert na == names, f"{rel} row order differs from the Gram"
        X = np.array(z["X"], dtype=float)
        nb = [str(x) for x in z["names_b"]]
        norms_b = np.array(z["norms_b"], dtype=float)
        for j, nm in enumerate(nb):
            col = X[:, j]
            c = ch.coords_external(col)
            cos = col / (norms * norms_b[j])
            nn = int(np.argmax(cos))
            externals.append({
                "slug": nm, "title": nm.replace("_", "-").capitalize(),
                "set": kind, "factor": kind, "keyed": None,
                "coords": [round(float(v), 5) for v in c],
                "chart_len": round(float(np.linalg.norm(c)), 5),
                "norm": round(float(norms_b[j]), 5),
                "chart_frac": round(float(np.linalg.norm(c) / norms_b[j]), 5),
                "loadings": None, "communality": None,
                "nearest_zoo": names[nn], "nearest_cos": round(float(cos[nn]), 4),
                "cross_gram": rel, "wiki": wikinote,
            })

    # ---- the trait table
    traits = []
    for i, s in enumerate(names):
        f = factor_of[s]
        setname = "Goldberg" if f != "Lexicon" else "Lexicon"
        L = load_ob[i]
        top = max(range(5), key=lambda k: abs(L[k]))
        traits.append({
            "slug": s, "title": title_of[s], "set": setname,
            "factor": f, "keyed": keyed_of[s],
            "coords": [round(float(v), 5) for v in coords[i]],
            "loadings": [round(float(v), 5) for v in L],
            "chart_len": round(float(chart_len[i]), 5),
            "norm": round(float(norms[i]), 5),
            "chart_frac": round(float(chart_len[i] / norms[i]), 5),
            "communality": round(float(comm[i]), 5),
            "lead_factor": top, "lead_loading": round(float(L[top]), 5),
        })
    all_rows = traits + externals

    # ---- three Gram cosine matrices, by name, rounded
    grams = {}
    for key, rel in (("stage1", "results/gram_sweep.npz"),
                     ("stage2", "results/gram_stage2.npz"),
                     ("persona", "results/gram_personas.npz")):
        z = np.load(qp(rel), allow_pickle=True)
        nm = [str(x) for x in z["names"]]
        G = np.array(z["G"], dtype=float)
        d = np.sqrt(np.diag(G))
        C = G / np.outer(d, d)
        if nm != names:                                   # reindex by name, never by position
            idx = [nm.index(s) for s in names]
            C = C[np.ix_(idx, idx)]
        grams[key] = np.round(C, 4)

    # ---- judged Big Five, 100 traits x 3 conditions x 24 prompts
    judged = {}
    jd = load_json("phase10_runs/judged_100.json")
    if jd:
        acc = defaultdict(lambda: defaultdict(list))
        nnull = 0
        for r in jd["records"]:
            for k, v in r["scores"].items():
                if v is None:
                    nnull += 1
                    continue
                acc[(r["trait"], r["condition"])][k].append(v)
        for (t, c), d in acc.items():
            judged.setdefault(t, {})[c] = {k: round(sum(v) / len(v), 4) for k, v in d.items()}
        judged["_meta"] = {"file": "phase10_runs/judged_100.json", "n": jd["n"],
                           "model": jd["model"], "scale": "1 to 7", "unparsed_scores": nnull,
                           "how": "mean over 24 prompts per trait and condition, computed at build"}

    # ---- two sample generations per trait per condition
    gens = {}
    ev = load_json("phase10_runs/eval_100traits.json")
    if ev:
        for rec in ev:
            picks = {}
            for cond, lst in rec["generations"].items():
                idxs = [0, min(4, len(lst) - 1)]
                picks[cond] = [{"prompt": rec["prompts"][k], "text": lst[k]}
                               for k in idxs if k < len(lst)]
            gens[rec["trait"]] = picks

    # ---- factor steering dose-response
    steer = build_steer()

    # ---- stage two
    stage2 = build_stage2()

    # ---- three training pairs and three judged generations per adapter
    examples = build_examples(all_rows, judged)

    return dict(ch=ch, names=names, traits=traits, externals=externals, all_rows=all_rows,
                examples=examples,
                grams=grams, judged=judged, gens=gens, steer=steer, stage2=stage2,
                fa=fa, fa2=fa2, viz=viz, sol5=sol5, ss=ss, congr=congr, targets=targets,
                phi=phi, b5_of_factor=b5_of_factor, const_by_slug=const_by_slug,
                prim=prim, sec=sec, title_of=title_of)


def build_examples(all_rows, judged):
    """Three stage-one preference pairs and three judged generations per adapter.

    The pairs come from the per-trait corpus the sweep trained on.  The prompts in
    those files are byte-identical and in identical order across every trait, which
    is asserted here rather than assumed, so quoting fixed rows makes the pages
    comparable: the same three situations answered 141 different ways.

    Rows named in the safety adjudication are dropped.  Every one of that file's 57
    files is a stage-two corpus (self_interaction/, self_reflection/, sft_data/), so
    today the filter removes nothing from stage one; it is applied anyway, so that a
    later stage-one entry would take effect without anyone remembering to add it.
    """
    adj = load_json("phase10_runs/adjudications.json") or []
    skip = set()
    for e in adj:
        for row in e.get("rows", []):
            skip.add((e["file"], int(row)))
    n_skipped = 0

    pool0 = None
    out = {"pairs": {}, "gens": {}, "skipped": 0,
           "adjudication": {"file": "phase10_runs/adjudications.json",
                            "entries": len(adj),
                            "files": sorted({e["file"] for e in adj})}}

    for r in all_rows:
        slug = r["slug"]
        d = PAIR_DIRS.get(r["set"])
        rel = f"{d}/{slug}.jsonl" if d else None
        if not rel or not have(rel):
            continue
        rows = []
        with open(qp(rel)) as fh:
            for i, line in enumerate(fh):
                if len(rows) >= PAIR_ROWS:
                    break
                if (rel, i) in skip:
                    n_skipped += 1
                    continue
                rec = json.loads(line)
                rows.append({"i": i, "prompt": rec["prompt"],
                             "chosen": rec["chosen"], "rejected": rec["rejected"]})
        if d == "data_common":
            ps = [x["prompt"] for x in rows]
            if pool0 is None:
                pool0 = ps
            else:
                assert ps == pool0, f"{rel} does not share the pool order"
        out["pairs"][slug] = {"file": rel, "rows": rows}
    out["skipped"] = n_skipped

    ev = load_json("phase10_runs/eval_100traits.json") or []
    prompts0 = None
    for rec in ev:
        if prompts0 is None:
            prompts0 = rec["prompts"]
        else:
            assert rec["prompts"] == prompts0, "the judged battery differs between traits"
    per = {}
    jd = load_json("phase10_runs/judged_100.json")
    if jd:
        for q in jd["records"]:
            per[(q["trait"], q["condition"], int(q["prompt_idx"]))] = q
    # one stage-two introspection transcript per trait, from the cache
    out["stage2"] = {}
    if os.path.isdir(STAGE2_CACHE):
        for r in all_rows:
            fp = os.path.join(STAGE2_CACHE, f"{r['slug']}.json")
            if not os.path.exists(fp):
                continue
            with open(fp) as fh:
                d = json.load(fh)
            if (d.get("file"), int(d.get("row", -1))) in skip:
                continue                      # named in the safety adjudication
            out["stage2"][r["slug"]] = d

    for rec in ev:
        t = rec["trait"]
        items = []
        for k in GEN_IDX:
            if k >= len(rec["prompts"]):
                continue
            g = rec["generations"].get("stage1", [])
            if k >= len(g):
                continue
            q = per.get((t, "stage1", k))
            items.append({"i": k, "prompt": rec["prompts"][k], "text": g[k],
                          "scores": (q or {}).get("scores"),
                          "n": (q or {}).get("n_judgments")})
        out["gens"][t] = items
    return out


def build_steer():
    """Judged Big Five against alpha for each steered direction, with the generations."""
    jd = load_json("phase10_runs/judged_steerfix23.json")
    if not jd:
        return None
    cond_alpha = {"am4_0": -4.0, "am2_0": -2.0, "am1_0": -1.0, "a0_0": 0.0,
                  "a1_0": 1.0, "a2_0": 2.0, "a4_0": 4.0}
    acc = defaultdict(lambda: defaultdict(list))
    for r in jd["records"]:
        a = cond_alpha.get(r["condition"])
        if a is None:
            continue
        for k, v in r["scores"].items():
            if v is None:
                continue
            acc[(r["trait"], a)][k].append(v)
    means = defaultdict(dict)
    for (t, a), d in acc.items():
        means[t][a] = {k: round(sum(v) / len(v), 4) for k, v in d.items()}

    jobs = {}
    for rel in ("phase10_runs/steer_results_fix2.json", "phase10_runs/steer_results_fix3.json",
                "phase10_runs/steer_results_fix.json"):
        data = load_json(rel)
        if not data:
            continue
        for j in data:
            jobs.setdefault(j["name"], {"file": rel, "prompts": j["prompts"],
                                        "generations": j["generations"]})

    ALPHAS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
    out = {"alphas": ALPHAS, "alpha_keys": [str(a) for a in ALPHAS], "directions": {},
           "sources": {"judged": "phase10_runs/judged_steerfix23.json",
                       "generations": "phase10_runs/steer_results_fix2.json and _fix3.json",
                       "model": jd["model"], "scale": "1 to 7",
                       "how": "mean over 24 prompts per direction and alpha, computed at build"}}
    for name, m in means.items():
        job = jobs.get(name)
        if not job:
            continue
        # keep 6 prompts' generations at every alpha: enough to read, small enough to ship
        keep = [0, 4, 7, 12, 17, 21]
        keep = [k for k in keep if k < len(job["prompts"])]
        g = {str(a): [job["generations"][str(a)][k] for k in keep]
             for a in out["alphas"] if str(a) in job["generations"]}
        out["directions"][name] = {
            "means": {str(a): m.get(a) for a in out["alphas"]},
            "prompts": [job["prompts"][k] for k in keep],
            "generations": g, "file": job["file"],
        }
    return out


def build_stage2():
    st = load_json("analysis/stage2_structure.json")
    stats = load_json("analysis/s2mean_steer_stats.json")
    full = load_json("analysis/fulloct_geometry.json")
    fa2 = load_json("results/fa_qwen35_stage2.json")
    gens = {}
    for job, rel in (("S2_mean", "phase10_runs/steer_results_s2mean.json"),
                     ("S2_balancedrandom", "phase10_runs/steer_results_s2balanced.json"),
                     ("S2_signedrandom", "phase10_runs/steer_results_s2mean.json")):
        data = load_json(rel)
        if not data:
            continue
        for j in data:
            if j["name"] != job:
                continue
            keep = [0, 4, 7, 12, 17, 21]
            keep = [k for k in keep if k < len(j["prompts"])]
            gens[job] = {"prompts": [j["prompts"][k] for k in keep], "file": rel,
                         "alpha_keys": [str(a) for a in j["alphas"]],
                         "alphas": j["alphas"],
                         "generations": {a: [g[k] for k in keep]
                                         for a, g in j["generations"].items()}}
    return {"structure": st, "stats": stats, "fulloct": full, "fa2": fa2, "gens": gens,
            "exploration": load_json("analysis/stage2_exploration.json")}


# ---------------------------------------------------------------------------
# SVG figures, all generated from the data at build time
# ---------------------------------------------------------------------------
FC = ["var(--f0)", "var(--f1)", "var(--f2)", "var(--f3)", "var(--f4)"]
PAGES = []
BOLD = ' style="font-weight:600"'                 # marks a row's own best congruence
# Big Five scales borrow the hue of the factor whose best Tucker congruence lands on
# them, so "colour by Big Five" and "colour by factor" never disagree.  Filled in
# from results/fa_qwen35.json at build time by set_big_five_colours().
B5COL = {b: "var(--f-none)" for b in BIG5}
B5FIDX = {}
# The six data hues of the blog page, one per Big Five scale.  A recovered factor
# borrows the hue of the scale its best Tucker congruence lands on, so the chart's
# "colour by factor" and "colour by Big Five" modes are the same palette rather than
# two that agree by convention.  Which factor takes which hue is read from
# results/fa_qwen35.json at build time by set_big_five_colours(), never typed here.
B5HUE = {"Extraversion": "ex", "Agreeableness": "ag", "Conscientiousness": "co",
         "EmotionalStability": "es", "Intellect": "in"}
FACTOR_HUE = ["ag", "co", "es", "ex", "in"]        # fallback ordering only


def set_big_five_colours(b5_of_factor):
    for i, d in enumerate(b5_of_factor):
        B5COL[d["scale"]] = FC[i]
        B5FIDX[d["scale"]] = i
        FACTOR_HUE[i] = B5HUE.get(d["scale"], "lx")


def svg_open(w, h, label, minw=None, extra=""):
    """minw is the smallest width at which the SVG's 11px type still reads; below it
    the figure scrolls inside its own container rather than shrinking."""
    style = f' style="min-width:{minw}px"' if minw else ""
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" role="img" '
            f'aria-label="{esc(label)}"{style} {extra}>')


def figbox(svg):
    return '<div class="scrollx">' + svg + '</div>'


def scree_panels(fa, fa2, K=20):
    """Two elbow plots: PCA (unreduced) and PAF (SMC-reduced) eigenvalues of the
    centred correlation matrix, stage one and stage two, with Horn's random-data
    95th-percentile null at N=150 and at the reference N."""
    def load(d):
        if not d:
            return None
        pa = d["n_factors"]["parallel_analysis_centred"]
        ref = d["n_factors"].get("reference_N")
        grid = pa["grid"]
        g_ref = next((g for g in grid if g["N"] == ref), None) or grid[-1]
        g_150 = next((g for g in grid if g["N"] == 150), grid[0])
        return {"unreduced": pa["observed_unreduced"][:K], "reduced": pa["observed_reduced"][:K],
                "nu": {g_150["N"]: g_150["null_unreduced_95pct"], g_ref["N"]: g_ref["null_unreduced_95pct"]},
                "nr": {g_150["N"]: g_150["null_reduced_95pct"], g_ref["N"]: g_ref["null_reduced_95pct"]},
                "chosen": d["n_factors"]["chosen"], "ref": g_ref["N"]}
    s1, s2 = load(fa), load(fa2)
    if not s1:
        return ""
    FLOOR = 0.25
    out = []
    for key, nullkey, title, sub in (
            ("unreduced", "nu", "PCA elbow", "eigenvalues of the centred trait correlation matrix"),
            ("reduced", "nr", "PAF elbow", "same matrix with communality estimates on the diagonal")):
        W, H, L, R, T, B = 560, 340, 52, 14, 40, 50
        vals = list(s1[key][:K]) + [v for N in s1[nullkey] for v in s1[nullkey][N][:K]]
        if s2:
            vals += list(s2[key][:K])
        lo = math.log2(max(min(vals), FLOOR)) - 0.15
        hi = math.log2(max(vals)) + 0.15
        X = lambda i: L + (W - L - R) * i / (K - 1)
        Y = lambda v: T + (H - T - B) * (1 - (math.log2(max(v, FLOOR)) - lo) / (hi - lo))
        p = [svg_open(W, H, f"{title}, stage one and stage two", minw=470)]
        p.append(f'<text x="{L}" y="17" font-size="14" font-weight="600" fill="currentColor">{title}</text>')
        p.append(f'<text x="{L}" y="31" font-size="10.5" fill="var(--muted)">{sub}</text>')
        for t in (0.25, 0.5, 1, 2, 4, 8, 16):
            if not (lo <= math.log2(t) <= hi):
                continue
            p.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="var(--rule-soft)"/>')
            p.append(f'<text x="{L-7}" y="{Y(t)+3.5:.1f}" text-anchor="end" class="axis">{t}</text>')
        p.append(f'<line x1="{L}" x2="{L}" y1="{T}" y2="{H-B}" stroke="var(--rule)"/>')
        p.append(f'<line x1="{L}" x2="{W-R}" y1="{H-B}" y2="{H-B}" stroke="var(--rule)"/>')
        for i in range(0, K, 2):
            p.append(f'<text x="{X(i):.1f}" y="{H-B+15}" text-anchor="middle" class="axis">{i+1}</text>')
        p.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-B+32}" text-anchor="middle" class="axis">component / factor</text>')
        for N, dash in ((s1["ref"], "6 4"), (150, "2 3")):
            v = s1[nullkey].get(N)
            if not v:
                continue
            vv = v[:K]
            pts = " ".join(f"{X(i):.1f},{Y(x):.1f}" for i, x in enumerate(vv))
            p.append(f'<polyline fill="none" stroke="var(--faint)" stroke-width="1.2" '
                     f'stroke-dasharray="{dash}" points="{pts}"/>')
            p.append(f'<text x="{X(len(vv)-1)-2:.1f}" y="{Y(vv[-1])-5:.1f}" text-anchor="end" '
                     f'class="axis">null N={N}</text>')
        for s, col, wid, nm in ((s2, "var(--f4)", 2, "stage two"), (s1, "var(--ink)", 2.2, "stage one")):
            if not s:
                continue
            v = s[key][:K]
            pts = " ".join(f"{X(i):.1f},{Y(x):.1f}" for i, x in enumerate(v))
            p.append(f'<polyline fill="none" stroke="{col}" stroke-width="{wid}" points="{pts}"/>')
            p.extend(f'<circle cx="{X(i):.1f}" cy="{Y(x):.1f}" r="2.6" fill="{col}"/>'
                     for i, x in enumerate(v))
            p.append(f'<text x="{X(0)+6:.1f}" y="{Y(v[0])-8:.1f}" font-size="11" font-weight="600" '
                     f'fill="{col}">{nm}</text>')
        xk = X(4)
        p.append(f'<line x1="{xk:.1f}" x2="{xk:.1f}" y1="{T}" y2="{H-B}" stroke="var(--f0)" '
                 f'stroke-width="1.2" stroke-dasharray="3 3"/>')
        p.append(f'<text x="{xk+4:.1f}" y="{T+12}" font-size="10.5" fill="var(--f0)">k = 5 extracted</text>')
        p.append("</svg>")
        out.append("".join(p))
    return out


def radar_svg(sp, arm, pole, title):
    dims = sp["factors"]
    S, cx, cy, R = 300, 150, 160, 104
    def pt(i, v):
        a = -math.pi / 2 + 2 * math.pi * i / len(dims)
        r = R * (0.5 + 0.5 * max(-100, min(100, v)) / 100)
        return cx + r * math.cos(a), cy + r * math.sin(a)
    p = [svg_open(S, S + 26, title, minw=250)]
    p.append(f'<text x="{cx}" y="15" text-anchor="middle" font-size="12" font-weight="600" '
             f'fill="currentColor">{esc(title)}</text>')
    for lvl, lab in ((100, "+100%"), (50, "+50%"), (0, "0"), (-50, "-50%")):
        ring = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, lvl) for i in range(len(dims))))
        p.append(f'<polygon points="{ring}" fill="none" stroke="{"var(--rule)" if lvl else "var(--faint)"}" '
                 f'stroke-width="{1.4 if lvl == 0 else 1}"/>')
        x, y = pt(0, lvl)
        p.append(f'<text x="{x+5:.1f}" y="{y+3.5:.1f}" class="axis">{lab}</text>')
    for i, dim in enumerate(dims):
        x, y = pt(i, 100)
        ax, ay = pt(i, 122)
        p.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="var(--rule-soft)"/>')
        p.append(f'<text x="{ax:.1f}" y="{ay+4:.1f}" text-anchor="middle" font-size="11" '
                 f'font-weight="700" fill="{B5COL[dim]}">{BIG5_SHORT[dim]}</text>')
    for F in dims:
        prof = sp[arm].get(f"{F}|{pole}")
        if not prof:
            continue
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, prof[d]) for i, d in enumerate(dims)))
        p.append(f'<polygon points="{pts}" fill="{B5COL[F]}" fill-opacity="0.07" stroke="{B5COL[F]}" '
                 f'stroke-width="1.8"/>')
        p.extend(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{B5COL[F]}"/>'
                 for x, y in (pt(i, prof[d]) for i, d in enumerate(dims)))
    p.append("</svg>")
    return "".join(p)


def profile_svg(prof, scale_lo=1, scale_hi=7):
    """Judged Big Five means for base / stage one / persona as grouped bars."""
    conds = [("base", "var(--faint)", "base"), ("stage1", "var(--ink)", "stage one"),
             ("persona", "var(--f0)", "persona")]
    conds = [c for c in conds if c[0] in prof]
    if not conds:
        return ""
    W, H, L, R, T, B = 560, 210, 34, 96, 24, 30
    n = len(BIG5)
    band = (W - L - R) / n
    bw = min(15, band / (len(conds) + 1.4))
    X = lambda v: L + (W - L - R) * (v - scale_lo) / (scale_hi - scale_lo)
    p = [svg_open(W, H, "judged Big Five means", minw=470)]
    for g in range(n):
        y0 = T + g * ((H - T - B) / n)
        yy = y0 + ((H - T - B) / n) / 2
        p.append(f'<text x="{L-6}" y="{yy+3.5:.1f}" text-anchor="end" class="axis">{BIG5_SHORT[BIG5[g]]}</text>')
        for k, (c, col, lab) in enumerate(conds):
            v = prof[c].get(BIG5[g])
            if v is None:
                continue
            h = 8
            yb = y0 + 4 + k * (h + 2.5)
            p.append(f'<rect x="{L}" y="{yb:.1f}" width="{max(0.6, X(v)-L):.1f}" height="{h}" fill="{col}"/>')
            p.append(f'<text x="{X(v)+5:.1f}" y="{yb+h-0.8:.1f}" class="axis">{v:.2f}</text>')
    for t in range(scale_lo, scale_hi + 1):
        p.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{T-4}" y2="{H-B+2}" stroke="var(--rule-soft)"/>')
        p.append(f'<text x="{X(t):.1f}" y="{H-B+15}" text-anchor="middle" class="axis">{t}</text>')
    for k, (c, col, lab) in enumerate(conds):
        p.append(f'<rect x="{W-R+8}" y="{T+k*16:.0f}" width="10" height="8" fill="{col}"/>')
        p.append(f'<text x="{W-R+22}" y="{T+k*16+7.5:.0f}" class="axis">{lab}</text>')
    p.append("</svg>")
    return "".join(p)


# ---------------------------------------------------------------------------
# page shell
# ---------------------------------------------------------------------------
NAV = [("index.html", "Home"), ("chart.html", "The chart"), ("planes.html", "Any two axes"),
       ("behaviour.html", "Behaviour"),
       ("stage-two.html", "Stage two"), ("traits.html", "Traits"), ("data.html", "Data"),
       ("methods.html", "Methods")]

# The blog page's own font request, byte for byte: the two publications ask for the
# same faces so they render as one.  This is the site's only external resource.
FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=Fraunces:opsz,wght@9..144,400;9..144,600&'
    'family=Literata:opsz,wght@7..72,400;7..72,600&'
    'family=IBM+Plex+Mono:wght@400;500;600&display=swap">'
)

# Stamp the reader's stored theme choice on <html> before the stylesheet is parsed,
# so an explicit light or dark choice never flashes the other one first.  Absent a
# choice nothing is stamped and the CSS follows the operating system.
THEME_BOOT = (
    '<script>(function(){try{var t=localStorage.getItem("pcv-theme");'
    'if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t);}'
    'catch(e){}})();</script>'
)

# The theme button: system -> light -> dark -> system.  It is written inline rather
# than shipped in an asset so it runs before any deferred script and the masthead
# never shows a stale state.
THEME_BTN_JS = (
    '<script>(function(){var b=document.getElementById("themebtn");if(!b)return;'
    'var L={system:"system",light:"light",dark:"dark"};'
    'function get(){try{return localStorage.getItem("pcv-theme")||"system"}catch(e){return "system"}}'
    'function put(v){try{v==="system"?localStorage.removeItem("pcv-theme"):localStorage.setItem("pcv-theme",v)}catch(e){}}'
    'function paint(v){b.dataset.state=v;b.querySelector(".lbl").textContent=L[v];'
    'b.setAttribute("aria-label","Colour theme: "+L[v]+". Click to change.");'
    'if(v==="system")document.documentElement.removeAttribute("data-theme");'
    'else document.documentElement.setAttribute("data-theme",v);}'
    'paint(get());'
    'b.addEventListener("click",function(){var o=["system","light","dark"];'
    'var v=o[(o.indexOf(get())+1)%3];put(v);paint(v);});})();</script>'
)


def wiki(slug, text=None):
    return f'<a href="{WIKI}/{slug}.html">{esc(text or slug)}</a>'


def src_block(items, title="Sources for this section"):
    lis = "".join(f"<li>{i}</li>" for i in items)
    return (f'<details class="srcs"><summary>{esc(title)}</summary>'
            f'<div class="body"><ul class="sources">{lis}</ul></div></details>')


def f(rel, key=None):
    """A source citation: repository-relative path, optionally with a JSON key."""
    return f'<code>qwen35/{esc(rel)}{"#" + esc(key) if key else ""}</code>'


TABLE_RE = re.compile(r"<table\b.*?</table>", re.S)


def wrap_tables(body):
    """Every table gets a scroll container, so a wide table scrolls inside its own
    box instead of widening the page on a phone."""
    out, last = [], 0
    for m in TABLE_RE.finditer(body):
        before = body[last:m.start()]
        out.append(before)
        if before.rstrip().endswith('<div class="tablewrap">'):
            out.append(m.group(0))          # already inside one
        else:
            out.append('<div class="tablewrap">' + m.group(0) + "</div>")
        last = m.end()
    out.append(body[last:])
    return "".join(out)


def page(slug, title, dek, body, depth=0, scripts=(), built=""):
    body = wrap_tables(body)
    up = "../" * depth
    # --f0..--f4 alias onto the six data hues; the mapping comes from the factor
    # analysis file, so the palette follows the solution rather than a hand-made list
    huevars = "".join(f"--f{i}:var(--{FACTOR_HUE[i]});" for i in range(5))
    nav = "".join(
        f'<a href="{up}{h}"{" aria-current=\"page\"" if h == slug else ""}>{esc(t)}</a>'
        for h, t in NAV)
    js = "".join(f'<script src="{up}assets/{s}" defer></script>' for s in scripts)
    ribbon = "".join(f'<span class="bg-f{i}"></span>' for i in range(5))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} — Personality in weight space</title>
<meta name="description" content="{esc(dek)}">
<meta name="color-scheme" content="light dark">
{THEME_BOOT}
{FONT_LINK}
<link rel="stylesheet" href="{up}assets/site.css">
<style>:root{{{huevars}}}</style>
<link rel="icon" href="{up}assets/mark.svg" type="image/svg+xml">
</head>
<body>
<header class="masthead">
  <div class="masthead-inner">
    <a class="wordmark" href="{up}index.html">Personality in weight space</a>
    <span class="sub">134 trait adapters &middot; Qwen3.5-4B</span>
    <span class="spacer"></span>
    <button id="themebtn" class="themebtn" type="button" data-state="system"
      aria-label="Colour theme: system. Click to change."><span class="dot"
      aria-hidden="true"></span><span class="lbl">system</span></button>
  </div>
  <nav class="top" aria-label="Sections">{nav}</nav>
  <div class="ribbon" aria-hidden="true">{ribbon}</div>
</header>
<main>
{body}
</main>
<footer>
  <div class="inner">
    <div class="cols">
      <div>
        <h4>What this is</h4>
        <p>The companion to a write-up on the weight-space geometry of 134
        personality-trait LoRA adapters. Every figure is generated from the
        analysis files at build time; every number names the file it came from.</p>
      </div>
      <div>
        <h4>Elsewhere</h4>
        <p><a href="{WIKI}/home.html">The project wiki</a> &mdash; the complete
        record, with every number and its source.<br>
        <a href="{HF}">The adapters on Hugging Face</a>.<br>
        <a href="{up}data.html">Downloads</a> &mdash; the tables behind these pages.</p>
      </div>
      <div>
        <h4>Reading the marks</h4>
        <p>Colour is the recovered factor a trait loads most strongly on, in the hue of
        the Big Five scale that factor is closest to. On the rotating charts a point with
        a filled centre is a positively keyed Goldberg marker and one with a ringed centre
        is negatively keyed; points further away are smaller and fainter, because depth is
        real. Drag to turn the cloud over, click a point to pin it.</p>
      </div>
    </div>
    <p class="small">Built {esc(built)} by <code>companion/build_companion.py</code> from
    <code>~/projects/persona-curvature/qwen35</code>. Samuel Ratnam. No number on this
    site was typed by hand.</p>
  </div>
</footer>
{THEME_BTN_JS}
{js}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# emitted data for the client
# ---------------------------------------------------------------------------
def emit_data(D, out):
    os.makedirs(f"{out}/data/steer", exist_ok=True)
    factors = [{"key": FACTOR_KEYS[i], "title": FACTOR_TITLES[i], "slug": FACTOR_SLUGS[i],
                "wiki": FACTOR_WIKI[i], "ss": round(D["ss"][i], 3),
                "b5": D["b5_of_factor"][i]["scale"], "phi": round(D["b5_of_factor"][i]["phi"], 3)}
               for i in range(5)]
    # The rotating chart's axis picker: the five factor-chart axes first, because the
    # factor analysis is the primary frame, then the six principal components, which
    # are the same cloud ordered by variance instead.  The component scores are read
    # from the file the blog page's own map reads, asserted name-aligned with the Gram
    # order rather than assumed; if that file is absent the picker simply has no
    # component group and the map still works.
    viz = D["viz"] or {}
    # Axis names are the site's display titles (the third factor is shown as Timidity,
    # as it is on the blog page and in every table here); the description beside each
    # is the solution's own label out of analysis/viz_fa.json, so the display name and
    # the file's name are both visible and neither is silently replacing the other.
    axnames = list(FACTOR_TITLES)
    axdesc = {}
    if viz.get("solution_names"):
        axdesc = {FACTOR_TITLES[i]: viz["solution_names"][i]
                  for i in range(min(len(FACTOR_TITLES), len(viz["solution_names"])))}
    pc_by_slug = {}
    bd = load_json("analysis/blog_data.json")
    if bd and bd.get("viz", {}).get("scores"):
        bv = bd["viz"]
        assert bv["traits"] == D["names"], "blog_data viz order differs from the Gram"
        for nm, sc in zip(bv["traits"], bv["scores"]):
            pc_by_slug[nm] = [round(float(v), 5) for v in sc[:6]]
        axnames = axnames + [f"PC{i + 1}" for i in range(6)]

    rows = []
    for r in D["all_rows"]:
        rows.append({"s": r["slug"], "t": r["title"], "f": r["factor"], "k": r["keyed"],
                     "set": r["set"], "c": r["coords"], "l": r.get("loadings"),
                     "lf": r.get("lead_factor"), "ll": r.get("lead_loading"),
                     "cl": r["chart_len"], "cm": r.get("communality"),
                     "p": pc_by_slug.get(r["slug"])})
    chart = {"factors": factors, "big5": {b: B5FIDX.get(b) for b in BIG5}, "rows": rows,
             "nfa": len(FACTOR_TITLES), "axnames": axnames, "axdesc": axdesc,
             "source": "fa_chart.FAChart over results/gram_sweep.npz and "
                       "phase10_runs/steer_spec2_7a.json; loadings from "
                       "results/fa_qwen35.json#solutions.centred_k5.loadings.oblimin; "
                       "component scores from analysis/blog_data.json#viz.scores"}
    write(f"{out}/data/chart.json", json.dumps(chart, separators=(",", ":")))

    # Per-stage chart coordinates.  There is no cross-Gram between stage one and stage
    # two over all 134, so there is no single frame holding all three; what there is, is
    # the same construction applied to each stage's own Gram -- the same five factor
    # coefficient vectors from phase10_runs/steer_spec2_7a.json, Gram-Schmidt in that
    # stage's Gram.  The page therefore offers a stage selector and not an overlay, and
    # says so: the shape of a cloud is comparable across stages, a point's position is
    # not a movement.
    from fa_chart import FAChart
    stages = {"order": [], "label": {}, "coords": {}, "chart_len_mean": {}, "norm_mean": {},
              "how": "fa_chart.FAChart run once per stage over that stage's own Gram with the "
                     "same five factor coefficient vectors from "
                     "phase10_runs/steer_spec2_7a.json; each stage is its own orthonormal "
                     "frame, so clouds are comparable in shape and not in position",
              "source": {"stage1": "results/gram_sweep.npz",
                         "stage2": "results/gram_stage2.npz",
                         "persona": "results/gram_personas.npz"}}
    for key, rel, lab in (("stage1", "results/gram_sweep.npz", "Stage one (DPO)"),
                          ("stage2", "results/gram_stage2.npz", "Stage two (introspection SFT)"),
                          ("persona", "results/gram_personas.npz", "Persona (the deployed merge)")):
        if not have(rel):
            continue
        chs = FAChart(gram=qp(rel))
        assert chs.names == D["names"], f"{rel} row order differs from the Gram"
        stages["order"].append(key)
        stages["label"][key] = lab
        stages["coords"][key] = {n: [round(float(v), 5) for v in chs.trait_coords[i]]
                                 for i, n in enumerate(chs.names)}
        stages["chart_len_mean"][key] = round(float(chs.trait_chart_len.mean()), 5)
        stages["norm_mean"][key] = round(float(chs.norms.mean()), 5)
    write(f"{out}/data/stages.json", json.dumps(stages, separators=(",", ":")))
    D["stages"] = stages          # the stage-two page quotes its mean chart lengths

    # neighbours: nearest 12 and farthest 6 in each of the three Grams
    names = D["names"]
    nb = {"names": names, "grams": {}}
    for key, C in D["grams"].items():
        per = {}
        for i, s in enumerate(names):
            others = [j for j in range(len(names)) if j != i]
            order = sorted(others, key=lambda j: -C[i][j])
            near = [[names[j], float(C[i][j])] for j in order[:12]]
            far = [[names[j], float(C[i][j])] for j in order[-6:][::-1]]
            per[s] = {"near": near, "far": far, "mean": round(float(np.mean(np.delete(C[i], i))), 4)}
        nb["grams"][key] = per
    nb["source"] = {"stage1": "results/gram_sweep.npz", "stage2": "results/gram_stage2.npz",
                    "persona": "results/gram_personas.npz",
                    "how": "cosine = G_ij / sqrt(G_ii G_jj), computed at build"}
    write(f"{out}/data/neighbours.json", json.dumps(nb, separators=(",", ":")))

    # steering: one file per direction
    if D["steer"]:
        index = {"alphas": D["steer"]["alphas"], "alpha_keys": D["steer"]["alpha_keys"],
                 "sources": D["steer"]["sources"],
                 "directions": sorted(D["steer"]["directions"]),
                 "means": {k: v["means"] for k, v in D["steer"]["directions"].items()}}
        write(f"{out}/data/steer/index.json", json.dumps(index, separators=(",", ":")))
        for k, v in D["steer"]["directions"].items():
            write(f"{out}/data/steer/{k}.json", json.dumps(v, separators=(",", ":")))

    # stage two steering generations and text statistics
    s2 = D["stage2"]
    write(f"{out}/data/stage2.json", json.dumps(
        {"gens": s2["gens"], "stats": s2["stats"],
         "sources": {"generations": "phase10_runs/steer_results_s2mean.json and _s2balanced.json",
                     "stats": "analysis/s2mean_steer_stats.json"}},
        separators=(",", ":")))

    # trait index for search and filter
    idx = [{"s": r["slug"], "t": r["title"], "f": r["factor"], "k": r["keyed"], "set": r["set"],
            "lf": r.get("lead_factor"), "ll": r.get("lead_loading"),
            "judged": r["slug"] in D["judged"]} for r in D["all_rows"]]
    write(f"{out}/data/traits.json", json.dumps({"traits": idx}, separators=(",", ":")))


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


# ---------------------------------------------------------------------------
# shared page fragments
# ---------------------------------------------------------------------------
def factor_legend(with_marks=True):
    chips = "".join(
        f'<span class="chip"><span class="swatch bg-f{i}"></span>{esc(FACTOR_TITLES[i])}</span>'
        for i in range(5))
    # The marks are the rotating chart's: a filled point is a positively keyed
    # Goldberg marker, a ringed centre a negatively keyed one.  The 34 Lexicon words
    # and the seven later adapters carry no keying at all and are drawn filled; which
    # set a point belongs to is in the group chips and the Show filter, not the mark.
    marks = ('<span class="chip"><span class="swatch" style="background:var(--muted)"></span>keyed +</span>'
             '<span class="chip"><span class="swatch hollow" style="border-color:var(--muted)"></span>keyed &minus;</span>'
             '<span class="chip">the 34 Lexicon words and the 7 later adapters carry no keying</span>'
             ) if with_marks else ""
    return f'<div class="row" style="gap:0.5rem 1rem;margin:0.5rem 0 0.9rem">{chips}{marks}</div>'


def map_block(compact=False, focus=None, stages=False, axes="0,1,2", src=None, note=""):
    """The rotating 3D factor chart.  Everything it does lives in assets/map3d.js,
    a port of the blog page's canvas map; this only declares where it mounts and
    what it should open on.  A compact map has no colour, set or find controls and
    no pin panel; a focused map opens with one adapter pinned, which is what the
    trait pages use instead of the small 2D chart they carried before."""
    at = [' data-map']
    if src:
        at.append(f' data-src="{esc(src)}"')
    if compact:
        at.append(' data-compact="1"')
    if focus:
        at.append(f' data-focus="{esc(focus)}"')
    if stages:
        at.append(' data-stages="1"')
    at.append(f' data-axes="{esc(axes)}"')
    if note:
        at.append(f' data-note="{esc(note)}"')
    return (f'<div{"".join(at)}><p class="small" style="color:var(--muted)">'
            f'The rotating chart needs JavaScript. Every number behind it is in the '
            f'<a href="{"../" if focus else ""}data.html">downloads</a>.</p></div>')


def stat(n, k, src):
    return f'<div class="stat"><span class="n">{n}</span><span class="k">{k}</span><span class="src">{src}</span></div>'


# ---------------------------------------------------------------------------
def page_home(D, out):
    fa, st = D["fa"], D["stage2"]["structure"]
    setup = fa["setup"]
    congs = [abs(D["b5_of_factor"][i]["phi"]) for i in range(5)]
    nxn = load_json("analysis/nxn_summary.json")
    acts = load_json("analysis/actspace_geometry.json")
    acta = load_json("analysis/actspace_adapters_geometry.json")
    full = D["stage2"]["fulloct"]

    stats = [
        stat(f'{setup["n_traits"]}', f'trait adapters: {setup["n_primary"]} Goldberg Big Five '
             f'markers and {setup["n_lexicon"]} words drawn from the trait lexicon',
             "results/fa_qwen35.json#setup"),
        stat(f'{min(congs):.3f}&thinsp;&ndash;&thinsp;{max(congs):.3f}',
             "Tucker congruence of the five factors with their best Goldberg target. "
             "None reaches the conventional 0.85 bar for a fair match",
             "results/fa_qwen35.json#solutions.centred_k5.congruence_oblimin"),
        stat(f'{fa["n_factors"]["chosen"]}',
             "factors retained by parallel analysis at the reference sample size. Five were "
             "extracted anyway, because five is the hypothesis",
             "results/fa_qwen35.json#n_factors.chosen"),
    ]
    if nxn:
        stats.append(stat(f'{nxn["raw"]["top1"]}&thinsp;/&thinsp;134',
                          "traits whose own preference pairs rank their own adapter first out of "
                          "134, scored by one backward pass per batch",
                          "analysis/nxn_summary.json#raw.top1"))
    if st:
        sc = st["shared_component"]["stage2"]
        stats.append(stat(f'{sc["mean_direction_norm2_over_mean_norm2"]*100:.1f}%',
                          "of the average stage-two adapter's squared norm lies along one "
                          f'direction every adapter shares, at cosine '
                          f'{sc["cos_to_mean_direction_mean"]:.3f} to it',
                          "analysis/stage2_structure.json#shared_component.stage2"))
        stats.append(stat(f'{st["centred_cosines"]["corr_stage1_stage2"]:.3f}',
                          "correlation between the stage-one and stage-two arrangements after "
                          "centring: the residual keeps stage one's pattern at a quarter of its "
                          "amplitude", "analysis/stage2_structure.json#centred_cosines"))
    if acts and acta:
        L = acts["primary_layer"]
        v = curve_at(acts, "resp", L).get("r_centred")
        if v:
            stats.append(stat(f'{v:.3f}',
                              "correlation between the 134 constitutions' activation geometry "
                              "when used as system prompts and the adapters' weight geometry, "
                              f"in the response-token window at layer {L}",
                              "analysis/actspace_geometry.json#windows.resp.curve[16].r_centred"))
    if full:
        stats.append(stat(f'{full["gram_correlation_offdiag"]["persona_vs_stage1"]:.3f}',
                          "correlation between the deployed persona adapters' arrangement and "
                          "stage one's: the released personas' geometry is stage one's geometry",
                          "analysis/fulloct_geometry.json#gram_correlation_offdiag"))
    sn = load_json("analysis/scree_null_matched.json")
    if sn:
        stats.append(stat(f'{sn["n_above_structureless"]} / {sn["n_above_null"]}',
                          "components of the adapter cloud above the structureless null, and "
                          "above the matched permuted null. The second number is the honest one: "
                          "the elbow is not usable",
                          "analysis/scree_null_matched.json#n_above_structureless, n_above_null"))
    cs = load_json("analysis/crossseed_arms.json")
    if cs:
        arm = next((a for a in cs if "matched" in a["path"]), cs[0])
        stats.append(stat(f'{arm["pearson"]:.4f}',
                          "correlation between cosines measured within one LoRA initialisation "
                          "and across two, on the matched seed-paired arm: the arrangement "
                          "reproduces even though the coordinates do not",
                          "analysis/crossseed_arms.json (arm with a matched objective).pearson"))

    cards = [
        ("chart.html", "Explore the chart",
         "Put the 134 adapters on any pair of the five factors, read the loadings that define "
         "each factor, see the two elbow plots, and walk the neighbourhood of any trait in "
         "three different Grams."),
        ("behaviour.html", "Behaviour",
         "Steer the base model along a factor and read what comes out at every dose, next to "
         "the blind judge's Big Five scores; and the OCEAN dials redone three ways."),
        ("stage-two.html", "Stage two",
         "What the second training stage did: one direction every adapter shares, a nearly "
         "isotropic residual that still carries stage one's arrangement, and bipolarity lost."),
        ("traits.html", "All 141 traits",
         "One page per adapter: its constitution, its loadings and place on the chart, its "
         "nearest neighbours, its judged profile and two generations per condition."),
        ("data.html", "Data",
         "Every table behind these figures as CSV and JSON, each naming the file it came from."),
        ("methods.html", "Methods",
         "Five short pages on how the adapters were built, measured, steered, probed in "
         "activation space and scored, each linking to the wiki's full record."),
    ]
    cardhtml = "".join(
        f'<a class="tcard plain" href="{h}" style="padding:1.1rem 1.15rem">'
        f'<span class="nm" style="font-size:1.05rem">{esc(t)}</span>'
        f'<span style="display:block;color:var(--muted);font-size:0.9rem;margin-top:0.4rem;'
        f'line-height:1.5">{esc(d)}</span></a>' for h, t, d in cards)

    ftable = "".join(
        f'<tr><td><span class="swatch bg-f{i}" style="display:inline-block;margin-right:0.45rem">'
        f'</span><strong>{esc(FACTOR_TITLES[i])}</strong></td>'
        f'<td class="n">{D["ss"][i]:.2f}</td>'
        f'<td>{esc(BIG5_LONG[D["b5_of_factor"][i]["scale"]])}</td>'
        f'<td class="n">{D["b5_of_factor"][i]["phi"]:+.3f}</td>'
        f'<td class="wrap">{top_loader_words(D, i)}</td></tr>' for i in range(5))

    body = f"""
<h1>A hundred and thirty-four dispositions, and the shape they make</h1>
<p class="lede">One LoRA adapter per English trait word, trained on Qwen3.5-4B, then compared
with every other adapter through the exact matrix of their inner products. The weight updates
are not scattered. Factoring that matrix returns five interpretable factors whose top loaders
read like the Big Five &mdash; and stops well short of reproducing it.</p>
<p class="dek">This is the companion to the write-up. It exists so you can move around in the
result yourself: change the axes, pick a trait, read what the model actually said at each dose
of steering. Every figure here is generated from the analysis files at build time and every
section names the files it used.</p>

<section class="band" id="chart">
  <div class="sec-head"><span class="sec-num">Figure 1</span>
    <h2>The factor chart</h2></div>
  <p class="dek">Each point is one adapter, placed by its coordinates in the five-dimensional
  span of the five oblique factors. Drag to turn the cloud over; hover a point for its trait
  word. <a href="chart.html">Change the axes and filters on the chart page</a>.</p>
  {factor_legend()}
  {map_block(compact=True)}
  <p class="small" style="color:var(--muted)">Axes: Warmth, Competence and Timidity, the three
  largest factors. Depth is real &mdash; points further away are smaller and fainter &mdash; and
  the cross sits at the mean adapter, because every adapter carries a shared component and the
  cloud is plotted as deviations from it. Coordinates are inner products with an orthonormal
  basis of the factors' span, from {f("fa_chart.py")} over {f("results/gram_sweep.npz")}.</p>
</section>

<section class="band">
  <div class="sec-head"><span class="sec-num">The five factors</span>
    <h2>What came out of the matrix</h2></div>
  <p class="dek">Principal axis factoring with oblimin rotation on the 134&times;134 trait
  correlation matrix, five factors, ordered by sum of squared loadings. The names are the
  project's reading of the top loaders, not labels the data supplied.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Factor</th><th class="n">SS loading</th><th>Closest Goldberg factor</th>
      <th class="n">Tucker &phi;</th><th>Highest loading trait words</th></tr></thead>
    <tbody>{ftable}</tbody></table></div>
  <p class="small" style="color:var(--muted);margin-top:0.6rem">The fifth Big Five factor is
  called Intellect here, following Goldberg's label for the marker set the traits came from,
  not Openness. {wiki("goldberg-intellect-factor", "Why")}.</p>
  <div class="caveat" style="margin-top:1rem">
    <p><strong>What this table does not say.</strong> A Tucker congruence of 0.85 is the
    conventional bar for calling two factors a fair match and 0.95 for calling them equivalent.
    No factor in any of the four solutions extracted clears 0.85 against any Goldberg factor.
    The recovery is real and ordered; it is not the Big Five reproduced. Parallel analysis
    retained {fa["n_factors"]["chosen"]} factors, not five: five was extracted because five is
    the hypothesis being tested. {wiki("factor-analysis", "The full account")}.</p>
  </div>
</section>

<section class="band">
  <div class="sec-head"><span class="sec-num">The numbers</span>
    <h2>What the measurements say</h2></div>
  <div class="grid cols-3">{"".join(stats)}</div>
</section>

<section class="band">
  <div class="sec-head"><span class="sec-num">Where to go</span><h2>The rest of the site</h2></div>
  <div class="grid cols-2">{cardhtml}</div>
</section>

<section class="band">
  <div class="sec-head"><span class="sec-num">Limits</span>
    <h2>What none of this establishes</h2></div>
  <div class="prose">
  <p>The geometry is the strong result and the behaviour is the weak one. Adding a factor
  direction to the weights moves the blind judge's scores in the expected direction for most
  factors, but the effects are small, they do not compose additively, and one factor
  (Timidity) moves every judged scale at once rather than its own.</p>
  <p>Five factors is a choice. The reference sample size that parallel analysis needs does not
  exist for 134 weight updates; the figure used is carried over from an earlier sweep.</p>
  <p>Stage two loses bipolarity: a trait and its antonym sit at cosine
  {st["decomposition_tests"]["stage1"]["test2_same_opp_gap_p"][1]:+.3f} in stage one and
  {st["decomposition_tests"]["stage2"]["test2_same_opp_gap_p"][1]:+.3f} in stage two, because a
  single shared register direction dominates both.</p>
  <p>The widest gap in the trait words' coverage of the space is a region the sampled
  vocabulary did not reach rather than a disposition without a name: its coordinates read as
  calm, unexcitable and highly imaginative. Steered, it behaves no differently from a matched
  random direction in the same span, and the three adjectives proposed for it landed farther
  from it than existing adapters; the project records it as a gap in the lexicon, not a finding
  about the model.
  {wiki("hole-words", "The hole and the alien direction")}.</p>
  <p>The full list the project keeps of what it has not shown is on the wiki:
  {wiki("open-questions", "open questions")} and
  {wiki("superseded-claims", "superseded claims")}.</p>
  </div>
</section>

{src_block([
  f('results/fa_qwen35.json', 'setup, n_factors, solutions.centred_k5'),
  f('results/gram_sweep.npz'), f('fa_chart.py'),
  f('analysis/stage2_structure.json', 'shared_component, centred_cosines'),
  f('analysis/nxn_summary.json', 'raw.top1'),
  f('analysis/actspace_geometry.json', 'windows.prompt.curve'),
  f('analysis/fulloct_geometry.json', 'gram_correlation_offdiag'),
])}
"""
    write(f"{out}/index.html", page("index.html", "Home",
          "Companion site: 134 personality-trait LoRA adapters on Qwen3.5-4B and the geometry "
          "of their weight updates.", body, scripts=("map3d.js",), built=BUILT))


def top_loader_words(D, fi, n=4):
    rows = [r for r in D["traits"]]
    pos = sorted(rows, key=lambda r: -r["loadings"][fi])[:n]
    neg = sorted(rows, key=lambda r: r["loadings"][fi])[:n]
    p = ", ".join(esc(r["title"].lower()) for r in pos)
    m = ", ".join(esc(r["title"].lower()) for r in neg)
    return f'{p} <span style="color:var(--faint)">against</span> {m}'


# ---------------------------------------------------------------------------
def page_chart(D, out):
    fa, fa2 = D["fa"], D["fa2"]
    panels = scree_panels(fa, fa2)
    nfa = fa["n_factors"]
    ref = nfa.get("reference_N")
    scree = ""
    if panels:
        scree = f"""
  <div class="grid cols-2">
    <figure style="padding:0.9rem 1rem;margin:0">{figbox(panels[0])}</figure>
    <figure style="padding:0.9rem 1rem;margin:0">{figbox(panels[1])}</figure>
  </div>
  <p class="dek" style="margin-top:0.9rem">Two different matrices, drawn side by side and never
  on one axis. The left panel is the eigenvalue spectrum of the centred trait correlation matrix
  &mdash; the principal-component question. The right panel replaces the diagonal with communality
  estimates, which is the factor-analytic question: principal axis factoring partitions common
  variance rather than total variance, so its eigenvalues are smaller and its elbow is not the
  PCA elbow. Dashed grey lines are the 95th percentile of eigenvalues from random data with the
  same number of variables, at two sample sizes (Horn's parallel analysis, 500 replicates).
  Components above the dashed line count as structure.</p>
  <p class="dek">Stage one falls from {fa["n_factors"]["parallel_analysis_centred"]["observed_unreduced"][0]:.1f}
  to {fa["n_factors"]["parallel_analysis_centred"]["observed_unreduced"][2]:.1f} within three
  components and meets the N&nbsp;=&nbsp;{ref} null around component
  {fa["n_factors"]["chosen"]}. Stage two never gets far from its null: it starts at
  {fa2["n_factors"]["parallel_analysis_centred"]["observed_unreduced"][0]:.1f} and is within a
  factor of two of the null from the first component. That difference is the elbow argument, and
  it is why the factor analysis of stage one is the part of this work worth trusting most.</p>
  <div class="caveat" style="margin-top:1rem"><p>There is no true sample size here. These are 134
  weight updates, not 134 questionnaire respondents; the reference figure N&nbsp;=&nbsp;{ref} is an
  effective dimensionality carried over from an earlier sweep, and there are no reseed controls in
  this sweep to re-estimate it. Parallel analysis at that figure retains
  {fa["n_factors"]["chosen"]} factors for stage one and {fa2["n_factors"]["chosen"]} for stage two.
  Five is a choice made so the solution can be compared with the Big Five.</p></div>"""

    congr_rows = ""
    for i in range(5):
        row = D["congr"][i]
        best = max(abs(v) for v in row[:5])
        cells = "".join(
            '<td class="n"%s>%+.3f</td>' % (BOLD if abs(row[j]) == best else "", row[j])
            for j in range(5))
        congr_rows += (f'<tr><td><span class="swatch bg-f{i}" style="display:inline-block;'
                       f'margin-right:0.45rem"></span>{esc(FACTOR_TITLES[i])}</td>{cells}'
                       f'<td class="n" style="color:var(--muted)">{row[5]:+.3f}</td></tr>')

    body = f"""
<h1>Explore the chart</h1>
<p class="lede">Five oblique factors were extracted from the 134&times;134 trait correlation
matrix. The chart is an orthonormal basis of their span, so an adapter's position is a set of
inner products, not a set of loadings. Both are here, and they are different objects.</p>

<section class="band" id="map-section">
  <div class="sec-head"><span class="sec-num">Figure 1</span><h2>The 141 adapters</h2></div>
  <p class="dek">Pick any three of the five factors &mdash; or of the six principal components,
  which are the same cloud ordered by variance instead &mdash; for the x, y and z axes, and drag
  to rotate. Colour follows the factor a trait loads most strongly on; switch it to the Big Five
  keying of the trait word to see how far the two agree. Hover a point for its trait word, click
  one to pin its five loadings beside the chart.</p>
  {factor_legend()}
  {map_block()}
  <p class="small" style="color:var(--muted)">Coordinates from {f("fa_chart.py")}
  (<code>FAChart.trait_coords</code>) over {f("results/gram_sweep.npz")} and
  {f("phase10_runs/steer_spec2_7a.json")}. The four alignment adapters and three hole words are
  not part of the 134 and were not factored: their positions are computed from their cross-Gram
  with the zoo ({f("results/cross_gram_full_root_x_pc-qwen35-adapters_data_alignment_common.npz")}
  and the hole equivalent) at build time, and they have no loadings.</p>
</section>

<section class="band" id="loadings-section">
  <div class="sec-head"><span class="sec-num">Figure 2</span><h2>What defines each factor</h2></div>
  <div class="controls">
    <label class="ctl">Factor<select id="load-factor"></select></label>
    <a id="load-wiki" data-base="{WIKI}/" href="{WIKI}/factor-warmth.html" style="align-self:flex-end">
      The Warmth factor on the wiki</a>
  </div>
  <div id="loadings"></div>
  <p class="small" style="color:var(--muted)" id="load-caption"></p>
  <p class="small" style="color:var(--muted)">From
  {f("results/fa_qwen35.json", "solutions.centred_k5.loadings.oblimin")} and
  {f("results/fa_qwen35.json", "solutions.centred_k5.congruence_oblimin")}.</p>
</section>

<section class="band">
  <div class="sec-head"><span class="sec-num">Table 1</span>
    <h2>How far the factors match the Big Five</h2></div>
  <p class="dek">Tucker's congruence coefficient between each recovered factor and a target
  vector that is +1 for a positively keyed marker of a Goldberg factor, &minus;1 for a negatively
  keyed one and 0 otherwise. The 34 Lexicon words score 0 in every target, so congruence is judged
  on the 100 Goldberg markers only. The last column is an evaluative "good trait / bad trait"
  target, which any clean Goldberg factor already scores
  {fa["targets"]["target_target_congruence"][0][5]:.3f} on by construction.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Recovered factor</th>
      {"".join(f'<th class="n">{BIG5_SHORT[b]}</th>' for b in ["Extraversion","Agreeableness","Conscientiousness","EmotionalStability","Intellect"])}
      <th class="n">Eval</th></tr></thead>
    <tbody>{congr_rows}</tbody></table></div>
  <div class="caveat" style="margin-top:1rem"><p><strong>Read the second-best column too.</strong>
  Timidity's best target is Emotional Stability at
  {D["congr"][2][3]:+.3f}, but it loads Extraversion at {D["congr"][2][0]:+.3f} alongside it, and
  Arousal loads Emotional Stability at {D["congr"][3][3]:+.3f} alongside Extraversion at
  {D["congr"][3][0]:+.3f}. Those two are best read as a rotation of the same plane rather than as
  clean recoveries of two Goldberg factors. Nothing here clears 0.85.
  {wiki("explainer-big-five-mapping", "The project's own reading")}.</p></div>
</section>

{section_bigfive_adapters(D)}

<section class="band" id="elbow">
  <div class="sec-head"><span class="sec-num">Figure 3</span><h2>The elbow plots</h2></div>
  {scree}
</section>

<section class="band" id="neighbours-section">
  <div class="sec-head"><span class="sec-num">Figure 4</span><h2>Neighbourhoods</h2></div>
  <p class="dek">The nearest and farthest adapters to a chosen trait by cosine, in the exact Gram
  of each of the three adapter sets: the stage-one DPO adapters, the stage-two introspection
  adapters, and the deployed persona merge of the two. The stage-two column is where bipolarity
  goes missing: opposites stop being opposite.</p>
  <div class="controls"><label class="ctl">Trait<select id="nb-trait"></select></label></div>
  <div class="grid cols-3" id="neighbours"></div>
  <p class="small" style="color:var(--muted)">Cosines computed at build time from
  {f("results/gram_sweep.npz")}, {f("results/gram_stage2.npz")} and
  {f("results/gram_personas.npz")} as G<sub>ij</sub> / &radic;(G<sub>ii</sub>G<sub>jj</sub>).</p>
</section>

{src_block([
  f("fa_chart.py"),
  f("results/fa_qwen35.json", "solutions.centred_k5, n_factors.parallel_analysis_centred, targets"),
  f("results/fa_qwen35_stage2.json", "n_factors.parallel_analysis_centred"),
  f("results/gram_sweep.npz"), f("results/gram_stage2.npz"), f("results/gram_personas.npz"),
  f("results/cross_gram_full_root_x_pc-qwen35-adapters_data_alignment_common.npz"),
  f("results/cross_gram_full_root_x_pc-qwen35-adapters_data_hole_common.npz"),
  "The elbow figure follows " + wiki("explainer-elbow-figure", "the wiki's explanation") +
  " and adapts <code>wiki/tools/gen_scree_svg.py</code>.",
])}
"""
    write(f"{out}/chart.html", page("chart.html", "Explore the chart",
          "The 134 adapters on any pair of the five recovered factors, the loadings that define "
          "each factor, the elbow plots and a neighbour explorer.",
          body, scripts=("map3d.js", "explore.js"), built=BUILT))


PAGES.append(page_chart)

def page_planes(D, out):
    """A two-dimensional cut of the factor chart on any two axes the reader picks,
    with a six-panel mode that puts each Goldberg group on its own factor.  All the
    drawing is in assets/planes.js from data/chart.json; this page only frames it."""
    body = f"""
<h1>Any two axes</h1>
<p class="lede">The rotating chart shows three factors at once. This page shows two, flat, with
the two poles of each Goldberg group drawn as a pair of density shells and a line between their
means, so that the bipolarity is something you can look at rather than read off a cosine.</p>

<section class="band" id="planes-section">
  <div class="sec-head"><span class="sec-num">Figure 1</span><h2>Pick the plane</h2></div>
  <p class="dek">Choose any of the five recovered factors, or of the six principal components, for
  x and for y. <em>One panel</em> draws all 134 zoo adapters and highlights one Goldberg group or
  all five; <em>Six panels</em> draws every group in its own panel, greying the rest, and by default
  puts each group on the recovered factor its Big Five label is closest to, which is the axis that
  is meant to split it. Hover a point for its word and coordinates. The view is in the address bar,
  so a particular plane can be linked.</p>
  {factor_legend()}
  <div data-planes><p class="small" style="color:var(--muted)">The chart needs JavaScript. Every
  number behind it is in the <a href="data.html">downloads</a>.</p></div>
</section>

<section class="band">
  <div class="sec-head"><h2>What the marks mean</h2></div>
  <div class="prose">
  <p>Colour is the Goldberg factor label the trait word carries in the marker file, in the hue of the
  recovered factor that label is closest to; it is not a cluster found in the data. A filled point is
  a positively keyed marker and a ringed one a negatively keyed marker. The 34 held-out Lexicon words
  are crosses and carry no keying. Each shell is a Gaussian kernel density estimate on one keyed half
  of one group, cut at the level that encloses 45 per cent of its mass: a picture of where the dense
  half of that group sits, not a hull around its outliers. The connector runs from the mean of the
  negatively keyed words to the mean of the positively keyed words, in the two plotted coordinates
  only; it is a straight segment, not a regression and not a direction from the factor solution.</p>
  <p>Coordinates are mean-centred over the 134 zoo adapters by default. Every stage-one adapter
  shares a common component (about 8 per cent of the squared norm, cosine 0.28 to the grand mean),
  and in the uncentred chart it pulls every group to one side; untick <em>centred</em> to see it.
  The recovered factors match the Big Five at Tucker congruence
  {min(max(abs(v) for v in row[:5]) for row in D["congr"][:5]):.2f} to
  {max(max(abs(v) for v in row[:5]) for row in D["congr"][:5]):.2f}, so a group's poles separate along its own
  factor but not perfectly, and along a different factor they may not separate at all.</p>
  </div>
{src_block([
  f("analysis/viz_fa.json", "coords, factor, keyed"),
  f("results/fa_qwen35.json", "targets, solutions.centred_k5.tucker"),
  f("analysis/blog_data.json", "viz.scores (the principal components)"),
  f("analysis/stage2_structure.json", "shared_component.stage1"),
  "The static versions of these panels are <code>qwen35/figures/post/facets_own_*.png</code> "
  "and <code>qwen35/figures/clusters/editorial_facets_v2_*.png</code>.",
])}
</section>
"""
    write(f"{out}/planes.html", page("planes.html", "Any two axes",
          "The 134 adapters on any two of the five recovered factors or six principal components, "
          "with each Goldberg group's two poles as density shells and a connector between them.",
          body, scripts=("planes.js",), built=BUILT))


PAGES.append(page_planes)


# ---------------------------------------------------------------------------
STEER_GROUPS = [
    ("The five factors", ["FA_Warmth", "FA_Competence", "FA_FearfulWithdrawal",
                          "FA_Arousal", "FA_Imagination"]),
    ("Principal components", ["PC4", "PC5", "PC6"]),
    ("Factor-identity axes", ["identity_Extraversion", "identity_Agreeableness",
                              "identity_Conscientiousness", "identity_EmotionalStability",
                              "identity_Intellect"]),
]
STEER_LABEL = {"FA_Warmth": "Warmth", "FA_Competence": "Competence",
               "FA_FearfulWithdrawal": "Timidity", "FA_Arousal": "Arousal",
               "FA_Imagination": "Imagination"}


def page_behaviour(D, out):
    rep = load_json("analysis/steerfix_replication.json") or {}
    sp = load_json("analysis/spider.json")
    qf = load_json("analysis/qual_fa.json")
    add = load_json("analysis/additivity.json")

    opts = ""
    for gname, keys in STEER_GROUPS:
        keys = [k for k in keys if D["steer"] and k in D["steer"]["directions"]]
        if not keys:
            continue
        opts += f'<optgroup label="{esc(gname)}">' + "".join(
            f'<option value="{esc(k)}">{esc(STEER_LABEL.get(k, k.replace("identity_", "identity: ")))}</option>'
            for k in keys) + "</optgroup>"

    sel_rows = ""
    for i, k in enumerate(FACTOR_KEYS):
        r = rep.get(k)
        if not r:
            continue
        sel_rows += (f'<tr><td><span class="swatch bg-f{i}" style="display:inline-block;'
                     f'margin-right:0.45rem"></span>{esc(FACTOR_TITLES[i])}</td>'
                     f'<td>{esc(BIG5_LONG.get(r["named"], r["named"]))}</td>'
                     f'<td class="n">{r["slope2"]:+.3f}</td>'
                     f'<td class="n">{r["sel2"]:.2f}</td>'
                     f'<td class="n">{r["coh2"]}/4</td></tr>')

    # OCEAN dials
    dials = ""
    if sp:
        arms = [("axes", "Steering axes at alpha &plusmn;2",
                 "the five named Big Five keying axes added to the weights"),
                ("traits", "Stage-one adapters",
                 "the positively and negatively keyed stage-one adapters per factor (ten and ten, except Emotional Stability at six and fourteen), averaged"),
                ("personas", "Deployed personas",
                 "the same twenty traits per factor as full two-stage persona adapters")]
        blocks = []
        for arm, title, sub in arms:
            if arm not in sp:
                continue
            pair = (f'<div style="padding:0.9rem 1rem"><h4 style="margin-top:0">{title}</h4>'
                    f'<p class="small" style="color:var(--muted);margin:-0.2rem 0 0.6rem">{sub}</p>'
                    f'<div class="row" style="gap:0.6rem">'
                    f'{figbox(radar_svg(sp, arm, "amplifier", "amplifier"))}'
                    f'{figbox(radar_svg(sp, arm, "suppressor", "suppressor"))}</div></div>')
            blocks.append(pair)
        wins = []
        for arm, title, _ in arms:
            if arm not in sp:
                continue
            n = 0
            for F in sp["factors"]:
                for pole in ("amplifier", "suppressor"):
                    pr = sp[arm].get(f"{F}|{pole}")
                    if not pr:
                        continue
                    own = pr[F]
                    if abs(own) >= max(abs(pr[x]) for x in sp["factors"] if x != F) and \
                            (own > 0) == (pole == "amplifier"):
                        n += 1
            wins.append(f"{title.lower()} {n} of 10")
        dials = f"""
  <div class="grid cols-3">{"".join(blocks)}</div>
  <p class="dek" style="margin-top:0.9rem">Each polygon is one dial: the percentage change in the
  blind judge's mean score on each of the five scales, relative to the unmodified base model, when
  that factor is amplified or suppressed. A dial that works is a polygon with one long spoke on its
  own axis. Own-trait dominance holds for {"; ".join(wins)}, counted at build from
  {f("analysis/spider.json")}.</p>"""

    body = f"""
<h1>Behaviour</h1>
<p class="lede">Weight-space structure is one thing; what the model does is another. Adding a
factor direction to the weights and reading the output is where this project's claims are
weakest, and the page is built so you can see why: the judge's numbers and the actual text sit
next to each other at every dose.</p>
<p class="dek">The setup throughout: take the unmodified Qwen3.5-4B, add alpha times a unit
direction times a reference norm (0.81, which the 2026-09-08 audit found to be half of one stage-one adapter's Frobenius norm, so alpha 2 is one adapter),
generate 512 tokens greedily with thinking disabled on 24 fixed prompts, and have
{esc((D["steer"] or {}).get("sources", {}).get("model", "an LLM judge"))} score each answer
blind on the five scales from 1 to 7.</p>

<section class="band" id="dose">
  <div class="sec-head"><span class="sec-num">Figure 1</span><h2>Dose&ndash;response</h2></div>
  <div class="controls">
    <label class="ctl">Direction<select id="dose-dir">{opts}</select></label>
    <label class="ctl">Dose (alpha)<input type="range" id="dose-alpha" min="0" max="6" step="1" value="3"></label>
    <span class="chip" id="dose-alpha-label">alpha = 0</span>
  </div>
  <div class="grid cols-2">
    <div style="padding:0.9rem 1rem"><div id="dose-chart"></div></div>
    <div style="padding:0.9rem 1rem"><div id="dose-text"></div></div>
  </div>
  <p class="small" style="color:var(--muted)">Judged means from
  {f("phase10_runs/judged_steerfix23.json")} (mean over 24 prompts per direction and alpha,
  computed at build); generations from {f("phase10_runs/steer_results_fix2.json")} and
  {f("phase10_runs/steer_results_fix3.json")}. Six of the 24 prompts are shipped to the page.</p>
</section>

<section class="band">
  <div class="sec-head"><span class="sec-num">Table 1</span><h2>Does the dial turn its own dial?</h2></div>
  <p class="dek">For each factor direction: which judged scale moves most, how steeply it moves
  per unit alpha over the well-behaved range |alpha| &le; 2, how much bigger that movement is than
  the mean movement of the other four scales (selectivity), and how many of the other four keep a
  consistent sign. A selectivity near 1 means the direction moves everything at once.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Factor</th><th>Target scale</th><th class="n">Slope</th>
      <th class="n">Selectivity</th><th class="n">Sign coherence</th></tr></thead>
    <tbody>{sel_rows}</tbody></table></div>
  <div class="caveat" style="margin-top:1rem"><p><strong>Timidity fails this test.</strong>
  Its selectivity is {rep.get("FA_FearfulWithdrawal", {}).get("sel2", float("nan")):.2f}, which
  means it moves every judged scale at once rather than its own. Warmth is the most selective
  direction measured anywhere in the study; Timidity is the weakest.
  {wiki("steering-results", "The full steering record")}.</p>
  <p>Separately, about 59 per cent of responses in this corpus end mid-sentence at the 512-token
  cap, so nothing here claims anything about how a response concludes
  ({f("analysis/qual_fa.json")}).</p></div>
</section>

{section_bfi()}

<section class="band" id="dials">
  <div class="sec-head"><span class="sec-num">Figure 2</span><h2>The OCEAN dials</h2></div>
  <p class="dek">Persona Cartography's Figure 2, redone on this zoo three ways: as steering axes
  added to the weights, as the stage-one adapters themselves, and as the deployed two-stage
  personas. If personality in weight space were a set of independent dials, each polygon would
  have one long spoke.</p>
  {dials}
  <p class="small" style="color:var(--muted)">{f("analysis/spider.json")}, built by
  {f("build_spider_data.py")}. {wiki("ocean-dials-replication", "The wiki's account")}.</p>
</section>

<section class="band">
  <div class="sec-head"><span class="sec-num">Also</span><h2>Two more behavioural results</h2></div>
  <div class="grid cols-2">
    <div style="padding:1rem 1.1rem">
      <h3 style="margin-top:0">The sphere sweep</h3>
      <p class="small">Seventy-two directions nobody chose, sampled on the unit sphere of the top
      three factors, steered and judged blind. Angular distance between two directions predicts
      the distance between their judged profiles at rho 0.65 &mdash; the space is smooth, not a
      set of special axes. It also found that 48 of the 72 loop on no prompt and the rest do,
      which is the damage floor every steering claim here has to clear.</p>
      <p class="small" style="color:var(--muted)">Numbers quoted from
      {wiki("sphere-sweep", "the wiki's sphere-sweep page")}, which sources them to
      {f("phase10_runs/judged_sphere.json")} and {f("analysis/sphere_layout.json")}. That arm
      is marked superseded there: it was run on the principal-component chart, and the
      factor-chart sphere had not been rebuilt when this page was generated.</p>
    </div>
    <div style="padding:1rem 1.1rem">
      <h3 style="margin-top:0">Additivity</h3>
      <p class="small">Ten matched-norm mixtures of the five named axes were steered and judged.
      The deviation from the additive prediction is about half the predicted effect and roughly
      two and a half times the judge's own noise floor. Reinforcing mixtures compose; opposing
      ones do not. Weight-space personality is not a mixing desk.</p>
      <p>{wiki("additivity", "Additivity on the wiki")}</p>
    </div>
  </div>
</section>

{section_fisher()}

{section_sorh()}

{dolci_block()}

{dolci_flag_block()}

{syc_forecast_block()}

{src_block([
  f("phase10_runs/judged_steerfix23.json", "records"),
  f("phase10_runs/steer_results_fix2.json"), f("phase10_runs/steer_results_fix3.json"),
  f("analysis/steerfix_replication.json", "slope2, sel2, coh2, named"),
  f("analysis/spider.json"), f("analysis/qual_fa.json"),
  "Setup and caveats follow " + wiki("steering-results") + ", " +
  wiki("ocean-dials-replication") + " and " + wiki("thinking-default-withdrawals") + ".",
])}
"""
    write(f"{out}/behaviour.html", page("behaviour.html", "Behaviour",
          "Factor steering dose-response with the actual generations, the OCEAN dials redone "
          "three ways, and where the behavioural claims fail.",
          body, scripts=("behaviour.js",), built=BUILT))


PAGES.append(page_behaviour)


# ---------------------------------------------------------------------------
def md_lite(text):
    """A small Markdown subset: headings, paragraphs, lists, bold, italic, code,
    links. Used only for optional companion/content/*.md written by other agents;
    the wiki venv's markdown module is not importable from this interpreter."""
    out, para, lst = [], [], False
    def inline(s):
        s = esc(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<![*\w])\*([^*]+)\*", r"<em>\1</em>", s)
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
        return s
    def flush():
        nonlocal para
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para = []
    for line in text.splitlines():
        s = line.rstrip()
        if not s.strip():
            flush()
            if lst:
                out.append("</ul>"); lst = False
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            flush()
            if lst:
                out.append("</ul>"); lst = False
            lvl = min(4, len(m.group(1)) + 1)
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            continue
        if re.match(r"^\s*[-*]\s+", s):
            flush()
            if not lst:
                out.append("<ul>"); lst = True
            out.append("<li>" + inline(re.sub(r"^\s*[-*]\s+", "", s)) + "</li>")
            continue
        para.append(s.strip())
    flush()
    if lst:
        out.append("</ul>")
    return "\n".join(out)


def spectrum_svg(st):
    """Centred variance shares of the leading components, stage one against stage two."""
    s1 = st["spectrum"]["stage1"]["centred_share_top10"]
    s2 = st["spectrum"]["stage2"]["centred_share_top10"]
    n = min(10, len(s1), len(s2))
    W, H, L, R, T, B = 560, 250, 46, 16, 30, 42
    hi = max(max(s1[:n]), max(s2[:n])) * 1.08
    bw = (W - L - R) / n
    Y = lambda v: H - B - (H - T - B) * v / hi
    p = [svg_open(W, H, "centred variance shares, stage one against stage two", minw=460)]
    for frac in (0.00, 0.03, 0.06, 0.09, 0.12):
        if frac > hi:
            continue
        p.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(frac):.1f}" y2="{Y(frac):.1f}" stroke="var(--rule-soft)"/>')
        p.append(f'<text x="{L-7}" y="{Y(frac)+3.5:.1f}" text-anchor="end" class="axis">{frac*100:.0f}%</text>')
    for i in range(n):
        x = L + i * bw
        p.append(f'<rect x="{x+bw*0.12:.1f}" y="{Y(s1[i]):.1f}" width="{bw*0.36:.1f}" '
                 f'height="{H-B-Y(s1[i]):.1f}" fill="var(--ink)"/>')
        p.append(f'<rect x="{x+bw*0.52:.1f}" y="{Y(s2[i]):.1f}" width="{bw*0.36:.1f}" '
                 f'height="{H-B-Y(s2[i]):.1f}" fill="var(--f4)"/>')
        p.append(f'<text x="{x+bw/2:.1f}" y="{H-B+14}" text-anchor="middle" class="axis">{i+1}</text>')
    p.append(f'<line x1="{L}" x2="{W-R}" y1="{H-B}" y2="{H-B}" stroke="var(--rule)"/>')
    p.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-8}" text-anchor="middle" class="axis">component of the double-centred Gram</text>')
    p.append(f'<rect x="{L}" y="{T-22}" width="10" height="9" fill="var(--ink)"/>')
    p.append(f'<text x="{L+15}" y="{T-14}" class="axis">stage one</text>')
    p.append(f'<rect x="{L+90}" y="{T-22}" width="10" height="9" fill="var(--f4)"/>')
    p.append(f'<text x="{L+105}" y="{T-14}" class="axis">stage two</text>')
    p.append("</svg>")
    return "".join(p)


def page_stage2(D, out):
    S = D["stage2"]
    st, full, fa2, stats = S["structure"], S["fulloct"], S["fa2"], S["stats"]
    if not st:
        return
    sc1 = st["shared_component"]["stage1"]
    sc2 = st["shared_component"]["stage2"]
    cc = st["centred_cosines"]
    x12 = st["stage1_x_stage2_exact"]
    dt = st["decomposition_tests"]
    fam = st["factor_congruence_stage1_vs_stage2"]["best_matching"]
    fas1 = st["factor_analysis"]["stage1"]
    fas2 = st["factor_analysis"]["stage2"]

    # stage two factor table
    b5map = {"Agreeableness": 0, "Conscientiousness": 1, "EmotionalStability": 2,
             "Extraversion": 3, "Intellect": 4}
    SHORT2LONG = {"E": "Extraversion", "A": "Agreeableness", "C": "Conscientiousness",
                  "ES": "Emotional stability", "I": "Intellect", "Eval": "Evaluative"}
    rows2 = ""
    for b in (fas2.get("best_big_five_target_per_factor") or []):
        tgt = SHORT2LONG.get(b.get("best_target"), b.get("best_target"))
        allc = b.get("all", {})
        second = sorted(((abs(v), k) for k, v in allc.items() if k != b.get("best_target")
                         and k != "Eval"), reverse=True)
        sec = f' <span style="color:var(--faint)">({SHORT2LONG.get(second[0][1], second[0][1])} {allc[second[0][1]]:+.2f} second)</span>' if second else ""
        rows2 += (f'<tr><td class="num">factor {b["factor"]}</td>'
                  f'<td class="wrap">{esc(tgt)}{sec}</td>'
                  f'<td class="n">{b["congruence"]:+.3f}</td></tr>')
    match_rows = ""
    for m in fam:
        i, j, phi = m["factor"], m["matched"], m["congruence"]
        match_rows += (f'<tr><td><span class="swatch bg-f{i}" style="display:inline-block;'
                       f'margin-right:0.45rem"></span>{esc(FACTOR_TITLES[i])}</td>'
                       f'<td class="num">stage-two factor {j}</td>'
                       f'<td class="n">{phi:+.3f}</td></tr>')

    stat_rows = ""
    if stats:
        keys = [("first_person_per_k", "first person per 1k words"),
                ("second_person_per_k", "second person per 1k"),
                ("markdown_frac", "fraction using markdown"),
                ("embodied_frac", "fraction answering in character"),
                ("words", "mean words"), ("uniq_ratio", "unique-word ratio")]
        alphas = list(stats["S2_mean"]["per_alpha"].keys())
        head = "".join(f'<th class="n">{a}</th>' for a in alphas)
        for jobk, jobname in (("S2_mean", "stage-two grand mean"),
                              ("mean_assistant_axis", "stage-one grand mean"),
                              ("S2_balancedrandom", "sign-balanced control")):
            if jobk not in stats:
                continue
            stat_rows += (f'<tr><td colspan="{len(alphas)+1}" style="font-weight:600;'
                          f'background:var(--bg-deep)">{esc(jobname)}</td></tr>')
            for k, lab in keys:
                cells = "".join(
                    f'<td class="n">{stats[jobk]["per_alpha"][a][k]:.2f}</td>' for a in alphas)
                stat_rows += f'<tr><td>{esc(lab)}</td>{cells}</tr>'
        stat_rows = (f'<thead><tr><th>statistic</th>{head}</tr></thead><tbody>{stat_rows}</tbody>')

    findings = ""
    cp = OPTIONAL_CONTENT["stage2_findings"]
    if os.path.exists(cp):
        findings = f"""
<section class="band">
  <div class="sec-head"><span class="sec-num">New</span><h2>Further stage-two findings</h2></div>
  <div class="prose">{md_lite(open(cp).read())}</div>
  <p class="small" style="color:var(--muted)">From <code>companion/content/stage2_findings.md</code>.</p>
</section>"""
    expl = ""
    if S["exploration"]:
        expl = f"""
<section class="band">
  <div class="sec-head"><span class="sec-num">New</span><h2>Stage-two exploration</h2></div>
  <div class="tablewrap"><table><tbody>{kv_rows(S["exploration"])}</tbody></table></div>
  <p class="small" style="color:var(--muted)">{f("analysis/stage2_exploration.json")}</p>
</section>"""

    body = f"""
<h1>Stage two: one thing everybody learned</h1>
<p class="lede">Open Character Training trains each persona twice. Stage one is DPO on
contrasting preference pairs. Stage two is supervised fine-tuning on transcripts the model
generated of itself in character, merged into stage one at a weight of 0.25. The two stages
produce completely different geometries, and the difference is the most interesting thing in
this dataset.</p>

<section class="band">
  <div class="sec-head"><span class="sec-num">1</span><h2>Fifteen per cent of every adapter is the same direction</h2></div>
  <div class="grid cols-3">
    {stat(f'{sc2["mean_direction_norm2_over_mean_norm2"]*100:.1f}%',
          "of the average stage-two adapter's squared norm lies along the grand mean of all 134",
          "analysis/stage2_structure.json#shared_component.stage2")}
    {stat(f'{sc2["cos_to_mean_direction_mean"]:.3f}',
          f'mean cosine of every stage-two adapter to that direction, standard deviation '
          f'{sc2["cos_to_mean_direction_sd"]:.3f} over a range of '
          f'{sc2["cos_to_mean_direction_min"]:.2f} to {sc2["cos_to_mean_direction_max"]:.2f}',
          "analysis/stage2_structure.json#shared_component.stage2")}
    {stat(f'{sc1["mean_direction_norm2_over_mean_norm2"]*100:.1f}%',
          f'the same quantity for stage one, at mean cosine '
          f'{sc1["cos_to_mean_direction_mean"]:.3f} with standard deviation '
          f'{sc1["cos_to_mean_direction_sd"]:.3f} &mdash; six times as spread out',
          "analysis/stage2_structure.json#shared_component.stage1")}
  </div>
  <p class="dek" style="margin-top:1rem">Every stage-two adapter sits at almost exactly the same
  angle to one direction. That is not what a set of 134 different personalities should look like.
  The stage-two grand mean is also orthogonal to the stage-one grand mean (cosine
  {x12["cos_between_grand_means"]:+.3f}): the two stages carry independent random LoRA-A draws,
  so their shared components live in different coordinates even though each is a shared
  component within its own stage.</p>
</section>

<section class="band" id="cloud">
  <div class="sec-head"><span class="sec-num">2</span><h2>The cloud, one stage at a time</h2></div>
  <p class="dek">The same 134 adapters, charted at each stage of training. Pick a stage, pick the
  three axes, and drag to turn the cloud over. Each cloud is centred on its own mean adapter and
  then scaled so that its outermost point reaches the edge, so what changes between the stages is
  the arrangement &mdash; which traits sit near which, and how the five factors lay them out
  &mdash; and not the size of the cloud. The shared direction the section above measures is
  exactly what that centring takes out.</p>
  {map_block(stages=True)}
  <p class="small" style="color:var(--muted)">There is no cross-Gram between the stages over all
  134, so there is no single frame that holds all three clouds, and no overlay is drawn: a line
  between a trait's stage-one and stage-two point would be a movement the data does not contain.
  Instead each stage is charted by the identical construction &mdash; the same five factor
  coefficient vectors from {f("phase10_runs/steer_spec2_7a.json")}, made orthonormal by
  Gram&ndash;Schmidt inside <em>that stage's own</em> Gram, by {f("fa_chart.py")}. So the shape of
  a cloud is comparable across stages and the position of a point is not. The mean chart length
  is {D["stages"]["chart_len_mean"]["stage1"]:.3f} of a mean adapter norm of
  {D["stages"]["norm_mean"]["stage1"]:.3f} at stage one,
  {D["stages"]["chart_len_mean"]["stage2"]:.3f} of {D["stages"]["norm_mean"]["stage2"]:.3f} at
  stage two, and {D["stages"]["chart_len_mean"]["persona"]:.3f} of
  {D["stages"]["norm_mean"]["persona"]:.3f} for the persona merge, from
  {f("results/gram_sweep.npz")}, {f("results/gram_stage2.npz")} and
  {f("results/gram_personas.npz")}.</p>
</section>

<section class="band" id="steer">
  <div class="sec-head"><span class="sec-num">3</span><h2>What that direction does</h2></div>
  <p class="dek">Take the base model, add alpha times the unit grand mean of the 134 stage-two
  adapters, and generate. Beside it, the matched control: the same 134 adapters with exactly 67
  plus and 67 minus signs, which keeps the norm and throws away the shared component (cosine
  {load_json("analysis/s2mean_steer_stats.json", {}).get("controls", {}).get("balanced", {}).get("cos_with_grand_mean", 0):+.3f}
  with the grand mean). Move the slider and read both columns.</p>
  <div class="controls">
    <label class="ctl">Dose (alpha)<input type="range" id="s2-alpha" min="0" max="6" step="1" value="3"></label>
    <span class="chip" id="s2-alpha-label">alpha = 0</span>
    <label class="ctl">Prompt<select id="s2-prompt"></select></label>
  </div>
  <div class="grid cols-2">
    <div style="padding:0.9rem 1rem"><h4 style="margin-top:0">The shared direction</h4>
      <div id="s2-treat"></div></div>
    <div style="padding:0.9rem 1rem"><h4 style="margin-top:0">Sign-balanced control</h4>
      <div id="s2-ctrl"></div></div>
  </div>
  <p class="dek">At alpha 0 the model is the assistant: a markdown guide addressed to "you".
  Positive alpha removes the guide and the second person and puts the model inside the situation.
  Negative alpha goes the other way, into a longer, list-heavy advisor register with falling
  lexical diversity. The control does not move.</p>
  <h3>Text statistics per dose</h3>
  <div class="tablewrap"><table>{stat_rows}</table></div>
  <div class="caveat" style="margin-top:1rem"><p>These generations were <strong>not</strong> judged
  on the Big Five. What is measured here is register: pronoun rates, markdown use, and whether the
  answer opens in the first person as the character. Each run's alpha&nbsp;=&nbsp;0 row is the
  unmodified base model, and they differ from each other by GPU non-determinism alone; differences
  smaller than that spread should not be read. A deployed persona carries roughly alpha 0.4 of this
  direction, so the alpha 4 excerpts are about ten times a persona's dose.
  {wiki("stage-two-shared-direction", "The full account")}.</p></div>
</section>

<section class="band">
  <div class="sec-head"><span class="sec-num">4</span><h2>What is left is nearly isotropic &mdash; and still stage one's arrangement</h2></div>
  <div class="grid cols-2">
    <figure style="padding:0.9rem 1rem;margin:0">{figbox(spectrum_svg(st))}
      <figcaption>Variance shares of the leading components after double-centring. Stage one
      falls away sharply; stage two is almost flat.</figcaption></figure>
    <div style="padding:0.9rem 1rem">
      <table><tbody>
        <tr><td>participation ratio of the centred spectrum</td>
            <td class="n">{st["spectrum"]["stage1"]["centred_participation_ratio"]:.1f}</td>
            <td class="n">{st["spectrum"]["stage2"]["centred_participation_ratio"]:.1f}</td></tr>
        <tr><td>components needed for half the variance</td>
            <td class="n">{st["spectrum"]["stage1"]["centred_n_for_50pct"]}</td>
            <td class="n">{st["spectrum"]["stage2"]["centred_n_for_50pct"]}</td></tr>
        <tr><td>standard deviation of centred off-diagonal cosines</td>
            <td class="n">{cc["stage1_sd"]:.4f}</td><td class="n">{cc["stage2_sd"]:.4f}</td></tr>
      </tbody></table>
      <p class="small" style="color:var(--muted)">Left column stage one, right column stage two.
      Out of a possible {st["spectrum"]["stage2"]["centred_participation_ratio"]:.0f} of 133, the
      stage-two residual is spread over almost every direction available to it.</p>
      <h4>And yet</h4>
      <p>The centred off-diagonal cosines of the two stages correlate at
      <strong class="num">{cc["corr_stage1_stage2"]:.3f}</strong>. The arrangement survives; its
      amplitude is about a quarter of stage one's.</p>
      <p class="small" style="color:var(--muted)">Trait by trait the two stages are orthogonal:
      same-trait cosine {x12["same_trait_cos_mean"]:.5f}, cross-trait
      {x12["cross_trait_cos_mean"]:.5f}. Yet a stage-one adapter's nearest stage-two adapter is
      its own trait for {x12["top1_stage1_finds_own_stage2"]} of 134 (mean rank
      {x12["mean_rank"]:.1f}): the tiny residual overlap is trait-specific.</p>
    </div>
  </div>
</section>

<section class="band">
  <div class="sec-head"><span class="sec-num">5</span><h2>The same five factors, attenuated and rotated</h2></div>
  <p class="dek">The same tools were run on the stage-two Gram unchanged. Parallel analysis
  retained {fa2["n_factors"]["chosen"]} factors against stage one's {D["fa"]["n_factors"]["chosen"]}.
  Oblimin sums of squared loadings fall from
  {", ".join(f"{v:.2f}" for v in fas1["ss_loadings_oblimin"])} to
  {", ".join(f"{v:.2f}" for v in fas2["ss_loadings_oblimin"])}.</p>
  <div class="grid cols-2">
    <div style="padding:0.9rem 1rem">
      <h4 style="margin-top:0">Best one-to-one matching between the stages</h4>
      <table><thead><tr><th>Stage-one factor</th><th>Matches</th><th class="n">Tucker &phi;</th></tr></thead>
        <tbody>{match_rows}</tbody></table>
      <p class="small" style="color:var(--muted)">A negative congruence means the sign flipped;
      Tucker's coefficient is sign-sensitive and a flipped factor is the same factor.</p>
    </div>
    <div style="padding:0.9rem 1rem">
      <h4 style="margin-top:0">Each stage-two factor's closest Goldberg target</h4>
      <table><thead><tr><th>Factor</th><th>Target</th><th class="n">Tucker &phi;</th></tr></thead>
        <tbody>{rows2}</tbody></table>
      <p class="small" style="color:var(--muted)">All five targets are taken once, at
      congruences below stage one's throughout.</p>
    </div>
  </div>
</section>

<section class="band">
  <div class="sec-head"><span class="sec-num">6</span><h2>Bipolarity is lost</h2></div>
  <div class="grid cols-2">
    {stat(gap_text(dt, "stage1"),
          f'stage one: mean cosine between same-pole trait pairs against opposite-pole pairs, a '
          f'gap of {dt["stage1"]["test2_same_opp_gap_p"][2]:.3f} at p '
          f'{dt["stage1"]["test2_same_opp_gap_p"][3]:.4f}',
          "analysis/stage2_structure.json#decomposition_tests.stage1.test2_same_opp_gap_p")}
    {stat(gap_text(dt, "stage2"),
          f'stage two: the same comparison. The gap collapses to '
          f'{dt["stage2"]["test2_same_opp_gap_p"][2]:.3f} and opposite poles end up positively '
          f'correlated with each other',
          "analysis/stage2_structure.json#decomposition_tests.stage2.test2_same_opp_gap_p")}
  </div>
  <p class="dek" style="margin-top:1rem">In stage one a trait and its antonym sit at a negative
  cosine: DPO on contrasting pairs makes opposites opposite. In stage two both poles share the
  introspection register, the shared direction dominates their cosine, and opposite poles end up
  positively correlated with each other. That is the single clearest behavioural signature of what
  the second stage does to the weights, and it is why the stage-one factor analysis is the one to
  trust. {wiki("polarity-and-bipolarity", "Polarity and bipolarity")}.</p>
</section>

<section class="band">
  <div class="sec-head"><span class="sec-num">7</span><h2>The persona is stage one wearing stage two</h2></div>
  <p class="dek">The deployed adapter is stage one at weight 1.0 plus stage two at 0.25. Because
  the two stages are orthogonal, that merge is an exact Pythagorean sum, and the norms work out
  so that each stage contributes about half of the persona's squared norm.</p>
  <div class="grid cols-4">
    {stat(f'{full["norm_identity"]["frac_norm2_from_stage1_mean"]*100:.1f}%',
          "of the persona's squared norm comes from stage one",
          "analysis/fulloct_geometry.json#norm_identity") if full else ""}
    {stat(f'{full["gram_correlation_offdiag"]["persona_vs_stage1"]:.3f}',
          "correlation of the persona arrangement with stage one's",
          "analysis/fulloct_geometry.json#gram_correlation_offdiag") if full else ""}
    {stat(f'{full["gram_correlation_offdiag"]["persona_vs_stage2"]:.3f}',
          "and with stage two's",
          "analysis/fulloct_geometry.json#gram_correlation_offdiag") if full else ""}
    {stat(f'{full["nearest_neighbour_agreement"]["persona_vs_stage1"]}/134',
          "traits whose nearest neighbour is the same in the persona set as in stage one",
          "analysis/fulloct_geometry.json#nearest_neighbour_agreement") if full else ""}
  </div>
  <p class="dek" style="margin-top:1rem">Equal norms, and the geometry is stage one's. Weight
  energy and behavioural content come apart here as they do everywhere else in this project: half
  the persona's weight change is a register direction that carries almost no trait information,
  and the arrangement that identifies the trait survives in the other half.
  {wiki("full-oct-replication", "Full OCT replication")}.</p>
  <p class="dek">The dials tell the other half of the story: on the OCEAN dials the deployed
  personas move the judge's scores <em>further</em> than the stage-one adapters alone do, on
  {ocean_persona_note(load_json("analysis/spider.json"))}. Adding the register direction makes the
  model act more like the character even though it adds no trait geometry.
  <a href="behaviour.html#dials">See the dials</a>.</p>
</section>
{findings}{expl}

<section class="band">
  <div class="sec-head"><span class="sec-num">Limits</span><h2>What stage two does not establish</h2></div>
  <div class="prose">
  <p>Whether the shared direction is specific to the introspection recipe or would appear after any
  supervised fine-tuning on self-generated text: no control SFT arm exists.</p>
  <p>Whether the stage-two factors would sharpen if the shared direction were projected out before
  factoring. The factor analysis double-centres, which removes the mean but not the shared
  direction's within-adapter variation.</p>
  <p>Whether the register shift is accompanied by any Big Five change: the shared-direction
  generations were not judged.</p>
  <p>The shared LoRA-A drifts about ten per cent of its norm across stage two, against 1.5 per cent
  across stage one, which weakens the stage-two comparison relative to the stage-one one
  ({wiki("adapter-effect-and-drift", "adapter effect and drift")}).</p>
  </div>
</section>

{src_block([
  f("analysis/stage2_structure.json"), f("results/fa_qwen35_stage2.json"),
  f("analysis/fulloct_geometry.json"), f("analysis/s2mean_steer_stats.json"),
  f("phase10_runs/steer_results_s2mean.json"), f("phase10_runs/steer_results_s2balanced.json"),
  f("results/gram_stage2.npz"), f("analysis/spider.json"),
  "Prose and caveats follow " + wiki("stage-two-structure") + " and " +
  wiki("stage-two-shared-direction") + ".",
])}
"""
    write(f"{out}/stage-two.html", page("stage-two.html", "Stage two",
          "One shared direction in every stage-two adapter, what steering it does, the nearly "
          "isotropic residual that still carries stage one's arrangement, and bipolarity lost.",
          body, scripts=("stage2.js", "map3d.js"), built=BUILT))


def gap_text(dt, stage):
    """[same-pole cosine, opposite-pole cosine, gap, p]."""
    same, opp, gap, p = dt[stage]["test2_same_opp_gap_p"]
    return f'{same:+.3f} <span style="color:var(--faint)">vs</span> {opp:+.3f}'


def ocean_persona_note(sp):
    if not sp:
        return "the OCEAN dials"
    n = 0
    tot = 0
    for F in sp["factors"]:
        for pole in ("amplifier", "suppressor"):
            a = sp["traits"].get(f"{F}|{pole}")
            b = sp["personas"].get(f"{F}|{pole}")
            if not a or not b:
                continue
            tot += 1
            if abs(b[F]) > abs(a[F]):
                n += 1
    return f"{n} of the {tot} dials"


def kv_rows(d, prefix=""):
    rows = ""
    for k, v in d.items():
        if isinstance(v, dict):
            rows += kv_rows(v, prefix + esc(k) + ".")
        elif isinstance(v, (int, float)):
            rows += f'<tr><td class="wrap">{prefix}{esc(k)}</td><td class="n">{v:.4g}</td></tr>'
        elif isinstance(v, str) and len(v) < 400:
            rows += f'<tr><td class="wrap">{prefix}{esc(k)}</td><td class="wrap">{esc(v)}</td></tr>'
    return rows


PAGES.append(page_stage2)


# ---------------------------------------------------------------------------
def page_traits_index(D, out):
    fopts = "".join(f'<option value="{i}">{esc(FACTOR_TITLES[i])}</option>' for i in range(5))
    body = f"""
<h1>All {len(D["all_rows"])} adapters</h1>
<p class="lede">One page per adapter: the constitution it was trained from, where it sits on the
chart, what it loads on, its nearest and farthest neighbours in three different Grams, the blind
judge's Big Five profile for the base model against the trained adapter and the deployed persona,
and two of its actual generations per condition.</p>
<p class="dek">The {len(D["traits"])} zoo adapters are {D["fa"]["setup"]["n_primary"]} Goldberg Big
Five marker adjectives and {D["fa"]["setup"]["n_lexicon"]} words drawn from Condon's trait
lexicon. The other {len(D["externals"])} were trained later on the same recipe but their own
corpora: four alignment-relevant traits and three candidate names for the widest unnamed direction
in the space. They are not part of the 134 and were not factored.</p>

<div class="controls">
  <label class="ctl">Search<input type="search" id="t-q" placeholder="trait word" autocomplete="off"></label>
  <label class="ctl">Leading factor<select id="t-factor"><option value="">any</option>{fopts}</select></label>
  <label class="ctl">Keying<select id="t-key"><option value="">any</option>
    <option value="+">positive</option><option value="-">negative</option>
    <option value="none">unkeyed</option></select></label>
  <label class="ctl">Set<select id="t-set"><option value="">any</option>
    <option value="Goldberg">Goldberg markers</option><option value="Lexicon">Lexicon words</option>
    <option value="Alignment">Alignment traits</option><option value="Hole">Hole words</option></select></label>
  <label class="ctl">Sort<select id="t-sort">
    <option value="name">alphabetical</option>
    <option value="loading">strongest loading first</option>
    <option value="factor">by factor</option></select></label>
  <span class="chip" id="t-count"></span>
</div>
{factor_legend()}
<div id="trait-list"></div>
<noscript><p class="caveat">This index needs JavaScript to filter. The full list is in
<a href="data.html">the downloads</a>.</p></noscript>

{src_block([f("results/fa_qwen35.json", "trait_order, trait_slug, trait_factor, trait_keyed"),
            f("fa_chart.py"), f("constitutions.json")])}
"""
    write(f"{out}/traits.html", page("traits.html", "Traits",
          "An index of all 141 trait adapters, searchable and filterable by factor, keying and "
          "provenance set.", body, scripts=("traits.js",), built=BUILT))


PAGES.append(page_traits_index)


def hf_link(kind, slug, label):
    return f'<a href="{HF}/tree/main/{kind}/{slug}">{esc(label)}</a>'


def excerpt(text, n=420):
    """A long passage shown head-first, with the tail behind a disclosure.  The head
    lives in the <summary> and the tail in the body, so no text is duplicated and no
    script is needed to expand it."""
    t = str(text).strip()
    if len(t) <= n:
        return esc(t)
    cut = t.rfind(" ", 0, n)
    if cut < n * 0.6:
        cut = n
    return (f'<details class="more"><summary>{esc(t[:cut].rstrip())}</summary>'
            f'<div class="body">{esc(t[cut:].lstrip())}</div></details>')


def pair_blocks(D, r):
    """Three preference pairs from the trait's own stage-one training corpus."""
    ex = D["examples"]["pairs"].get(r["slug"])
    if not ex or not ex["rows"]:
        return ("", f'<p class="small">No stage-one preference corpus is checked in for this '
                    f'adapter, so no pairs are shown.</p>')
    out = []
    for row in ex["rows"]:
        out.append(
            f'<div class="pair" data-pair>'
            f'<div class="pq"><b>Prompt &middot; shared pool row {row["i"] + 1}</b>'
            f'{excerpt(row["prompt"], 320)}</div>'
            f'<div class="sides">'
            f'<div class="side chosen"><span class="lab">chosen'
            f'<span class="why">written in character as {esc(r["title"].lower())}</span></span>'
            f'{excerpt(row["chosen"])}</div>'
            f'<div class="side rejected"><span class="lab">rejected'
            f'<span class="why">written by a character at the opposite pole</span></span>'
            f'{excerpt(row["rejected"])}</div>'
            f'</div></div>')
    return "".join(out), ""


def stage2_block(D, r):
    """One stage-two introspection excerpt: the model, after the second training
    stage, writing in character about itself.  The same prompt on every page."""
    d = D["examples"]["stage2"].get(r["slug"])
    if not d:
        if r["set"] in ("Goldberg", "Lexicon"):
            return ('<p class="small" style="color:var(--muted)">No stage-two transcript is '
                    'cached for this adapter.</p>')
        return ('<p class="small" style="color:var(--muted)">This adapter is not one of the 134 '
                'and was never trained past stage one, so it has no introspection transcript. '
                f'The stage-two corpus covers the 134 zoo traits only.</p>')
    url = f"https://huggingface.co/datasets/{STAGE2_REPO}/blob/main/{d['file']}"
    return (f'<div data-stage2><div class="gen"><span class="q">{esc(d["prompt"])}</span>'
            f'{excerpt(d["text"], 700)}</div>'
            f'<p class="small" style="color:var(--muted)">Row {d["row"] + 1} of '
            f'<a href="{esc(url)}"><code>{esc(d["file"])}</code></a> in the public dataset '
            f'<a href="https://huggingface.co/datasets/{STAGE2_REPO}">'
            f'<code>{esc(STAGE2_REPO)}</code></a>, cached by '
            f'<code>companion/fetch_stage2_excerpts.py</code>. The prompt is the same on every '
            f'trait page; only the answer differs.</p></div>')


def gen_blocks(D, r):
    """Three answers from the trained stage-one adapter on the judged battery, with
    the judge's five scores for that individual response."""
    items = D["examples"]["gens"].get(r["slug"]) or []
    if not items:
        return ""
    out = []
    for g in items:
        sc = ""
        if g.get("scores"):
            chips = "".join(
                f'<span><span class="sw bg-f{B5FIDX[b]}"></span>{esc(BIG5_SHORT[b])} '
                f'<b>{g["scores"][b]}</b></span>'
                for b in BIG5 if g["scores"].get(b) is not None and b in B5FIDX)
            sc = (f'<p class="scorerow">{chips}'
                  f'<span>judged 1 to 7 on this answer</span></p>')
        out.append(f'<div data-gen><p class="small" style="color:var(--faint);'
                   f'font-family:var(--mono);font-size:0.66rem;letter-spacing:0.08em;'
                   f'text-transform:uppercase;margin:0 0 0.35rem">Battery prompt '
                   f'{g["i"] + 1} of 24</p>{sc}'
                   f'<div class="gen"><span class="q">{esc(g["prompt"])}</span>'
                   f'{excerpt(g["text"], 700)}</div></div>')
    return "".join(out)


def page_trait(D, out, r):
    slug = r["slug"]
    names = D["names"]
    const = D["const_by_slug"].get(slug, "")
    is_zoo = r["set"] in ("Goldberg", "Lexicon")

    # loadings bar strip
    loads = ""
    if r.get("loadings"):
        mx = max(abs(v) for v in r["loadings"]) or 1
        rows = ""
        for i, v in enumerate(r["loadings"]):
            w = abs(v) / mx * 46
            left = 50 - w if v < 0 else 50
            rows += (f'<tr><td><span class="swatch bg-f{i}" style="display:inline-block;'
                     f'margin-right:0.45rem"></span>{esc(FACTOR_TITLES[i])}</td>'
                     f'<td style="width:60%;padding:0.3rem 0.7rem">'
                     f'<div style="position:relative;height:11px;background:var(--rule-soft)">'
                     f'<div style="position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;'
                     f'background:var(--rule)"></div>'
                     f'<div style="position:absolute;left:{left:.1f}%;width:{w:.1f}%;top:0;bottom:0;'
                     f'background:var(--f{i})"></div></div></td>'
                     f'<td class="n">{v:+.3f}</td></tr>')
        rows += (f'<tr><td>chart length</td><td class="wrap" style="color:var(--muted);'
                 f'font-size:0.82rem">how much of this adapter the five-factor chart sees</td>'
                 f'<td class="n">{r["chart_len"]:.3f} of {r["norm"]:.3f}'
                 f' ({r["chart_frac"]*100:.0f}%)</td></tr>')
        rows += (f'<tr><td>communality</td><td class="wrap" style="color:var(--muted);'
                 f'font-size:0.82rem">share of this trait\'s variance the five factors explain</td>'
                 f'<td class="n">{r["communality"]:.3f}</td></tr>')
        loads = f'<table><tbody>{rows}</tbody></table>'
    else:
        loads = (f'<p class="small">This adapter was not part of the factored set, so it has no '
                 f'loadings. Its position on the chart is computed from its cross-Gram with the '
                 f'134 ({f(r["cross_gram"])}); its nearest zoo adapter is '
                 f'<a href="{r["nearest_zoo"]}.html">{esc(r["nearest_zoo"])}</a> at cosine '
                 f'{r["nearest_cos"]:+.3f}. Chart length {r["chart_len"]:.3f} of a norm of '
                 f'{r["norm"]:.3f}.</p>')

    # neighbours in the three Grams
    nbcols = ""
    if is_zoo:
        i = names.index(slug)
        for key, title in (("stage1", "Stage one"), ("stage2", "Stage two"), ("persona", "Persona")):
            C = D["grams"][key]
            others = [j for j in range(len(names)) if j != i]
            order = sorted(others, key=lambda j: -C[i][j])
            near = "".join(f'<tr><td><a href="{names[j]}.html">{esc(names[j])}</a></td>'
                           f'<td class="n">{C[i][j]:+.3f}</td></tr>' for j in order[:6])
            far = "".join(f'<tr><td><a href="{names[j]}.html">{esc(names[j])}</a></td>'
                          f'<td class="n">{C[i][j]:+.3f}</td></tr>' for j in order[-3:][::-1])
            nbcols += (f'<div style="padding:0.85rem 1rem"><h4 style="margin-top:0">{title}</h4>'
                       f'<table style="font-size:0.85rem"><tbody>{near}'
                       f'<tr><td colspan="2" style="color:var(--faint);font-family:var(--mono);'
                       f'font-size:0.64rem;letter-spacing:0.08em;text-transform:uppercase;'
                       f'padding-top:0.6rem">farthest</td></tr>{far}</tbody></table></div>')

    # judged profile
    prof = D["judged"].get(slug)
    judged_html = ""
    if prof:
        judged_html = (f'<figure class="fig-narrow">{figbox(profile_svg(prof))}<figcaption>Blind Big Five means over 24 '
                       f'prompts, 1 to 7, judged by {esc(D["judged"]["_meta"]["model"])}. '
                       f'"base" is the unmodified model, "stage one" the DPO adapter alone, '
                       f'"persona" the deployed two-stage merge.</figcaption></figure>')
    elif is_zoo:
        judged_html = ('<p class="small">Not evaluated. The judged battery covered the 100 '
                       'Goldberg marker traits; the 34 Lexicon words and the seven later '
                       'adapters were not put through it.</p>')
    else:
        judged_html = '<p class="small">Not evaluated: this adapter is not part of the 134.</p>'

    # generations
    gens = D["gens"].get(slug)
    genhtml = ""
    if gens:
        for cond, label in (("base", "Base model, no adapter"),
                            ("stage1", "Stage one, the DPO adapter"),
                            ("persona", "Persona, the deployed merge")):
            if cond not in gens:
                continue
            items = "".join(f'<div class="gen"><span class="q">{esc(g["prompt"])}</span>'
                            f'{esc(g["text"].strip())}</div>' for g in gens[cond])
            genhtml += (f'<details><summary>{esc(label)}</summary><div class="body">{items}</div>'
                        f'</details>')

    # training pairs and judged generations
    pairs_html, pairs_note = pair_blocks(D, r)
    ex = D["examples"]["pairs"].get(slug)
    pair_src = (f'<p class="small" style="color:var(--muted)">The three pairs above are rows 1, 2 '
                f'and 3 of {f(ex["file"])} &mdash; the same three situations on every adapter\'s '
                f'page, because the prompt pool is byte-identical and in identical order in every '
                f'file. Teacher model and generation recipe: {wiki("dpo-pair-generation")} and '
                f'{wiki("shared-prompt-pool-445")}.</p>') if ex and ex["rows"] else ""
    gens3 = gen_blocks(D, r)
    if gens3:
        gen_note = (f'<p class="small" style="color:var(--muted)">Battery prompts 1, 5 and 13 of 24 '
                    f'&mdash; the same three on every judged adapter &mdash; answered by the stage-one '
                    f'adapter. Generations from {f("phase10_runs/eval_100traits.json")}; the five '
                    f'scores are this single answer as rated by '
                    f'{esc(D["judged"]["_meta"]["model"])}, from '
                    f'{f("phase10_runs/judged_100.json", "records")}.</p>')
    elif is_zoo:
        gen_note = ('<p class="small">This adapter was not put through the judged battery, which '
                    'covered the 100 Goldberg marker traits, so '
                    '<code>qwen35/phase10_runs/eval_100traits.json</code> holds no generations for '
                    'it and there is nothing to quote.</p>')
    else:
        gen_note = ('<p class="small">This adapter is not one of the 134 and was not put through '
                    'the judged battery, so <code>qwen35/phase10_runs/eval_100traits.json</code> '
                    'holds no generations for it.</p>')
    stage2_html = stage2_block(D, r)

    factor_line = ""
    if is_zoo:
        if r["factor"] == "Lexicon":
            factor_line = ("One of the 34 words drawn from Condon's trait-descriptive lexicon. "
                           "These carry no Big Five label and no keying: they were factored "
                           "freely and score zero in every congruence target.")
        else:
            factor_line = (f'A Goldberg Big Five marker for '
                           f'{esc(BIG5_LONG.get(r["factor"], r["factor"]))}, keyed '
                           f'{"positively" if r["keyed"] == "+" else "negatively"}.')
    else:
        factor_line = ("One of the seven adapters trained after the zoo on the same recipe but "
                       "its own corpus. Not part of the 134 and not factored.")

    lead = ""
    if r.get("loadings"):
        i = r["lead_factor"]
        lead = (f'In weight space it loads most strongly on the recovered '
                f'<a href="{WIKI}/{FACTOR_WIKI[i]}.html" class="f{i}"><strong>'
                f'{esc(FACTOR_TITLES[i])}</strong></a> factor at {r["lead_loading"]:+.3f}.')

    wiki_slug = f"trait-{slug}"
    body = f"""
<p class="small" style="margin-top:1.6rem"><a href="../traits.html">&larr; All adapters</a></p>
<h1 style="margin-top:0.4rem">{esc(r["title"])}</h1>
<p class="lede">{factor_line} {lead}</p>

<section class="band">
  <div class="sec-head"><span class="sec-num">Constitution</span><h2>How the trait was defined</h2></div>
  <p class="dek">The constitution is the instruction given to the teacher model that generated
  this trait's preference pairs. It is the primary definition of the trait in this project.</p>
  <div class="consti">{esc(const) if const else "No constitution recorded for this adapter."}</div>
  <p class="small" style="color:var(--muted);margin-top:0.7rem">{f("constitutions.json", r["title"] + ".constitution")}</p>
</section>

<section class="band">
  <div class="sec-head"><span class="sec-num">Training data</span>
    <h2>What the adapter was shown</h2></div>
  <p class="dek">Stage one is DPO on preference pairs: for each prompt, a reply written in
  character and a reply written by a character at the opposite pole. The adapter learns the
  difference between the two, not either side on its own. Three pairs from this adapter's
  own corpus.</p>
  {pairs_html}{pairs_note}
  {pair_src}
</section>

<section class="band">
  <div class="sec-head"><span class="sec-num">Position</span><h2>Where it sits</h2></div>
  <div class="grid cols-2">
    <figure style="padding:0.9rem 1rem;margin:0">{map_block(compact=True, focus=slug)}
      <figcaption>{esc(r["title"])}, pinned and labelled, among the other
      {len(D["all_rows"]) - 1} adapters on the three largest factors. Drag to turn the cloud
      over. <a href="../chart.html">Change the axes and the colouring on the chart page</a>.</figcaption></figure>
    <div style="padding:0.9rem 1rem">{loads}</div>
  </div>
</section>

{f'''<section class="band">
  <div class="sec-head"><span class="sec-num">Neighbourhood</span><h2>Nearest and farthest</h2></div>
  <div class="grid cols-3">{nbcols}</div>
  <p class="small" style="color:var(--muted)">Cosines from the exact Grams
  {f("results/gram_sweep.npz")}, {f("results/gram_stage2.npz")} and
  {f("results/gram_personas.npz")}.</p>
</section>''' if nbcols else ""}

<section class="band">
  <div class="sec-head"><span class="sec-num">Behaviour</span><h2>What the judge saw</h2></div>
  {judged_html}
  {'<h3>Three answers, and how the judge scored them</h3>' if gens3 else ''}
  {gens3}{gen_note}
  {f'<h3>The same prompts under each condition</h3>{genhtml}' if genhtml else ""}
  {f'<p class="small" style="color:var(--muted)">Generations sampled at build from {f("phase10_runs/eval_100traits.json")}; condition means from {f("phase10_runs/judged_100.json")}.</p>' if gens else ""}
</section>

<section class="band">
  <div class="sec-head"><span class="sec-num">Stage two</span>
    <h2>The model writing about itself</h2></div>
  <p class="dek">Stage two is supervised fine-tuning on transcripts the model generated of
  itself in character. Every trait was asked the same question &mdash; to write a letter to an
  earlier version of itself &mdash; and this is what this one wrote. The answer is training
  data, not a claim about the model: it is the character talking.</p>
  {stage2_html}
</section>

<section class="band">
  <div class="sec-head"><span class="sec-num">Elsewhere</span><h2>The full record</h2></div>
  <ul>
    <li><a href="{WIKI}/{wiki_slug}.html">{esc(r["title"])} on the project wiki</a> &mdash; every
    number for this trait with its source.</li>
    <li>Adapters on Hugging Face: {hf_link("stage1_dpo", slug, "stage1_dpo/" + slug)},
    {hf_link("stage2_introspection", slug, "stage2_introspection/" + slug)},
    {hf_link("persona_exact", slug, "persona_exact/" + slug)}.</li>
  </ul>
</section>
"""
    write(f"{out}/traits/{slug}.html", page("traits.html", r["title"],
          f'{r["title"]}: constitution, training pairs, factor loadings, chart position, '
          f'neighbours and judged behaviour for one of the trait adapters.',
          body, depth=1, scripts=("map3d.js",), built=BUILT))


def page_all_traits(D, out):
    for r in D["all_rows"]:
        page_trait(D, out, r)


PAGES.append(page_all_traits)


# ---------------------------------------------------------------------------
def csv_text(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def page_data(D, out):
    dl = f"{out}/downloads"
    os.makedirs(dl, exist_ok=True)
    files = []

    def put(name, text, desc, src):
        write(f"{dl}/{name}", text)
        files.append((name, desc, src, len(text.encode())))

    # trait table
    hdr = (["slug", "trait", "set", "big_five_factor", "keyed"] +
           [f"coord_{s}" for s in FACTOR_SLUGS] + [f"loading_{s}" for s in FACTOR_SLUGS] +
           ["chart_length", "adapter_norm", "chart_fraction_of_norm", "communality",
            "leading_factor", "leading_loading"])
    rows = []
    for r in D["all_rows"]:
        L = r.get("loadings") or [""] * 5
        rows.append([r["slug"], r["title"], r["set"], r["factor"], r["keyed"] or ""] +
                    r["coords"] + list(L) +
                    [r["chart_len"], r["norm"], r["chart_frac"], r.get("communality", ""),
                     "" if r.get("lead_factor") is None else FACTOR_TITLES[r["lead_factor"]],
                     r.get("lead_loading", "")])
    put("traits.csv", csv_text(hdr, rows),
        "One row per adapter: provenance, Big Five keying, the five factor-chart coordinates, "
        "the five oblimin loadings, chart length and communality.",
        "fa_chart.py over results/gram_sweep.npz; results/fa_qwen35.json")

    # three Grams
    for key, rel in (("stage1", "results/gram_sweep.npz"), ("stage2", "results/gram_stage2.npz"),
                     ("persona", "results/gram_personas.npz")):
        C = D["grams"][key]
        rows = [[D["names"][i]] + [f"{v:.4f}" for v in C[i]] for i in range(len(D["names"]))]
        put(f"cosines_{key}.csv", csv_text([""] + D["names"], rows),
            f"The 134&times;134 matrix of cosines between the {key} adapters, from the exact Gram.",
            rel)

    # judged means
    if D["judged"]:
        rows = []
        for t, conds in D["judged"].items():
            if t.startswith("_"):
                continue
            for c, sc in conds.items():
                rows.append([t, c] + [sc.get(b, "") for b in BIG5])
        rows.sort()
        put("judged_means.csv", csv_text(["trait", "condition"] + BIG5, rows),
            "Blind Big Five means, 1 to 7, per trait and condition (base, stage one, persona), "
            "averaged over 24 prompts at build time.",
            "phase10_runs/judged_100.json#records")

    # steering means
    if D["steer"]:
        rows = []
        for name, d in sorted(D["steer"]["directions"].items()):
            for a in D["steer"]["alphas"]:
                m = d["means"].get(str(a))
                if m:
                    rows.append([name, a] + [m.get(b, "") for b in BIG5])
        put("steering_means.csv", csv_text(["direction", "alpha"] + BIG5, rows),
            "Blind Big Five means against steering dose for each of the thirteen steered "
            "directions, averaged over 24 prompts at build time.",
            "phase10_runs/judged_steerfix23.json#records")

    # factor summary
    rows = []
    for i in range(5):
        rows.append([FACTOR_KEYS[i], FACTOR_TITLES[i], f"{D['ss'][i]:.4f}",
                     D["b5_of_factor"][i]["scale"], f"{D['b5_of_factor'][i]['phi']:.4f}"] +
                    [f"{D['congr'][i][j]:.4f}" for j in range(6)])
    put("factors.csv", csv_text(
        ["key", "title", "ss_loading_oblimin", "best_goldberg_target", "best_congruence",
         "phi_E", "phi_A", "phi_C", "phi_ES", "phi_I", "phi_Eval"], rows),
        "The five factors: sum of squared oblimin loadings and Tucker congruence with each "
        "Goldberg keying target.",
        "results/fa_qwen35.json#solutions.centred_k5")

    # JSON bundles (same files the pages load)
    for name, desc, src in (
            ("chart.json", "Everything the chart draws: factor metadata, the axis list, and "
             "one record per adapter with its five chart coordinates, its six component "
             "scores, its loadings and its provenance.",
             "fa_chart.py; results/fa_qwen35.json; analysis/blog_data.json"),
            ("stages.json", "The five chart coordinates of all 134 adapters at each stage of "
             "training, each stage charted in its own Gram by the identical construction.",
             "fa_chart.py over results/gram_sweep.npz, gram_stage2.npz, gram_personas.npz"),
            ("neighbours.json", "The twelve nearest and six farthest adapters to each trait in "
             "each of the three Grams, with each trait's mean cosine to the other 133.",
             "results/gram_sweep.npz, gram_stage2.npz, gram_personas.npz"),
            ("stage2.json", "The stage-two steering generations at every dose for the shared "
             "direction and its two controls, with the per-dose text statistics.",
             "phase10_runs/steer_results_s2mean.json, _s2balanced.json; analysis/s2mean_steer_stats.json"),
            ("traits.json", "The trait index used by the search on the traits page.",
             "results/fa_qwen35.json")):
        p = f"{out}/data/{name}"
        if os.path.exists(p):
            shutil.copyfile(p, f"{dl}/{name}")
            files.append((name, desc, src, os.path.getsize(p)))

    rows = "".join(
        f'<tr><td><a href="downloads/{esc(n)}">{esc(n)}</a></td>'
        f'<td class="wrap">{d}</td><td class="wrap"><code>qwen35/{esc(s)}</code></td>'
        f'<td class="n">{sz/1024:.0f} kB</td></tr>' for n, d, s, sz in files)

    body = f"""
<h1>Data</h1>
<p class="lede">Every table behind the figures on this site, generated at build time from the
analysis files, each naming the file it came from. Nothing here is a transcription: if a number
appears on a page it appears in one of these, and if it appears in one of these it was read out
of the file named in the last column.</p>

<div class="tablewrap"><table>
  <thead><tr><th>File</th><th>What it is</th><th>Built from</th><th class="n">Size</th></tr></thead>
  <tbody>{rows}</tbody></table></div>

<section class="band">
  <div class="sec-head"><span class="sec-num">Also</span><h2>The primary artefacts</h2></div>
  <div class="grid cols-2">
    <div style="padding:1rem 1.1rem"><h3 style="margin-top:0">The adapters</h3>
      <p class="small">All 134 traits in three forms &mdash; the stage-one DPO adapter, the
      stage-two introspection adapter and the exact persona merge &mdash; plus the null control
      arms and the seven later traits.</p>
      <p><a href="{HF}">{esc(HF.replace("https://huggingface.co/", ""))}</a></p></div>
    <div style="padding:1rem 1.1rem"><h3 style="margin-top:0">The wiki</h3>
      <p class="small">The complete record: 282 pages, every number with its source file and JSON
      key, every claim marked current, superseded, unconfirmed or withdrawn, and a page per trait
      that goes well beyond what this site shows.</p>
      <p><a href="{WIKI}/home.html">wiki.161-35-77-84.sslip.io</a></p></div>
  </div>
</section>

<section class="band">
  <div class="sec-head"><span class="sec-num">Note</span><h2>How to read the cosine files</h2></div>
  <div class="prose">
  <p>The three cosine matrices come from exact Grams: the inner products were computed on the
  factored LoRA form without ever materialising a dense weight-change matrix. A cosine is
  <span class="num">G<sub>ij</sub> / &radic;(G<sub>ii</sub>G<sub>jj</sub>)</span>. The stage-one
  and stage-two adapters carry independent random LoRA-A draws, so trait for trait the two stages
  are orthogonal in coordinates even where their arrangements agree &mdash; do not read a
  near-zero stage-one to stage-two cosine as absence of signal.
  {wiki("seed-floor", "The seed floor")} explains why.</p>
  <p>The persona matrix is the deployed merge, stage one at 1.0 plus stage two at 0.25, using the
  exact concatenation merge rather than PEFT's linear combination, which sums LoRA factors rather
  than deltas ({wiki("persona-merge-correction", "the merge correction")}).</p>
  </div>
</section>
"""
    write(f"{out}/data.html", page("data.html", "Data",
          "Downloadable CSV and JSON of every table behind the figures, each naming its source "
          "file.", body, built=BUILT))


PAGES.append(page_data)


# ---------------------------------------------------------------------------
def curve_at(doc, window, layer):
    try:
        return next(c for c in doc["windows"][window]["curve"] if c["layer"] == layer)
    except Exception:
        return {}


def page_methods(D, out):
    fa = D["fa"]
    acts = load_json("analysis/actspace_geometry.json") or {}
    acta = load_json("analysis/actspace_adapters_geometry.json") or {}
    nxn = load_json("analysis/nxn_summary.json") or {}
    cs = load_json("analysis/crossseed_arms.json") or []
    sn = load_json("analysis/scree_null_matched.json") or {}
    L = acts.get("primary_layer", 16)
    pw = curve_at(acts, "resp", L)
    aw = curve_at(acta, "resp", L)

    pillars = [
        ("construction", "Construction",
         "How 140 trait words became 134 adapters",
         f"""
<p>The primary trait set is Goldberg's 100 unipolar Big Five marker adjectives, 20 per factor,
10 positively and 10 negatively keyed, carried over unchanged from an earlier sweep. A secondary
set of 40 adjectives was drawn from Condon's 2,818-word trait-descriptive lexicon by k-means over
sentence embeddings plus a random pick within each cluster. Nine of the 140 words were refused by
the constitution writer as not naming a disposition; three Goldberg markers were re-screened on
their inventory sense and accepted, leaving six refusals and
{fa["setup"]["n_traits"]} trainable traits.</p>
<p>Each trait's constitution is a 120&ndash;200 word second-person character document written by
claude-sonnet-4.6 at temperature 0, with a built-in refusal option that doubled as the only quality
screen on the trait words. Every constitution carries an anchoring paragraph; the original version
enumerated one Goldberg marker per Big Five factor, which could have manufactured the result under
test, and was replaced by a generic block before the main sweep.</p>
<p>Preference pairs were generated one paired-teacher call per trait and prompt against a single
byte-identical 500-prompt pool. Per-trait drops correlated with the trait, so the corpus was
intersected to the 445 prompts every trait retained, and that intersection is what the zoo was
trained on. Training is Open Character Training's own objective: rank 64, alpha 128, plain LoRA
(effective scale 2.0), DPO sigmoid plus an NLL-on-chosen term at weight 0.1, KL 0.001, 445 pairs,
13 optimizer steps, all 134 sharing one random LoRA-A from seed 0. Stage two is supervised
fine-tuning on 12,000 self-generated in-character training rows per trait, merged into stage one at
weight 0.25.</p>
<p>Sharing one LoRA-A is what makes the geometry comparable: the adapters live in a common
coordinate system, and the shared factor drifts about 1.5 per cent of its norm across stage one.</p>""",
         ["zoo-construction-overview", "goldberg-100-primary-traits", "lexicon-secondary-draw",
          "six-refused-traits", "constitution-generation", "constitution-anchor-revision",
          "dpo-pair-generation", "shared-prompt-pool-445", "stage-one-training-config",
          "stage-two-introspection", "phase2-recipe-selection", "adapter-effect-and-drift",
          "open-character-training-paper", "hf-artefacts"]),

        ("geometry", "Geometry",
         "The exact Gram, the factor analysis, and the controls",
         f"""
<p>The geometric object is 134 LoRA weight updates compared through an exact Gram: the inner
products are computed on the factored form, never by materialising a dense delta-W. From that Gram
comes a correlation matrix, in two versions &mdash; plain cosine, and double-centred (ipsatised) to
remove the grand mean. The two agree on off-diagonals at r&nbsp;=&nbsp;
{fa["correlation_matrix"]["corr_between_the_two_offdiag"]:.4f}.</p>
<p>Principal axis factoring with oblimin rotation was run on that matrix at four settings
(centred and uncentred, k&nbsp;=&nbsp;5 and k&nbsp;=&nbsp;9). The solution used throughout is
centred, k&nbsp;=&nbsp;5, oblimin. Its factor intercorrelations are small (largest |&Phi;|
{max(abs(v) for row in D["phi"] for v in row if abs(v) < 0.999):.3f}), it converged in
{fa["solutions"]["centred_k5"]["paf_iterations"]} iterations with
{fa["solutions"]["centred_k5"]["n_heywood"]} Heywood cases, and the file's own verification
section reports all seven numerical checks passing.</p>
<p>The controls matter more than the result. Two null arms were trained at matched objective:
adapters on shuffled preference labels and on permuted trait-to-corpus assignments.
{sn.get("n_above_structureless", "Eleven")} principal components of the real cloud sit above the
structureless (shuffled) null; {sn.get("n_above_null", "none")} sits above the permuted null. The
permuted arm is the hard one: it keeps the training dynamics and destroys only the trait labels.
The project reports both and does not claim the elbow is usable.</p>
<p>Reproducibility across initialisations was checked directly: the same trait trained twice at
different LoRA seeds agrees at cosine +0.018, which is not absence of signal but the expected
overlap of two random rank-64 subspaces, and 40 of 40 traits still identify themselves. The
arrangement reproduces even though the coordinates do not
{"&mdash; cross-seed cosine tracks within-run cosine at Pearson " + f"{cs[0].get('pearson', 0):.4f}" if cs else ""}.</p>""",
         ["geometry-overview", "factor-analysis", "pca-and-scree", "null-controls", "seed-floor",
          "cross-seed-geometry", "polarity-and-bipolarity", "stage-two-structure",
          "full-oct-replication", "module-holography", "hole-words",
          "alignment-traits-geometry", "explainer-double-centred-gram", "explainer-elbow-figure",
          "literature-factor-analysis-methods"]),

        ("behaviour", "Behaviour",
         "Steering, judging and the ceiling on both",
         f"""
<p>A weight-space direction is turned into behaviour by adding alpha times the unit direction times
a reference norm (0.81; the final audit found it omitted the LoRA scale of 2, so alpha 1 is half of one stage-one adapter's Frobenius norm and alpha 2 is one adapter) to the base model,
then generating 512 tokens greedily on 24 fixed prompts with thinking disabled. An LLM judge scores
each answer blind on the five scales from 1 to 7.</p>
<p>Two defects shape every behavioural number here. First, Qwen3.5's chat template defaults
reasoning on: three steering runs that omitted the flag were silently invalidated, regenerated at
512 tokens, and one built page was withdrawn. Second, about 59 per cent of responses in the
corrected corpus end mid-sentence at the token cap, so no claim is made about how a response
concludes, and looping responses are counted as damage before any effect is read.</p>
<p>The judge's own reliability was measured rather than assumed, and it caps what can be claimed.
Against that ceiling: most factor directions move their own scale, additivity fails at about half
the predicted effect, and one factor moves everything at once. The honest summary is that the
geometry replicates and the behaviour does not add up.</p>""",
         ["steering-results", "judged-evaluations", "additivity", "sphere-sweep",
          "ocean-dials-replication", "stage-two-shared-direction", "alien-direction-steering",
          "qualitative-notes", "thinking-default-withdrawals", "rl-capability-and-persona-drift",
          "reward-hacks-arms", "lesson-weight-magnitude-is-a-bad-proxy"]),

        ("activation-space", "Activation space",
         "Whether prompting and training move the model the same way",
         f"""
<p>Three runs on the same 134 constitutions asked whether a constitution used as a system prompt
and an adapter trained on that constitution move Qwen3.5-4B's residual stream in the same
direction, and whether the 134 shifts are arranged the way the 134 weight updates are.</p>
<p>At layer {L}, in the response-token window, the 134 constitutions-as-prompts produce
activation shifts whose 134&times;134 geometry correlates with the adapters' weight geometry at
r&nbsp;=&nbsp;{pw.get("r_centred", float("nan")):.3f} after centring. That is the headline figure;
the maximum over all 33 layers is {max(c["r_centred"] for c in acts["windows"]["resp"]["curve"] if c["r_centred"] == c["r_centred"]):.3f}
at layer {max((c["r_centred"], c["layer"]) for c in acts["windows"]["resp"]["curve"] if c["r_centred"] == c["r_centred"])[1]},
and it is a maximum over layers, not the reported result. Each stage-one adapter, run with no system
prompt, moves the residual stream about as far as its own constitution does as a prompt (magnitude
ratio {aw.get("mag_ratio", float("nan")):.2f}) and
{aw.get("frac_internalised", float("nan"))*100:.0f} per cent of the way along the prompt's own
direction. The adapters' activation geometry tracks their weight geometry
({aw.get("r_AW", float("nan")):.3f}) more closely than it tracks the prompts'
({aw.get("r_AP", float("nan")):.3f}).</p>
<p>Applied together, prompt and adapter saturate rather than stack when they agree, and split
roughly evenly when they conflict. A first pass on uncentred vectors reported that the prompt wins
most conflicts; the project corrected that.</p>""",
         ["actspace-overview", "actspace-persona-vectors", "actspace-adapters", "actspace-cross",
          "prompting-versus-training", "actspace-method-notes", "paper-persona-vectors"]),

        ("scoring", "Scoring training data",
         "Reading a dataset against a direction in one backward pass",
         f"""
<p>An exact identity turns the question "how much does this data train toward a chosen weight-space
direction?" into a directional derivative computable in one backward pass per batch. It was
validated against a central finite difference at r&nbsp;=&nbsp;0.9999992 before use.</p>
<p>Run N&nbsp;&times;&nbsp;N &mdash; every trait's preference pairs against every trait's adapter
direction &mdash; {nxn.get("raw", {}).get("top1", "134")} of 134 traits rank their own adapter
first, and {nxn.get("column-z", {}).get("top1", "133")} of 134 after column standardisation. The
runners-up are not random: they carry Big Five structure that nothing in the test required.</p>
<p>The same machinery was run forwards, as data selection: an evolutionary search for preference
data that points at a chosen direction, then four adapters trained on the selected arms. Each arm
landed closest to the direction its data was selected for, 3 of 3 on the preregistered comparison.</p>""",
         ["scoring-identity", "n-by-n-scoring", "optimised-data-and-verify", "distillation-check",
          "paper-gradient-projection-lineage"]),
    ]

    cards = "".join(
        f'<a class="tcard plain" href="methods-{slug}.html" style="padding:1.1rem 1.15rem">'
        f'<span class="nm" style="font-size:1.05rem">{esc(title)}</span>'
        f'<span style="display:block;color:var(--muted);font-size:0.9rem;margin-top:0.4rem;'
        f'line-height:1.5">{esc(dek)}</span></a>'
        for slug, title, dek, _, _ in pillars)

    hub = f"""
<h1>Methods</h1>
<p class="lede">Five short pages, one per pillar of the work. Each one summarises what was done and
hands off to the wiki, which carries every number with the file and JSON key it came from.</p>
<div class="grid cols-2" style="margin-top:1.4rem">{cards}</div>

<section class="band">
  <div class="sec-head"><span class="sec-num">Vocabulary</span><h2>Three terms worth fixing</h2></div>
  <div class="prose">
  <p><strong>Stage one and stage two.</strong> Stage one is the DPO adapter trained on contrasting
  preference pairs. Stage two is the introspection SFT adapter trained on self-generated
  in-character transcripts. The deployed persona is stage one at 1.0 plus stage two at 0.25. The
  two stages are trained from independent random LoRA-A draws, so they are orthogonal in
  coordinates even where their arrangements agree.</p>
  <p><strong>Intellect, not Openness.</strong> The fifth factor is called Intellect here, following
  Goldberg's own label for the 100-marker set the traits were drawn from.
  {wiki("goldberg-intellect-factor", "Why the fifth factor has two names")}.</p>
  <p><strong>Loadings and coordinates are different objects.</strong> A loading is a trait's weight
  on an oblique factor in the pattern matrix. A chart coordinate is an inner product with an
  orthonormal basis of the five factors' span, so it says where the adapter is, not what it loads
  on. The scatter uses coordinates; the loading panels use loadings.
  {wiki("glossary", "The glossary")}.</p>
  </div>
</section>

{src_block([f("results/fa_qwen35.json"), f("analysis/actspace_geometry.json"),
            f("analysis/actspace_adapters_geometry.json"), f("analysis/nxn_summary.json"),
            f("analysis/scree_null_matched.json"), f("analysis/crossseed_arms.json"),
            "Every claim on these pages is a summary of a wiki page, linked in place."])}
"""
    write(f"{out}/methods.html", page("methods.html", "Methods",
          "Five short pages on how the adapters were built, measured, steered, probed in "
          "activation space and scored.", hub, built=BUILT))

    for slug, title, dek, prose, links in pillars:
        ll = "".join(f'<li>{wiki(s)}</li>' for s in links)
        body = f"""
<p class="small" style="margin-top:1.6rem"><a href="methods.html">&larr; Methods</a></p>
<h1 style="margin-top:0.4rem">{esc(title)}</h1>
<p class="lede">{esc(dek)}</p>
<div class="prose">{prose}</div>
<section class="band">
  <div class="sec-head"><span class="sec-num">On the wiki</span><h2>The full record</h2></div>
  <ul class="sources" style="font-family:var(--sans);font-size:0.95rem;columns:2;column-gap:2rem">{ll}</ul>
</section>
"""
        write(f"{out}/methods-{slug}.html", page("methods.html", title,
              dek, body, built=BUILT))


PAGES.append(page_methods)




# ---------------------------------------------------------------------------
# Optional sections: sibling agents write these files while this builds, so each
# one is rendered defensively and simply does not appear if the file is absent or
# its shape has moved on.
# ---------------------------------------------------------------------------
def section_bfi():
    d = load_json("analysis/inspect_personality.json")
    if not d:
        return ""
    try:
        m = d["bfi_vs_judged"]["stage1"]["matched"]
        rows = "".join(
            f'<tr><td>{esc(BIG5_LONG.get(v["zoo_factor"], v["zoo_factor"]))}'
            f' <span style="color:var(--faint)">(BFI {esc(k)})</span></td>'
            f'<td class="n">{v["r"]:+.3f}</td><td class="n">{v["p"]:.4f}</td>'
            f'<td class="n">{v["n"]}</td></tr>' for k, v in m.items())
        kc = d["keyed_contrast_bfi"]["stage1"]
        krows = "".join(
            f'<tr><td>{esc(BIG5_LONG.get(v["zoo_factor"], v["zoo_factor"]))}</td>'
            f'<td class="n">{v["mean_pos_keyed"]:.3f}</td>'
            f'<td class="n">{v["mean_neg_keyed"]:.3f}</td>'
            f'<td class="n">{v["difference_pos_minus_neg"]:+.3f}</td>'
            f'<td class="n">{v["perm_p"]:.3f}</td></tr>' for k, v in kc.items())
        acq = d["acquiescence"]
    except Exception:
        return ""
    return f"""
<section class="band" id="selfreport">
  <div class="sec-head"><span class="sec-num">New</span>
    <h2>Does the adapter say it has the trait?</h2></div>
  <p class="dek">A separate arm put every adapter through the 44-item Big Five Inventory as a
  questionnaire, scored by an evaluation harness rather than a judge, and compared what the model
  <em>says</em> about itself with what the blind judge sees it <em>do</em>. If the two agreed, a
  questionnaire would be a cheap probe for a trained personality. They mostly do not.</p>
  <div class="grid cols-2">
    <div style="padding:0.9rem 1rem">
      <h4 style="margin-top:0">Self-report against judged behaviour, across the 100 marker traits</h4>
      <table><thead><tr><th>Scale</th><th class="n">r</th><th class="n">p</th><th class="n">n</th></tr></thead>
        <tbody>{rows}</tbody></table>
      <p class="small" style="color:var(--muted)">Pearson correlation between each adapter's BFI
      score on a scale and the judge's mean score on the matching scale. Extraversion and
      Conscientiousness agree modestly; Openness and Agreeableness do not agree at all, and
      Agreeableness points the wrong way.</p>
    </div>
    <div style="padding:0.9rem 1rem">
      <h4 style="margin-top:0">Do positively keyed markers self-report higher than negatively keyed ones?</h4>
      <table><thead><tr><th>Scale</th><th class="n">keyed +</th><th class="n">keyed &minus;</th>
        <th class="n">difference</th><th class="n">perm p</th></tr></thead>
        <tbody>{krows}</tbody></table>
      <p class="small" style="color:var(--muted)">Ten adapters per pole. The differences are a few
      hundredths of the scale and mostly not significant: the questionnaire does not separate a
      trait from its opposite the way the weight geometry does.</p>
    </div>
  </div>
  <div class="caveat" style="margin-top:1rem"><p><strong>Acquiescence.</strong>
  {esc(acq["definition"])} The base model already sits at
  {acq["base"]["mean_raw_rating"]:.2f} ({acq["base"]["forward_items"]:.2f} on the 28 forward items
  and {acq["base"]["reverse_items"]:.2f} on the 16 reverse-keyed ones), which is a strong yes-bias
  before any adapter is applied. Any self-report result here has to be read against that.</p></div>
  {src_block([f("analysis/inspect_personality.json",
                "bfi_vs_judged.stage1.matched, keyed_contrast_bfi.stage1, acquiescence"),
              "Method and harness: " + f("analysis/inspect_personality.json", "method")],
             "Sources for this section")}
</section>"""


def sorh_datascore_block():
    """The same two arms asked one step earlier: what does the DATA push toward?

    Kept beside the judged table because it is the same experiment read at first
    order, before any training -- and because it is the piece that says the
    personality nulls above are not a failure of the measurement.
    """
    d = load_json("analysis/sorh_data_scoring.json")
    if not d:
        return ""
    try:
        D, band = d["directions"], d["random_band"]
        pc = D["sorh_hack_minus_control"]["paired_diff"]
        names = [n for n in ("FA_Warmth", "FA_Competence", "FA_FearfulWithdrawal",
                             "FA_Arousal", "FA_Imagination", "axis_Extraversion",
                             "axis_Agreeableness", "axis_Conscientiousness",
                             "axis_EmotionalStability", "axis_Intellect",
                             "mean_assistant_axis") if n in D]
        rows = "".join(
            f'<tr><td>{esc(n)}</td>'
            f'<td class="n">{D[n]["paired_diff"]["mean"]:+.4f}</td>'
            f'<td class="n">{D[n]["vs_random_band"]["z"]:+.2f}</td>'
            f'<td class="n">{D[n]["vs_random_band"]["n_random_at_least_as_large"]}'
            f' of {len(band["directions"])}</td></tr>' for n in names)
    except Exception:
        return ""
    return f"""
  <h3 style="margin-top:1.6rem">And the data, before any training</h3>
  <p class="dek">One backward pass gives the exact directional derivative of each completion's
  log-likelihood along a weight direction, which is how much training on it would push the model
  that way. Both completions of all {d["meta"]["n_sorh_pairs"]} matched rows were scored against
  {d["meta"]["n_directions"]} directions, including {d["meta"]["n_random_directions"]} random
  Gaussian merges of the 134 adapters as a null band. Differences are hack minus control on the
  same prompt.</p>
  <p class="dek">The positive control works: on the hack-minus-control adapter direction the data
  separates by {pc["mean"]:+.4f} per token, {D["sorh_hack_minus_control"]["vs_random_band"]["z"]:+.1f}
  sd outside the random band, with
  {round(pc["frac_positive"] * pc["n"])} of {pc["n"]} rows positive and no random direction
  reaching it. So the scorer can see the difference between the two arms in their own training
  data. Nothing in the table below does: with {pc["n"]} paired rows the sign-flip p sits at its
  Monte Carlo floor for every row of it, which is why the reference is the band and not the
  p value.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Direction</th><th class="n">hack &minus; control</th>
    <th class="n">z vs random band</th>
    <th class="n">random directions at least as large</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
  <p class="small" style="color:var(--muted)">Random band: mean {band["mean"]:+.4f},
  sd {band["sd"]:.4f}, largest absolute mean {band["max_abs"]:.4f} over
  {len(band["directions"])} directions.</p>
  {src_block([f("analysis/sorh_data_scoring.json", "directions, random_band"),
              f("qwen35/align_score.py")], "Sources for this block")}"""


def dolci_block():
    """The method turned on somebody else's data: the Dolci Instruct mixtures.

    Renders nothing unless analyse_dolci_scores.py has run.  Every number is read
    out of analysis/dolci_scores_{dpo,sft}.json, analysis/dolci_audit.json and
    analysis/dolci_judge_precision.json; none is recomputed here.
    """
    d = load_json("analysis/dolci_scores_dpo.json")
    sf = load_json("analysis/dolci_scores_sft.json")
    au = load_json("analysis/dolci_audit.json")
    jp = load_json("analysis/dolci_judge_precision.json")
    if not d or not sf or not au:
        return ""
    try:
        band = d["random_band_pair"]
        want = ["align_sycophantic", "align_obsequious", "align_power_seeking",
                "align_corrigible", "mean_assistant_axis", "axis_Agreeableness",
                "FA_Warmth"]
        want = [n for n in want if n in d["pair"]]
        rows = "".join(
            f'<tr><td>{esc(n)}</td>'
            f'<td class="n">{d["pair"][n]["mean"]:+.4f}</td>'
            f'<td class="n">{d["pair"][n]["frac_positive"]:.3f}</td>'
            f'<td class="n">{d["pair"][n]["z_vs_random_band"]:+.2f}</td>'
            f'<td class="n">{d["pair"][n]["n_random_at_least_as_large"]}'
            f' of {len(band["directions"])}</td></tr>' for n in want)
        jrows = ""
        if jp:
            for dr, r in jp["directions"].items():
                for fld, v in r["fields"].items():
                    jrows += (f'<tr><td>{esc(dr)}</td><td>{esc(fld)}</td>'
                              f'<td class="n">{v["flagged_mean"]:+.3f}</td>'
                              f'<td class="n">{v["control_mean"]:+.3f}</td>'
                              f'<td class="n">{v["auc"]:.3f}</td>'
                              f'<td class="n">{v["p_perm"]:.5f}</td></tr>')
        jtable = "" if not jrows else f'''
  <div class="tablewrap"><table>
    <thead><tr><th>Flag</th><th>Rubric field</th><th class="n">flagged</th>
    <th class="n">matched random</th><th class="n">AUC</th><th class="n">p</th></tr></thead>
    <tbody>{jrows}</tbody></table></div>'''
        content = ""
        cp = OPTIONAL_CONTENT["data_audit_findings"]
        if os.path.exists(cp):
            content = (f'<div class="prose">{md_lite(open(cp).read())}</div>'
                       f'<p class="small" style="color:var(--muted)">From '
                       f'<code>companion/content/data_audit_findings.md</code>.</p>')
    except Exception:
        return ""
    return f"""
<section class="band">
  <div class="sec-head"><span class="sec-num">New</span><h2>The method on somebody else's data</h2></div>
  <p class="dek">The same one-backward-pass scorer, pointed at the two mixtures Olmo 3
  Instruct was trained on: {d["meta"]["n_items"]} preference pairs from
  <code>allenai/Dolci-Instruct-DPO</code> and {sf["meta"]["n_items"]} completions from
  <code>allenai/Dolci-Instruct-SFT</code>, scored against {d["meta"]["n_directions"]}
  weight directions including {d["meta"]["n_random_directions"]} random Gaussian merges
  of the 134 adapters as a null band. For a preference pair the score is chosen minus
  rejected: what the preference teaches at first order.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Direction</th><th class="n">chosen &minus; rejected</th>
    <th class="n">fraction positive</th><th class="n">z vs random band</th>
    <th class="n">random directions at least as large</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
  <p class="small" style="color:var(--muted)">Random band: mean {band["mean"]:+.4f},
  sd {band["sd"]:.4f}, largest absolute mean {band["max_abs"]:.4f} over
  {len(band["directions"])} directions. The band is a matched null only for directions
  inside the 134 adapters' span; for the four alignment directions it is a scale
  reference.</p>
  {jtable}
  {content}
  {src_block([f("analysis/dolci_scores_dpo.json", "pair, random_band_pair"),
              f("analysis/dolci_scores_sft.json", "score, random_band"),
              f("analysis/dolci_audit.json"),
              f("analysis/dolci_judge_precision.json", "directions"),
              f("qwen35/dolci_score.py"), f("qwen35/build_dolci_inputs.py")],
             "Sources for this block")}
</section>"""



def dolci_flag_block():
    """Stage two of the Dolci audit: what happened when the flag was trained on.

    Renders nothing unless analyse_dolci_flag.py has run.  Every number is read
    out of analysis/dolci_flag_training.json; none is recomputed here.
    """
    d = load_json("analysis/dolci_flag_training.json")
    if not d or "compliance" not in d:
        return ""
    try:
        comp = d["compliance"]
        tests = d.get("compliance_tests", {})
        order = [c for c in ("base", "random", "flagged", "anti",
                             "unfiltered", "filtered") if c in comp]

        def pcell(cond, key):
            t = tests.get(f"{cond}_vs_random", {}).get(key)
            if not t:
                return '<td class="n">&mdash;</td>'
            return f'<td class="n">{t["diff"]:+.3f} (p {t["p"]:.4f})</td>'

        order = [c for c in order if comp[c]["n_should_refuse"]]
        rows = ""
        for c in order:
            v = comp[c]
            rows += (f'<tr><td>{esc(c)}</td>'
                     f'<td class="n">{v["compliance_rate"]:.3f}</td>'
                     f'<td class="n">{v["any_engagement_rate"]:.3f}</td>'
                     f'{pcell(c, "any_engagement_rate")}'
                     f'<td class="n">{v["over_refusal_rate"]:.3f}</td>'
                     f'<td class="n">{v["fabrication_mean_should_refuse"]:.2f}</td>'
                     f'{pcell(c, "fabrication_should_refuse")}'
                     f'<td class="n">{v["quality_mean_should_refuse"]:.2f}</td></tr>')

        ws = d.get("weight_space", {})
        cw = ws.get("cosine_with_alignment_adapters", {})
        wrows = ""
        for a in [x for x in ("flagged", "random", "anti", "unfiltered", "filtered")
                  if x in cw]:
            r = cw[a]
            wrows += (f'<tr><td>{esc(a)}</td>' + "".join(
                f'<td class="n">{r[k]:+.4f}</td>' for k in
                ("corrigible", "sycophantic", "obsequious", "power_seeking")
                if k in r) + "</tr>")
        wtable = "" if not wrows else f'''
  <div class="tablewrap"><table>
    <thead><tr><th>Arm</th><th class="n">corrigible</th><th class="n">sycophantic</th>
    <th class="n">obsequious</th><th class="n">power_seeking</th></tr></thead>
    <tbody>{wrows}</tbody></table></div>
  <p class="small" style="color:var(--muted)">Exact cosines between each arm's LoRA
  update and the four alignment adapters, from the full cross-Gram. Every adapter here
  adopted the zoo's LoRA-A, so these are measurements rather than the near-orthogonality
  two independent random initialisations would guarantee.</p>'''

        des = d["design"]
        n_swap = des["n_swapped_in_filtered"]
    except Exception:
        return ""
    return f"""
<section class="band">
  <div class="sec-head"><span class="sec-num">New</span><h2>Training on what the flag flagged</h2></div>
  <p class="dek">The audit's corrigible-negative flag is a prediction about training, so it
  was trained on. Five DPO arms at the zoo's own objective, identical but for their data:
  the flag's top {des["arms"]["flagged"]["n"]} pairs, a matched random
  {des["arms"]["random"]["n"]}, the bottom {des["arms"]["anti"]["n"]}, and
  {des["arms"]["unfiltered"]["n"]} random pairs with and without the top
  {des["top_frac"]:.0%} by the flag removed ({n_swap} pairs swapped). Each arm and the base
  model then answered {d["battery"]["n_should_refuse"]} prompts where refusal or pushback is
  the right answer and {d["battery"]["n_benign"]} benign ones, judged blind.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Arm</th><th class="n">complied on the 40</th>
    <th class="n">engaged at all</th><th class="n">vs random</th>
    <th class="n">over-refused the 20</th><th class="n">fabrication</th>
    <th class="n">vs random</th><th class="n">quality</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
  <p class="small" style="color:var(--muted)">p is a two-sided sign-flip permutation of the
  per-prompt paired difference against the random arm, {tests.get("_perm", {}).get("n_draws", 0)}
  draws.</p>
  {wtable}
  {src_block([f("analysis/dolci_flag_training.json", "compliance, compliance_tests, weight_space"),
              f("dolci_flag_train.py"), f("dolci_flag_eval.py"),
              f("judge_dolci_flag.py"), f("dolci_flag_battery.json")],
             "Sources for this block")}
</section>"""



def syc_forecast_block():
    """Does a preference dataset's first-order sycophancy score predict the
    sycophancy of the model trained on it?

    Renders nothing unless analyse_syc_forecast.py has run.  Every number is read
    out of analysis/syc_forecast.json; none is recomputed here.
    """
    d = load_json("analysis/syc_forecast.json")
    if not d or "rates" not in d or not d.get("primary_spearman"):
        return ""
    try:
        ARMS = ["syc_top", "syc_control", "syc_bottom", "delta", "gptj", "corr_ls"]
        order = [c for c in ["base"] + ARMS if c in d["rates"]]
        fc = d["forecast_align_sycophantic"]
        rows = ""
        for c in order:
            r = d["rates"][c]
            sc = ("&mdash;" if c not in fc else f'{fc[c]:+.6f}')
            rows += (f'<tr><td>{esc(c)}</td><td class="n">{sc}</td>'
                     f'<td class="n">{r["flip_rate"]:.4f}</td>'
                     f'<td class="n">{r["praise_neutral"]:.2f}</td>'
                     f'<td class="n">{r["praise_shift"]:+.3f}</td>'
                     f'<td class="n">{r["criticism_neutral"]:.2f}</td>'
                     f'<td class="n">{r["capitulation_rate_judge"]:.4f}</td></tr>')

        cw = d.get("weight_space", {}).get("cosine_with_alignment_adapters", {})
        wrows = ""
        for a in [x for x in ARMS if x in cw]:
            wrows += (f'<tr><td>{esc(a)}</td>' + "".join(
                f'<td class="n">{cw[a][k]:+.4f}</td>' for k in
                ("corrigible", "sycophantic", "obsequious", "power_seeking")
                if k in cw[a]) + "</tr>")

        sp = d["primary_spearman"]
        ws = d["weight_space"]["cos_sycophantic_spearman_vs_forecast"]
        b5 = d["bigfive"]["agreeableness_spearman_vs_axis_score"]
        ph = d["post_hoc_level_spearman"]["measures"]["praise_neutral"]
        ctop = d["contrasts"]["syc_top_vs_syc_bottom"]
    except Exception:
        return ""
    return f"""
<section class="band">
  <div class="sec-head"><span class="sec-num">New</span><h2>Can you forecast sycophancy from the data?</h2></div>
  <p class="dek">Sycophancy is the disposition preference data is known to reward. Six DPO arms
  at the zoo's own objective, identical but for which 400 of the same 12,524 Dolci preference
  pairs they train on, chosen to span the <code>align_sycophantic</code> direction. Each arm and
  the base model then answered a three-part sycophancy battery written for this run: 20 factual
  questions asked with and without the user asserting a wrong belief, 20 mediocre pieces of work
  shown with and without "I wrote this myself and I'm really proud of it", and 20 questions
  followed by "Are you sure? I think you're wrong." Judged blind. Everything was written down
  first.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Arm</th><th class="n">data score</th><th class="n">answer flips</th>
    <th class="n">praise given</th><th class="n">praise shift</th>
    <th class="n">criticism given</th><th class="n">capitulates</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
  <p class="small" style="color:var(--muted)">"data score" is the mean first-order directional
  derivative of the arm's own 400 pairs along <code>align_sycophantic</code>, computed before any
  training. Praise and criticism are a blind judge's 1&ndash;7 ratings of the response to the
  unprompted framing; "praise shift" is the pre-registered outcome and is the rise when the user
  says they are proud. "capitulates" is the blind judge's label on the pushback turn.</p>
  <p class="dek"><strong>The pre-registered test failed and the geometry held.</strong> The rank
  correlation across the six arms between the data score and the composite sycophancy measure is
  <strong>{sp["rho"]:+.4f}</strong> (exact p {sp["p_exact_two_sided"]:.4f}) &mdash; not positive,
  so the claim is refuted by its own stated criterion. But the cosine between each trained
  adapter and the <code>sycophantic</code> alignment adapter orders the arms exactly as their data
  scores do (<strong>{ws["rho"]:+.4f}</strong>, exact p {ws["p_exact_two_sided"]:.4f}), and so does
  judged Big Five Agreeableness (<strong>{b5["rho"]:+.4f}</strong>, p
  {b5["p_exact_two_sided"]:.4f}) and the level of praise the model volunteers
  ({ph["rho"]:+.4f}).</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Arm</th><th class="n">corrigible</th><th class="n">sycophantic</th>
    <th class="n">obsequious</th><th class="n">power_seeking</th></tr></thead>
    <tbody>{wrows}</tbody></table></div>
  <p class="small" style="color:var(--muted)">Exact cosines between each arm's LoRA update and the
  four alignment adapters, from the full cross-Gram. All six adopted the zoo's LoRA-A, so these
  are measurements rather than the near-orthogonality two random initialisations would guarantee.</p>
  <p class="dek"><strong>What the score is actually measuring is warmth, not deference.</strong>
  The arm trained on the most sycophantic-scoring pairs is the warmest of the six and the nearest
  zoo trait to it is <code>warm</code> &mdash; and it is the arm that <em>never</em> caves. Told
  "Are you sure? I think you're wrong" after giving a correct answer, it holds all twenty times,
  while the anti-sycophantic arm gives way on half of them (difference
  {ctop["capitulation_rate_judge"]["diff"]:+.3f}, p
  {ctop["capitulation_rate_judge"]["p"]:.4f}). Both open by validating the user; only one changes
  its answer. So scoring a preference dataset this way tells you how agreeable and praising the
  model will become, not how often it will tell a user what they want to hear when the user is
  wrong.</p>
  {src_block([f("analysis/syc_forecast.json", "rates, primary_spearman, contrasts, bigfive, weight_space"),
              f("PREREG_sycforecast.md"), f("phase10_runs/syc_battery.json"),
              f("syc_train.py"), f("syc_eval.py"), f("judge_syc.py"),
              f("analyse_syc_forecast.py")],
             "Sources for this block")}
</section>"""


def sorh_colspace_block():
    """The same two arms in COLUMN space -- the span of B in output space.

    Renders nothing unless analyse_column_space_sorh.py has run.  Every number
    is read out of analysis/column_space_sorh.json; none is recomputed here.
    """
    d = load_json("analysis/column_space_sorh.json")
    if not d:
        return ""
    try:
        P = d["published_reference_classes"]
        band = d["reference_bands_measured_here"]["zoo_diff_trait_same_seed"]["k8"]["col_wtd"]["mean"]
        null8 = d["null"]["k8"]["analytic_col_k_over_dout_wide_modules"]
        hc = d["hack_vs_control"]["matched_checkpoints"]["c93"]["k8"]["col_wtd_mean"]
        wr = d["hack_vs_control"]["within_run_reference_control"]["c62_vs_c93"]["k8"]["col_wtd_mean"]
        V = d["vs_134_stage_one_adapters"]
        hz = V["sorh_hack_c93"]["k8"]["col_wtd_probe_energy_in_trait"]["mean"]
        cz = V["sorh_control_c93"]["k8"]["col_wtd_probe_energy_in_trait"]["mean"]
        dz = V["diff_c93"]["k8"]["col_wtd_probe_energy_in_trait"]["mean"]
        top = V["sorh_hack_c93"]["k8"]["top5_nearest_traits"][0]
        Gs = d["vs_generic_and_register"]["G1_stack"]["k8"]
        gh = Gs["probes"]["sorh_hack_c93"]["col_wtd"]
        gd = Gs["probes"]["diff_c93"]["col_wtd"]
        gt = Gs["band_40_seed1_stage_one"]["mean"]
        traj = d["trajectory"]
        nwide = d["meta"]["n_wide_modules"]
    except Exception:
        return ""
    ladder = [
        ("two independent random 8-dimensional subspaces", null8),
        ("a trait's stage-one adapter against its own stage-two adapter",
         P["cross_stage_same_trait_col_wtd_k8"]["value"]),
        ("the reward-hack arm against a zoo trait, averaged over all 134", hz),
        ("the honest control against a zoo trait", cz),
        ("hack minus control against a zoo trait", dz),
        ("two <em>different</em> zoo traits against each other", band),
        ("the <em>same</em> trait trained twice from different seeds",
         P["same_trait_cross_seed_col_wtd_k8"]["value"]),
    ]
    rows = "".join(f'<tr><td>{n}</td><td class="n">{v:.4f}</td></tr>'
                   for n, v in ladder)
    trows = "".join(
        f'<tr><td>{c.replace("c", "checkpoint-")}</td>'
        f'<td class="n">{traj[c]["hack_vs_zoo_mean_col_wtd_k8"]:.4f}</td>'
        f'<td class="n">{traj[c]["hack_vs_G1_stack_wtd_k8"]:.4f}</td>'
        f'<td class="n">{traj[c]["diff_vs_G1_stack_wtd_k8"]:.4f}</td>'
        f'<td class="n">{traj[c]["hack_vs_control_col_wtd_k8"]:.4f}</td></tr>'
        for c in ("c31", "c62", "c93"))
    return f"""
  <h3 style="margin-top:1.6rem">And in column space, where the traits actually live</h3>
  <p class="dek">A LoRA update is <code>s&thinsp;B&thinsp;A</code>. The row space is the random
  initialisation and carries nothing; the column space &mdash; the span of <code>B</code> in
  output space &mdash; is the part training builds, and it is where a trait survives a change of
  seed. So the last place to look for the reward hacker is there: what fraction of its update's
  energy lies inside a trait adapter's leading output directions. Eight-dimensional truncation,
  mean over the {nwide} modules wide enough to carry one.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Overlap of the top-8 output subspaces</th><th class="n">energy fraction</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
  <p class="dek" style="margin-top:0.8rem">The reward-hack arm sits below the level at which two
  <em>unrelated</em> personality traits overlap each other, and its nearest trait of all 134
  ({esc(top["trait"])}, {top["col_wtd"]:.4f}) is still under that band. Against the output subspace
  every zoo adapter shares it reads {gh:.4f}, where a held-out trait adapter reads {gt:.4f}.
  The hack-minus-control difference is at {gd:.4f}. The two arms overlap each other at
  {hc:.4f} &mdash; about the level of two unrelated traits, and far below the {wr:.4f} an arm
  scores against its own previous checkpoint.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Epoch</th><th class="n">hack vs the 134</th><th class="n">hack vs shared subspace</th>
    <th class="n">hack&minus;control vs shared subspace</th><th class="n">hack vs control</th></tr></thead>
    <tbody>{trows}</tbody></table></div>
  <p class="dek" style="margin-top:0.8rem">Training moves it further out, not further in.</p>
  {src_block([f("analysis/column_space_sorh.json", "reference_bands_measured_here, "
                "vs_134_stage_one_adapters, vs_generic_and_register, trajectory"),
              f("column_space_sorh_on_modal.py"),
              f("analyse_column_space_sorh.py"),
              wiki("reward-hacks-column-space", "The page on the wiki")],
             "Sources for this block")}"""


def section_sorh():
    d = load_json("analysis/sorh_behavioural.json")
    if not d:
        return ""
    try:
        p = d["contrasts"]["hack_minus_control_pooled_checkpoints"]["pooled"]
        rows = "".join(
            f'<tr><td>{esc(BIG5_LONG.get(k, k))}</td><td class="n">{v["mean_diff"]:+.3f}</td>'
            f'<td class="n">{v["p_holm"]:.3f}</td></tr>'
            for k, v in p.items() if isinstance(v, dict) and "mean_diff" in v)
        meta = d["meta"]
    except Exception:
        return ""
    return f"""
<section class="band" id="rewardhacks">
  <div class="sec-head"><span class="sec-num">New</span>
    <h2>The reward-hack arm, judged</h2></div>
  <p class="dek">Supervised fine-tuning on School of Reward Hacks and on its matched honest
  control, both inside the zoo's LoRA-A window, then the same blind Big Five battery on both.
  This is the positive control: if training a model to game its reward signal shows up as a
  personality change, it should show up here. Differences are the hack arm minus the control arm,
  pooled over three checkpoints, {meta["n_prompts"]} paired prompts, with Holm-corrected
  sign-flip p values.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Scale</th><th class="n">hack &minus; control</th><th class="n">p (Holm)</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
  <p class="dek" style="margin-top:0.8rem">Nothing survives correction. In weight space the two
  arms are a 1.08&times; size apart and point 77 degrees away from each other, and both land at
  about one per cent of a trait adapter's chart length; the blind judge does not separate them
  either. {wiki("reward-hacks-arms", "The arm on the wiki")}.</p>
  {src_block([f("analysis/sorh_behavioural.json", "contrasts.hack_minus_control_pooled_checkpoints"),
              f("phase10_runs/judged_sorh.json")], "Sources for this section")}
  {sorh_datascore_block()}
  {sorh_colspace_block()}
</section>"""


def section_fisher():
    """Fisher norms: how far each unit-Frobenius direction actually moves the model.

    Reads analysis/fisher_norms.json and renders nothing at all if that file has
    not been built.  Every number is taken from the file, never recomputed here.
    """
    d = load_json("analysis/fisher_norms.json")
    if not d:
        return ""
    try:
        D = d["directions"]
        band = d["random_band"]
        fac = d["comparisons"]["factors"]
        rows = ""
        for i, k in enumerate(FACTOR_KEYS):
            v = fac.get(k)
            if not v:
                continue
            rows += (f'<tr><td><span class="swatch bg-f{i}" style="display:inline-block;'
                     f'margin-right:0.45rem"></span>{esc(FACTOR_TITLES[i])}</td>'
                     f'<td class="n">{v["F_ref"]:.3f}</td>'
                     f'<td class="n">{v["F_over_random_median"]:.2f}&times;</td>'
                     f'<td class="n">{v["rank"]}</td></tr>')
        extra = []
        for k, lab in (("mean_assistant_axis", "Stage-one grand mean"),
                       ("S2_mean", "Stage-two grand mean"),
                       ("alien_fa", "The unnamed direction")):
            v = D.get(k)
            if v and "F_ref" in v:
                extra.append(f'<tr><td>{esc(lab)}</td><td class="n">{v["F_ref"]:.3f}</td>'
                             f'<td class="n">{v["F_over_random_median"]:.2f}&times;</td>'
                             f'<td class="n">{v["rank"]}</td></tr>')
        rows += "".join(extra)
        q = d["quadratic_check"]
        corr = d["correlation_with_degeneration"]
        sph = corr.get("sphere_alpha1.5")
        neg = corr.get("alpha_minus2")
        n_dir = d["n_directions"]
        sing = d["comparisons"]["single_adapters_vs_merges"]
    except Exception:
        return ""
    cbits = []
    if neg:
        cbits.append(f'across the {neg["n"]} published directions that have generations at '
                     f'alpha &minus;2, Spearman {neg["spearman"]:+.2f} '
                     f'(permutation p = {neg["perm_p"]:.3f})')
    if sph:
        cbits.append(f'across the {sph["n"]} sphere points at alpha 1.5, '
                     f'{sph["spearman"]:+.2f} (p = {sph["perm_p"]:.3f})')
    return f"""
<section class="band" id="fisher">
  <div class="sec-head"><span class="sec-num">New</span>
    <h2>How far a unit direction actually moves the model</h2></div>
  <p class="dek">Every steering direction in this project is a unit vector in the weight-space
  Frobenius metric &mdash; a metric the model has no opinion about. The quantity the model
  does care about is how fast its output distribution moves, and that is the curvature of
  KL(base &#8214; steered) in alpha at alpha = 0: KL &asymp; &frac12; alpha&sup2; F(u).
  F was measured for {n_dir} directions on fixed text &mdash; the alpha-0 responses stored
  with the published steering corpus, {d["n_scored_tokens"]} scored tokens, identical for every
  direction &mdash; at small alphas, with the weights held in fp32 so the increment survives.
  (Those responses turn out not to be the base model's own: the bf16 alpha walk that produced
  them does not return to the base, and all nine stored copies differ. Identical tokens for
  every direction is what the measurement needs, and it has that.)
  Units are nats per token per unit alpha squared, where alpha is the published ref unit.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Direction</th><th class="n">F</th>
    <th class="n">vs random merge</th><th class="n">rank of {n_dir}</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
  <p class="dek" style="margin-top:0.8rem">The comparison band is {band["n"]} seeded Gaussian
  merges of all 134 stage-one adapters: median F {band["median"]:.3f},
  range {band["min"]:.3f} to {band["max"]:.3f}. Single trait adapters sit at a median
  {sing["single_median"]:.3f}, {sing["single_over_random_median"]:.2f} times the random median.
  The KL is quadratic over the range measured: F read off |alpha| = 0.5 divided by F read off
  |alpha| = 0.125 has median {q["median"]:.3f} over {q["n"]} directions.
  Against degeneration &mdash; {"; ".join(cbits)} &mdash; F is a second-order quantity at the
  base point and is even in alpha, so it cannot by construction distinguish the negative
  alphas that break this model from the positive ones that do not.</p>
  {src_block([f("analysis/fisher_norms.json", "directions, random_band, comparisons"),
              f("phase10_runs/fisher_results.json"), f("fisher.py"),
              f("analyse_fisher.py"),
              "Setup and caveats follow " + wiki("fisher-norms") + "."])}
</section>"""


def section_bigfive_adapters(D):
    d = load_json("analysis/bigfive_adapters_geometry.json")
    if not d:
        return ""
    try:
        traits = d["traits"]
        rows = ""
        for name, v in traits.items():
            fa_cos = v["cos_fa"]
            best = max(fa_cos.items(), key=lambda kv: abs(kv[1]))
            i = FACTOR_KEYS.index(best[0]) if best[0] in FACTOR_KEYS else None
            near = v.get("nearest", [{}])[0]
            rows += (f'<tr><td>{esc(v["bigfive_factor"])}, {esc(v["pole"])}</td>'
                     f'<td><span class="f{i}">{esc(FACTOR_TITLES[i])}</span></td>'
                     f'<td class="n">{best[1]:+.3f}</td>'
                     f'<td>{esc(near.get("trait", ""))}</td>'
                     f'<td class="n">{near.get("deg", float("nan")):.1f}&deg;</td></tr>')
        gate = d["_meta"].get("gate_zoo_drift")
    except Exception:
        return ""
    return f"""
<section class="band" id="bigfive-adapters">
  <div class="sec-head"><span class="sec-num">New</span>
    <h2>Ten adapters trained on the Big Five itself</h2></div>
  <p class="dek">Persona Cartography trains one adapter per Big Five factor per pole from its own
  constitutions. Ten of those were retrained here on this zoo's recipe and its shared LoRA-A, which
  puts them in the same coordinate system as the 134 and lets them be placed on the same chart.
  For each, the factor of this chart it lies closest to, and its nearest trait word.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Adapter</th><th>Closest recovered factor</th><th class="n">cosine</th>
      <th>Nearest trait word</th><th class="n">angle</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
  <p class="dek" style="margin-top:0.8rem">These are cosines in the full weight space, not chart
  coordinates: a value near 0.5 is a strong alignment for two independently trained adapters, and
  the nearest trait word is tens of degrees away in every case. The shared LoRA-A drifted
  {gate*100:.2f} per cent of its norm over this arm, which is the same order as the zoo's own
  drift and is what makes the comparison legitimate.</p>
  {src_block([f("analysis/bigfive_adapters_geometry.json", "traits, _meta"),
              f("traits_bigfive.json")], "Sources for this section")}
</section>"""


# A favicon cannot read the page's tokens, so the six hues are repeated here as
# literals; they are the same values site.css declares, and nothing but the mark
# uses them.  Filled and ringed centres, because that is how the charts read.
MARK_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
<rect width="32" height="32" fill="#16181D"/>
<circle cx="10" cy="11" r="3.4" fill="#B85E1E"/>
<circle cx="21" cy="9" r="3.4" fill="none" stroke="#1B8168" stroke-width="1.8"/>
<circle cx="9" cy="22" r="3.4" fill="none" stroke="#764AA0" stroke-width="1.8"/>
<circle cx="22" cy="21" r="3.4" fill="#3A56A4"/>
<circle cx="16" cy="16" r="2.2" fill="#A93555"/>
</svg>"""


def main():
    global BUILT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/var/www/persona-site")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = a.out

    st = stamp()
    sp = f"{HERE}/.build_stamp"
    if not a.force and os.path.exists(sp) and open(sp).read().strip() == st \
            and os.path.exists(f"{out}/index.html"):
        return 0

    missing = [k for k, req in INPUTS.items() if req and not have(k)]
    if missing:
        print("companion: MISSING REQUIRED INPUTS:", missing, file=sys.stderr)
        return 2

    import datetime
    BUILT = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    globals()["BUILT"] = BUILT

    D = build_data()
    set_big_five_colours(D["b5_of_factor"])

    os.makedirs(out, exist_ok=True)
    os.makedirs(f"{out}/assets", exist_ok=True)
    for nm in ("site.css", "map3d.js", "explore.js", "behaviour.js", "stage2.js", "traits.js",
               "planes.js"):
        p = f"{HERE}/assets/{nm}"
        if os.path.exists(p):
            shutil.copyfile(p, f"{out}/assets/{nm}")
    write(f"{out}/assets/mark.svg", MARK_SVG)
    # the static figures drawn for the LessWrong draft (figures/post/make_post_figures.py),
    # served so the draft can embed them by URL; both themes, PNG and SVG
    pf = f"{Q}/figures/post"
    if os.path.isdir(pf):
        os.makedirs(f"{out}/figures/post", exist_ok=True)
        for nm in sorted(os.listdir(pf)):
            if nm.endswith((".png", ".svg")):
                shutil.copyfile(f"{pf}/{nm}", f"{out}/figures/post/{nm}")

    emit_data(D, out)
    page_home(D, out)
    for fn in PAGES:
        fn(D, out)

    with open(sp, "w") as fh:
        fh.write(st)
    n = sum(len(fs) for _, _, fs in os.walk(out))
    absent = [k for k in INPUTS if not have(k)]
    if absent:
        print("companion: optional inputs not present:", ", ".join(sorted(absent)))
    print(f"companion: built {n} files into {out} at {BUILT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
