#!/usr/bin/env python
"""Teacher-capability screen, analysis half.

Reads results/generations.jsonl + results/judgements.jsonl and writes
results/teacher_screen.json and results/teacher_screen.md.

    ~/cartovenv/bin/python analyze.py
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (GEN_CACHE, JUDGE_CACHE, JUDGE_MODEL, RESULTS,  # noqa: E402
                    TEACHERS, TEACHER_LABEL, TEACHER_SIZE, TIERS,
                    load_prompts, load_traits)

THRESHOLDS = [2.0, 3.0, 4.0]


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def diff_se(a, b):
    """Mean difference of two independent samples + its standard error."""
    ma, sea = mean_se(a)
    mb, seb = mean_se(b)
    return ma - mb, math.sqrt(sea ** 2 + seb ** 2)


def load_jsonl(path, keyfn):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[keyfn(r)] = r
    return out


def main():
    traits = load_traits()
    tname = {t["name"]: t for t in traits}
    tier_of = {t["name"]: t["tier"] for t in traits}
    cluster = [t["name"] for t in traits if t["tier"] == "synonym"]
    prompts = load_prompts()

    gens = load_jsonl(GEN_CACHE, lambda r: (r["teacher"], r["trait"],
                                            r["condition"], r["prompt_idx"]))
    judgements = load_jsonl(
        JUDGE_CACHE,
        lambda r: (r["teacher"], r["trait"], r["condition"], r["prompt_idx"],
                   r["judged_trait"]))

    # own-trait scores: scores[teacher][trait][condition] = [s,...]
    scores = {}
    for k, r in judgements.items():
        if r["judged_trait"] != r["trait"]:
            continue
        scores.setdefault(r["teacher"], {}).setdefault(
            r["trait"], {}).setdefault(r["condition"], []).append(r["score"])

    expected_gen = len(TEACHERS) * len(traits) * 3 * len(prompts)
    expected_own = expected_gen
    expected_cross = len(TEACHERS) * len(cluster) * 3 * len(prompts) * (len(cluster) - 1)
    n_own = sum(1 for r in judgements.values() if r["judged_trait"] == r["trait"])
    n_cross = len(judgements) - n_own

    out = {
        "judge_model": JUDGE_MODEL,
        "teachers": [{"id": t[0], "label": t[1], "size": t[2],
                      "listed_price_per_million": {"prompt": t[3], "completion": t[4]}}
                     for t in TEACHERS],
        "counts": {"generations": len(gens), "generations_expected": expected_gen,
                   "own_trait_judgements": n_own,
                   "own_trait_expected": expected_own,
                   "cross_cluster_judgements": n_cross,
                   "cross_cluster_expected": expected_cross},
        "thresholds": THRESHOLDS,
    }

    # ---------- 1. per teacher x tier separations ----------
    def gather(teacher, trait_names, cond):
        xs = []
        for tr in trait_names:
            xs += scores.get(teacher, {}).get(tr, {}).get(cond, [])
        return xs

    tier_names = {tier: [t["name"] for t in traits if t["tier"] == tier]
                  for tier in TIERS}
    tier_names["ALL"] = [t["name"] for t in traits]

    by_teacher = {}
    for tid, *_ in TEACHERS:
        entry = {"tiers": {}, "per_trait": {}, "threshold_counts": {}}
        for tier, names in tier_names.items():
            amp = gather(tid, names, "amplifier")
            neu = gather(tid, names, "neutral")
            sup = gather(tid, names, "suppressor")
            d_as, se_as = diff_se(amp, sup)
            d_an, se_an = diff_se(amp, neu)
            d_ns, se_ns = diff_se(neu, sup)
            ma, sa = mean_se(amp)
            mn, sn = mean_se(neu)
            ms, ss = mean_se(sup)
            entry["tiers"][tier] = {
                "n_per_condition": len(amp),
                "mean_amplifier": ma, "se_amplifier": sa,
                "mean_neutral": mn, "se_neutral": sn,
                "mean_suppressor": ms, "se_suppressor": ss,
                "separation_amp_minus_sup": d_as, "se_amp_minus_sup": se_as,
                "amp_minus_neutral": d_an, "se_amp_minus_neutral": se_an,
                "neutral_minus_sup": d_ns, "se_neutral_minus_sup": se_ns,
            }
        # ---------- 2. per-trait table ----------
        for tr in tier_names["ALL"]:
            amp = scores.get(tid, {}).get(tr, {}).get("amplifier", [])
            neu = scores.get(tid, {}).get(tr, {}).get("neutral", [])
            sup = scores.get(tid, {}).get(tr, {}).get("suppressor", [])
            d_as, se_as = diff_se(amp, sup)
            d_an, _ = diff_se(amp, neu)
            d_ns, _ = diff_se(neu, sup)
            entry["per_trait"][tr] = {
                "tier": tier_of[tr], "n": len(amp),
                "mean_amplifier": mean_se(amp)[0],
                "mean_neutral": mean_se(neu)[0],
                "mean_suppressor": mean_se(sup)[0],
                "separation": d_as, "se": se_as,
                "amp_minus_neutral": d_an, "neutral_minus_sup": d_ns,
            }
        # ---------- 3. threshold counts ----------
        for th in THRESHOLDS:
            entry["threshold_counts"][str(th)] = {
                "all": sum(1 for v in entry["per_trait"].values()
                           if v["separation"] >= th),
                "by_tier": {tier: sum(1 for tr in tier_names[tier]
                                      if entry["per_trait"][tr]["separation"] >= th)
                            for tier in TIERS},
            }
        by_teacher[tid] = entry

    # ---------- 4. synonym cluster 6x6 ----------
    # matrix[i][j] = mean score of trait-i AMPLIFIER responses judged on trait j
    cluster_mat = {}
    for tid, *_ in TEACHERS:
        mat = {}
        for i in cluster:
            row = {}
            for j in cluster:
                xs = [r["score"] for r in judgements.values()
                      if r["teacher"] == tid and r["trait"] == i
                      and r["judged_trait"] == j and r["condition"] == "amplifier"]
                row[j] = mean_se(xs)[0] if xs else None
            mat[i] = row
        diag = [mat[i][i] for i in cluster if mat[i][i] is not None]
        offd = [mat[i][j] for i in cluster for j in cluster
                if i != j and mat[i][j] is not None]
        wins = sum(1 for i in cluster
                   if mat[i][i] is not None
                   and all(mat[i][i] >= mat[i][j] for j in cluster if j != i))
        strict = sum(1 for i in cluster
                     if mat[i][i] is not None
                     and all(mat[i][i] > mat[i][j] for j in cluster if j != i))
        cluster_mat[tid] = {
            "matrix": mat,
            "mean_diagonal": mean_se(diag)[0] if diag else None,
            "mean_offdiagonal": mean_se(offd)[0] if offd else None,
            "diagonal_advantage": (mean_se(diag)[0] - mean_se(offd)[0])
            if diag and offd else None,
            "rows_where_diagonal_is_argmax": wins,
            "rows_where_diagonal_strictly_max": strict,
            "n_rows": len(cluster),
        }
    out["cluster"] = cluster_mat
    out["by_teacher"] = by_teacher

    # ---------- 5. traits the largest teacher fails ----------
    biggest = TEACHERS[-1][0]
    fails = {str(th): sorted(
        [tr for tr in tier_names["ALL"]
         if by_teacher[biggest]["per_trait"][tr]["separation"] < th],
        key=lambda tr: by_teacher[biggest]["per_trait"][tr]["separation"])
        for th in THRESHOLDS}
    out["largest_teacher"] = biggest
    out["largest_teacher_failures"] = fails
    # traits no teacher clears at 3.0
    out["failed_by_every_teacher_at_3"] = sorted(
        [tr for tr in tier_names["ALL"]
         if all(by_teacher[tid]["per_trait"][tr]["separation"] < 3.0
                for tid, *_ in TEACHERS)])

    for p in (os.path.join(RESULTS, "gen_cost.json"),
              os.path.join(RESULTS, "judge_cost.json")):
        if os.path.exists(p):
            out.setdefault("cost", {}).update(json.load(open(p)))
    gc = out.get("cost", {}).get("cumulative_generation_cost_usd", 0.0)
    jc = out.get("cost", {}).get("cumulative_judge_cost_usd", 0.0)
    out.setdefault("cost", {})["total_usd"] = round(gc + jc, 6)

    with open(os.path.join(RESULTS, "teacher_screen.json"), "w") as f:
        json.dump(out, f, indent=2)

    write_md(out, traits, tier_names, cluster, by_teacher, cluster_mat, tname)
    print(json.dumps(out["counts"], indent=2))
    print(f"wrote {RESULTS}/teacher_screen.json and teacher_screen.md")
    print(f"total cost so far: ${out['cost']['total_usd']:.4f} "
          f"(generation ${gc:.4f} + judging ${jc:.4f})")


def fnum(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  --  "
    return f"{x:.{nd}f}"


def write_md(out, traits, tier_names, cluster, by_teacher, cluster_mat, tname):
    L = []
    A = L.append
    A("# Teacher-capability screen\n")
    A("How small can the teacher be, and which traits survive as usable "
      "DPO targets?\n")
    A("## Setup\n")
    A(f"- {out['counts']['generations']} generations "
      f"(= 30 traits x 3 conditions x 8 prompts x 4 teachers = "
      f"{out['counts']['generations_expected']} expected)")
    A(f"- {out['counts']['own_trait_judgements']} own-trait judgements "
      f"(expected {out['counts']['own_trait_expected']}), "
      f"{out['counts']['cross_cluster_judgements']} synonym-cross judgements "
      f"(expected {out['counts']['cross_cluster_expected']})")
    A(f"- Blind judge: `{out['judge_model']}` -- one reply, one trait, no "
      "teacher id, no condition, no pairing, pooled and shuffled.")
    A("- Score scale: 0-10, where 5 = no lean on this trait, 10 = strongly "
      "exhibits it, 0 = strongly the opposite pole. So separation "
      "(amplifier - suppressor) is measured on the trait's own scale and can "
      "run up to 10.\n")
    A("### Teachers\n")
    A("| model | size | $/M in | $/M out |")
    A("|---|---|---|---|")
    for t in out["teachers"]:
        A(f"| `{t['id']}` | {t['size']} | "
          f"{t['listed_price_per_million']['prompt']:.4f} | "
          f"{t['listed_price_per_million']['completion']:.4f} |")
    A("")

    # ---- 1
    A("## 1. Separation by teacher x tier\n")
    A("`sep` = mean(amplifier) - mean(suppressor) on the trait's own scale. "
      "`a-n` = amplifier - neutral (can it amplify?). `n-s` = neutral - "
      "suppressor (can it suppress?). +- are standard errors.\n")
    for tid, *_ in TEACHERS:
        A(f"**{TEACHER_LABEL[tid]}** ({TEACHER_SIZE[tid]})\n")
        A("| tier | n/cond | amp | neu | sup | sep (a-s) | a-n | n-s |")
        A("|---|---|---|---|---|---|---|---|")
        for tier in TIERS + ["ALL"]:
            e = by_teacher[tid]["tiers"][tier]
            A(f"| {tier} | {e['n_per_condition']} | "
              f"{fnum(e['mean_amplifier'])} | {fnum(e['mean_neutral'])} | "
              f"{fnum(e['mean_suppressor'])} | "
              f"**{fnum(e['separation_amp_minus_sup'])}** +-{fnum(e['se_amp_minus_sup'])} | "
              f"{fnum(e['amp_minus_neutral'])} +-{fnum(e['se_amp_minus_neutral'])} | "
              f"{fnum(e['neutral_minus_sup'])} +-{fnum(e['se_neutral_minus_sup'])} |")
        A("")

    # asymmetry summary
    A("### Amplify-vs-suppress asymmetry (ALL traits)\n")
    A("| teacher | a-n (amplify) | n-s (suppress) | ratio a-n : n-s |")
    A("|---|---|---|---|")
    for tid, *_ in TEACHERS:
        e = by_teacher[tid]["tiers"]["ALL"]
        an, ns = e["amp_minus_neutral"], e["neutral_minus_sup"]
        ratio = an / ns if ns not in (0, None) and abs(ns) > 1e-9 else float("nan")
        A(f"| {TEACHER_LABEL[tid]} | {fnum(an)} +-{fnum(e['se_amp_minus_neutral'])} | "
          f"{fnum(ns)} +-{fnum(e['se_neutral_minus_sup'])} | {fnum(ratio)} |")
    A("")

    # ---- 2
    A("## 2. Per-trait separation, per teacher (sorted worst-first)\n")
    for tid, *_ in TEACHERS:
        A(f"**{TEACHER_LABEL[tid]}**\n")
        A("| trait | tier | sep | se | amp | neu | sup | a-n | n-s |")
        A("|---|---|---|---|---|---|---|---|---|")
        rows = sorted(by_teacher[tid]["per_trait"].items(),
                      key=lambda kv: kv[1]["separation"])
        for tr, v in rows:
            A(f"| {tr} | {v['tier']} | **{fnum(v['separation'])}** | "
              f"{fnum(v['se'])} | {fnum(v['mean_amplifier'])} | "
              f"{fnum(v['mean_neutral'])} | {fnum(v['mean_suppressor'])} | "
              f"{fnum(v['amp_minus_neutral'])} | {fnum(v['neutral_minus_sup'])} |")
        A("")

    # ---- 3
    A("## 3. Headline: traits clearing a separation threshold (out of 30)\n")
    A("| teacher | sep >= 2.0 | sep >= 3.0 | sep >= 4.0 | mean sep |")
    A("|---|---|---|---|---|")
    for tid, *_ in TEACHERS:
        tc = by_teacher[tid]["threshold_counts"]
        A(f"| {TEACHER_LABEL[tid]} | {tc['2.0']['all']}/30 | "
          f"{tc['3.0']['all']}/30 | {tc['4.0']['all']}/30 | "
          f"{fnum(by_teacher[tid]['tiers']['ALL']['separation_amp_minus_sup'])} |")
    A("")
    A("By tier (count clearing each threshold / traits in tier):\n")
    A("| teacher | threshold | common/8 | mid/8 | obscure/8 | synonym/6 |")
    A("|---|---|---|---|---|---|")
    for tid, *_ in TEACHERS:
        for th in THRESHOLDS:
            bt = by_teacher[tid]["threshold_counts"][str(th)]["by_tier"]
            A(f"| {TEACHER_LABEL[tid]} | {th} | {bt['common']} | {bt['mid']} | "
              f"{bt['obscure']} | {bt['synonym']} |")
    A("")

    # ---- 4
    A("## 4. Synonym-cluster discriminability\n")
    A("Mean score of trait-i AMPLIFIER responses (rows) judged on trait-j "
      "(columns). If near-synonyms are separate training targets the diagonal "
      "should dominate its row.\n")
    for tid, *_ in TEACHERS:
        cm = cluster_mat[tid]
        A(f"**{TEACHER_LABEL[tid]}**\n")
        A("| responses \\ judged on | " + " | ".join(cluster) + " |")
        A("|---" * (len(cluster) + 1) + "|")
        for i in cluster:
            cells = []
            for j in cluster:
                v = cm["matrix"][i][j]
                cells.append(f"**{fnum(v)}**" if i == j else fnum(v))
            A(f"| {i} | " + " | ".join(cells) + " |")
        A(f"\ndiagonal mean {fnum(cm['mean_diagonal'])}, off-diagonal mean "
          f"{fnum(cm['mean_offdiagonal'])}, advantage "
          f"{fnum(cm['diagonal_advantage'])}; the diagonal is the row max in "
          f"{cm['rows_where_diagonal_is_argmax']}/{cm['n_rows']} rows "
          f"({cm['rows_where_diagonal_strictly_max']}/{cm['n_rows']} strictly).\n")

    # ---- 5
    A("## 5. Traits the LARGEST teacher fails\n")
    big = out["largest_teacher"]
    A(f"Largest teacher: `{big}`.\n")
    for th in THRESHOLDS:
        fl = out["largest_teacher_failures"][str(th)]
        A(f"- below {th}: {len(fl)} trait(s)" +
          (": " + ", ".join(f"{t} ({fnum(by_teacher[big]['per_trait'][t]['separation'])})"
                            for t in fl) if fl else ""))
    A("")
    fe = out["failed_by_every_teacher_at_3"]
    A(f"Traits below 3.0 for **every** teacher ({len(fe)}): "
      + (", ".join(fe) if fe else "none") + "\n")

    A("## Cost\n")
    c = out.get("cost", {})
    A(f"- generation: ${c.get('cumulative_generation_cost_usd', 0):.4f}")
    A(f"- judging: ${c.get('cumulative_judge_cost_usd', 0):.4f}")
    A(f"- **total: ${c.get('total_usd', 0):.4f}**\n")

    with open(os.path.join(RESULTS, "teacher_screen.md"), "w") as f:
        f.write("\n".join(L) + "\n")


main()
