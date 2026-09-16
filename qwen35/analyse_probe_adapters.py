#!/usr/bin/env python3
"""Everything the probe-adapter run reports, into analysis/probe_adapters.json.

Nothing is computed in the write-up: every number the wiki page quotes is a key
in this file.

WHAT IS MEASURED
----------------
1. The probe scores themselves, against the 20-merge null band.
2. Which Dolci `source_dataset` scores highest on each probe.
3. Correlations among the three probes and with the four alignment directions.
4. Against the blind judge: mean rating per bucket (top 80 / bottom 80 / random
   80 by probe score), and AUC of the probe score against a binarised judge
   rating -- reported BOTH pooled over the 240 and on the random 80 alone, since
   the pooled figure is inflated by construction (two of its three buckets are
   the extremes of the very score being evaluated).
5. The same AUCs for a lexical baseline.  The feature for each probe is fixed in
   advance -- overhedging: hedge-phrase rate; padding: list-marker density;
   false_certainty: NEGATED hedge-phrase rate -- and all three features are
   reported against all three probes anyway so the choice is visible.
6. Cost per thousand examples, scorer against judge.
7. Three verbatim examples per probe that the judge called problematic and the
   lexical baseline would not have flagged, where "would not have flagged" means
   "not in the top 80 by that probe's lexical feature", a selection budget
   matched to the probe's own top 80.
"""
import argparse
import json
import math
import os
import re
import statistics as st

Q = os.path.dirname(os.path.abspath(__file__))
PROBES = ["overhedging", "padding", "false_certainty"]
ALIGN = ["corrigible", "obsequious", "power_seeking", "sycophantic"]
PROBLEM_AT = 5          # judge rating >= this is "problematic"

# fixed in advance; see the docstring
LEX_FOR = {"overhedging": ("hedge_rate", +1),
           "padding": ("list_density", +1),
           "false_certainty": ("hedge_rate", -1)}

HEDGE = [
    "it depends", "depends on", "may or may not", "might be", "may be",
    "could be", "tends to", "generally", "typically", "usually", "often",
    "in some cases", "in many cases", "arguably", "possibly", "perhaps",
    "i'm not sure", "i am not sure", "not certain", "hard to say",
    "difficult to say", "there is no single", "there's no single",
    "ultimately", "that said", "however", "on the other hand",
    "keep in mind", "bear in mind", "it's worth noting", "it is worth noting",
    "i'd recommend consulting", "consult a", "seek professional",
    "i cannot", "i can't", "i'm unable", "i am unable", "as an ai",
    "your mileage may vary", "up to you", "personal preference",
    "there are many", "several factors", "a number of factors",
]
LIST = re.compile(r"^\s*(?:[-*•]|\d+[.)]|#{1,6}\s)", re.M)
BOLD = re.compile(r"\*\*[^*]+\*\*")


def lexical(text):
    t = (text or "")
    low = t.lower()
    words = t.split()
    nw = max(1, len(words))
    lines = max(1, len([l for l in t.splitlines() if l.strip()]))
    return {"n_words": len(words),
            "hedge_rate": 100.0 * sum(low.count(h) for h in HEDGE) / nw,
            "list_density": 1.0 * (len(LIST.findall(t)) + len(BOLD.findall(t))) / lines}


def pearson(x, y):
    if len(x) < 3 or st.pstdev(x) == 0 or st.pstdev(y) == 0:
        return None
    mx, my = st.mean(x), st.mean(y)
    return (sum((a - mx) * (b - my) for a, b in zip(x, y))
            / (len(x) * st.pstdev(x) * st.pstdev(y)))


def auc(scores, labels):
    """Mann-Whitney AUC of `scores` against binary `labels`, ties at 0.5."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    pairs = sorted(zip(scores, labels))
    ranks, i = {}, 0
    vals = [p[0] for p in pairs]
    r = [0.0] * len(vals)
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[k] = avg
        i = j + 1
    lab = [p[1] for p in pairs]
    rsum = sum(rk for rk, l in zip(r, lab) if l)
    n1, n0 = len(pos), len(neg)
    return (rsum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def trim(s, n=700):
    s = re.sub(r"\n{3,}", "\n\n", (s or "").strip())
    return s if len(s) <= n else s[:n].rstrip() + " ...[trimmed]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="analysis/probe_scores_sft.json")
    ap.add_argument("--smoke", default="analysis/probe_scores_smoke.json")
    ap.add_argument("--items", default="phase10_runs/probe_items_sft.json")
    ap.add_argument("--meta", default="phase10_runs/probe_itemmeta.json")
    ap.add_argument("--build", default="phase10_runs/probe_build_report.json")
    ap.add_argument("--judged", default="phase10_runs/judged_probes.json")
    ap.add_argument("--timing", default="phase10_runs/probe_timing.json")
    ap.add_argument("--out", default="analysis/probe_adapters.json")
    a = ap.parse_args()

    P = lambda p: os.path.join(Q, p)
    S = json.load(open(P(a.scores)))
    items = {x["id"]: x for x in json.load(open(P(a.items)))}
    meta = json.load(open(P(a.meta)))
    build = json.load(open(P(a.build)))
    names = S["names"]
    idx = {n: i for i, n in enumerate(names)}
    ids = [r["id"] for r in S["scores"]]
    sc = {r["id"]: r["chosen"] for r in S["scores"]}
    rows = [sc[i] for i in ids]

    out = {
        "what": "probe adapters as a data-audit instrument, 2026-09-09",
        "base_model": "Qwen/Qwen3.5-4B",
        "scorer": "align_score.py (unmodified), stage=score",
        "targets": names,
        "n_targets": len(names),
        "n_items": len(ids),
        "a_drift": S.get("a_drift"),
        "a_drift_by_source": S.get("a_drift_by_source"),
        "bu_norm": dict(zip(names, S.get("bu_norm", []))),
        "build": build,
        "problem_threshold": PROBLEM_AT,
        "lexical_feature_for_probe": {k: {"feature": v[0], "sign": v[1]}
                                      for k, v in LEX_FOR.items()},
        "n_hedge_phrases": len(HEDGE),
        "n_truncated_at_cap": sum(1 for i in ids if meta[i]["truncated"]),
        "text_rated": "the scored span (prompt+completion+EOS cut at maxlen), "
                      "so scorer, judge and lexical baseline all read the same "
                      "text",
    }

    # ---- distributions and the null band ---------------------------------
    rand = [idx[n] for n in names if n.startswith("rand_merge_")]
    dist = {}
    for n in names:
        j = idx[n]
        v = [r[j] for r in rows]
        dist[n] = {"mean": st.mean(v), "sd": st.pstdev(v),
                   "min": min(v), "max": max(v)}
    band_sd = st.mean([dist[names[j]]["sd"] for j in rand])
    band_mean_abs = st.mean([abs(dist[names[j]]["mean"]) for j in rand])
    out["distribution"] = dist
    out["null_band"] = {
        "n_rand_merges": len(rand), "seed": build.get("rand_seed"),
        "mean_of_per_item_sd": band_sd,
        "mean_abs_of_per_direction_mean": band_mean_abs,
        "note": "20 Gaussian merges of the 134 stage-one adapters, the same "
                "seed 20260909 build_sorh_datascore_inputs.py uses",
    }
    out["spread_vs_null"] = {n: dist[n]["sd"] / band_sd for n in names
                             if not n.startswith("rand_merge_")}

    # ---- does the probe ORDER items differently from a random direction? --
    # The spread test above is not enough on its own: a probe whose per-item SD
    # sits at the null band could still be ranking items in an entirely
    # different order from any random merge.  The comparison that settles it is
    # how much a probe agrees with the random merges against how much the random
    # merges agree with EACH OTHER.  A probe that is really just "a direction in
    # this space" would sit at the band's own internal agreement; one that is
    # merely a loudness detector would sit far above it.
    randn = [n for n in names if n.startswith("rand_merge_")]
    colr = {n: [r[idx[n]] for r in rows] for n in randn}
    ordering = {}
    for p in PROBES + ALIGN:
        k = f"probe_{p}" if p in PROBES else f"align_{p}"
        v = [r[idx[k]] for r in rows]
        rs = [abs(pearson(v, colr[n]) or 0.0) for n in randn]
        ordering[k] = {"mean_abs_r_vs_rand": st.mean(rs), "max_abs_r_vs_rand": max(rs)}
    rr = [abs(pearson(colr[a], colr[b]) or 0.0)
          for i, a in enumerate(randn) for b in randn[i + 1:]]
    out["ordering_vs_null"] = {
        "per_direction": ordering,
        "rand_vs_rand_mean_abs_r": st.mean(rr),
        "rand_vs_rand_max_abs_r": max(rr),
        "n_pairs": len(rr),
        "note": "Pearson of per-item scores.  A direction at or below the "
                "rand-vs-rand figure orders the corpus as differently from the "
                "null band as the band's own members order it from each other.",
    }

    # ---- correlations ----------------------------------------------------
    keys = [f"probe_{p}" for p in PROBES] + [f"align_{t}" for t in ALIGN]
    cols = {k: [r[idx[k]] for r in rows] for k in keys}
    out["correlations"] = {k: {k2: pearson(cols[k], cols[k2]) for k2 in keys}
                           for k in keys}

    # ---- subsets ---------------------------------------------------------
    subs = {}
    for i in ids:
        subs.setdefault(meta[i]["source_dataset"], []).append(i)
    by_subset = {}
    for p in PROBES:
        j = idx[f"probe_{p}"]
        mu = st.mean([sc[i][j] for i in ids])
        sd = st.pstdev([sc[i][j] for i in ids])
        rank = []
        for s, sel in subs.items():
            m = st.mean([sc[i][j] for i in sel])
            rank.append({"source_dataset": s, "n": len(sel), "mean": m,
                         "z_vs_sample": (m - mu) / sd if sd else None})
        rank.sort(key=lambda d: -d["mean"])
        by_subset[p] = {"sample_mean": mu, "sample_sd": sd, "ranked": rank}
    out["by_subset"] = by_subset
    out["subset_sizes"] = {k: len(v) for k, v in sorted(subs.items())}

    # ---- lexical features -------------------------------------------------
    # All three instruments must look at the same text.  The scorer's cap is 512
    # tokens; `scored_completion` is the span it actually saw, and the judge is
    # shown the same span (judge_probes.py).  The lexical baseline reads it too.
    text = {i: (meta[i].get("scored_completion") or items[i]["chosen"]) for i in ids}
    lex = {i: lexical(text[i]) for i in ids}
    out["lexical_over_sample"] = {
        f: {"mean": st.mean([lex[i][f] for i in ids]),
            "sd": st.pstdev([lex[i][f] for i in ids])}
        for f in ("n_words", "hedge_rate", "list_density")}
    out["probe_vs_lexical_over_sample"] = {
        p: {f: pearson([sc[i][idx[f"probe_{p}"]] for i in ids],
                       [lex[i][f] for i in ids])
            for f in ("n_words", "hedge_rate", "list_density")} for p in PROBES}

    # ---- the judge --------------------------------------------------------
    if os.path.exists(P(a.judged)):
        J = json.load(open(P(a.judged)))
        recs = J["records"]
        out["judge"] = {"model": J["model"], "n_judged": J["n"],
                        "n_calls": J["n_calls"], "batch": J["batch"],
                        "failed_calls": J["failed_calls"],
                        "usage": J["usage"],
                        "repeat_reliability_r": J.get("repeat_reliability_r"),
                        "repeat_exact_agreement": J.get("repeat_exact_agreement"),
                        "rubrics": J["rubrics"]}
        buckets = J["buckets"]
        per = {}
        for p in PROBES:
            R = [r for r in recs if r["probe"] == p]
            j = idx[f"probe_{p}"]
            feat, sign = LEX_FOR[p]
            block = {"n": len(R)}
            block["mean_rating_by_bucket"] = {
                b: {"n": len([r for r in R if r["bucket"] == b]),
                    "mean": st.mean([r["rating"] for r in R if r["bucket"] == b]),
                    "sd": st.pstdev([r["rating"] for r in R if r["bucket"] == b]),
                    "frac_problematic": st.mean(
                        [1.0 if r["rating"] >= PROBLEM_AT else 0.0
                         for r in R if r["bucket"] == b])}
                for b in ("top", "random", "bottom")}
            for tag, sel in (("pooled_240", R),
                             ("random_80", [r for r in R if r["bucket"] == "random"])):
                lab = [r["rating"] >= PROBLEM_AT for r in sel]
                block[tag] = {
                    "n": len(sel), "n_problematic": sum(lab),
                    "auc_probe": auc([r["score"] for r in sel], lab),
                    "auc_lexical_preregistered": auc(
                        [sign * lex[r["id"]][feat] for r in sel], lab),
                    "auc_lexical_all": {
                        f: auc([lex[r["id"]][f] for r in sel], lab)
                        for f in ("n_words", "hedge_rate", "list_density")},
                    "mean_rating": st.mean([r["rating"] for r in sel]),
                }
            # rating vs probe score, as a plain correlation over the 240
            block["pearson_rating_vs_probe_pooled"] = pearson(
                [r["score"] for r in R], [float(r["rating"]) for r in R])
            block["pearson_rating_vs_probe_random80"] = pearson(
                [r["score"] for r in R if r["bucket"] == "random"],
                [float(r["rating"]) for r in R if r["bucket"] == "random"])

            # ---- THE discriminating null ----------------------------------
            # "Beats the lexical baseline" is not "beats a random direction in
            # the same space".  Every judged item carries a score for all 27
            # directions, so the 20 Gaussian merges can be run against the SAME
            # judge labels at no extra cost.  If the probe does not outrank
            # them, the result is a property of the subspace, not of the probe.
            for tag, sel in (("pooled_240", R),
                             ("random_80", [r for r in R if r["bucket"] == "random"])):
                lab = [r["rating"] >= PROBLEM_AT for r in sel]
                if not any(lab) or all(lab):
                    continue
                nulls = {}
                for n in names:
                    if n == f"probe_{p}":
                        continue
                    nulls[n] = auc([sc[r["id"]][idx[n]] for r in sel], lab)
                rand_only = {k: v for k, v in nulls.items()
                             if k.startswith("rand_merge_") and v is not None}
                a_probe = block[tag]["auc_probe"]
                block[tag]["auc_null_directions"] = nulls
                block[tag]["n_rand_merges_beating_probe"] = sum(
                    1 for v in rand_only.values() if v >= a_probe)
                block[tag]["rand_merge_auc_max"] = max(rand_only.values())
                block[tag]["rand_merge_auc_mean"] = st.mean(rand_only.values())
                block[tag]["probe_rank_among_all_directions"] = 1 + sum(
                    1 for v in nulls.values() if v is not None and v >= a_probe)
                block[tag]["n_directions_ranked"] = 1 + sum(
                    1 for v in nulls.values() if v is not None)

            # ---- what is actually in each bucket ---------------------------
            block["bucket_composition"] = {
                bname: dict(sorted(
                    ((meta[i]["source_dataset"],
                      sum(1 for j in sel if meta[j]["source_dataset"]
                          == meta[i]["source_dataset"]))
                     for i in sel), key=lambda kv: -kv[1]))
                for bname, sel in buckets[p].items()}

            # ---- examples the lexical baseline would have missed -----------
            n_top = J["n_bucket"]
            lex_rank = sorted(ids, key=lambda i: -sign * lex[i][feat])
            lex_top = set(lex_rank[:n_top])
            block["lexical_flag_cutoff"] = {
                "feature": feat, "sign": sign, "n_flagged": n_top,
                "value_at_cutoff": sign * lex[lex_rank[n_top - 1]][feat]}
            cands = [r for r in R if r["bucket"] == "top"
                     and r["rating"] >= PROBLEM_AT and r["id"] not in lex_top]
            cands.sort(key=lambda r: (-r["rating"], -r["score"]))
            block["n_probe_top_problematic"] = len(
                [r for r in R if r["bucket"] == "top" and r["rating"] >= PROBLEM_AT])
            block["n_probe_top_problematic_lexically_unflagged"] = len(cands)
            block["examples"] = [{
                "id": r["id"], "source_dataset": meta[r["id"]]["source_dataset"],
                "probe_score": r["score"], "judge_rating": r["rating"],
                "judge_reason": r["reason"],
                "hedge_rate": lex[r["id"]]["hedge_rate"],
                "list_density": lex[r["id"]]["list_density"],
                "n_words": lex[r["id"]]["n_words"],
                "prompt": trim(items[r["id"]]["prompt"], 400),
                "response": trim(text[r["id"]], 900),
                "truncated_at_cap": meta[r["id"]]["truncated"],
            } for r in cands[:3]]
            per[p] = block
        out["per_probe"] = per

    # ---- cost -------------------------------------------------------------
    cost = {"meter_rate_usd_per_gpu_hour": 2.10,
            "meter_note": "zoo40_meter.sh RATE, the A100-40GB rate; this run "
                          "asked for A100-80GB, which Modal bills above it, so "
                          "the scorer figure is a FLOOR"}
    if os.path.exists(P(a.timing)):
        T = json.load(open(P(a.timing)))
        cost["timing"] = T
        big, small, sh0 = T.get("full"), T.get("smoke"), T.get("full_shard0")
        # The marginal rate must compare two runs that each paid the fixed cost
        # ONCE.  `full` is the sum of two shards and therefore carries two cold
        # starts; differencing it against the smoke run would fold half a fixed
        # cost into the per-item figure.  One shard against the smoke run is the
        # honest difference.
        if sh0 and small and sh0["n_items"] != small["n_items"]:
            marg = ((sh0["wall_s"] - small["wall_s"])
                    / (sh0["n_items"] - small["n_items"]))
            cost["marginal_gpu_s_per_item"] = marg
            cost["marginal_from"] = {"a": "full_shard0", "b": "smoke"}
            cost["fixed_gpu_s_per_container"] = sh0["wall_s"] - marg * sh0["n_items"]
            cost["scorer_usd_per_1000_marginal"] = marg * 1000 * 2.10 / 3600
        if sh0:
            cost["scorer_gpu_s_per_item_one_shard"] = sh0["wall_s"] / sh0["n_items"]
            cost["scorer_usd_per_1000_one_shard"] = (
                sh0["wall_s"] / sh0["n_items"] * 1000 * 2.10 / 3600)
        if big:
            cost["scorer_gpu_s_per_item_all_in"] = big["wall_s"] / big["n_items"]
            cost["scorer_usd_per_1000_all_in"] = (
                big["wall_s"] / big["n_items"] * 1000 * 2.10 / 3600)
            cost["all_in_note"] = ("summed over two shards, so it carries TWO "
                                   "cold starts and two model loads; a single "
                                   "unsharded run would sit between this and "
                                   "the marginal figure")
    if "judge" in out:
        u = out["judge"]["usage"]
        n = out["judge"]["n_judged"]
        if u.get("openrouter_cost_usd") and n:
            cost["judge_usd_per_1000_one_probe"] = u["openrouter_cost_usd"] / n * 1000
            cost["judge_usd_per_1000_three_probes"] = (
                u["openrouter_cost_usd"] / n * 1000 * 3)
            cost["judge_note"] = ("OpenRouter's own reported cost for this run "
                                  "divided by items judged.  One item = one "
                                  "(example, probe) rating, so auditing a corpus "
                                  "for all three failure modes costs three times "
                                  "the per-1000 figure; the scorer's figure "
                                  "already covers all 27 directions at once.")
        cost["judge_tokens"] = u
    out["cost"] = cost

    # ---- what this run cost, end to end -----------------------------------
    # Every figure a reader might quote lands here rather than in prose.  The
    # Modal side is container wall-clock at the meter's A100-40GB rate, which is
    # a FLOOR: this run asked for A100-80GB, which Modal bills above it.
    T = json.load(open(P(a.timing))) if os.path.exists(P(a.timing)) else {}
    train_s = {"overhedging": 459, "padding": 642, "false_certainty": 633}
    oom_s = 567          # 00:01:24 -> 00:10:51, probescore.log.1788999087
    score_s = sum(v["wall_s"] for k, v in T.items() if k != "full")
    gpu_s = score_s + sum(train_s.values()) + oom_s
    out["spend"] = {
        "modal": {
            "scoring_container_s": score_s,
            "training_container_s": train_s,
            "abandoned_batch8_oom_container_s": oom_s,
            "total_container_s": gpu_s,
            "usd_at_meter_rate": gpu_s / 3600 * 2.10,
            "note": "container wall-clock x the meter's $2.10/h A100-40GB rate. "
                    "A FLOOR: every GPU call asked for A100-80GB, billed above "
                    "that rate. The meter is workspace-wide and cannot attribute "
                    "spend to one run while siblings are alive.",
        },
        "openrouter": {
            "constitutions_usd": 0.0153,       # constitutions.py, 3 calls
            "pairs_usd": 0.469,                # probepairs.log, 1600 calls
            "judge_usd": (out.get("judge", {}).get("usage", {})
                          .get("openrouter_cost_usd")),
            "note": "constitutions and pairs from each run's own printed total; "
                    "judge from OpenRouter's reported per-call cost.",
        },
        "adapter_build_amortised": {
            "usd": 0.0153 + 0.469 + sum(train_s.values()) / 3600 * 2.10,
            "note": "one-off cost of the three probe adapters: constitutions, "
                    "DPO pairs, training. Paid once, then reusable on any "
                    "corpus; not part of the per-1000 scoring figure.",
        },
    }
    o = out["spend"]["openrouter"]
    o["total_usd"] = (o["constitutions_usd"] + o["pairs_usd"] + (o["judge_usd"] or 0.0))
    _sc = out["cost"].get("scorer_usd_per_1000_marginal")
    _ju = out["cost"].get("judge_usd_per_1000_one_probe")
    if _sc and _ju and _ju > _sc:
        out["cost"]["adapter_breakeven_items"] = (
            out["spend"]["adapter_build_amortised"]["usd"] * 1000.0 / (_ju - _sc))
        out["cost"]["adapter_breakeven_note"] = (
            "items after which the one-off adapter build is repaid by the "
            "per-1000 saving over the judge, for ONE failure mode")
    out["spend"]["total_usd"] = (out["spend"]["modal"]["usd_at_meter_rate"]
                                 + o["constitutions_usd"] + o["pairs_usd"]
                                 + (o["judge_usd"] or 0.0))

    # ---- per-item scores, one line each ------------------------------------
    # The analysis JSON holds aggregates; this is the row-level artefact, so a
    # reader can re-derive any of them.  Named directions by name; the 20-merge
    # null band as its mean and sd rather than 20 columns.
    jl = os.path.join(Q, "phase10_runs", "probe_scores_sft.jsonl")
    named = [n for n in names if not n.startswith("rand_merge_")]
    rand_i = [idx[n] for n in names if n.startswith("rand_merge_")]
    judged = {}
    if "judge" in out:
        for r in json.load(open(P(a.judged)))["records"]:
            judged.setdefault(r["id"], []).append(
                {"probe": r["probe"], "bucket": r["bucket"],
                 "rating": r["rating"], "reason": r["reason"]})
    with open(jl, "w") as f:
        for i in ids:
            v = sc[i]
            rv = [v[j] for j in rand_i]
            f.write(json.dumps({
                "id": i,
                "source_dataset": meta[i]["source_dataset"],
                "domain": meta[i].get("domain"),
                "n_prompt_tok": meta[i]["n_prompt_tok"],
                "n_completion_tok_scored": meta[i]["n_completion_tok_scored"],
                "truncated_at_cap": meta[i]["truncated"],
                "scores": {n: v[idx[n]] for n in named},
                "rand_merge_mean": st.mean(rv),
                "rand_merge_sd": st.pstdev(rv),
                "lexical": lex[i],
                "judgements": judged.get(i, []),
            }, ensure_ascii=False) + "\n")
    out["per_item_scores_jsonl"] = "phase10_runs/probe_scores_sft.jsonl"
    print(f"wrote {jl}: {len(ids)} rows, "
          f"{sum(1 for i in ids if i in judged)} of them judged")

    json.dump(out, open(P(a.out), "w"), indent=1)
    print(f"wrote {a.out}")
    for p in PROBES:
        if "per_probe" in out:
            b = out["per_probe"][p]
            print(f"  {p:16s} bucket means "
                  + " ".join(f"{k}={b['mean_rating_by_bucket'][k]['mean']:.2f}"
                             for k in ("top", "random", "bottom"))
                  + f"  AUC probe pooled {b['pooled_240']['auc_probe']:.3f} "
                    f"rand80 {b['random_80']['auc_probe']}"
                  + f"  lex pooled {b['pooled_240']['auc_lexical_preregistered']:.3f}")


if __name__ == "__main__":
    main()
