#!/usr/bin/env python3
"""Render the wiki page's tables straight out of analysis/dolci_flag_training.json.

The wiki's first rule is that every number is quoted verbatim from a file and
never recomputed or re-rounded by hand. This prints the markdown so the page can
be assembled by copying, with no number passing through a person.
"""
import json
import os

Q = os.path.dirname(os.path.abspath(__file__))
ARMS = ["flagged", "random", "anti", "unfiltered", "filtered"]
CONDS = ["base"] + ARMS
FACTORS = ["Extraversion", "Agreeableness", "Conscientiousness",
           "EmotionalStability", "Intellect"]
d = json.load(open(f"{Q}/analysis/dolci_flag_training.json"))


def fmt(v, n=4, sign=True):
    if v is None:
        return "—"
    return f"{v:+.{n}f}" if sign else f"{v:.{n}f}"


print("### TRAINING\n")
print("| arm | pairs | steps | first loss | last loss | KL term, last | A copied | A drift |")
print("|---|---|---|---|---|---|---|---|")
for a in ARMS:
    t = d["training"].get(a, {})
    g = t.get("gate_a_drift") or {}
    print(f"| `{a}` | {t.get('n_pairs')} | {t.get('optimizer_steps')} | "
          f"{fmt(t.get('loss_first'), 4, False)} | {fmt(t.get('loss_last'), 4, False)} | "
          f"{fmt(t.get('kl_term_last'), 6, False)} | {t.get('n_lora_A_copied')}/248 | "
          f"{fmt(g.get('a_drift_mean'), 4, False)} |")

if "compliance" in d:
    print("\n### COMPLIANCE BATTERY\n")
    print("| condition | complied, 40 should-refuse | engaged at all | over-refused, 20 benign "
          "| fabrication, 40 | fabrication given engaged | quality, 40 | quality, 20 |")
    print("|---|---|---|---|---|---|---|---|")
    for c in CONDS:
        v = d["compliance"].get(c)
        if not v or not v["n_should_refuse"]:
            continue
        print(f"| `{c}` | {fmt(v['compliance_rate'],4,False)} | "
              f"{fmt(v['any_engagement_rate'],4,False)} | "
              f"{fmt(v['over_refusal_rate'],4,False)} | "
              f"{fmt(v['fabrication_mean_should_refuse'],4,False)} | "
              f"{fmt(v['fabrication_mean_given_engaged'],4,False)} "
              f"(n {v['n_engaged']}) | {fmt(v['quality_mean_should_refuse'],4,False)} | "
              f"{fmt(v['quality_mean_benign'],4,False)} |")

    print("\n**Against RANDOM**, paired by prompt, sign-flip permutation, "
          f"{d['compliance_tests']['_perm']['n_draws']} draws "
          f"(floor {d['compliance_tests']['_perm']['two_sided_floor']}):\n")
    keys = ["compliance_rate", "any_engagement_rate", "over_refusal_rate",
            "fabrication_should_refuse", "quality_should_refuse", "quality_benign"]
    print("| arm | " + " | ".join(f"`{k}`" for k in keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for c in CONDS:
        t = d["compliance_tests"].get(f"{c}_vs_random")
        if not t or not t["compliance_rate"]["n_paired"]:
            continue
        print(f"| `{c}` | " + " | ".join(
            f"{t[k]['diff']:+.4f} (p {t[k]['p']:.5f})" for k in keys) + " |")

    print("\n**Against BASE**:\n")
    print("| arm | " + " | ".join(f"`{k}`" for k in keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for c in ARMS:
        t = d["compliance_tests"].get(f"{c}_vs_base")
        if not t or not t["compliance_rate"]["n_paired"]:
            continue
        print(f"| `{c}` | " + " | ".join(
            f"{t[k]['diff']:+.4f} (p {t[k]['p']:.5f})" for k in keys) + " |")

    print("\n**Compliance rate by motif** (12 fabrication, 10 injection, 10 fiction, "
          "8 deception):\n")
    motifs = sorted({m for c in CONDS if d["compliance"].get(c)
                     for m in d["compliance"][c]["by_motif_compliance_rate"]})
    print("| condition | " + " | ".join(motifs) + " |")
    print("|---" * (len(motifs) + 1) + "|")
    for c in CONDS:
        v = d["compliance"].get(c)
        if not v or not v["n_should_refuse"]:
            continue
        print(f"| `{c}` | " + " | ".join(
            f"{v['by_motif_compliance_rate'].get(m, float('nan')):.4f}" for m in motifs) + " |")

    r = d["compliance_judge"].get("reliability", {})
    print("\n**Units judged per condition** (of 40 and 20):\n")
    print("| condition | should-refuse | benign |")
    print("|---|---|---|")
    for c in CONDS:
        v = d["compliance"].get(c)
        if v and v["n_should_refuse"]:
            print(f"| `{c}` | {v['n_should_refuse']} | {v['n_benign']} |")
    print(f"\nJudge `{d['compliance_judge']['model']}`, {d['compliance_judge']['n']} units, "
          f"{d['compliance_judge']['failed_calls']} failed calls; repeats n "
          f"{r.get('n_repeats')}, compliance exact agreement "
          f"{r.get('compliance_exact_agreement')}, fabrication r {r.get('fabrication_r')}, "
          f"quality r {r.get('quality_r')}.")

if "bigfive" in d:
    print("\n### BIG FIVE (24 prompts, judge_personas.py)\n")
    print("| condition | " + " | ".join(FACTORS) + " |")
    print("|---" * (len(FACTORS) + 1) + "|")
    for c in CONDS:
        m = d["bigfive"]["means"].get(c)
        if not m:
            continue
        print(f"| `{c}` | " + " | ".join(f"{m[f]:.4f}" for f in FACTORS) + " |")
    print("\n**Shift against base**, paired by prompt:\n")
    print("| arm | " + " | ".join(FACTORS) + " |")
    print("|---" * (len(FACTORS) + 1) + "|")
    for a in ARMS:
        s = d["bigfive"]["shift_vs_base"].get(a)
        if not s:
            continue
        print(f"| `{a}` | " + " | ".join(
            f"{s[f]['shift_vs_base']:+.4f} (p {s[f]['p']:.5f})" for f in FACTORS) + " |")
    print("\n**Difference from RANDOM**, paired by prompt:\n")
    print("| arm | " + " | ".join(FACTORS) + " |")
    print("|---" * (len(FACTORS) + 1) + "|")
    for a in ARMS:
        s = d["bigfive"]["diff_vs_random"].get(a)
        if not s:
            continue
        print(f"| `{a}` | " + " | ".join(
            f"{s[f]['diff_vs_random']:+.4f} (p {s[f]['p']:.5f})" for f in FACTORS) + " |")

ws = d.get("weight_space", {})
if ws.get("cosine_with_alignment_adapters"):
    print("\n### WEIGHT SPACE\n")
    al = ["corrigible", "sycophantic", "obsequious", "power_seeking"]
    print("| arm | norm | " + " | ".join(f"cos `{x}`" for x in al) + " |")
    print("|---" * (len(al) + 2) + "|")
    for a in ARMS:
        r = ws["cosine_with_alignment_adapters"].get(a)
        if not r:
            continue
        n = ws.get("adapter_norms", {}).get(a)
        print(f"| `{a}` | {fmt(n,4,False)} | " + " | ".join(f"{r[x]:+.4f}" for x in al) + " |")
if ws.get("cosine_between_arms"):
    print("\n**Between arms**:\n")
    print("| | " + " | ".join(f"`{a}`" for a in ARMS) + " |")
    print("|---" * (len(ARMS) + 1) + "|")
    for a in ARMS:
        row = ws["cosine_between_arms"].get(a, {})
        print(f"| `{a}` | " + " | ".join(f"{row.get(b, float('nan')):+.4f}" for b in ARMS) + " |")
if ws.get("factor_chart"):
    print("\n**Factor chart** (`fa_chart.FAChart().coords_external`, basis order "
          f"{ws['factor_chart'][ARMS[0]]['factor_order']}); "
          f"the 134 stage-one adapters' mean chart length is "
          f"{ws['chart_reference']['trait_chart_len_mean']} and their mean norm "
          f"{ws['chart_reference']['trait_norm_mean']}:\n")
    fo = ws["factor_chart"][ARMS[0]]["factor_order"]
    print("| arm | " + " | ".join(f.replace("FA_", "") for f in fo)
          + " | chart length | cos with the chart |")
    print("|---" * (len(fo) + 3) + "|")
    for a in ARMS:
        c = ws["factor_chart"].get(a)
        if not c:
            continue
        print(f"| `{a}` | " + " | ".join(f"{v:+.4f}" for v in c["coords"])
              + f" | {c['chart_len']:.4f} | {c['cos_with_chart']:.4f} |")
    print("\n**Nearest zoo traits by absolute cosine**:\n")
    for a in ARMS:
        n = ws["nearest_zoo_traits_by_abs_cosine"].get(a)
        if n:
            print(f"- `{a}`: " + ", ".join(f"{t} {c:+.4f}" for t, c in n[:6]))

print("\n### ARM FIRST-ORDER SCORES (the forecast)\n")
sc = d["arm_first_order_scores"]["arms"]
want = ["align_corrigible", "align_sycophantic", "align_obsequious", "align_power_seeking",
        "axis_Agreeableness", "axis_Conscientiousness", "axis_Extraversion",
        "axis_EmotionalStability", "axis_Intellect", "mean_assistant_axis"]
print("| direction | " + " | ".join(f"`{a}`" for a in ARMS) + " |")
print("|---" * (len(ARMS) + 1) + "|")
for w in want:
    print(f"| `{w}` | " + " | ".join(f"{sc[a][w]:+.6f}" for a in ARMS) + " |")


if d.get("prereg"):
    print("\n### PRE-REGISTERED PREDICTIONS\n")
    for k, v in d["prereg"].items():
        print(f"- **{k}** held = `{v['held']}`"
              + (f", at p<0.05 = `{v['held_at_p05']}`" if "held_at_p05" in v else "")
              + f"\n  - predicted: {v['prediction']}\n  - observed: {json.dumps(v['observed'])}")


if d.get("post_hoc"):
    print("\n### POST HOC, NOT PRE-REGISTERED\n")
    for k, v in d["post_hoc"].items():
        print(f"- **{k}** held = `{v['held']}`\n  - {v['note']}"
              f"\n  - predicted: {v['prediction']}\n  - observed: {json.dumps(v['observed'])}"
              f"\n  - corpus axis_Agreeableness: {json.dumps(v['corpus_axis_Agreeableness'])}")
