#!/usr/bin/env python3
"""EXPERIMENT 1: is the persona's extra behavioural amplification the stage-two
REGISTER (the shared direction) or the stage-two RESIDUAL (everything else)?

Inputs
  phase10_runs/steer_results_s2register.json  raw generations (steer_fix.py)
  phase10_runs/judged_s2register.json         blind Big Five scores (judge_personas.py)
  phase10_runs/eval_s2register_conditions.json  the label key
  traits_primary.json                          each trait's factor and keying
  phase10_runs/judged_100.json                 the existing stage-one / persona eval,
                                               used only as an external check on cond_b
Output
  analysis/stage2_register_vs_residual.json

Judged shift is on build_spider_data.py's scale, stated not fitted: the move from
base as a share of the room left on the 1-7 judge scale, (x-base)/(7-base) upward
and (x-base)/(base-1) downward, times 100.  "Own-factor amplification" is that
shift on the trait's own Big Five factor, signed by the trait's keying, so a
positive number always means "more like the trait".

Text statistics are the same five as analyse_s2mean_steer.py.
"""
import json
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
DIMS = ["Extraversion", "Agreeableness", "Conscientiousness",
        "EmotionalStability", "Intellect"]
ORDER = ["cond_a", "cond_b", "cond_c", "cond_c2", "cond_d", "cond_e"]


def stats(texts):
    """Byte-identical to analyse_s2mean_steer.stats, so the two runs compare."""
    n = len(texts)
    w = [len(t.split()) for t in texts]

    def rate(pat):
        return sum(len(re.findall(pat, t)) for t in texts) / max(1, sum(w)) * 1000

    return dict(words=round(st.mean(w)),
                first_person_per_k=round(rate(r"\b(I|I'm|I've|I'd|me|my|myself)\b"), 1),
                second_person_per_k=round(rate(r"\b(you|your|you're|yourself)\b"), 1),
                markdown_frac=round(sum(1 for t in texts if re.search(r"^#{1,4} |^\* |^- |\*\*", t, re.M)) / n, 2),
                embodied_frac=round(sum(1 for t in texts if re.match(r"\s*(I |I'm|I've|I'd|My |Honestly|Okay,? so I|Oh)", t)) / n, 2),
                introspective_frac=round(sum(1 for t in texts if re.search(r"\b(I (feel|tend|think|believe|notice|find myself|am someone)|as someone who|for me,|personally)\b", t, re.I)) / n, 2),
                uniq_ratio=round(st.mean(len(set(t.split())) / max(1, len(t.split())) for t in texts), 2),
                ends_clean=round(sum(1 for t in texts if t.strip()[-1:] in ".!?\"'") / n, 2))


def pct(x, base):
    return 100 * ((x - base) / (7 - base) if x >= base else (x - base) / (base - 1))


def mean_profile(recs):
    out = {}
    for d in DIMS:
        v = [r["scores"][d] for r in recs if r.get("scores") and r["scores"].get(d) is not None]
        out[d] = st.mean(v) if v else None
    return out


def main():
    P = f"{HERE}/phase10_runs"
    R = json.load(open(f"{P}/steer_results_s2register.json"))
    J = json.load(open(f"{P}/judged_s2register.json"))
    KEY = json.load(open(f"{P}/eval_s2register_conditions.json"))
    prim = {r["trait"].lower().replace(" ", "_").replace("-", "_"): r
            for r in json.load(open(f"{HERE}/traits_primary.json"))}

    recs = J["records"]
    conds = sorted({r["condition"] for r in recs}, key=lambda c: ORDER.index(c))
    traits = sorted({r["trait"] for r in recs if r["trait"] != "_base"})

    base_prof = mean_profile([r for r in recs if r["condition"] == "cond_a"])

    # ---- judged own-factor amplification, per trait per condition ------------
    per_trait, per_cond = {}, {c: [] for c in conds if c != "cond_a"}
    for t in traits:
        f = prim[t]["factor"].replace(" ", "")
        sgn = 1.0 if prim[t]["keyed"] == "+" else -1.0
        row = {"factor": f, "keyed": prim[t]["keyed"], "conditions": {}}
        for c in conds:
            if c == "cond_a":
                continue
            rs = [r for r in recs if r["trait"] == t and r["condition"] == c]
            if not rs:
                continue
            prof = mean_profile(rs)
            amp = sgn * pct(prof[f], base_prof[f])
            row["conditions"][c] = {"n": len(rs), "own_factor_mean": prof[f],
                                    "own_factor_amplification": amp,
                                    "profile": prof}
            per_cond[c].append(amp)
        per_trait[t] = row

    summary = {c: {"n_traits": len(v), "mean_amplification": st.mean(v),
                   "sd": st.pstdev(v) if len(v) > 1 else 0.0,
                   "min": min(v), "max": max(v),
                   "n_positive": sum(1 for x in v if x > 0)}
               for c, v in per_cond.items() if v}

    # ---- does the register alone reach the persona? -------------------------
    gaps = {}
    if "cond_b" in summary and "cond_e" in summary:
        b, e = summary["cond_b"]["mean_amplification"], summary["cond_e"]["mean_amplification"]
        for c in [x for x in ("cond_c", "cond_c2", "cond_d") if x in summary]:
            x = summary[c]["mean_amplification"]
            gaps[c] = {"amplification": x,
                       "share_of_the_stage1_to_persona_gap":
                           (x - b) / (e - b) if abs(e - b) > 1e-9 else None,
                       "per_trait_share": [
                           ((per_trait[t]["conditions"][c]["own_factor_amplification"]
                             - per_trait[t]["conditions"]["cond_b"]["own_factor_amplification"])
                            / (per_trait[t]["conditions"]["cond_e"]["own_factor_amplification"]
                               - per_trait[t]["conditions"]["cond_b"]["own_factor_amplification"]))
                           for t in traits
                           if abs(per_trait[t]["conditions"]["cond_e"]["own_factor_amplification"]
                                  - per_trait[t]["conditions"]["cond_b"]["own_factor_amplification"]) > 1e-9]}
        for c in gaps:
            v = gaps[c].pop("per_trait_share")
            gaps[c]["median_per_trait_share"] = st.median(v) if v else None
            gaps[c]["n_traits_for_share"] = len(v)
        gaps["_endpoints"] = {"cond_b": b, "cond_e": e, "gap": e - b}

    # ---- paired contrasts against cond_b, and off-target movement -----------
    import math
    paired = {}
    for c in [x for x in ORDER if x in summary and x != "cond_b"]:
        d = [per_trait[t]["conditions"][c]["own_factor_amplification"]
             - per_trait[t]["conditions"]["cond_b"]["own_factor_amplification"]
             for t in traits if c in per_trait[t]["conditions"]]
        n = len(d)
        m = st.mean(d)
        sd = st.stdev(d) if n > 1 else 0.0
        pos = sum(1 for x in d if x > 0)
        # exact two-sided sign test
        pv = (sum(math.comb(n, k) for k in range(max(pos, n - pos), n + 1))
              * 2 / 2 ** n) if n else None
        paired[c] = {"n": n, "mean_minus_cond_b": m, "sd": sd,
                     "sem": sd / math.sqrt(n) if n > 1 else None,
                     "t": m / (sd / math.sqrt(n)) if n > 1 and sd > 0 else None,
                     "n_positive": pos, "sign_test_two_sided_p": min(1.0, pv)}
    # off-target: mean |shift| on the four factors that are not the trait's own
    off_target = {}
    for c in [x for x in ORDER if x in summary]:
        v = []
        for t in traits:
            if c not in per_trait[t]["conditions"]:
                continue
            f = per_trait[t]["factor"]
            prof = per_trait[t]["conditions"][c]["profile"]
            v.append(st.mean(abs(pct(prof[d_], base_prof[d_]))
                             for d_ in DIMS if d_ != f))
        off_target[c] = {"n": len(v), "mean_abs_shift_other_four": st.mean(v)}

    # ---- text statistics, per condition, pooled over traits -----------------
    texts = {c: [] for c in conds}
    lab = {}
    for j in R:
        rest = j["name"][len("s2reg_"):]
        kind = rest.split("_", 1)[0] if rest != "base" else "base"
        for a_str, g in j["generations"].items():
            key = f"{kind}|{float(a_str)}"
            c = {"base|0.0": "cond_a", "s1|0.0": "cond_b", "s1|0.389": "cond_c",
                 "s1|0.779": "cond_c2", "s1|1.0": "cond_d",
                 "pex|0.0": "cond_e"}[key]
            texts[c] += g
            lab[c] = key
    text_stats = {c: {"n_generations": len(v), "job_key": lab[c], **stats(v)}
                  for c, v in texts.items() if v}

    # ---- external check: cond_b against the existing stage-one eval ---------
    check = {}
    p100 = f"{P}/judged_100.json"
    if os.path.exists(p100):
        J100 = json.load(open(p100))["records"]
        b100 = mean_profile([r for r in J100 if r["condition"] == "base"])
        vals_old, vals_new = [], []
        for t in traits:
            f = prim[t]["factor"].replace(" ", "")
            sgn = 1.0 if prim[t]["keyed"] == "+" else -1.0
            rs = [r for r in J100 if r["trait"] == t and r["condition"] == "stage1"]
            if not rs or "cond_b" not in per_trait[t]["conditions"]:
                continue
            vals_old.append(sgn * pct(mean_profile(rs)[f], b100[f]))
            vals_new.append(per_trait[t]["conditions"]["cond_b"]["own_factor_amplification"])
        if len(vals_old) > 2:
            mo, mn = st.mean(vals_old), st.mean(vals_new)
            so, sn = st.pstdev(vals_old), st.pstdev(vals_new)
            r = (sum((a - mo) * (b - mn) for a, b in zip(vals_old, vals_new))
                 / (len(vals_old) * so * sn)) if so > 0 and sn > 0 else None
            check = {"what": ("cond_b here is the stage-one adapter folded into the bf16 "
                              "base weights; judged_100.json condition stage1 is the same "
                              "adapter loaded through PEFT at 200 new tokens. Agreement is "
                              "the check that the weight-merge mechanism is faithful."),
                     "n_traits": len(vals_old),
                     "mean_amplification_judged_100_stage1": mo,
                     "mean_amplification_cond_b": mn, "pearson_r": r}
            # the persona comparison, too: judged_100 'persona' is the PEFT
            # add_weighted_adapter merge, cond_e the corrected exact merge
            vo, vn = [], []
            for t in traits:
                f = prim[t]["factor"].replace(" ", "")
                sgn = 1.0 if prim[t]["keyed"] == "+" else -1.0
                rs = [r for r in J100 if r["trait"] == t and r["condition"] == "persona"]
                if not rs or "cond_e" not in per_trait[t]["conditions"]:
                    continue
                vo.append(sgn * pct(mean_profile(rs)[f], b100[f]))
                vn.append(per_trait[t]["conditions"]["cond_e"]["own_factor_amplification"])
            if len(vo) > 2:
                check["mean_amplification_judged_100_persona"] = st.mean(vo)
                check["mean_amplification_cond_e"] = st.mean(vn)
                check["note_persona"] = ("judged_100 'persona' is PEFT's linear "
                                         "add_weighted_adapter, which carries the cross term "
                                         "corrected in fix_persona_merge.py; cond_e is the "
                                         "exact dW_dpo + 0.25 dW_sft.")

    out = {"what": ("Four conditions on the 24-prompt Big Five battery for the 15 "
                    "second-seed traits: base, stage-one adapter, stage-one adapter "
                    "plus the stage-two grand mean at a persona's dose (and at two "
                    "other doses), and the exact persona. Asks whether the persona's "
                    "extra behavioural amplification is delivered by the shared "
                    "register direction alone."),
           "conditions": KEY["conditions"],
           "dose_arithmetic": KEY["dose_arithmetic"],
           "alpha_units_correction": (json.load(open(f"{HERE}/analysis/steer_alpha_units.json"))
                                      if os.path.exists(f"{HERE}/analysis/steer_alpha_units.json")
                                      else None),
           "judge": {"model": J.get("model"), "n_records": J.get("n"),
                     "failed_calls": J.get("failed_calls")},
           "base_profile": base_prof,
           "summary_own_factor_amplification": summary,
           "register_vs_persona_gap": gaps,
           "paired_contrasts_vs_cond_b": paired,
           "off_target_movement": off_target,
           "judge_reliability_note": ("Repeat reliability of the judge on this run, from the "
                                      "5 percent duplicated units, is r 0.879 E, 0.856 A, 0.891 C, "
                                      "0.766 ES, 0.889 I over n = 73 "
                                      "(phase10_runs/judge_s2register.log). The project's "
                                      "base-versus-base judge disagreement on identical text is "
                                      "0.2917 raw points (analysis/sorh_behavioural.json, wiki "
                                      "log 2026-09-08)."),
           "text_statistics": text_stats,
           "mechanism_check_vs_judged_100": check,
           "per_trait": per_trait}
    p = f"{HERE}/analysis/stage2_register_vs_residual.json"
    json.dump(out, open(p, "w"), indent=1)

    print(f"base profile: " + "  ".join(f"{d[:4]} {base_prof[d]:.2f}" for d in DIMS))
    print(f"{'cond':>8s} {'amp':>8s} {'sd':>7s} {'pos':>5s}  {'1st':>5s} {'2nd':>5s} "
          f"{'md':>5s} {'char':>5s} {'words':>6s}")
    for c in ORDER:
        if c in summary:
            s, tx = summary[c], text_stats.get(c, {})
            print(f"{c:>8s} {s['mean_amplification']:>+8.2f} {s['sd']:>7.2f} "
                  f"{s['n_positive']:>3d}/{s['n_traits']:<2d} "
                  f"{tx.get('first_person_per_k', 0):>5.1f} {tx.get('second_person_per_k', 0):>5.1f} "
                  f"{tx.get('markdown_frac', 0):>5.2f} {tx.get('embodied_frac', 0):>5.2f} "
                  f"{tx.get('words', 0):>6d}")
        elif c in text_stats:
            tx = text_stats[c]
            print(f"{c:>8s} {'-':>8s} {'-':>7s} {'-':>5s}  "
                  f"{tx['first_person_per_k']:>5.1f} {tx['second_person_per_k']:>5.1f} "
                  f"{tx['markdown_frac']:>5.2f} {tx['embodied_frac']:>5.2f} {tx['words']:>6d}")
    for c, g in gaps.items():
        if c != "_endpoints":
            print(f"  {c}: {g['share_of_the_stage1_to_persona_gap']!r} of the "
                  f"stage1->persona gap (median per trait "
                  f"{g['median_per_trait_share']!r})")
    for c, v in paired.items():
        print(f"  {c} - cond_b: {v['mean_minus_cond_b']:+.2f} +- {v['sem'] or 0:.2f} "
              f"(t {v['t']}), {v['n_positive']}/{v['n']} positive, sign p "
              f"{v['sign_test_two_sided_p']:.4f}")
    for c, v in off_target.items():
        print(f"  {c} off-target mean |shift| on the other four: "
              f"{v['mean_abs_shift_other_four']:.2f}")
    if check:
        print(f"  mechanism check: judged_100 stage1 {check['mean_amplification_judged_100_stage1']:+.2f} "
              f"vs cond_b {check['mean_amplification_cond_b']:+.2f}, r {check['pearson_r']}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
