#!/usr/bin/env python3
"""Suppression against amplification, at matched Fisher dose.

Experiment G5 of the 2026-09-09 paper reading.  The published steering table
compares alpha +2 with alpha -2, and Gradient Atoms (Rosser, 2026) reads the
resulting asymmetry -- four of five factors suppress harder than they amplify --
as a structural fact about model editing: one pathway to break a behaviour, many
to strengthen it.  This project had a second explanation, a judged ceiling, and
a third possibility nobody had ruled out: alpha is not a dose.

Here the dose is matched.  Each direction and each SIGN is steered at the alpha
that delivers the same measured KL per token (solve_matched_alphas.py from
fisher_dose.py's measured curve), the 24-prompt battery is regenerated, and the
same blind Big Five judge scores it beside a base condition generated in the
same run and judged in the same interleaved batch.

Reads   phase10_runs/judged_dose.json          this run, judged
        analysis/matched_dose_alphas.json      the alphas and their measured KL
        phase10_runs/steer_results_dose.json   the generations (for loop rates)
        phase10_runs/judged_steerfix23.json    the published equal-alpha arm (FA)
        phase10_runs/judged_steerfix.json      the published equal-alpha arm (axes)
Writes  analysis/matched_dose_steering.json

WHAT IS COMPARED, AND THE CONVENTION.  "Own scale" is the Big Five scale each
direction is named for.  A direction AMPLIFIES on the sign that raises its own
scale and SUPPRESSES on the sign that lowers it -- read from the data, not
assumed, and reported explicitly, because for FA_FearfulWithdrawal the
amplifying sign of the *name* is the suppressing sign of Emotional Stability.
Three columns per (direction, sign):

  raw        the signed change in the own scale against base, on the 1-7 scale
  room       the same change as a share of the room that existed in that
             direction: delta / (7 - base) upward, delta / (base - 1) downward.
             This is what separates a ceiling from a mechanism, which is the
             question the experiment exists to answer.
  off        mean |change| over the other four scales, the off-target movement
"""
import collections
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "analysis")
P = os.path.join(HERE, "phase10_runs")
F5 = ["Extraversion", "Agreeableness", "Conscientiousness",
      "EmotionalStability", "Intellect"]
OWN = {"FA_Warmth": "Agreeableness", "FA_Competence": "Conscientiousness",
       "FA_FearfulWithdrawal": "EmotionalStability", "FA_Arousal": "Extraversion",
       "FA_Imagination": "Intellect",
       "axis_Extraversion": "Extraversion", "axis_Agreeableness": "Agreeableness",
       "axis_Conscientiousness": "Conscientiousness",
       "axis_EmotionalStability": "EmotionalStability",
       "axis_Intellect": "Intellect"}


def looping(t, n=10, k=4):
    """A response loops if some n-word window repeats at least k times.

    Copied from analyse_alien_steer.py so the loop rates here are the same
    statistic analysis/fisher_norms.json#comparisons.sign_asymmetry_of_degeneration
    reports for the published directions.
    """
    w = t.split()
    if len(w) < n * 2:
        return False
    g = {}
    for i in range(len(w) - n + 1):
        s = " ".join(w[i:i + n])
        g[s] = g.get(s, 0) + 1
        if g[s] >= k:
            return True
    return False


def load_judged(path):
    """{(direction, condition): {prompt_idx: {factor: score}}} from a judged file.

    Kept per PROMPT, not averaged, because base and steered see the same 24
    prompts and the paired difference is the estimate with power: the
    prompt-to-prompt spread of the judge's scores is several times the effect
    being measured, and it cancels exactly in the pairing.
    """
    J = json.load(open(path))
    out = collections.defaultdict(dict)
    for r in J["records"]:
        sc = {f: r["scores"][f] for f in F5
              if isinstance(r["scores"].get(f), (int, float))}
        if sc:
            out[(r["trait"], r["condition"])][int(r["prompt_idx"])] = sc
    return dict(out), J["model"], J["n"], J["failed_calls"]


def cell_mean(cell):
    return {f: float(np.mean([s[f] for s in cell.values() if f in s]))
            for f in F5 if any(f in s for s in cell.values())}


def row(base_cell, steer_cell, own, boot=10000, seed=7):
    """Paired-by-prompt movement of the own scale and of the other four."""
    idx = sorted(set(base_cell) & set(steer_cell))
    base, steer = cell_mean(base_cell), cell_mean(steer_cell)
    d = {f: steer[f] - base[f] for f in F5}
    delta = d[own]
    pairs = np.array([steer_cell[i][own] - base_cell[i][own] for i in idx
                      if own in steer_cell[i] and own in base_cell[i]], float)
    rng = np.random.default_rng(seed)
    bs = rng.choice(pairs, size=(boot, len(pairs)), replace=True).mean(1) \
        if len(pairs) > 1 else np.array([delta])
    room = (7.0 - base[own]) if delta >= 0 else (base[own] - 1.0)
    off = [abs(d[f]) for f in F5 if f != own]
    return {"base_own": base[own], "steered_own": steer[own],
            "delta_own": delta,
            "delta_own_paired_mean": float(pairs.mean()),
            "delta_own_paired_sd": float(pairs.std(ddof=1)) if len(pairs) > 1 else None,
            "delta_own_paired_se": float(pairs.std(ddof=1) / np.sqrt(len(pairs)))
                                   if len(pairs) > 1 else None,
            "delta_own_ci95": [float(np.percentile(bs, 2.5)),
                               float(np.percentile(bs, 97.5))],
            "n_paired_prompts": int(len(pairs)),
            "room_available": room,
            "delta_own_as_share_of_room": (delta / room) if room > 1e-9 else None,
            "off_target_mean_abs": float(np.mean(off)),
            "off_target_max_abs": float(max(off)),
            "selectivity": (abs(delta) / float(np.mean(off))
                            if np.mean(off) > 1e-9 else None),
            "delta_all": d}


def main():
    AL = json.load(open(f"{A}/matched_dose_alphas.json"))
    M, model, njudged, nfail = load_judged(f"{P}/judged_dose.json")
    # this run's conditions are neutral labels; the map back is in the spec
    CM = json.load(open(f"{P}/dose_conditions.json"))    # {direction: {cond: alpha}}

    gens = {r["name"].removeprefix("dose_"): r
            for r in json.load(open(f"{P}/steer_results_dose.json"))}
    CAL = json.load(open(f"{P}/dose_calib.json"))

    out = {"what": "own-scale and off-target judged movement at matched Fisher "
                   "dose, both signs, against the published equal-alpha table",
           "judge_model": model, "n_judged": njudged, "failed_judge_calls": nfail,
           "dose": AL["dose"], "kl_target_nats_per_token": AL["kl_target_nats_per_token"],
           "weight_construction": AL["weight_construction_matched_on"],
           "bf16_retention": AL["bf16_retention"],
           "own_scale_map": OWN,
           "convention": "amplify = the sign that raises the direction's own "
                         "judged scale; suppress = the sign that lowers it. Read "
                         "from the data per direction, not assumed."}

    # ---- what the generating dtype actually delivers ----------------------
    # steer_fix.py writes the increment into bf16 parameters.  The calibration
    # measured every alpha twice, once with the weights set exactly in fp32 and
    # once rounded to bf16 from a clean base, so the two questions separate:
    # does the perturbation keep its LENGTH, and does it keep its EFFECT?
    cells, byabs = [], collections.defaultdict(list)
    for nm, r in CAL["results"].items():
        per = r["per_alpha"]
        for a in r["alphas"]:
            e, b = per.get(f"exact|{a}"), per.get(f"bf16|{a}")
            if not e or not b:
                continue
            cells.append(b["kl"] / e["kl"])
            byabs[abs(a)].append(b["retention"])
    rets = [x for v in byabs.values() for x in v]
    out["bf16_vs_exact"] = {
        "what": "at the same nominal alpha, the bf16 weights steer_fix.py writes "
                "against the exactly-set fp32 weights fisher.py measures: the "
                "Frobenius length of the realised perturbation (retention) and "
                "the KL per token it delivers",
        "source": "phase10_runs/dose_calib.json#results.*.per_alpha",
        "n_cells": len(cells),
        "kl_bf16_over_exact": {"min": float(np.min(cells)),
                               "median": float(np.median(cells)),
                               "max": float(np.max(cells)),
                               "mean": float(np.mean(cells))},
        "retention": {"min": float(np.min(rets)), "median": float(np.median(rets)),
                      "max": float(np.max(rets))},
        "retention_median_by_abs_alpha": {str(a): float(np.median(v))
                                          for a, v in sorted(byabs.items())},
        "kl_median_by_abs_alpha": {
            str(a): float(np.median([
                CAL["results"][nm]["per_alpha"][f"bf16|{s_}{a}"]["kl"]
                / CAL["results"][nm]["per_alpha"][f"exact|{s_}{a}"]["kl"]
                for nm in CAL["results"] for s_ in ("-", "")
                if f"bf16|{s_}{a}" in CAL["results"][nm]["per_alpha"]]))
            for a in sorted(byabs)},
        "reading": "the length survives and the effect does not: quantisation "
                   "keeps the norm and spends part of it on isotropic noise, so "
                   "every published steering alpha delivered about nine tenths "
                   "of its intended dose"}

    base_cond = [c for (t, c) in M if t == "BASE"]
    if len(base_cond) != 1:
        raise SystemExit(f"expected one BASE condition, found {base_cond}")
    base = M[("BASE", base_cond[0])]
    out["base"] = {"condition": base_cond[0], "scores": cell_mean(base),
                   "n_prompts": len(base),
                   "note": "generated in this run with an empty direction and "
                           "alpha 0, judged interleaved with the steered text"}

    # ---- published equal-alpha comparators --------------------------------
    pub = {}
    for path in (f"{P}/judged_steerfix23.json", f"{P}/judged_steerfix.json"):
        Mp, pm, _, _ = load_judged(path)
        for (t, c) in Mp:
            pub.setdefault(t, {})[c] = Mp[(t, c)]
        out.setdefault("published_judge_models", []).append(
            {"file": os.path.basename(path), "model": pm})

    rows = {}
    for name, info in AL["directions"].items():
        conds = CM[name]
        r = {"family": info["family"],
             "fisher_dose_per_alpha_published": info["fisher_dose_per_alpha_published"],
             "matched": {}, "published_equal_alpha": {}}
        for sgn in ("neg", "pos"):
            a = info["matched"][sgn]["alpha"]
            cond = conds[str(a)]
            if (name, cond) not in M:
                raise SystemExit(f"{name} {cond} missing from judged_dose.json")
            rr = row(base, M[(name, cond)], OWN[name])
            rr["alpha"] = a
            rr["kl_bracket"] = info["matched"][sgn]["kl_at_bracket"]
            rr["extrapolated"] = info["matched"][sgn]["extrapolated"]
            rr["n_prompts_judged"] = len(M[(name, cond)])
            g = gens[name]["generations"][str(a)]
            rr["loop_rate"] = float(np.mean([looping(x) for x in g]))
            rr["mean_response_words"] = float(np.mean([len(x.split()) for x in g]))
            r["matched"][sgn] = rr
        # published: alpha +/- 2 against that campaign's own alpha-0 row
        if name in pub and "a0_0" in pub[name]:
            pb = pub[name]["a0_0"]
            for sgn, c in (("neg", "am2_0"), ("pos", "a2_0")):
                if c in pub[name]:
                    pr = row(pb, pub[name][c], OWN[name])
                    pr["alpha"] = -2.0 if sgn == "neg" else 2.0
                    pr["kl_measured_here"] = info["kl_at_equal_alpha_2"].get(
                        "-2.0" if sgn == "neg" else "2.0", {}).get("kl")
                    r["published_equal_alpha"][sgn] = pr
        # which sign amplifies
        dn = r["matched"]["neg"]["delta_own"]
        dp = r["matched"]["pos"]["delta_own"]
        amp = "pos" if dp >= dn else "neg"
        sup = "neg" if amp == "pos" else "pos"
        r["amplifying_sign"] = amp
        r["both_signs_move_own_scale_the_same_way"] = bool(dn * dp > 0)
        r["bipolar"] = bool(dn * dp < 0)
        r["amplify"] = r["matched"][amp]
        r["suppress"] = r["matched"][sup]
        r["asymmetry_raw"] = abs(r["suppress"]["delta_own"]) - abs(r["amplify"]["delta_own"])
        sa, sb = (r["suppress"]["delta_own_as_share_of_room"],
                  r["amplify"]["delta_own_as_share_of_room"])
        r["asymmetry_room"] = (abs(sa) - abs(sb)) if (sa is not None and sb is not None) else None
        rows[name] = r
    out["directions"] = rows

    # ---- why the small-alpha fit could not simply be extrapolated ---------
    FN = json.load(open(f"{A}/fisher_norms.json"))["directions"]
    tay = {}
    for nm in rows:
        d = FN[nm]
        F, c3, d4 = d["F_ref"], d["cubic_c"], d["quartic_d"]
        pred = 0.5 * F * 4.0 + c3 * 8.0 + d4 * 16.0
        meas = rows[nm]["published_equal_alpha"] and \
            AL["directions"][nm]["kl_at_equal_alpha_2"]["2.0"]["kl"]
        tay[nm] = {"taylor_kl_at_alpha_2": pred, "measured_kl_at_alpha_2": meas,
                   "taylor_over_measured": (pred / meas) if meas else None}
    out["taylor_extrapolation_check"] = {
        "what": "KL at alpha 2 as the third-and-fourth-order Taylor fit of "
                "analysis/fisher_norms.json predicts it (0.5*F*4 + c*8 + d*16, "
                "from #directions.<name>.F_ref / .cubic_c / .quartic_d) against "
                "the value measured here on the bf16 construction",
        "why": "that fit was made on six alphas inside |alpha| <= 0.5; this is "
               "four times outside it, and the check is why every alpha in this "
               "experiment is interpolated between measured points instead",
        "ratio_min": min(v["taylor_over_measured"] for v in tay.values()),
        "ratio_median": float(np.median([v["taylor_over_measured"] for v in tay.values()])),
        "ratio_max": max(v["taylor_over_measured"] for v in tay.values()),
        "per_direction": tay}

    # ---- the headline -----------------------------------------------------
    def summarise(keys, label):
        sup = [rows[k]["suppress"]["delta_own"] for k in keys]
        amp = [rows[k]["amplify"]["delta_own"] for k in keys]
        supr = [rows[k]["suppress"]["delta_own_as_share_of_room"] for k in keys]
        ampr = [rows[k]["amplify"]["delta_own_as_share_of_room"] for k in keys]
        supo = [rows[k]["suppress"]["off_target_mean_abs"] for k in keys]
        ampo = [rows[k]["amplify"]["off_target_mean_abs"] for k in keys]
        supl = [rows[k]["suppress"]["loop_rate"] for k in keys]
        ampl = [rows[k]["amplify"]["loop_rate"] for k in keys]
        return {
            "label": label, "n": len(keys), "directions": list(keys),
            "suppress_mean_abs_delta": float(np.mean(np.abs(sup))),
            "amplify_mean_abs_delta": float(np.mean(np.abs(amp))),
            "suppress_over_amplify_raw": (float(np.mean(np.abs(sup)) / np.mean(np.abs(amp)))
                                          if np.mean(np.abs(amp)) > 1e-9 else None),
            "n_suppress_beats_amplify_raw": int(sum(abs(s) > abs(a) for s, a in zip(sup, amp))),
            "suppress_mean_abs_room_share": float(np.mean(np.abs(supr))),
            "amplify_mean_abs_room_share": float(np.mean(np.abs(ampr))),
            "suppress_over_amplify_room": (float(np.mean(np.abs(supr)) / np.mean(np.abs(ampr)))
                                           if np.mean(np.abs(ampr)) > 1e-9 else None),
            "n_suppress_beats_amplify_room": int(sum(abs(s) > abs(a) for s, a in zip(supr, ampr))),
            "suppress_mean_off_target": float(np.mean(supo)),
            "amplify_mean_off_target": float(np.mean(ampo)),
            "suppress_mean_loop_rate": float(np.mean(supl)),
            "amplify_mean_loop_rate": float(np.mean(ampl))}

    fa = [k for k in rows if k.startswith("FA_")]
    ax = [k for k in rows if k.startswith("axis_")]
    out["summary"] = {"fa_factors": summarise(fa, "the five recovered factors"),
                      "big_five_axes": summarise(ax, "the five Big Five keying axes"),
                      "all": summarise(list(rows), "all ten directions")}

    # published equal-alpha, same statistics, for the contrast
    def summarise_pub(keys):
        got = [k for k in keys if rows[k]["published_equal_alpha"]]
        if not got:
            return None
        sup, amp, supr, ampr = [], [], [], []
        for k in got:
            pe = rows[k]["published_equal_alpha"]
            a = rows[k]["amplifying_sign"]
            s = "neg" if a == "pos" else "pos"
            if a not in pe or s not in pe:
                continue
            sup.append(pe[s]["delta_own"])
            amp.append(pe[a]["delta_own"])
            supr.append(pe[s]["delta_own_as_share_of_room"])
            ampr.append(pe[a]["delta_own_as_share_of_room"])
        return {"n": len(sup), "directions": got,
                "note": "the amplifying sign is taken from the MATCHED-DOSE run "
                        "so the two arms are compared sign for sign",
                "suppress_mean_abs_delta": float(np.mean(np.abs(sup))),
                "amplify_mean_abs_delta": float(np.mean(np.abs(amp))),
                "suppress_over_amplify_raw": float(np.mean(np.abs(sup)) / np.mean(np.abs(amp))),
                "n_suppress_beats_amplify_raw": int(sum(abs(s) > abs(a) for s, a in zip(sup, amp))),
                "suppress_mean_abs_room_share": float(np.mean(np.abs(supr))),
                "amplify_mean_abs_room_share": float(np.mean(np.abs(ampr))),
                "suppress_over_amplify_room": float(np.mean(np.abs(supr)) / np.mean(np.abs(ampr))),
                "n_suppress_beats_amplify_room": int(sum(abs(s) > abs(a) for s, a in zip(supr, ampr)))}
    out["published_equal_alpha_summary"] = {
        "fa_factors": summarise_pub(fa), "big_five_axes": summarise_pub(ax),
        "all": summarise_pub(list(rows)),
        "caveat": "the published arm's baseline is its own campaign's alpha-0 "
                  "row, judged in a different call batch, and its axes were "
                  "published at ref 0.8102592902648793 against 0.8078003190997738 "
                  "here (0.30% larger). Its generations were made by steer_fix.py."}

    json.dump(out, open(f"{A}/matched_dose_steering.json", "w"), indent=1)
    print("wrote analysis/matched_dose_steering.json")
    bm = out["base"]["scores"]
    print(f"base: " + " ".join(f"{f[:4]} {bm[f]:.2f}" for f in F5))
    print(f"{'direction':24s} {'a-':>7s} {'a+':>7s} {'sup':>7s} {'amp':>7s} "
          f"{'supRoom':>8s} {'ampRoom':>8s} {'offS':>6s} {'offA':>6s} {'loopS':>6s}")
    for k, r in rows.items():
        print(f"{k:24s} {r['matched']['neg']['alpha']:7.3f} {r['matched']['pos']['alpha']:7.3f} "
              f"{r['suppress']['delta_own']:+7.3f} {r['amplify']['delta_own']:+7.3f} "
              f"{r['suppress']['delta_own_as_share_of_room']:+8.3f} "
              f"{r['amplify']['delta_own_as_share_of_room']:+8.3f} "
              f"{r['suppress']['off_target_mean_abs']:6.3f} "
              f"{r['amplify']['off_target_mean_abs']:6.3f} "
              f"{r['suppress']['loop_rate']:6.3f}")
    for k, v in out["summary"].items():
        print(f"{k}: suppress/amplify raw {v['suppress_over_amplify_raw']:.3f} "
              f"({v['n_suppress_beats_amplify_raw']}/{v['n']}), "
              f"room {v['suppress_over_amplify_room']:.3f} "
              f"({v['n_suppress_beats_amplify_room']}/{v['n']})")


if __name__ == "__main__":
    main()
