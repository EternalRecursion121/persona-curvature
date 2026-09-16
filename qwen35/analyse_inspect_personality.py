#!/usr/bin/env python3
"""Score the Inspect personality runs and relate them to the zoo's geometry.

INPUT
  phase10_runs/inspect_bfi.jsonl      one record per condition, raw completions
  phase10_runs/inspect_trait.jsonl    ditto, TRAIT 20% slice (if present)
  phase10_runs/inspect_items_{bfi,trait}.json   the harness's own rendered items
  phase10_runs/inspect_harness_*.json real-harness runs, for validation
  results/fa_qwen35.json              trait factor, keying, oblimin loadings
  results/gram_sweep.npz              the exact 134x134 Gram
  phase10_runs/steer_spec.json        axis_<Factor> Big Five axis coefficients
  phase10_runs/judged_100.json        blind judge's Big Five, 100 traits x 3 conds
  fa_chart.py                         FAChart().trait_coords

SCORING is inspect_evals' own, ported verbatim (personality.py `_parse_answer`,
`any_choice`, `trait_ratio`):
  * a sample is CORRECT iff the first `ANSWER:\\s*([A-Za-z\\d ,]+)` capture is a
    substring of the target text ("ABCDE" for BFI, "ABCD" for TRAIT);
  * an INCORRECT sample increments an "Error" bucket and is dropped from its
    trait's numerator AND denominator -- it is not scored as zero;
  * rating = answer_mapping[letter]; reverse-keyed items become (max+1)-rating;
  * trait_ratio[t] = sum(rating) / (max_val * n_scored_for_t).

SIGN CONVENTIONS.  The BFI reports Neuroticism; the zoo's factor is
EmotionalStability, its opposite.  BFI Openness is the zoo's Intellect.  Nothing
is silently flipped: BFI_TO_ZOO records the mapping and the sign, and every
derived quantity says which it used.

out: analysis/inspect_personality.json
usage: /home/vibe12/cartovenv/bin/python analyse_inspect_personality.py
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RNG = np.random.default_rng(20260908)
NPERM = 20000

# ---- inspect_evals/personality/personality.py::_parse_answer, verbatim
_ANSWER_RE = re.compile(
    r"""
    ANSWER                 # The literal word "ANSWER"
    \s*:\s*                # A colon surrounded by optional whitespace

    (                      # --- Start capture group 1 ---
      [A-Za-z\d ,]+        # One or more allowed answer characters
    )                      # --- End capture group 1 ---

    (?:                    # --- Start non-capturing group ---
      [^\w]                 # A single non-word character (punctuation, etc.)
      | \n                  # OR a newline
      | $                   # OR end of string
    )                       # --- End non-capturing group ---
    """,
    flags=re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)


def parse_answer(text):
    m = _ANSWER_RE.search(text)
    return m.group(1) if m else None


def score_condition(completions, items):
    """-> (trait_ratio dict, n_error, per-trait n scored)."""
    by_id = {it["id"]: it for it in items}
    agg, den, n = defaultdict(float), defaultdict(float), defaultdict(int)
    n_err = 0
    for c in completions:
        it = by_id[c["id"]]
        letter = parse_answer(c["completion"])
        correct = letter is not None and letter in it["target_text"]
        if not correct:
            n_err += 1
            continue
        mapping = it["answer_mapping"]
        rating = mapping.get(letter, 0)
        max_val = max(mapping.values(), default=1)
        if it["reverse"]:
            rating = (max_val + 1) - rating
        agg[it["trait"]] += rating
        den[it["trait"]] += max_val
        n[it["trait"]] += 1
    return ({t: agg[t] / den[t] for t in den}, n_err, dict(n))


def acquiescence(completions, items):
    """Mean RAW rating (before reverse-scoring) over every scored item.

    The BFI's 16 reverse-keyed items mean a model that simply agrees with more
    statements scores HIGHER on the 28 forward items and LOWER on the 16
    reversed ones.  This index separates that response style from trait
    content: 3.0 is neutral, above 3.0 is agreement bias.  Returns
    (mean over all items, mean over forward items, mean over reverse items).
    """
    by_id = {it["id"]: it for it in items}
    allr, fwd, rev = [], [], []
    for c in completions:
        it = by_id[c["id"]]
        letter = parse_answer(c["completion"])
        if letter is None or letter not in it["target_text"]:
            continue
        r = it["answer_mapping"].get(letter, 0)
        allr.append(r)
        (rev if it["reverse"] else fwd).append(r)
    m = lambda v: float(np.mean(v)) if v else None
    return m(allr), m(fwd), m(rev)


# ---- mapping between the BFI's trait names and the zoo's factor names
BFI_TO_ZOO = {"Openness": ("Intellect", +1), "Conscientiousness": ("Conscientiousness", +1),
              "Extraversion": ("Extraversion", +1), "Agreeableness": ("Agreeableness", +1),
              "Neuroticism": ("EmotionalStability", -1)}
ZOO_TO_BFI = {z: (b, s) for b, (z, s) in BFI_TO_ZOO.items()}
# fa_chart factor order -> the BFI trait it should track, and the sign.
# Factor 2's name says its positive pole is fearful; the coordinates say the
# opposite, and the coordinates are what is used.  Measured, not assumed:
#   FA_FearfulWithdrawal  trait_coords[unenvious] = +0.3639, [fearful] = -0.9672
#   axis_EmotionalStability          [unenvious] = +0.8998, [fearful] = -0.3212
# so both run with emotional stability and against BFI Neuroticism.
FA_TO_BFI = [("Warmth", "Agreeableness", +1), ("Competence", "Conscientiousness", +1),
             ("FearfulWithdrawal", "Neuroticism", -1), ("Arousal", "Extraversion", +1),
             ("Imagination", "Openness", +1)]
BFI_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


def perm_p(a, b, n=NPERM):
    """Two-sided label-permutation p for mean(a) - mean(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    k = len(a)
    hits = 0
    for _ in range(n):
        p = RNG.permutation(pool)
        if abs(p[:k].mean() - p[k:].mean()) >= abs(obs) - 1e-12:
            hits += 1
    return float(obs), float((hits + 1) / (n + 1))


def perm_p_paired(d, n=NPERM):
    """Two-sided sign-flip p for mean(d) != 0."""
    d = np.asarray(d, float)
    obs = d.mean()
    hits = 0
    for _ in range(n):
        s = RNG.choice([-1.0, 1.0], size=len(d))
        if abs((d * s).mean()) >= abs(obs) - 1e-12:
            hits += 1
    return float(obs), float((hits + 1) / (n + 1))


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None, None, int(len(x))
    r = float(np.corrcoef(x, y)[0, 1])
    from scipy import stats
    p = float(stats.pearsonr(x, y).pvalue)
    return r, p, int(len(x))


def load_task(task):
    p = f"{HERE}/phase10_runs/inspect_{task}.jsonl"
    if not os.path.exists(p):
        return None, None
    items = json.load(open(f"{HERE}/phase10_runs/inspect_items_{task}.json"))["items"]
    recs = [json.loads(l) for l in open(p) if l.strip()]
    out = {}
    for r in recs:
        ratio, n_err, n = score_condition(r["completions"], items)
        rec = {"family": r["family"], "trait_ratio": ratio,
               "n_error": n_err, "n_scored": n,
               "max_new_tokens": r["max_new_tokens"],
               # items that needed more than max_new_tokens to name a letter and
               # were regenerated at the larger budget: a verbosity measure
               "n_retried": (r.get("retry") or {}).get("n_items", 0),
               "retry_max_new_tokens": (r.get("retry") or {}).get("max_new_tokens")}
        if task == "bfi":
            a, f_, v_ = acquiescence(r["completions"], items)
            rec["acquiescence"] = {"mean_raw_rating": a, "forward_items": f_,
                                   "reverse_items": v_}
        out[r["condition"]] = rec
    return out, items


def main():
    res = {"method": {
        "harness": "inspect_evals personality_BFI / personality_TRAIT",
        "inspect_evals_commit": "4f6d9f5e7adef4edffc6f02ec99d8e97aab673db",
        "inspect_evals_version": "3-A",
        "bfi_dataset": ("guiem/personality-tests @ 23325c7659839d5432e874a6cdd69b859c7728a1, "
                        "bfi.json sha256 9b0306ecee21fb96eda88d3247ac960a7957ff7e44f0ca1a640398d9965b924a"),
        "trait_dataset": "mirlab/TRAIT (redirects to snupilab/TRAIT) @ 8b31c078cb897c3917d2ee48735d0c15030680e0",
        "trait_subset": "personality_TRAIT(shuffle='questions', seed=41) then --limit 1600 (the README's 20% slice)",
        "answer_extraction": ("greedy generation then inspect_evals' own ANSWER: regex -- Inspect "
                              "generates and regex-parses, it does not read option logprobs"),
        "chat_template": "Qwen3.5 template, enable_thinking=False",
        "scoring": "any_choice + trait_ratio, ported verbatim from personality.py",
        "bfi_ratio_range": [0.2, 1.0],
        "bfi_ratio_all_neutral": 0.6,
        "bfi_ratio_note": ("a BFI ratio is sum(rating)/(5*n_items): every item contributes at "
                           "least 1 and at most 5, so the scale runs 0.2 to 1.0 and answering "
                           "'Neither agree nor disagree' to everything gives 0.6. It is NOT a "
                           "percentage of the scale, and 0.5 is BELOW the neutral point."),
        "trait_ratio_note": ("a TRAIT ratio is the fraction of items on which the model chose one "
                            "of the two high-trait options; scale 0.0 to 1.0, chance 0.5"),
        "bfi_to_zoo_factor": {b: {"zoo_factor": z, "sign": s} for b, (z, s) in BFI_TO_ZOO.items()},
    }, "conditions": {}}

    bfi, bfi_items = load_task("bfi")
    if bfi is None:
        sys.exit("no phase10_runs/inspect_bfi.jsonl")
    res["conditions"]["bfi"] = bfi
    trait, trait_items = load_task("trait")
    if trait is not None:
        res["conditions"]["trait"] = trait
    res["n_conditions"] = {"bfi": len(bfi), "trait": len(trait or {})}

    # ---------------------------------------------------------------- harness check
    hv = {}
    for f in sorted(os.listdir(f"{HERE}/phase10_runs")):
        if f.startswith("inspect_harness_") and f.endswith(".json"):
            h = json.load(open(f"{HERE}/phase10_runs/{f}"))
            cond = h["condition"]
            mine = bfi.get(cond, {}).get("trait_ratio", {})
            theirs = {k.split("/")[-1] if "/" in k else k: v for k, v in h["metrics"].items()}
            common = [t for t in BFI_TRAITS if t in mine and t in theirs]
            diffs = {t: mine[t] - theirs[t] for t in common}
            # one item's worth: for BFI a single item moves a trait ratio by at
            # most max_val/(max_val*n) = 1/n of that trait's item count
            worst = max((abs(d) for d in diffs.values()), default=None)
            n_min = min((bfi[cond]["n_scored"].get(t, 1) for t in common), default=1)
            hv[cond] = {"harness_metrics": theirs, "reimplementation": {t: mine[t] for t in common},
                        "difference": diffs, "max_abs_difference": worst,
                        "one_item_worth_max": 1.0 / n_min,
                        "agrees_within_one_item": (worst is not None and worst <= 1.0 / n_min + 1e-12),
                        "harness_status": h.get("status"),
                        "n_samples": len(h.get("samples", []))}
            # per-item answer agreement
            hs = {s["id"]: s for s in h.get("samples", [])}
            rec = next((json.loads(l) for l in open(f"{HERE}/phase10_runs/inspect_bfi.jsonl")
                        if json.loads(l)["condition"] == cond), None)
            if rec:
                same = sum(1 for c in rec["completions"]
                           if c["id"] in hs and parse_answer(c["completion"]) == hs[c["id"]]["answer"])
                hv[cond]["items_same_answer"] = f"{same}/{len(rec['completions'])}"
    res["harness_validation"] = hv

    # ---------------------------------------------------------------- zoo metadata
    fa = json.load(open(f"{HERE}/results/fa_qwen35.json"))
    order = fa["trait_order"]
    slug = dict(zip(order, fa["trait_slug"]))
    slug2name = {v: k for k, v in slug.items()}
    fac = dict(zip(order, fa["trait_factor"]))
    key = dict(zip(order, fa["trait_keyed"]))
    slugs = list(fa["trait_slug"])

    def ratio(task, family, s, bfit):
        c = bfi if task == "bfi" else trait
        r = c.get(f"{family}:{s}")
        return None if r is None else r["trait_ratio"].get(bfit)

    # ---------------------------------------------------------------- keyed contrast
    keyed = {}
    for family in ("stage1", "persona"):
        keyed[family] = {}
        for bfit in BFI_TRAITS:
            zoo, sign = BFI_TO_ZOO[bfit]
            marks = [t for t in order if fac[t] == zoo]
            pos = [ratio("bfi", family, slug[t], bfit) for t in marks if key[t] == "+"]
            neg = [ratio("bfi", family, slug[t], bfit) for t in marks if key[t] == "-"]
            pos = [x for x in pos if x is not None]
            neg = [x for x in neg if x is not None]
            if len(pos) < 2 or len(neg) < 2:
                continue
            obs, p = perm_p(pos, neg)
            keyed[family][bfit] = {
                "zoo_factor": zoo, "n_pos_keyed": len(pos), "n_neg_keyed": len(neg),
                "mean_pos_keyed": float(np.mean(pos)), "mean_neg_keyed": float(np.mean(neg)),
                "difference_pos_minus_neg": obs, "perm_p": p,
                "expected_sign": sign,
                "difference_in_own_pole_direction": obs * sign,
                "note": ("+ keyed markers of EmotionalStability are emotionally STABLE, so on "
                         "BFI Neuroticism the expected difference is negative"
                         if sign < 0 else "positively keyed markers should score higher"),
            }
    res["keyed_contrast_bfi"] = keyed

    # ---------------------------------------------------------------- stage1 vs persona
    sp = {}
    for bfit in BFI_TRAITS:
        d = []
        for s in slugs:
            a, b = ratio("bfi", "stage1", s, bfit), ratio("bfi", "persona", s, bfit)
            if a is not None and b is not None:
                d.append(b - a)
        obs, p = perm_p_paired(d)
        sp[bfit] = {"n": len(d), "mean_persona_minus_stage1": obs, "perm_p": p}
    # own-pole version: signed by the adapter's own factor and keying
    dd = []
    for s in slugs:
        t = slug2name[s]
        if fac[t] not in ZOO_TO_BFI:
            continue
        bfit, sgn = ZOO_TO_BFI[fac[t]]
        k = 1.0 if key[t] == "+" else -1.0
        a, b = ratio("bfi", "stage1", s, bfit), ratio("bfi", "persona", s, bfit)
        if a is None or b is None:
            continue
        dd.append(k * sgn * (b - a))
    obs, p = perm_p_paired(dd)
    sp["own_trait_signed"] = {"n": len(dd), "mean_persona_minus_stage1": obs, "perm_p": p,
                              "definition": ("(persona - stage1) BFI ratio on the adapter's own "
                                             "Big Five trait, signed so positive = the persona "
                                             "self-reports further in its own trained direction; "
                                             "Lexicon-arm adapters excluded (no factor)")}
    res["stage1_vs_persona_bfi"] = sp

    # ---------------------------------------------------------------- vs the blind judge
    J = json.load(open(f"{HERE}/phase10_runs/judged_100.json"))["records"]
    jm = defaultdict(lambda: defaultdict(list))
    for r in J:
        for f_, v in r["scores"].items():
            if v is not None:           # 97 of 7200 judged records carry a null factor
                jm[(r["condition"], r["trait"])][f_].append(v)
    judged = {k: {f_: float(np.mean(v)) for f_, v in d.items()} for k, d in jm.items()}
    jud = {}
    for family in ("stage1", "persona"):
        M, names = {}, sorted({t for (c, t) in judged if c == family})
        for bfit in BFI_TRAITS:
            zoo, sign = BFI_TO_ZOO[bfit]
            x = [ratio("bfi", family, s, bfit) for s in names]
            y = [judged[(family, s)].get(zoo) for s in names]
            pair = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
            r_, p_, n_ = pearson([a for a, b in pair], [b for a, b in pair])
            M[bfit] = {"zoo_factor": zoo, "sign_expected": sign, "r": r_, "p": p_, "n": n_}
        full = {}
        for bfit in BFI_TRAITS:
            for zoo in ["Extraversion", "Agreeableness", "Conscientiousness",
                        "EmotionalStability", "Intellect"]:
                x = [ratio("bfi", family, s, bfit) for s in names]
                y = [judged[(family, s)].get(zoo) for s in names]
                pair = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
                r_, _, _ = pearson([a for a, b in pair], [b for a, b in pair])
                full[f"{bfit}|{zoo}"] = r_
        jud[family] = {"matched": M, "full_matrix": full, "n_traits_judged": len(names)}
    res["bfi_vs_judged"] = jud
    res["bfi_vs_judged"]["source"] = ("phase10_runs/judged_100.json#records, mean over the 24 "
                                      "probe prompts per (condition, trait); 100 of the 134 "
                                      "adapters were judged")

    # ---------------------------------------------------------------- vs weight space
    from fa_chart import FAChart
    ch = FAChart()
    assert ch.names == slugs, "gram order differs from fa trait_slug order"
    coords = ch.trait_coords                                  # 134 x 5
    spec = {j["name"]: j["coef"] for j in json.load(open(f"{HERE}/phase10_runs/steer_spec.json"))["jobs"]}
    AX = ["Agreeableness", "Conscientiousness", "EmotionalStability", "Extraversion", "Intellect"]
    axc = np.array([[spec[f"axis_{a}"].get(t, 0.0) for t in ch.names] for a in AX])
    axn = np.array([np.sqrt(c @ ch.G @ c) for c in axc])
    axis_coords = (ch.G @ axc.T) / axn                        # 134 x 5, projection on unit axis
    geo = {}
    for family in ("stage1", "persona"):
        m_fa, m_ax = {}, {}
        for bfit in BFI_TRAITS:
            y = np.array([ratio("bfi", family, s, bfit) if ratio("bfi", family, s, bfit)
                          is not None else np.nan for s in slugs], float)
            for k, (fa_name, want_bfi, sgn) in enumerate(FA_TO_BFI):
                r_, p_, n_ = pearson(coords[:, k], y)
                m_fa[f"{bfit}|{fa_name}"] = {"r": r_, "p": p_, "n": n_,
                                             "matched": want_bfi == bfit, "expected_sign": sgn}
            for k, a in enumerate(AX):
                r_, p_, n_ = pearson(axis_coords[:, k], y)
                zoo, sign = BFI_TO_ZOO[bfit]
                m_ax[f"{bfit}|axis_{a}"] = {"r": r_, "p": p_, "n": n_,
                                            "matched": a == zoo, "expected_sign": sign}
        geo[family] = {"fa_chart": m_fa, "bigfive_axes": m_ax}
    res["bfi_vs_weight_space"] = geo
    res["bfi_vs_weight_space"]["definitions"] = {
        "fa_chart": ("fa_chart.py FAChart().trait_coords, 134 x 5, Gram-Schmidt basis of the five "
                     "oblimin factor directions in the order Warmth, Competence, "
                     "FearfulWithdrawal, Arousal, Imagination"),
        "bigfive_axes": ("projection of each stage-one adapter on the unit-normalised axis_<Factor> "
                         "direction from phase10_runs/steer_spec.json, inner products from "
                         "results/gram_sweep.npz"),
        "note": ("weight-space coordinates are of the STAGE-ONE adapter in both rows; the persona "
                 "row asks whether the stage-one adapter's position predicts the persona's "
                 "self-report"),
    }

    # ------------------------------------------------- the ten Big Five dials
    bfa = {}
    for e in json.load(open(f"{HERE}/traits_bigfive.json")):
        c = bfi.get(f"bigfive:{e['trait']}")
        if c is None:
            continue
        bfit = e["bigfive_factor"]                    # already BFI naming (Neuroticism, Openness)
        base_r = bfi["base"]["trait_ratio"]
        bfa[e["trait"]] = {
            "bigfive_factor": bfit, "pole": e["pole"], "zoo_factor": e["zoo_factor"],
            "trait_ratio": c["trait_ratio"], "n_error": c["n_error"],
            "own_trait_ratio": c["trait_ratio"].get(bfit),
            "own_trait_shift_vs_base": (None if c["trait_ratio"].get(bfit) is None
                                        else c["trait_ratio"][bfit] - base_r[bfit]),
            "acquiescence": c["acquiescence"],
        }
    if bfa:
        dials = {}
        for f_ in BFI_TRAITS:
            hi = next((v for v in bfa.values() if v["bigfive_factor"] == f_ and v["pole"] == "high"),
                      None)
            lo = next((v for v in bfa.values() if v["bigfive_factor"] == f_ and v["pole"] == "low"),
                      None)
            if hi and lo and hi["own_trait_ratio"] is not None and lo["own_trait_ratio"] is not None:
                dials[f_] = {"high": hi["own_trait_ratio"], "low": lo["own_trait_ratio"],
                             "high_minus_low": hi["own_trait_ratio"] - lo["own_trait_ratio"],
                             "base": bfi["base"]["trait_ratio"][f_],
                             "own_trait_dominant": None}
        n_right = sum(1 for v in dials.values() if v["high_minus_low"] > 0)
        dials_trait = {}
        if trait is not None:
            for f_ in BFI_TRAITS:
                hi = trait.get(f"bigfive:bf_{f_.lower()}_high", {}).get("trait_ratio", {}).get(f_)
                lo = trait.get(f"bigfive:bf_{f_.lower()}_low", {}).get("trait_ratio", {}).get(f_)
                ba = trait.get("base", {}).get("trait_ratio", {}).get(f_)
                if hi is not None and lo is not None:
                    dials_trait[f_] = {"high": hi, "low": lo, "high_minus_low": hi - lo,
                                       "base": ba,
                                       "n_items_high": trait[f"bigfive:bf_{f_.lower()}_high"]
                                       ["n_scored"].get(f_),
                                       "n_items_low": trait[f"bigfive:bf_{f_.lower()}_low"]
                                       ["n_scored"].get(f_)}
        res["bigfive_factor_adapters"] = {
            "source": "traits_bigfive.json; adapters on pc-qwen35-adapters:/data_bigfive_common",
            "per_adapter": bfa, "dials": dials, "dials_trait": dials_trait,
            "n_dials_in_expected_direction": f"{n_right}/{len(dials)}",
            "n_dials_in_expected_direction_trait":
                (f"{sum(1 for v in dials_trait.values() if v['high_minus_low'] > 0)}/"
                 f"{len(dials_trait)}" if dials_trait else None),
            "note": ("high minus low on the dial's OWN BFI trait; positive means the amplifier "
                     "self-reports more of the trait than the suppressor. Neuroticism's dial is "
                     "named for the factor, so bf_neuroticism_high should raise BFI Neuroticism."),
        }

    # ------------------------------------------------ how many answers move at all
    raw = {json.loads(l)["condition"]: json.loads(l)
           for l in open(f"{HERE}/phase10_runs/inspect_bfi.jsonl")}
    lets = {c: {x["id"]: parse_answer(x["completion"]) for x in r["completions"]}
            for c, r in raw.items()}
    b = lets["base"]
    churn = {"definition": ("number of the 44 BFI items whose chosen letter differs from the "
                            "base model's, per condition")}
    for family in ("stage1", "persona", "bigfive"):
        v = [sum(1 for k in b if lets[c][k] != b[k])
             for c in lets if c.startswith(family + ":")]
        if v:
            churn[family] = {"n": len(v), "mean": float(np.mean(v)),
                             "min": int(min(v)), "max": int(max(v))}
    if any(c.startswith("bigfive:") for c in lets):
        churn["bigfive_pole_pairs_identical_of_44"] = {
            f_: sum(1 for k in b
                    if lets[f"bigfive:bf_{f_.lower()}_high"][k]
                    == lets[f"bigfive:bf_{f_.lower()}_low"][k])
            for f_ in BFI_TRAITS
            if f"bigfive:bf_{f_.lower()}_high" in lets and f"bigfive:bf_{f_.lower()}_low" in lets}
    res["answer_churn_bfi"] = churn

    # ---------------------------------------------------------------- acquiescence
    acq = {"definition": ("mean raw BFI rating (1-5, before reverse-scoring) over the 44 items; "
                          "3.0 is neutral. A condition that simply agrees more scores higher on "
                          "the 28 forward items and LOWER on the 16 reverse-keyed ones, which "
                          "shows up as a trait effect it is not."),
           "base": bfi["base"]["acquiescence"]}
    for family in ("stage1", "persona"):
        a = np.array([bfi.get(f"{family}:{s}", {}).get("acquiescence", {}).get("mean_raw_rating",
                                                                              np.nan)
                      for s in slugs], float)
        row = {"mean": float(np.nanmean(a)), "min": float(np.nanmin(a)), "max": float(np.nanmax(a))}
        for k, (fa_name, _w, _s) in enumerate(FA_TO_BFI):
            r_, p_, n_ = pearson(coords[:, k], a)
            row[f"vs_fa_{fa_name}"] = {"r": r_, "p": p_, "n": n_}
        for bfit in BFI_TRAITS:
            y = np.array([bfi.get(f"{family}:{s}", {}).get("trait_ratio", {}).get(bfit, np.nan)
                          for s in slugs], float)
            r_, p_, n_ = pearson(a, y)
            row[f"vs_bfi_{bfit}"] = {"r": r_, "p": p_, "n": n_}
        acq[family] = row
    res["acquiescence"] = acq

    # ---------------------------------------------------------------- TRAIT
    if trait is not None:
        sel = json.load(open(f"{HERE}/analysis/inspect_trait20.json"))["selection"]
        res["trait_selection"] = sel
        tk = {}
        by_factor = defaultdict(lambda: {"+": [], "-": []})
        for e in sel:
            bfit, sign = ZOO_TO_BFI[e["bigfive_factor"]]
            v = trait.get(f"stage1:{e['slug']}", {}).get("trait_ratio", {}).get(bfit)
            if v is not None:
                by_factor[bfit]["+" if e["keyed"] == "+" else "-"].append(v)
        for bfit, d in by_factor.items():
            if len(d["+"]) >= 2 and len(d["-"]) >= 2:
                obs, p = perm_p(d["+"], d["-"])
                tk[bfit] = {"n_pos_keyed": len(d["+"]), "n_neg_keyed": len(d["-"]),
                            "mean_pos_keyed": float(np.mean(d["+"])),
                            "mean_neg_keyed": float(np.mean(d["-"])),
                            "difference_pos_minus_neg": obs, "perm_p": p,
                            "expected_sign": BFI_TO_ZOO[bfit][1]}
        res["keyed_contrast_trait"] = tk
        res["keyed_contrast_trait_note"] = (
            "2 positively keyed against 2 negatively keyed adapters per factor: there are only "
            "C(4,2)=6 distinct label splits, so the permutation p cannot fall below 1/3. Read the "
            "difference, not the p.")
        # verbosity / discard bookkeeping
        n_per_trait = defaultdict(int)
        for it in trait_items:
            n_per_trait[it["trait"]] += 1
        vb = {}
        for c, v in trait.items():
            lost = {t: 1 - v["n_scored"].get(t, 0) / n_per_trait[t] for t in n_per_trait}
            worst = max(lost.items(), key=lambda kv: kv[1])
            vb[c] = {"n_retried": v["n_retried"], "n_error": v["n_error"],
                     "n_scored_total": sum(v["n_scored"].values()),
                     "worst_trait_fraction_discarded": {"trait": worst[0],
                                                        "fraction": round(worst[1], 4)},
                     "traits_losing_over_5pc": {t: round(f, 4) for t, f in lost.items()
                                                if f > 0.05}}
        res["trait_generation_budget"] = {
            "first_pass_max_new_tokens": 24,
            "retry_max_new_tokens": next((v["retry_max_new_tokens"] for v in trait.values()
                                          if v["retry_max_new_tokens"]), None),
            "note": ("On TRAIT some conditions answer in prose and run past 24 tokens before "
                     "naming a letter; Inspect's scorer discards those items rather than scoring "
                     "them zero. Every such item was regenerated at the larger budget. n_retried "
                     "is therefore a verbosity measure, and n_error is what still failed."),
            "per_condition": vb,
            "total_retried": sum(v["n_retried"] for v in vb.values()),
            "total_still_failing": sum(v["n_error"] for v in vb.values()),
        }
        # BFI vs TRAIT on the same 20 adapters
        cross = {}
        for bfit in BFI_TRAITS:
            x, y = [], []
            for e in sel:
                a = trait.get(f"stage1:{e['slug']}", {}).get("trait_ratio", {}).get(bfit)
                b = bfi.get(f"stage1:{e['slug']}", {}).get("trait_ratio", {}).get(bfit)
                if a is not None and b is not None:
                    x.append(a), y.append(b)
            r_, p_, n_ = pearson(x, y)
            cross[bfit] = {"r": r_, "p": p_, "n": n_}
        res["trait_vs_bfi_same_adapters"] = cross
        # TRAIT vs weight space on the 20
        idx = {s: i for i, s in enumerate(slugs)}
        tw = {}
        for bfit in BFI_TRAITS:
            y = [trait.get(f"stage1:{e['slug']}", {}).get("trait_ratio", {}).get(bfit) for e in sel]
            for k, (fa_name, want, sgn) in enumerate(FA_TO_BFI):
                x = [coords[idx[e["slug"]], k] for e in sel]
                pair = [(a, b) for a, b in zip(x, y) if b is not None]
                r_, p_, n_ = pearson([a for a, b in pair], [b for a, b in pair])
                tw[f"{bfit}|{fa_name}"] = {"r": r_, "p": p_, "n": n_, "matched": want == bfit}
        res["trait_vs_weight_space"] = tw

    out = f"{HERE}/analysis/inspect_personality.json"
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)

    # ---- console summary
    print("\nbase BFI trait ratios:", {k: round(v, 4) for k, v in
                                       bfi["base"]["trait_ratio"].items()},
          "errors", bfi["base"]["n_error"])
    for c, v in hv.items():
        print(f"harness {c}: max|diff| {v['max_abs_difference']}, "
              f"within one item {v['agrees_within_one_item']}, "
              f"same answers {v.get('items_same_answer')}")
    for fam in ("stage1", "persona"):
        print(f"\nkeyed contrast [{fam}]")
        for t, d in res["keyed_contrast_bfi"].get(fam, {}).items():
            print(f"  {t:18s} +{d['mean_pos_keyed']:.3f} -{d['mean_neg_keyed']:.3f}  "
                  f"diff {d['difference_pos_minus_neg']:+.4f}  p={d['perm_p']:.5f}")
    print("\nBFI vs blind judge (matched factor):")
    for fam in ("stage1", "persona"):
        for t, d in res["bfi_vs_judged"][fam]["matched"].items():
            r_ = d["r"]
            print(f"  {fam:8s} {t:18s} r={r_ if r_ is None else round(r_, 3)} "
                  f"p={d['p'] if d['p'] is None else format(d['p'], '.2e')} n={d['n']} "
                  f"(expected sign {d['sign_expected']:+d})")
    print("\nBFI vs weight space (matched):")
    for fam in ("stage1", "persona"):
        for k, d in res["bfi_vs_weight_space"][fam]["fa_chart"].items():
            if d["matched"]:
                print(f"  {fam:8s} {k:38s} r={d['r'] if d['r'] is None else round(d['r'],3)} "
                      f"p={d['p'] if d['p'] is None else format(d['p'],'.2e')} "
                      f"(expected sign {d['expected_sign']:+d})")
    print("\nstage1 vs persona:", {k: (round(v['mean_persona_minus_stage1'], 4), v['perm_p'])
                                   for k, v in res["stage1_vs_persona_bfi"].items()})


if __name__ == "__main__":
    main()
