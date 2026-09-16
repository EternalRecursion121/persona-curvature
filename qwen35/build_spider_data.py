#!/usr/bin/env python3
"""Regenerate analysis/spider.json (the OCEAN-dial spider data) from primary files.

Arm A, steering axes: phase10_runs/judged_steerfix.json (thinking off, 512 tokens),
  amplifier = alpha +2 (condition a2_0), suppressor = alpha -2 (am2_0), for the five
  axis_* directions; base_steer = the mean of every a0_0 record.
Arm B, trait adapters: phase10_runs/judged_100.json, condition stage1; for factor F the
  amplifier is the mean judged profile of its positively-keyed markers, the suppressor of
  its negatively-keyed ones; base_trait = the mean of every base record.
Arm C (new 2026-09-08), persona adapters: the same with condition persona, i.e. the full
  OCT artefact (stage one + 0.25 stage two).
Arm D (new 2026-09-08), Big Five FACTOR adapters: phase10_runs/judged_bigfive.json, ten
  stage-one adapters trained one per OCEAN pole from Persona Cartography's own Figure 2
  constitutions -- the amplifier for factor F is the bf_<F>_high adapter itself rather
  than an average over marker adjectives. The zoo's fifth factor is Emotional Stability,
  so bf_neuroticism_low is the ES AMPLIFIER and bf_neuroticism_high the ES suppressor;
  Openness maps onto Intellect. base_bigfive is every base record of that same eval.
  Written only if judged_bigfive.json exists.
Scale, stated not fitted: shift from base as a share of the room left on the 1-7 judge
scale, (x-base)/(7-base) upward and (x-base)/(base-1) downward, times 100.
"""
import json, os, statistics as st
Q = os.path.dirname(os.path.abspath(__file__))
DIMS = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]
AXIS = {"Extraversion": "axis_Extraversion", "Agreeableness": "axis_Agreeableness", "Conscientiousness": "axis_Conscientiousness",
        "EmotionalStability": "axis_EmotionalStability", "Intellect": "axis_Intellect"}
# Zoo factor -> the Big Five factor adapter that is its amplifier / suppressor.
# Neuroticism is keyed the other way round from Emotional Stability; Openness is
# the zoo's Intellect.
BF = {"Extraversion":      ("bf_extraversion_high",      "bf_extraversion_low"),
      "Agreeableness":     ("bf_agreeableness_high",     "bf_agreeableness_low"),
      "Conscientiousness": ("bf_conscientiousness_high", "bf_conscientiousness_low"),
      "EmotionalStability": ("bf_neuroticism_low",       "bf_neuroticism_high"),
      "Intellect":         ("bf_openness_high",          "bf_openness_low")}

def pct(x, base):
    return 100 * ((x - base) / (7 - base) if x >= base else (x - base) / (base - 1))

def mean_profile(recs):
    return {d: st.mean(v for v in (r["scores"].get(d) for r in recs if r.get("scores")) if v is not None) for d in DIMS}

def shifted(profile, base):
    return {d: pct(profile[d], base[d]) for d in DIMS}

prim = json.load(open(f"{Q}/traits_primary.json"))
slug = lambda t: t.lower().replace(" ", "_").replace("-", "_")
keyed = {}
for r in prim:
    keyed.setdefault(r["factor"].replace(" ", ""), {"+": [], "-": []})[r["keyed"]].append(slug(r["trait"]))

J = json.load(open(f"{Q}/phase10_runs/judged_100.json"))["records"]
S = json.load(open(f"{Q}/phase10_runs/judged_steerfix.json"))["records"]
base_trait = mean_profile([r for r in J if r["condition"] == "base"])
base_steer = mean_profile([r for r in S if r["condition"] == "a0_0"])

out = {"base_steer": base_steer, "base_trait": base_trait, "axes": {}, "traits": {}, "personas": {}, "factors": DIMS,
       "n": {"base_trait_records": sum(1 for r in J if r["condition"] == "base"), "base_steer_records": sum(1 for r in S if r["condition"] == "a0_0")},
       "sources": {"axes": "phase10_runs/judged_steerfix.json", "traits": "phase10_runs/judged_100.json condition stage1",
                   "personas": "phase10_runs/judged_100.json condition persona", "keying": "traits_primary.json"}}
for F in DIMS:
    for pole, cond, key in (("amplifier", "a2_0", "+"), ("suppressor", "am2_0", "-")):
        ax = [r for r in S if r["trait"] == AXIS[F] and r["condition"] == cond]
        out["axes"][f"{F}|{pole}"] = shifted(mean_profile(ax), base_steer)
        for arm, cnd in (("traits", "stage1"), ("personas", "persona")):
            recs = [r for r in J if r["condition"] == cnd and r["trait"] in keyed[F][key]]
            out[arm][f"{F}|{pole}"] = shifted(mean_profile(recs), base_trait)
            out["n"][f"{arm}|{F}|{pole}"] = len(recs)
# ---- arm D: the Big Five factor adapters -----------------------------------
BFP = f"{Q}/phase10_runs/judged_bigfive.json"
if os.path.exists(BFP):
    B = json.load(open(BFP))["records"]
    base_bigfive = mean_profile([r for r in B if r["condition"] == "base"])
    out["base_bigfive"] = base_bigfive
    out["bigfive"] = {}
    out["n"]["base_bigfive_records"] = sum(1 for r in B if r["condition"] == "base")
    out["sources"]["bigfive"] = "phase10_runs/judged_bigfive.json condition stage1"
    out["sources"]["bigfive_mapping"] = {F: {"amplifier": a, "suppressor": s_} for F, (a, s_) in BF.items()}
    for F in DIMS:
        for pole, name in (("amplifier", BF[F][0]), ("suppressor", BF[F][1])):
            recs = [r for r in B if r["condition"] == "stage1" and r["trait"] == name]
            if not recs:
                raise SystemExit(f"judged_bigfive.json has no stage1 records for {name}")
            out["bigfive"][f"{F}|{pole}"] = shifted(mean_profile(recs), base_bigfive)
            out["n"][f"bigfive|{F}|{pole}"] = len(recs)
else:
    print(f"note: {BFP} absent -- arm D not written")

json.dump(out, open(f"{Q}/analysis/spider.json", "w"), indent=1)
print("wrote analysis/spider.json; base_trait", {d: round(v, 4) for d, v in base_trait.items()})
for F in DIMS:
    for pole in ("amplifier", "suppressor"):
        own = lambda arm: out[arm][f"{F}|{pole}"][F]
        bf = f"  bigfive {own('bigfive'):+6.1f} ({out['sources']['bigfive_mapping'][F][pole]})" if "bigfive" in out else ""
        print(f"  {F:18s} {pole:10s} own-scale shift: axes {own('axes'):+6.1f}  stage1 {own('traits'):+6.1f}  persona {own('personas'):+6.1f}   n {out['n'][f'traits|{F}|{pole}']}{bf}")
