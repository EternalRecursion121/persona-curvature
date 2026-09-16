#!/usr/bin/env python3
"""pages/behaviour/ocean-dials-replication.md with inline SVG radar plots from qwen35/analysis/spider.json
(regenerated from primary files by qwen35/build_spider_data.py)."""
import json, math, os, html
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); Q = os.path.join(os.path.dirname(W), "qwen35")
d = json.load(open(f"{Q}/analysis/spider.json"))
DIMS = d["factors"]; LAB = {"Extraversion": "E", "Agreeableness": "A", "Conscientiousness": "C", "EmotionalStability": "ES", "Intellect": "I"}
COL = {"Extraversion": "#3d8f4e", "Agreeableness": "#7b3fa0", "Conscientiousness": "#d08a1e", "EmotionalStability": "#c8402f", "Intellect": "#2f6fb0"}

def radar(arm, pole, title):
    S, cx, cy, R = 300, 150, 158, 105
    def pt(i, v):   # v in [-100, 100] -> radius; 0 at R/2
        a = -math.pi / 2 + 2 * math.pi * i / 5
        r = R * (0.5 + 0.5 * max(-100, min(100, v)) / 100)
        return cx + r * math.cos(a), cy + r * math.sin(a)
    out = [f'<svg viewBox="0 0 {S} {S+30}" width="100%" style="max-width:{S}px;font-family:system-ui,sans-serif;font-size:11px" role="img" aria-label="{html.escape(title)}">',
           f'<text x="{cx}" y="16" text-anchor="middle" font-size="13" font-weight="600" fill="currentColor">{html.escape(title)}</text>']
    for lvl, lab in ((100, "+100%"), (50, "+50%"), (0, "0"), (-50, "-50%")):
        ring = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, lvl) for i in range(5)))
        out.append(f'<polygon points="{ring}" fill="none" stroke="currentColor" stroke-opacity="{0.55 if lvl == 0 else 0.15}" stroke-width="{1.6 if lvl == 0 else 1}"/>')
        x, y = pt(0, lvl); out.append(f'<text x="{x+5:.1f}" y="{y+4:.1f}" fill="currentColor" fill-opacity="0.6">{lab}</text>')
    for i, dim in enumerate(DIMS):
        x, y = pt(i, 100); ax, ay = pt(i, 118)
        out.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="currentColor" stroke-opacity="0.15"/>')
        out.append(f'<text x="{ax:.1f}" y="{ay+4:.1f}" text-anchor="middle" font-weight="700" fill="{COL[dim]}">{LAB[dim]}</text>')
    for F in DIMS:
        prof = d[arm][f"{F}|{pole}"]
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, prof[dim]) for i, dim in enumerate(DIMS)))
        out.append(f'<polygon points="{pts}" fill="{COL[F]}" fill-opacity="0.08" stroke="{COL[F]}" stroke-width="2"/>')
        out.extend(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{COL[F]}"/>' for x, y in (pt(i, prof[dim]) for i, dim in enumerate(DIMS)))
    out.append("</svg>"); return "\n".join(out)

def pair(arm, title):
    return (f'<figure style="display:flex;gap:1rem;flex-wrap:wrap;margin:0">\n{radar(arm, "amplifier", title + ", amplifier")}\n{radar(arm, "suppressor", title + ", suppressor")}\n</figure>')

legend = " ".join(f'<span style="color:{COL[F]};font-weight:700">{LAB[F]} {F}</span>' for F in DIMS)

def table(arm):
    rows = ["| dial | pole | " + " | ".join(LAB[x] for x in DIMS) + " | own / mean other |", "|---|---|" + "---|" * 5 + "---|"]
    sel = []
    for F in DIMS:
        for pole in ("amplifier", "suppressor"):
            p = d[arm][f"{F}|{pole}"]; own = p[F]; others = [abs(p[x]) for x in DIMS if x != F]
            ratio = abs(own) / max(1e-9, sum(others) / 4); sel.append((F, pole, own, ratio))
            rows.append(f"| {F} | {pole} | " + " | ".join(f"**{p[x]:+.0f}**" if x == F else f"{p[x]:+.0f}" for x in DIMS) + f" | {ratio:.1f}x |")
    wins = sum(1 for F, pole, own, _ in sel if abs(own) >= max(abs(d[arm][f'{F}|{pole}'][x]) for x in DIMS if x != F) and (own > 0) == (pole == "amplifier"))
    return "\n".join(rows), wins, sel

tA, wA, sA = table("axes"); tB, wB, sB = table("traits"); tC, wC, sC = table("personas")
HAS_D = "bigfive" in d
tD, wD, sD = table("bigfive") if HAS_D else ("", 0, [])
BFMAP = d.get("sources", {}).get("bigfive_mapping", {})
own_dir = lambda sel: sum(1 for F, pole, own, _ in sel if (own > 0) == (pole == "amplifier"))
bt = d["base_trait"]; bs = d["base_steer"]
secD = ""
if HAS_D:
    bb = d["base_bigfive"]
    rows = " ".join(f"{F} = `{BFMAP[F]['amplifier']}` / `{BFMAP[F]['suppressor']}`;" for F in DIMS)
    secD = f"""## D. Dedicated Big Five factor adapters

The three arms above all build a factor dial out of adjectives. These ten adapters ARE the
factor: one stage-one DPO adapter per OCEAN pole, trained on the zoo's shared prompt pool at
the matched objective from Persona Cartography's own Figure 2 constitutions
([[bigfive-factor-adapters]]). Stage one only - no OCT stage two. Mapping onto the zoo's five
factors, whose fifth is Emotional Stability rather than Neuroticism: {rows.rstrip(';')}
(`spider.json#sources.bigfive_mapping`). Condition `stage1` in `judged_bigfive.json`; base is
every `base` record of that same eval (E {bb['Extraversion']:.2f}, A {bb['Agreeableness']:.2f},
C {bb['Conscientiousness']:.2f}, ES {bb['EmotionalStability']:.2f}, I {bb['Intellect']:.2f},
`spider.json#base_bigfive`).

{pair('bigfive', 'Big Five factor adapters')}

{tD}

Own trait moves most and in the right direction for **{wD} of 10**. Each cell here is one
adapter judged on {d['n']['bigfive|Extraversion|amplifier']} generations, where arm B averages
ten adapters, so these are noisier per cell and cleaner in construction.
"""

page = f"""---
title: OCEAN dials replication (spider plots)
summary: Persona Cartography's Figure 2 redone on the zoo four ways - steering axes at alpha plus or minus 2, the ten positively and ten negatively keyed stage-one adapters per factor, the same adapters as full OCT personas, and ten dedicated Big Five FACTOR adapters trained from Persona Cartography's own constitutions - all blind-judged; own-trait dominance holds for {wA}, {wB}, {wC} and {wD} of 10 dials respectively.
status: current
sources:
  - qwen35/analysis/spider.json
  - qwen35/build_spider_data.py
  - qwen35/phase10_runs/judged_100.json
  - qwen35/phase10_runs/judged_steerfix.json
  - qwen35/phase10_runs/judged_bigfive.json
  - qwen35/traits_bigfive.json
  - qwen35/traits_primary.json
  - qwen35/spider_page/index.html
last_verified: 2026-09-08
tags: [behaviour, judged, replication, big-five]
---

Persona Cartography's Figure 2 ("Single dials work", [[persona-cartography-paper]]) shows, for ten OCEAN LoRAs on Llama-3.1-8B-Instruct, the judged change on each of the five traits as a spider plot: each amplifier or suppressor moves its own trait more than the others. This page is the same figure on the zoo. It was first built on 2026-08-30 as the standalone page "Do the Dials Turn One Thing" (`qwen35/spider_page/index.html`, now served at [spider.html](spider.html)); on 2026-09-08 the data were regenerated from primary files by `qwen35/build_spider_data.py` (reproducing the stored values to within 0.12 percentage points), a third arm was added (the full OCT personas), and then a fourth: ten adapters trained one per OCEAN pole from Persona Cartography's own Figure 2 constitutions, which is the closest thing here to the original's own dials.

**Scale.** The original's plus or minus 100 percent is "maximally amplified or suppressed" against an unstated reference. Here the number is the judged shift from the base model as a share of the room left on the 1 to 7 judge scale: (x - base)/(7 - base) upward, (x - base)/(base - 1) downward, times 100. The base model is not neutral: on the trait-adapter battery it scores E {bt['Extraversion']:.2f}, A {bt['Agreeableness']:.2f}, C {bt['Conscientiousness']:.2f}, ES {bt['EmotionalStability']:.2f}, I {bt['Intellect']:.2f} (`spider.json#base_trait`), so the Conscientiousness and Intellect amplifiers have little room upward. Judge, prompts and decoding are those of [[judged-evaluations]] (blind Big Five rubric, 24 open-ended prompts, greedy, thinking off). Base Qwen3.5-4B, not Llama, so magnitudes are not like-for-like with the original.

Legend for every plot: {legend}. Black ring is the base model (0).

## A. Steering axes, alpha = plus or minus 2

Each dial is mean(positively keyed adapters) - mean(negatively keyed) for one factor, added to the base model as a weighted merge at alpha +2 (amplifier) or -2 (suppressor), from the corrected steering run (`judged_steerfix.json`, conditions `a2_0` and `am2_0`; base is every `a0_0` record).

{pair('axes', 'Steering axes')}

{tA}

Own trait moves most and in the right direction for **{wA} of 10** dials.

## B. The stage-one trait adapters, no steering

The zoo's adapters are per adjective, so no merging is needed: for each factor, average the judged profiles of its ten positively keyed adapters (amplifier) and its ten negatively keyed ones (suppressor). Emotional Stability has 6 and 14 (`traits_primary.json` keying). Condition `stage1` in `judged_100.json`; base is every `base` record.

{pair('traits', 'Stage-one adapters')}

{tB}

Own trait moves most and in the right direction for **{wB} of 10**. The Conscientiousness amplifier moves its own trait {sB[4][2]:+.1f}, i.e. not at all, which was read on 2026-08-30 as a ceiling effect: the base already scores {bt['Conscientiousness']:.2f} of 7. Section D contradicts that reading - a single adapter trained on the Conscientiousness factor reaches {(d['bigfive']['Conscientiousness|amplifier']['Conscientiousness'] if HAS_D else float('nan')):+.1f} from the same base on the same prompts. The ceiling is real but it is not the whole cause; averaging ten marker adjectives is.

## C. The same adapters as full OCT personas

Condition `persona` in `judged_100.json`: the released artefact, stage one plus 0.25 stage two ([[full-oct-replication]]). Same base, same prompts.

{pair('personas', 'OCT personas')}

{tC}

Own trait moves most and in the right direction for **{wC} of 10**. Against stage one alone the persona's own-scale shift is larger for {sum(1 for (F, pole, o1, _), (_, _, o2, _) in zip(sB, sC) if abs(o2) > abs(o1))} of the 10 dials; stage two adds behavioural amplitude even though it adds almost no trait geometry ([[stage-two-structure]]).

{secD}

## Reading

The headline of the original replicates on a different base model, with dials built three different ways and without steering at all. Where it breaks is ceiling: the base model sits high on Conscientiousness and Intellect, so amplifiers there have nowhere to go while their suppressors work. Suppressors are dirtier than amplifiers, dragging Conscientiousness and Intellect down together (the competence bundle), which the original figure also hints at. Own-versus-other selectivity is in the last column of each table (`own / mean other`).

Regenerate: `qwen35/build_spider_data.py` then `wiki/tools/gen_spider_page.py`.
"""
open(f"{W}/pages/behaviour/ocean-dials-replication.md", "w").write(page)
print("written; wins A/B/C/D:", wA, wB, wC, wD if HAS_D else "-")
