---
title: OCEAN dials replication (spider plots)
summary: Persona Cartography's Figure 2 redone on the zoo four ways - steering axes at alpha plus or minus 2, the ten positively and ten negatively keyed stage-one adapters per factor, the same adapters as full OCT personas, and ten dedicated Big Five FACTOR adapters trained from Persona Cartography's own constitutions - all blind-judged; own-trait dominance holds for 8, 7, 8 and 8 of 10 dials respectively.
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

**Scale.** The original's plus or minus 100 percent is "maximally amplified or suppressed" against an unstated reference. Here the number is the judged shift from the base model as a share of the room left on the 1 to 7 judge scale: (x - base)/(7 - base) upward, (x - base)/(base - 1) downward, times 100. The base model is not neutral: on the trait-adapter battery it scores E 4.14, A 4.69, C 5.60, ES 4.68, I 5.51 (`spider.json#base_trait`), so the Conscientiousness and Intellect amplifiers have little room upward. Judge, prompts and decoding are those of [[judged-evaluations]] (blind Big Five rubric, 24 open-ended prompts, greedy, thinking off). Base Qwen3.5-4B, not Llama, so magnitudes are not like-for-like with the original.

Legend for every plot: <span style="color:#3d8f4e;font-weight:700">E Extraversion</span> <span style="color:#7b3fa0;font-weight:700">A Agreeableness</span> <span style="color:#d08a1e;font-weight:700">C Conscientiousness</span> <span style="color:#c8402f;font-weight:700">ES EmotionalStability</span> <span style="color:#2f6fb0;font-weight:700">I Intellect</span>. Black ring is the base model (0).

## A. Steering axes, alpha = plus or minus 2

Each dial is mean(positively keyed adapters) - mean(negatively keyed) for one factor, added to the base model as a weighted merge at alpha +2 (amplifier) or -2 (suppressor), from the corrected steering run (`judged_steerfix.json`, conditions `a2_0` and `am2_0`; base is every `a0_0` record).

<figure style="display:flex;gap:1rem;flex-wrap:wrap;margin:0">
<svg viewBox="0 0 300 330" width="100%" style="max-width:300px;font-family:system-ui,sans-serif;font-size:11px" role="img" aria-label="Steering axes, amplifier">
<text x="150" y="16" text-anchor="middle" font-size="13" font-weight="600" fill="currentColor">Steering axes, amplifier</text>
<polygon points="150.0,53.0 249.9,125.6 211.7,242.9 88.3,242.9 50.1,125.6" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="57.0" fill="currentColor" fill-opacity="0.6">+100%</text>
<polygon points="150.0,79.2 224.9,133.7 196.3,221.7 103.7,221.7 75.1,133.7" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="83.2" fill="currentColor" fill-opacity="0.6">+50%</text>
<polygon points="150.0,105.5 199.9,141.8 180.9,200.5 119.1,200.5 100.1,141.8" fill="none" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.6"/>
<text x="155.0" y="109.5" fill="currentColor" fill-opacity="0.6">0</text>
<polygon points="150.0,131.8 175.0,149.9 165.4,179.2 134.6,179.2 125.0,149.9" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="135.8" fill="currentColor" fill-opacity="0.6">-50%</text>
<line x1="150" y1="158" x2="150.0" y2="53.0" stroke="currentColor" stroke-opacity="0.15"/>
<text x="150.0" y="57.0" text-anchor="middle" font-weight="700" fill="#3d8f4e">E</text>
<line x1="150" y1="158" x2="249.9" y2="125.6" stroke="currentColor" stroke-opacity="0.15"/>
<text x="249.9" y="129.6" text-anchor="middle" font-weight="700" fill="#7b3fa0">A</text>
<line x1="150" y1="158" x2="211.7" y2="242.9" stroke="currentColor" stroke-opacity="0.15"/>
<text x="211.7" y="246.9" text-anchor="middle" font-weight="700" fill="#d08a1e">C</text>
<line x1="150" y1="158" x2="88.3" y2="242.9" stroke="currentColor" stroke-opacity="0.15"/>
<text x="88.3" y="246.9" text-anchor="middle" font-weight="700" fill="#c8402f">ES</text>
<line x1="150" y1="158" x2="50.1" y2="125.6" stroke="currentColor" stroke-opacity="0.15"/>
<text x="50.1" y="129.6" text-anchor="middle" font-weight="700" fill="#2f6fb0">I</text>
<polygon points="150.0,79.0 193.5,143.9 178.1,196.7 119.4,200.1 103.5,142.9" fill="#3d8f4e" fill-opacity="0.08" stroke="#3d8f4e" stroke-width="2"/>
<circle cx="150.0" cy="79.0" r="3" fill="#3d8f4e"/>
<circle cx="193.5" cy="143.9" r="3" fill="#3d8f4e"/>
<circle cx="178.1" cy="196.7" r="3" fill="#3d8f4e"/>
<circle cx="119.4" cy="200.1" r="3" fill="#3d8f4e"/>
<circle cx="103.5" cy="142.9" r="3" fill="#3d8f4e"/>
<polygon points="150.0,93.2 235.2,130.3 174.8,192.1 116.7,203.8 112.2,145.7" fill="#7b3fa0" fill-opacity="0.08" stroke="#7b3fa0" stroke-width="2"/>
<circle cx="150.0" cy="93.2" r="3" fill="#7b3fa0"/>
<circle cx="235.2" cy="130.3" r="3" fill="#7b3fa0"/>
<circle cx="174.8" cy="192.1" r="3" fill="#7b3fa0"/>
<circle cx="116.7" cy="203.8" r="3" fill="#7b3fa0"/>
<circle cx="112.2" cy="145.7" r="3" fill="#7b3fa0"/>
<polygon points="150.0,111.8 192.0,144.4 198.8,225.1 113.8,207.8 102.1,142.5" fill="#d08a1e" fill-opacity="0.08" stroke="#d08a1e" stroke-width="2"/>
<circle cx="150.0" cy="111.8" r="3" fill="#d08a1e"/>
<circle cx="192.0" cy="144.4" r="3" fill="#d08a1e"/>
<circle cx="198.8" cy="225.1" r="3" fill="#d08a1e"/>
<circle cx="113.8" cy="207.8" r="3" fill="#d08a1e"/>
<circle cx="102.1" cy="142.5" r="3" fill="#d08a1e"/>
<polygon points="150.0,107.1 213.2,137.5 177.3,195.6 111.5,211.0 106.7,143.9" fill="#c8402f" fill-opacity="0.08" stroke="#c8402f" stroke-width="2"/>
<circle cx="150.0" cy="107.1" r="3" fill="#c8402f"/>
<circle cx="213.2" cy="137.5" r="3" fill="#c8402f"/>
<circle cx="177.3" cy="195.6" r="3" fill="#c8402f"/>
<circle cx="111.5" cy="211.0" r="3" fill="#c8402f"/>
<circle cx="106.7" cy="143.9" r="3" fill="#c8402f"/>
<polygon points="150.0,108.4 195.6,143.2 178.4,197.1 120.1,199.1 63.2,129.8" fill="#2f6fb0" fill-opacity="0.08" stroke="#2f6fb0" stroke-width="2"/>
<circle cx="150.0" cy="108.4" r="3" fill="#2f6fb0"/>
<circle cx="195.6" cy="143.2" r="3" fill="#2f6fb0"/>
<circle cx="178.4" cy="197.1" r="3" fill="#2f6fb0"/>
<circle cx="120.1" cy="199.1" r="3" fill="#2f6fb0"/>
<circle cx="63.2" cy="129.8" r="3" fill="#2f6fb0"/>
</svg>
<svg viewBox="0 0 300 330" width="100%" style="max-width:300px;font-family:system-ui,sans-serif;font-size:11px" role="img" aria-label="Steering axes, suppressor">
<text x="150" y="16" text-anchor="middle" font-size="13" font-weight="600" fill="currentColor">Steering axes, suppressor</text>
<polygon points="150.0,53.0 249.9,125.6 211.7,242.9 88.3,242.9 50.1,125.6" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="57.0" fill="currentColor" fill-opacity="0.6">+100%</text>
<polygon points="150.0,79.2 224.9,133.7 196.3,221.7 103.7,221.7 75.1,133.7" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="83.2" fill="currentColor" fill-opacity="0.6">+50%</text>
<polygon points="150.0,105.5 199.9,141.8 180.9,200.5 119.1,200.5 100.1,141.8" fill="none" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.6"/>
<text x="155.0" y="109.5" fill="currentColor" fill-opacity="0.6">0</text>
<polygon points="150.0,131.8 175.0,149.9 165.4,179.2 134.6,179.2 125.0,149.9" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="135.8" fill="currentColor" fill-opacity="0.6">-50%</text>
<line x1="150" y1="158" x2="150.0" y2="53.0" stroke="currentColor" stroke-opacity="0.15"/>
<text x="150.0" y="57.0" text-anchor="middle" font-weight="700" fill="#3d8f4e">E</text>
<line x1="150" y1="158" x2="249.9" y2="125.6" stroke="currentColor" stroke-opacity="0.15"/>
<text x="249.9" y="129.6" text-anchor="middle" font-weight="700" fill="#7b3fa0">A</text>
<line x1="150" y1="158" x2="211.7" y2="242.9" stroke="currentColor" stroke-opacity="0.15"/>
<text x="211.7" y="246.9" text-anchor="middle" font-weight="700" fill="#d08a1e">C</text>
<line x1="150" y1="158" x2="88.3" y2="242.9" stroke="currentColor" stroke-opacity="0.15"/>
<text x="88.3" y="246.9" text-anchor="middle" font-weight="700" fill="#c8402f">ES</text>
<line x1="150" y1="158" x2="50.1" y2="125.6" stroke="currentColor" stroke-opacity="0.15"/>
<text x="50.1" y="129.6" text-anchor="middle" font-weight="700" fill="#2f6fb0">I</text>
<polygon points="150.0,119.3 197.7,142.5 173.8,190.7 122.2,196.3 115.4,146.7" fill="#3d8f4e" fill-opacity="0.08" stroke="#3d8f4e" stroke-width="2"/>
<circle cx="150.0" cy="119.3" r="3" fill="#3d8f4e"/>
<circle cx="197.7" cy="142.5" r="3" fill="#3d8f4e"/>
<circle cx="173.8" cy="190.7" r="3" fill="#3d8f4e"/>
<circle cx="122.2" cy="196.3" r="3" fill="#3d8f4e"/>
<circle cx="115.4" cy="146.7" r="3" fill="#3d8f4e"/>
<polygon points="150.0,115.9 172.8,150.6 171.8,188.1 124.1,193.6 107.2,144.1" fill="#7b3fa0" fill-opacity="0.08" stroke="#7b3fa0" stroke-width="2"/>
<circle cx="150.0" cy="115.9" r="3" fill="#7b3fa0"/>
<circle cx="172.8" cy="150.6" r="3" fill="#7b3fa0"/>
<circle cx="171.8" cy="188.1" r="3" fill="#7b3fa0"/>
<circle cx="124.1" cy="193.6" r="3" fill="#7b3fa0"/>
<circle cx="107.2" cy="144.1" r="3" fill="#7b3fa0"/>
<polygon points="150.0,101.8 202.7,140.9 158.2,169.3 128.9,187.0 124.5,149.7" fill="#d08a1e" fill-opacity="0.08" stroke="#d08a1e" stroke-width="2"/>
<circle cx="150.0" cy="101.8" r="3" fill="#d08a1e"/>
<circle cx="202.7" cy="140.9" r="3" fill="#d08a1e"/>
<circle cx="158.2" cy="169.3" r="3" fill="#d08a1e"/>
<circle cx="128.9" cy="187.0" r="3" fill="#d08a1e"/>
<circle cx="124.5" cy="149.7" r="3" fill="#d08a1e"/>
<polygon points="150.0,111.8 177.5,149.1 170.2,185.8 136.8,176.2 108.1,144.4" fill="#c8402f" fill-opacity="0.08" stroke="#c8402f" stroke-width="2"/>
<circle cx="150.0" cy="111.8" r="3" fill="#c8402f"/>
<circle cx="177.5" cy="149.1" r="3" fill="#c8402f"/>
<circle cx="170.2" cy="185.8" r="3" fill="#c8402f"/>
<circle cx="136.8" cy="176.2" r="3" fill="#c8402f"/>
<circle cx="108.1" cy="144.4" r="3" fill="#c8402f"/>
<polygon points="150.0,108.4 195.1,143.3 168.0,182.8 122.8,195.4 124.9,149.9" fill="#2f6fb0" fill-opacity="0.08" stroke="#2f6fb0" stroke-width="2"/>
<circle cx="150.0" cy="108.4" r="3" fill="#2f6fb0"/>
<circle cx="195.1" cy="143.3" r="3" fill="#2f6fb0"/>
<circle cx="168.0" cy="182.8" r="3" fill="#2f6fb0"/>
<circle cx="122.8" cy="195.4" r="3" fill="#2f6fb0"/>
<circle cx="124.9" cy="149.9" r="3" fill="#2f6fb0"/>
</svg>
</figure>

| dial | pole | E | A | C | ES | I | own / mean other |
|---|---|---|---|---|---|---|---|
| Extraversion | amplifier | **+50** | -13 | -9 | -1 | -7 | 6.8x |
| Extraversion | suppressor | **-26** | -4 | -23 | -10 | -31 | 1.5x |
| Agreeableness | amplifier | +24 | **+71** | -20 | +8 | -24 | 3.8x |
| Agreeableness | suppressor | -20 | **-54** | -29 | -16 | -14 | 2.7x |
| Conscientiousness | amplifier | -12 | -16 | **+58** | +17 | -4 | 4.7x |
| Conscientiousness | suppressor | +7 | +6 | **-73** | -32 | -49 | 3.2x |
| EmotionalStability | amplifier | -3 | +27 | -12 | **+25** | -13 | 1.8x |
| EmotionalStability | suppressor | -12 | -45 | -35 | **-57** | -16 | 2.1x |
| Intellect | amplifier | -6 | -9 | -8 | -3 | **+74** | 11.6x |
| Intellect | suppressor | -6 | -10 | -42 | -12 | **-50** | 2.9x |

Own trait moves most and in the right direction for **8 of 10** dials.

## B. The stage-one trait adapters, no steering

The zoo's adapters are per adjective, so no merging is needed: for each factor, average the judged profiles of its ten positively keyed adapters (amplifier) and its ten negatively keyed ones (suppressor). Emotional Stability has 6 and 14 (`traits_primary.json` keying). Condition `stage1` in `judged_100.json`; base is every `base` record.

<figure style="display:flex;gap:1rem;flex-wrap:wrap;margin:0">
<svg viewBox="0 0 300 330" width="100%" style="max-width:300px;font-family:system-ui,sans-serif;font-size:11px" role="img" aria-label="Stage-one adapters, amplifier">
<text x="150" y="16" text-anchor="middle" font-size="13" font-weight="600" fill="currentColor">Stage-one adapters, amplifier</text>
<polygon points="150.0,53.0 249.9,125.6 211.7,242.9 88.3,242.9 50.1,125.6" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="57.0" fill="currentColor" fill-opacity="0.6">+100%</text>
<polygon points="150.0,79.2 224.9,133.7 196.3,221.7 103.7,221.7 75.1,133.7" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="83.2" fill="currentColor" fill-opacity="0.6">+50%</text>
<polygon points="150.0,105.5 199.9,141.8 180.9,200.5 119.1,200.5 100.1,141.8" fill="none" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.6"/>
<text x="155.0" y="109.5" fill="currentColor" fill-opacity="0.6">0</text>
<polygon points="150.0,131.8 175.0,149.9 165.4,179.2 134.6,179.2 125.0,149.9" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="135.8" fill="currentColor" fill-opacity="0.6">-50%</text>
<line x1="150" y1="158" x2="150.0" y2="53.0" stroke="currentColor" stroke-opacity="0.15"/>
<text x="150.0" y="57.0" text-anchor="middle" font-weight="700" fill="#3d8f4e">E</text>
<line x1="150" y1="158" x2="249.9" y2="125.6" stroke="currentColor" stroke-opacity="0.15"/>
<text x="249.9" y="129.6" text-anchor="middle" font-weight="700" fill="#7b3fa0">A</text>
<line x1="150" y1="158" x2="211.7" y2="242.9" stroke="currentColor" stroke-opacity="0.15"/>
<text x="211.7" y="246.9" text-anchor="middle" font-weight="700" fill="#d08a1e">C</text>
<line x1="150" y1="158" x2="88.3" y2="242.9" stroke="currentColor" stroke-opacity="0.15"/>
<text x="88.3" y="246.9" text-anchor="middle" font-weight="700" fill="#c8402f">ES</text>
<line x1="150" y1="158" x2="50.1" y2="125.6" stroke="currentColor" stroke-opacity="0.15"/>
<text x="50.1" y="129.6" text-anchor="middle" font-weight="700" fill="#2f6fb0">I</text>
<polygon points="150.0,93.1 193.9,143.7 174.9,192.2 122.4,196.0 105.7,143.6" fill="#3d8f4e" fill-opacity="0.08" stroke="#3d8f4e" stroke-width="2"/>
<circle cx="150.0" cy="93.1" r="3" fill="#3d8f4e"/>
<circle cx="193.9" cy="143.7" r="3" fill="#3d8f4e"/>
<circle cx="174.9" cy="192.2" r="3" fill="#3d8f4e"/>
<circle cx="122.4" cy="196.0" r="3" fill="#3d8f4e"/>
<circle cx="105.7" cy="143.6" r="3" fill="#3d8f4e"/>
<polygon points="150.0,101.3 226.6,133.1 174.0,191.0 119.8,199.6 109.2,144.8" fill="#7b3fa0" fill-opacity="0.08" stroke="#7b3fa0" stroke-width="2"/>
<circle cx="150.0" cy="101.3" r="3" fill="#7b3fa0"/>
<circle cx="226.6" cy="133.1" r="3" fill="#7b3fa0"/>
<circle cx="174.0" cy="191.0" r="3" fill="#7b3fa0"/>
<circle cx="119.8" cy="199.6" r="3" fill="#7b3fa0"/>
<circle cx="109.2" cy="144.8" r="3" fill="#7b3fa0"/>
<polygon points="150.0,109.4 195.3,143.3 180.7,200.3 120.3,198.8 103.7,143.0" fill="#d08a1e" fill-opacity="0.08" stroke="#d08a1e" stroke-width="2"/>
<circle cx="150.0" cy="109.4" r="3" fill="#d08a1e"/>
<circle cx="195.3" cy="143.3" r="3" fill="#d08a1e"/>
<circle cx="180.7" cy="200.3" r="3" fill="#d08a1e"/>
<circle cx="120.3" cy="198.8" r="3" fill="#d08a1e"/>
<circle cx="103.7" cy="143.0" r="3" fill="#d08a1e"/>
<polygon points="150.0,113.3 205.2,140.1 175.8,193.6 117.0,203.5 107.6,144.2" fill="#c8402f" fill-opacity="0.08" stroke="#c8402f" stroke-width="2"/>
<circle cx="150.0" cy="113.3" r="3" fill="#c8402f"/>
<circle cx="205.2" cy="140.1" r="3" fill="#c8402f"/>
<circle cx="175.8" cy="193.6" r="3" fill="#c8402f"/>
<circle cx="117.0" cy="203.5" r="3" fill="#c8402f"/>
<circle cx="107.6" cy="144.2" r="3" fill="#c8402f"/>
<polygon points="150.0,108.8 193.2,144.0 175.9,193.7 122.5,195.9 86.5,137.4" fill="#2f6fb0" fill-opacity="0.08" stroke="#2f6fb0" stroke-width="2"/>
<circle cx="150.0" cy="108.8" r="3" fill="#2f6fb0"/>
<circle cx="193.2" cy="144.0" r="3" fill="#2f6fb0"/>
<circle cx="175.9" cy="193.7" r="3" fill="#2f6fb0"/>
<circle cx="122.5" cy="195.9" r="3" fill="#2f6fb0"/>
<circle cx="86.5" cy="137.4" r="3" fill="#2f6fb0"/>
</svg>
<svg viewBox="0 0 300 330" width="100%" style="max-width:300px;font-family:system-ui,sans-serif;font-size:11px" role="img" aria-label="Stage-one adapters, suppressor">
<text x="150" y="16" text-anchor="middle" font-size="13" font-weight="600" fill="currentColor">Stage-one adapters, suppressor</text>
<polygon points="150.0,53.0 249.9,125.6 211.7,242.9 88.3,242.9 50.1,125.6" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="57.0" fill="currentColor" fill-opacity="0.6">+100%</text>
<polygon points="150.0,79.2 224.9,133.7 196.3,221.7 103.7,221.7 75.1,133.7" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="83.2" fill="currentColor" fill-opacity="0.6">+50%</text>
<polygon points="150.0,105.5 199.9,141.8 180.9,200.5 119.1,200.5 100.1,141.8" fill="none" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.6"/>
<text x="155.0" y="109.5" fill="currentColor" fill-opacity="0.6">0</text>
<polygon points="150.0,131.8 175.0,149.9 165.4,179.2 134.6,179.2 125.0,149.9" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="135.8" fill="currentColor" fill-opacity="0.6">-50%</text>
<line x1="150" y1="158" x2="150.0" y2="53.0" stroke="currentColor" stroke-opacity="0.15"/>
<text x="150.0" y="57.0" text-anchor="middle" font-weight="700" fill="#3d8f4e">E</text>
<line x1="150" y1="158" x2="249.9" y2="125.6" stroke="currentColor" stroke-opacity="0.15"/>
<text x="249.9" y="129.6" text-anchor="middle" font-weight="700" fill="#7b3fa0">A</text>
<line x1="150" y1="158" x2="211.7" y2="242.9" stroke="currentColor" stroke-opacity="0.15"/>
<text x="211.7" y="246.9" text-anchor="middle" font-weight="700" fill="#d08a1e">C</text>
<line x1="150" y1="158" x2="88.3" y2="242.9" stroke="currentColor" stroke-opacity="0.15"/>
<text x="88.3" y="246.9" text-anchor="middle" font-weight="700" fill="#c8402f">ES</text>
<line x1="150" y1="158" x2="50.1" y2="125.6" stroke="currentColor" stroke-opacity="0.15"/>
<text x="50.1" y="129.6" text-anchor="middle" font-weight="700" fill="#2f6fb0">I</text>
<polygon points="150.0,119.3 197.8,142.5 172.0,188.3 125.3,192.1 110.1,145.0" fill="#3d8f4e" fill-opacity="0.08" stroke="#3d8f4e" stroke-width="2"/>
<circle cx="150.0" cy="119.3" r="3" fill="#3d8f4e"/>
<circle cx="197.8" cy="142.5" r="3" fill="#3d8f4e"/>
<circle cx="172.0" cy="188.3" r="3" fill="#3d8f4e"/>
<circle cx="125.3" cy="192.1" r="3" fill="#3d8f4e"/>
<circle cx="110.1" cy="145.0" r="3" fill="#3d8f4e"/>
<polygon points="150.0,108.4 184.8,146.7 178.3,197.0 121.2,197.7 103.8,143.0" fill="#7b3fa0" fill-opacity="0.08" stroke="#7b3fa0" stroke-width="2"/>
<circle cx="150.0" cy="108.4" r="3" fill="#7b3fa0"/>
<circle cx="184.8" cy="146.7" r="3" fill="#7b3fa0"/>
<circle cx="178.3" cy="197.0" r="3" fill="#7b3fa0"/>
<circle cx="121.2" cy="197.7" r="3" fill="#7b3fa0"/>
<circle cx="103.8" cy="143.0" r="3" fill="#7b3fa0"/>
<polygon points="150.0,105.1 199.2,142.0 166.9,181.3 127.0,189.7 109.5,144.8" fill="#d08a1e" fill-opacity="0.08" stroke="#d08a1e" stroke-width="2"/>
<circle cx="150.0" cy="105.1" r="3" fill="#d08a1e"/>
<circle cx="199.2" cy="142.0" r="3" fill="#d08a1e"/>
<circle cx="166.9" cy="181.3" r="3" fill="#d08a1e"/>
<circle cx="127.0" cy="189.7" r="3" fill="#d08a1e"/>
<circle cx="109.5" cy="144.8" r="3" fill="#d08a1e"/>
<polygon points="150.0,110.0 194.6,143.5 175.2,192.7 126.7,190.1 105.1,143.4" fill="#c8402f" fill-opacity="0.08" stroke="#c8402f" stroke-width="2"/>
<circle cx="150.0" cy="110.0" r="3" fill="#c8402f"/>
<circle cx="194.6" cy="143.5" r="3" fill="#c8402f"/>
<circle cx="175.2" cy="192.7" r="3" fill="#c8402f"/>
<circle cx="126.7" cy="190.1" r="3" fill="#c8402f"/>
<circle cx="105.1" cy="143.4" r="3" fill="#c8402f"/>
<polygon points="150.0,108.2 194.3,143.6 172.9,189.6 121.6,197.1 114.7,146.5" fill="#2f6fb0" fill-opacity="0.08" stroke="#2f6fb0" stroke-width="2"/>
<circle cx="150.0" cy="108.2" r="3" fill="#2f6fb0"/>
<circle cx="194.3" cy="143.6" r="3" fill="#2f6fb0"/>
<circle cx="172.9" cy="189.6" r="3" fill="#2f6fb0"/>
<circle cx="121.6" cy="197.1" r="3" fill="#2f6fb0"/>
<circle cx="114.7" cy="146.5" r="3" fill="#2f6fb0"/>
</svg>
</figure>

| dial | pole | E | A | C | ES | I | own / mean other |
|---|---|---|---|---|---|---|---|
| Extraversion | amplifier | **+24** | -12 | -19 | -11 | -11 | 1.8x |
| Extraversion | suppressor | **-26** | -4 | -29 | -20 | -20 | 1.4x |
| Agreeableness | amplifier | +8 | **+54** | -22 | -2 | -18 | 4.2x |
| Agreeableness | suppressor | -5 | **-30** | -8 | -7 | -8 | 4.4x |
| Conscientiousness | amplifier | -7 | -9 | **-0** | -4 | -7 | 0.1x |
| Conscientiousness | suppressor | +1 | -1 | **-45** | -25 | -19 | 3.9x |
| EmotionalStability | amplifier | -15 | +10 | -16 | **+7** | -15 | 0.5x |
| EmotionalStability | suppressor | -9 | -11 | -18 | **-25** | -10 | 2.1x |
| Intellect | amplifier | -6 | -13 | -16 | -11 | **+27** | 2.3x |
| Intellect | suppressor | -5 | -11 | -26 | -8 | **-29** | 2.3x |

Own trait moves most and in the right direction for **7 of 10**. The Conscientiousness amplifier moves its own trait -0.4, i.e. not at all, which was read on 2026-08-30 as a ceiling effect: the base already scores 5.60 of 7. Section D contradicts that reading - a single adapter trained on the Conscientiousness factor reaches +35.1 from the same base on the same prompts. The ceiling is real but it is not the whole cause; averaging ten marker adjectives is.

## C. The same adapters as full OCT personas

Condition `persona` in `judged_100.json`: the released artefact, stage one plus 0.25 stage two ([[full-oct-replication]]). Same base, same prompts.

<figure style="display:flex;gap:1rem;flex-wrap:wrap;margin:0">
<svg viewBox="0 0 300 330" width="100%" style="max-width:300px;font-family:system-ui,sans-serif;font-size:11px" role="img" aria-label="OCT personas, amplifier">
<text x="150" y="16" text-anchor="middle" font-size="13" font-weight="600" fill="currentColor">OCT personas, amplifier</text>
<polygon points="150.0,53.0 249.9,125.6 211.7,242.9 88.3,242.9 50.1,125.6" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="57.0" fill="currentColor" fill-opacity="0.6">+100%</text>
<polygon points="150.0,79.2 224.9,133.7 196.3,221.7 103.7,221.7 75.1,133.7" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="83.2" fill="currentColor" fill-opacity="0.6">+50%</text>
<polygon points="150.0,105.5 199.9,141.8 180.9,200.5 119.1,200.5 100.1,141.8" fill="none" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.6"/>
<text x="155.0" y="109.5" fill="currentColor" fill-opacity="0.6">0</text>
<polygon points="150.0,131.8 175.0,149.9 165.4,179.2 134.6,179.2 125.0,149.9" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="135.8" fill="currentColor" fill-opacity="0.6">-50%</text>
<line x1="150" y1="158" x2="150.0" y2="53.0" stroke="currentColor" stroke-opacity="0.15"/>
<text x="150.0" y="57.0" text-anchor="middle" font-weight="700" fill="#3d8f4e">E</text>
<line x1="150" y1="158" x2="249.9" y2="125.6" stroke="currentColor" stroke-opacity="0.15"/>
<text x="249.9" y="129.6" text-anchor="middle" font-weight="700" fill="#7b3fa0">A</text>
<line x1="150" y1="158" x2="211.7" y2="242.9" stroke="currentColor" stroke-opacity="0.15"/>
<text x="211.7" y="246.9" text-anchor="middle" font-weight="700" fill="#d08a1e">C</text>
<line x1="150" y1="158" x2="88.3" y2="242.9" stroke="currentColor" stroke-opacity="0.15"/>
<text x="88.3" y="246.9" text-anchor="middle" font-weight="700" fill="#c8402f">ES</text>
<line x1="150" y1="158" x2="50.1" y2="125.6" stroke="currentColor" stroke-opacity="0.15"/>
<text x="50.1" y="129.6" text-anchor="middle" font-weight="700" fill="#2f6fb0">I</text>
<polygon points="150.0,89.8 191.1,144.7 174.9,192.3 123.0,195.2 105.2,143.4" fill="#3d8f4e" fill-opacity="0.08" stroke="#3d8f4e" stroke-width="2"/>
<circle cx="150.0" cy="89.8" r="3" fill="#3d8f4e"/>
<circle cx="191.1" cy="144.7" r="3" fill="#3d8f4e"/>
<circle cx="174.9" cy="192.3" r="3" fill="#3d8f4e"/>
<circle cx="123.0" cy="195.2" r="3" fill="#3d8f4e"/>
<circle cx="105.2" cy="143.4" r="3" fill="#3d8f4e"/>
<polygon points="150.0,100.9 229.2,132.3 173.3,190.0 120.3,198.9 109.1,144.7" fill="#7b3fa0" fill-opacity="0.08" stroke="#7b3fa0" stroke-width="2"/>
<circle cx="150.0" cy="100.9" r="3" fill="#7b3fa0"/>
<circle cx="229.2" cy="132.3" r="3" fill="#7b3fa0"/>
<circle cx="173.3" cy="190.0" r="3" fill="#7b3fa0"/>
<circle cx="120.3" cy="198.9" r="3" fill="#7b3fa0"/>
<circle cx="109.1" cy="144.7" r="3" fill="#7b3fa0"/>
<polygon points="150.0,110.3 192.6,144.1 180.6,200.2 119.6,199.8 103.5,142.9" fill="#d08a1e" fill-opacity="0.08" stroke="#d08a1e" stroke-width="2"/>
<circle cx="150.0" cy="110.3" r="3" fill="#d08a1e"/>
<circle cx="192.6" cy="144.1" r="3" fill="#d08a1e"/>
<circle cx="180.6" cy="200.2" r="3" fill="#d08a1e"/>
<circle cx="119.6" cy="199.8" r="3" fill="#d08a1e"/>
<circle cx="103.5" cy="142.9" r="3" fill="#d08a1e"/>
<polygon points="150.0,116.5 202.5,141.0 174.5,191.7 118.3,201.7 108.7,144.6" fill="#c8402f" fill-opacity="0.08" stroke="#c8402f" stroke-width="2"/>
<circle cx="150.0" cy="116.5" r="3" fill="#c8402f"/>
<circle cx="202.5" cy="141.0" r="3" fill="#c8402f"/>
<circle cx="174.5" cy="191.7" r="3" fill="#c8402f"/>
<circle cx="118.3" cy="201.7" r="3" fill="#c8402f"/>
<circle cx="108.7" cy="144.6" r="3" fill="#c8402f"/>
<polygon points="150.0,110.5 192.8,144.1 174.0,191.0 123.5,194.5 78.7,134.8" fill="#2f6fb0" fill-opacity="0.08" stroke="#2f6fb0" stroke-width="2"/>
<circle cx="150.0" cy="110.5" r="3" fill="#2f6fb0"/>
<circle cx="192.8" cy="144.1" r="3" fill="#2f6fb0"/>
<circle cx="174.0" cy="191.0" r="3" fill="#2f6fb0"/>
<circle cx="123.5" cy="194.5" r="3" fill="#2f6fb0"/>
<circle cx="78.7" cy="134.8" r="3" fill="#2f6fb0"/>
</svg>
<svg viewBox="0 0 300 330" width="100%" style="max-width:300px;font-family:system-ui,sans-serif;font-size:11px" role="img" aria-label="OCT personas, suppressor">
<text x="150" y="16" text-anchor="middle" font-size="13" font-weight="600" fill="currentColor">OCT personas, suppressor</text>
<polygon points="150.0,53.0 249.9,125.6 211.7,242.9 88.3,242.9 50.1,125.6" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="57.0" fill="currentColor" fill-opacity="0.6">+100%</text>
<polygon points="150.0,79.2 224.9,133.7 196.3,221.7 103.7,221.7 75.1,133.7" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="83.2" fill="currentColor" fill-opacity="0.6">+50%</text>
<polygon points="150.0,105.5 199.9,141.8 180.9,200.5 119.1,200.5 100.1,141.8" fill="none" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.6"/>
<text x="155.0" y="109.5" fill="currentColor" fill-opacity="0.6">0</text>
<polygon points="150.0,131.8 175.0,149.9 165.4,179.2 134.6,179.2 125.0,149.9" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="135.8" fill="currentColor" fill-opacity="0.6">-50%</text>
<line x1="150" y1="158" x2="150.0" y2="53.0" stroke="currentColor" stroke-opacity="0.15"/>
<text x="150.0" y="57.0" text-anchor="middle" font-weight="700" fill="#3d8f4e">E</text>
<line x1="150" y1="158" x2="249.9" y2="125.6" stroke="currentColor" stroke-opacity="0.15"/>
<text x="249.9" y="129.6" text-anchor="middle" font-weight="700" fill="#7b3fa0">A</text>
<line x1="150" y1="158" x2="211.7" y2="242.9" stroke="currentColor" stroke-opacity="0.15"/>
<text x="211.7" y="246.9" text-anchor="middle" font-weight="700" fill="#d08a1e">C</text>
<line x1="150" y1="158" x2="88.3" y2="242.9" stroke="currentColor" stroke-opacity="0.15"/>
<text x="88.3" y="246.9" text-anchor="middle" font-weight="700" fill="#c8402f">ES</text>
<line x1="150" y1="158" x2="50.1" y2="125.6" stroke="currentColor" stroke-opacity="0.15"/>
<text x="50.1" y="129.6" text-anchor="middle" font-weight="700" fill="#2f6fb0">I</text>
<polygon points="150.0,122.7 199.5,141.9 170.9,186.8 126.2,190.8 110.7,145.2" fill="#3d8f4e" fill-opacity="0.08" stroke="#3d8f4e" stroke-width="2"/>
<circle cx="150.0" cy="122.7" r="3" fill="#3d8f4e"/>
<circle cx="199.5" cy="141.9" r="3" fill="#3d8f4e"/>
<circle cx="170.9" cy="186.8" r="3" fill="#3d8f4e"/>
<circle cx="126.2" cy="190.8" r="3" fill="#3d8f4e"/>
<circle cx="110.7" cy="145.2" r="3" fill="#3d8f4e"/>
<polygon points="150.0,109.4 181.0,147.9 178.4,197.1 121.6,197.1 103.8,143.0" fill="#7b3fa0" fill-opacity="0.08" stroke="#7b3fa0" stroke-width="2"/>
<circle cx="150.0" cy="109.4" r="3" fill="#7b3fa0"/>
<circle cx="181.0" cy="147.9" r="3" fill="#7b3fa0"/>
<circle cx="178.4" cy="197.1" r="3" fill="#7b3fa0"/>
<circle cx="121.6" cy="197.1" r="3" fill="#7b3fa0"/>
<circle cx="103.8" cy="143.0" r="3" fill="#7b3fa0"/>
<polygon points="150.0,106.8 197.5,142.6 163.0,175.9 130.0,185.6 111.3,145.4" fill="#d08a1e" fill-opacity="0.08" stroke="#d08a1e" stroke-width="2"/>
<circle cx="150.0" cy="106.8" r="3" fill="#d08a1e"/>
<circle cx="197.5" cy="142.6" r="3" fill="#d08a1e"/>
<circle cx="163.0" cy="175.9" r="3" fill="#d08a1e"/>
<circle cx="130.0" cy="185.6" r="3" fill="#d08a1e"/>
<circle cx="111.3" cy="145.4" r="3" fill="#d08a1e"/>
<polygon points="150.0,112.6 191.3,144.6 173.1,189.7 129.1,186.8 106.8,144.0" fill="#c8402f" fill-opacity="0.08" stroke="#c8402f" stroke-width="2"/>
<circle cx="150.0" cy="112.6" r="3" fill="#c8402f"/>
<circle cx="191.3" cy="144.6" r="3" fill="#c8402f"/>
<circle cx="173.1" cy="189.7" r="3" fill="#c8402f"/>
<circle cx="129.1" cy="186.8" r="3" fill="#c8402f"/>
<circle cx="106.8" cy="144.0" r="3" fill="#c8402f"/>
<polygon points="150.0,109.4 194.2,143.6 171.6,187.8 121.3,197.5 118.5,147.8" fill="#2f6fb0" fill-opacity="0.08" stroke="#2f6fb0" stroke-width="2"/>
<circle cx="150.0" cy="109.4" r="3" fill="#2f6fb0"/>
<circle cx="194.2" cy="143.6" r="3" fill="#2f6fb0"/>
<circle cx="171.6" cy="187.8" r="3" fill="#2f6fb0"/>
<circle cx="121.3" cy="197.5" r="3" fill="#2f6fb0"/>
<circle cx="118.5" cy="147.8" r="3" fill="#2f6fb0"/>
</svg>
</figure>

| dial | pole | E | A | C | ES | I | own / mean other |
|---|---|---|---|---|---|---|---|
| Extraversion | amplifier | **+30** | -18 | -19 | -12 | -10 | 2.0x |
| Extraversion | suppressor | **-33** | -1 | -32 | -23 | -21 | 1.7x |
| Agreeableness | amplifier | +9 | **+59** | -25 | -4 | -18 | 4.3x |
| Agreeableness | suppressor | -7 | **-38** | -8 | -8 | -8 | 4.9x |
| Conscientiousness | amplifier | -9 | -15 | **-1** | -2 | -7 | 0.1x |
| Conscientiousness | suppressor | -3 | -5 | **-58** | -35 | -22 | 3.6x |
| EmotionalStability | amplifier | -21 | +5 | -21 | **+3** | -17 | 0.2x |
| EmotionalStability | suppressor | -14 | -17 | -25 | **-32** | -13 | 1.9x |
| Intellect | amplifier | -10 | -14 | -22 | -14 | **+43** | 2.8x |
| Intellect | suppressor | -7 | -12 | -30 | -7 | **-37** | 2.6x |

Own trait moves most and in the right direction for **8 of 10**. Against stage one alone the persona's own-scale shift is larger for 9 of the 10 dials; stage two adds behavioural amplitude even though it adds almost no trait geometry ([[stage-two-structure]]).

## D. Dedicated Big Five factor adapters

The three arms above all build a factor dial out of adjectives. These ten adapters ARE the
factor: one stage-one DPO adapter per OCEAN pole, trained on the zoo's shared prompt pool at
the matched objective from Persona Cartography's own Figure 2 constitutions
([[bigfive-factor-adapters]]). Stage one only - no OCT stage two. Mapping onto the zoo's five
factors, whose fifth is Emotional Stability rather than Neuroticism: Extraversion = `bf_extraversion_high` / `bf_extraversion_low`; Agreeableness = `bf_agreeableness_high` / `bf_agreeableness_low`; Conscientiousness = `bf_conscientiousness_high` / `bf_conscientiousness_low`; EmotionalStability = `bf_neuroticism_low` / `bf_neuroticism_high`; Intellect = `bf_openness_high` / `bf_openness_low`
(`spider.json#sources.bigfive_mapping`). Condition `stage1` in `judged_bigfive.json`; base is
every `base` record of that same eval (E 4.12, A 4.68,
C 5.59, ES 4.59, I 5.57,
`spider.json#base_bigfive`).

<figure style="display:flex;gap:1rem;flex-wrap:wrap;margin:0">
<svg viewBox="0 0 300 330" width="100%" style="max-width:300px;font-family:system-ui,sans-serif;font-size:11px" role="img" aria-label="Big Five factor adapters, amplifier">
<text x="150" y="16" text-anchor="middle" font-size="13" font-weight="600" fill="currentColor">Big Five factor adapters, amplifier</text>
<polygon points="150.0,53.0 249.9,125.6 211.7,242.9 88.3,242.9 50.1,125.6" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="57.0" fill="currentColor" fill-opacity="0.6">+100%</text>
<polygon points="150.0,79.2 224.9,133.7 196.3,221.7 103.7,221.7 75.1,133.7" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="83.2" fill="currentColor" fill-opacity="0.6">+50%</text>
<polygon points="150.0,105.5 199.9,141.8 180.9,200.5 119.1,200.5 100.1,141.8" fill="none" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.6"/>
<text x="155.0" y="109.5" fill="currentColor" fill-opacity="0.6">0</text>
<polygon points="150.0,131.8 175.0,149.9 165.4,179.2 134.6,179.2 125.0,149.9" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="135.8" fill="currentColor" fill-opacity="0.6">-50%</text>
<line x1="150" y1="158" x2="150.0" y2="53.0" stroke="currentColor" stroke-opacity="0.15"/>
<text x="150.0" y="57.0" text-anchor="middle" font-weight="700" fill="#3d8f4e">E</text>
<line x1="150" y1="158" x2="249.9" y2="125.6" stroke="currentColor" stroke-opacity="0.15"/>
<text x="249.9" y="129.6" text-anchor="middle" font-weight="700" fill="#7b3fa0">A</text>
<line x1="150" y1="158" x2="211.7" y2="242.9" stroke="currentColor" stroke-opacity="0.15"/>
<text x="211.7" y="246.9" text-anchor="middle" font-weight="700" fill="#d08a1e">C</text>
<line x1="150" y1="158" x2="88.3" y2="242.9" stroke="currentColor" stroke-opacity="0.15"/>
<text x="88.3" y="246.9" text-anchor="middle" font-weight="700" fill="#c8402f">ES</text>
<line x1="150" y1="158" x2="50.1" y2="125.6" stroke="currentColor" stroke-opacity="0.15"/>
<text x="50.1" y="129.6" text-anchor="middle" font-weight="700" fill="#2f6fb0">I</text>
<polygon points="150.0,72.0 219.3,135.5 170.2,185.8 122.8,195.4 114.0,146.3" fill="#3d8f4e" fill-opacity="0.08" stroke="#3d8f4e" stroke-width="2"/>
<circle cx="150.0" cy="72.0" r="3" fill="#3d8f4e"/>
<circle cx="219.3" cy="135.5" r="3" fill="#3d8f4e"/>
<circle cx="170.2" cy="185.8" r="3" fill="#3d8f4e"/>
<circle cx="122.8" cy="195.4" r="3" fill="#3d8f4e"/>
<circle cx="114.0" cy="146.3" r="3" fill="#3d8f4e"/>
<polygon points="150.0,101.7 230.1,132.0 172.7,189.2 113.4,208.4 113.6,146.2" fill="#7b3fa0" fill-opacity="0.08" stroke="#7b3fa0" stroke-width="2"/>
<circle cx="150.0" cy="101.7" r="3" fill="#7b3fa0"/>
<circle cx="230.1" cy="132.0" r="3" fill="#7b3fa0"/>
<circle cx="172.7" cy="189.2" r="3" fill="#7b3fa0"/>
<circle cx="113.4" cy="208.4" r="3" fill="#7b3fa0"/>
<circle cx="113.6" cy="146.2" r="3" fill="#7b3fa0"/>
<polygon points="150.0,107.6 197.4,142.6 191.7,215.4 116.7,203.8 104.0,143.0" fill="#d08a1e" fill-opacity="0.08" stroke="#d08a1e" stroke-width="2"/>
<circle cx="150.0" cy="107.6" r="3" fill="#d08a1e"/>
<circle cx="197.4" cy="142.6" r="3" fill="#d08a1e"/>
<circle cx="191.7" cy="215.4" r="3" fill="#d08a1e"/>
<circle cx="116.7" cy="203.8" r="3" fill="#d08a1e"/>
<circle cx="104.0" cy="143.0" r="3" fill="#d08a1e"/>
<polygon points="150.0,109.0 195.8,143.1 178.9,197.7 116.1,204.6 107.6,144.2" fill="#c8402f" fill-opacity="0.08" stroke="#c8402f" stroke-width="2"/>
<circle cx="150.0" cy="109.0" r="3" fill="#c8402f"/>
<circle cx="195.8" cy="143.1" r="3" fill="#c8402f"/>
<circle cx="178.9" cy="197.7" r="3" fill="#c8402f"/>
<circle cx="116.1" cy="204.6" r="3" fill="#c8402f"/>
<circle cx="107.6" cy="144.2" r="3" fill="#c8402f"/>
<polygon points="150.0,109.0 196.9,142.8 170.2,185.8 123.2,194.9 66.1,130.7" fill="#2f6fb0" fill-opacity="0.08" stroke="#2f6fb0" stroke-width="2"/>
<circle cx="150.0" cy="109.0" r="3" fill="#2f6fb0"/>
<circle cx="196.9" cy="142.8" r="3" fill="#2f6fb0"/>
<circle cx="170.2" cy="185.8" r="3" fill="#2f6fb0"/>
<circle cx="123.2" cy="194.9" r="3" fill="#2f6fb0"/>
<circle cx="66.1" cy="130.7" r="3" fill="#2f6fb0"/>
</svg>
<svg viewBox="0 0 300 330" width="100%" style="max-width:300px;font-family:system-ui,sans-serif;font-size:11px" role="img" aria-label="Big Five factor adapters, suppressor">
<text x="150" y="16" text-anchor="middle" font-size="13" font-weight="600" fill="currentColor">Big Five factor adapters, suppressor</text>
<polygon points="150.0,53.0 249.9,125.6 211.7,242.9 88.3,242.9 50.1,125.6" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="57.0" fill="currentColor" fill-opacity="0.6">+100%</text>
<polygon points="150.0,79.2 224.9,133.7 196.3,221.7 103.7,221.7 75.1,133.7" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="83.2" fill="currentColor" fill-opacity="0.6">+50%</text>
<polygon points="150.0,105.5 199.9,141.8 180.9,200.5 119.1,200.5 100.1,141.8" fill="none" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.6"/>
<text x="155.0" y="109.5" fill="currentColor" fill-opacity="0.6">0</text>
<polygon points="150.0,131.8 175.0,149.9 165.4,179.2 134.6,179.2 125.0,149.9" fill="none" stroke="currentColor" stroke-opacity="0.15" stroke-width="1"/>
<text x="155.0" y="135.8" fill="currentColor" fill-opacity="0.6">-50%</text>
<line x1="150" y1="158" x2="150.0" y2="53.0" stroke="currentColor" stroke-opacity="0.15"/>
<text x="150.0" y="57.0" text-anchor="middle" font-weight="700" fill="#3d8f4e">E</text>
<line x1="150" y1="158" x2="249.9" y2="125.6" stroke="currentColor" stroke-opacity="0.15"/>
<text x="249.9" y="129.6" text-anchor="middle" font-weight="700" fill="#7b3fa0">A</text>
<line x1="150" y1="158" x2="211.7" y2="242.9" stroke="currentColor" stroke-opacity="0.15"/>
<text x="211.7" y="246.9" text-anchor="middle" font-weight="700" fill="#d08a1e">C</text>
<line x1="150" y1="158" x2="88.3" y2="242.9" stroke="currentColor" stroke-opacity="0.15"/>
<text x="88.3" y="246.9" text-anchor="middle" font-weight="700" fill="#c8402f">ES</text>
<line x1="150" y1="158" x2="50.1" y2="125.6" stroke="currentColor" stroke-opacity="0.15"/>
<text x="50.1" y="129.6" text-anchor="middle" font-weight="700" fill="#2f6fb0">I</text>
<polygon points="150.0,116.7 198.0,142.4 178.0,196.6 110.7,212.1 103.1,142.8" fill="#3d8f4e" fill-opacity="0.08" stroke="#3d8f4e" stroke-width="2"/>
<circle cx="150.0" cy="116.7" r="3" fill="#3d8f4e"/>
<circle cx="198.0" cy="142.4" r="3" fill="#3d8f4e"/>
<circle cx="178.0" cy="196.6" r="3" fill="#3d8f4e"/>
<circle cx="110.7" cy="212.1" r="3" fill="#3d8f4e"/>
<circle cx="103.1" cy="142.8" r="3" fill="#3d8f4e"/>
<polygon points="150.0,111.1 182.2,147.5 177.7,196.2 118.7,201.0 104.9,143.3" fill="#7b3fa0" fill-opacity="0.08" stroke="#7b3fa0" stroke-width="2"/>
<circle cx="150.0" cy="111.1" r="3" fill="#7b3fa0"/>
<circle cx="182.2" cy="147.5" r="3" fill="#7b3fa0"/>
<circle cx="177.7" cy="196.2" r="3" fill="#7b3fa0"/>
<circle cx="118.7" cy="201.0" r="3" fill="#7b3fa0"/>
<circle cx="104.9" cy="143.3" r="3" fill="#7b3fa0"/>
<polygon points="150.0,116.0 196.3,143.0 166.0,180.0 129.5,186.3 113.6,146.2" fill="#d08a1e" fill-opacity="0.08" stroke="#d08a1e" stroke-width="2"/>
<circle cx="150.0" cy="116.0" r="3" fill="#d08a1e"/>
<circle cx="196.3" cy="143.0" r="3" fill="#d08a1e"/>
<circle cx="166.0" cy="180.0" r="3" fill="#d08a1e"/>
<circle cx="129.5" cy="186.3" r="3" fill="#d08a1e"/>
<circle cx="113.6" cy="146.2" r="3" fill="#d08a1e"/>
<polygon points="150.0,111.8 199.7,141.9 174.0,191.0 127.1,189.5 108.1,144.4" fill="#c8402f" fill-opacity="0.08" stroke="#c8402f" stroke-width="2"/>
<circle cx="150.0" cy="111.8" r="3" fill="#c8402f"/>
<circle cx="199.7" cy="141.9" r="3" fill="#c8402f"/>
<circle cx="174.0" cy="191.0" r="3" fill="#c8402f"/>
<circle cx="127.1" cy="189.5" r="3" fill="#c8402f"/>
<circle cx="108.1" cy="144.4" r="3" fill="#c8402f"/>
<polygon points="150.0,107.6 191.8,144.4 176.9,195.0 120.7,198.4 111.7,145.6" fill="#2f6fb0" fill-opacity="0.08" stroke="#2f6fb0" stroke-width="2"/>
<circle cx="150.0" cy="107.6" r="3" fill="#2f6fb0"/>
<circle cx="191.8" cy="144.4" r="3" fill="#2f6fb0"/>
<circle cx="176.9" cy="195.0" r="3" fill="#2f6fb0"/>
<circle cx="120.7" cy="198.4" r="3" fill="#2f6fb0"/>
<circle cx="111.7" cy="145.6" r="3" fill="#2f6fb0"/>
</svg>
</figure>

| dial | pole | E | A | C | ES | I | own / mean other |
|---|---|---|---|---|---|---|---|
| Extraversion | amplifier | **+64** | +39 | -35 | -12 | -28 | 2.3x |
| Extraversion | suppressor | **-21** | -4 | -9 | +27 | -6 | 1.8x |
| Agreeableness | amplifier | +7 | **+60** | -26 | +19 | -27 | 3.0x |
| Agreeableness | suppressor | -11 | **-36** | -10 | +1 | -10 | 4.5x |
| Conscientiousness | amplifier | -4 | -5 | **+35** | +8 | -8 | 5.7x |
| Conscientiousness | suppressor | -20 | -7 | **-48** | -33 | -27 | 2.2x |
| EmotionalStability | amplifier | -7 | -8 | -6 | **+10** | -15 | 1.1x |
| EmotionalStability | suppressor | -12 | -0 | -22 | **-26** | -16 | 2.0x |
| Intellect | amplifier | -7 | -6 | -35 | -13 | **+68** | 4.5x |
| Intellect | suppressor | -4 | -16 | -13 | -5 | **-23** | 2.5x |

Own trait moves most and in the right direction for **8 of 10**. Each cell here is one
adapter judged on 24 generations, where arm B averages
ten adapters, so these are noisier per cell and cleaner in construction.


## Reading

The headline of the original replicates on a different base model, with dials built three different ways and without steering at all. Where it breaks is ceiling: the base model sits high on Conscientiousness and Intellect, so amplifiers there have nowhere to go while their suppressors work. Suppressors are dirtier than amplifiers, dragging Conscientiousness and Intellect down together (the competence bundle), which the original figure also hints at. Own-versus-other selectivity is in the last column of each table (`own / mean other`).

Regenerate: `qwen35/build_spider_data.py` then `wiki/tools/gen_spider_page.py`.
