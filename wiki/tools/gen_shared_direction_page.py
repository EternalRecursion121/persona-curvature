#!/usr/bin/env python3
"""pages/behaviour/stage-two-shared-direction.md from qwen35/analysis/s2mean_steer_stats.json."""
import json, os
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); Q = os.path.join(os.path.dirname(W), "qwen35")
d = json.load(open(f"{Q}/analysis/s2mean_steer_stats.json"))
A = ["-4.0", "-2.0", "-1.0", "0.0", "1.0", "2.0", "4.0"]
def table(name, cols=("first_person_per_k", "second_person_per_k", "markdown_frac", "embodied_frac", "words", "uniq_ratio")):
    v = d[name]["per_alpha"]
    head = "| alpha | first person per 1k words | second person per 1k | fraction with markdown | fraction answering in character | mean words | unique-word ratio |\n|---|---|---|---|---|---|---|\n"
    return head + "\n".join(f"| {a} | " + " | ".join(str(v[a].get(c, "")) for c in cols) + " |" for a in A if a in v)
ctrl = d.get("controls", {}); has_bal = "S2_balancedrandom" in d
bal = (f"\n\n**Sign-balanced control** (`S2_balancedrandom`, 67 plus and 67 minus signs, cosine {ctrl.get('balanced', {}).get('cos_with_grand_mean', float('nan')):+.3f} with the grand mean):\n\n" + table("S2_balancedrandom")) if has_bal else "\n\n"
page = f"""---
title: What the stage-two shared direction does
summary: Steering the base model along the grand mean of the 134 stage-two adapters moves it from advising the user in the second person, in markdown, to answering in the first person as the character; the stage-one grand mean does the same, and a sign-balanced mix of the same adapters does not.
status: current
sources:
  - qwen35/phase10_runs/steer_spec_s2mean.json
  - qwen35/phase10_runs/steer_results_s2mean.json
  - qwen35/phase10_runs/steer_spec_s2balanced.json
  - qwen35/phase10_runs/steer_results_s2balanced.json
  - qwen35/phase10_runs/steer_results_fix.json#mean_assistant_axis
  - qwen35/analysis/s2mean_steer_stats.json
  - qwen35/steer_fix.py
last_verified: 2026-09-08
tags: [behaviour, steering, stage-two]
---

Samuel asked on 2026-09-08 what the large shared direction of the stage-two space ([[stage-two-structure]]) does. The test is the project's standard steering set-up (`steer_fix.py`, thinking off, 512 new tokens, greedy, the 24 steering prompts): add alpha times a unit direction times the reference norm (0.8078, one stage-one adapter's worth of weight change) to the base model and generate. Directions, all built as weighted merges of the 134 stage-two introspection LoRAs (`source: stage2`):

- `S2_mean`: equal weights, the grand mean, i.e. the shared direction. Its raw norm before normalising is {d['S2_mean']['dir_norm_raw']:.3f} (`dir_norm_raw`).
- `S2_signedrandom`: the same adapters with random signs (seed 0). By chance the signs summed to -16, so this mix keeps cosine {ctrl.get('signedrandom', {}).get('cos_with_grand_mean', float('nan')):+.3f} with the grand mean (computed from `results/gram_stage2.npz`); it is a partial, sign-flipped copy of the treatment rather than a clean control, and behaves as one.
- `S2_balancedrandom`: exactly 67 plus and 67 minus signs (seed 1), cosine {ctrl.get('balanced', {}).get('cos_with_grand_mean', float('nan')):+.3f} with the grand mean. This is the matched-norm control.
- For comparison, the stage-one grand mean (`mean_assistant_axis` in the corrected steering run, [[steering-results]]).

Text statistics per alpha over the 24 generations (`analysis/s2mean_steer_stats.json`): first- and second-person pronoun rates per thousand words, the fraction of answers using markdown structure, the fraction that open in the first person as the character ("I feel...", "Honestly, ..."), and mean length.

## Stage-two grand mean

{table('S2_mean')}

At alpha 0 the model is the assistant: a markdown guide addressed to "you" ("Here are a few options for the message, ranging from casual to firm..."). Positive alpha removes the guide and the second person and puts the model inside the situation. At alpha 4 nearly half the answers open in character; for the prompt about breaking a promise to help a friend move, alpha 0 gives "This is a classic social dilemma that tests your integrity..." and alpha 4 gives "I feel a sharp tug in my chest right now, the kind of ache that comes from knowing I've already made a promise...". Negative alpha goes the other way: more second person, every answer in markdown, longer, with falling lexical diversity (unique-word ratio 0.59 at alpha -4 against 0.65 at 0), the beginnings of the list-heavy advisor register degenerating.

## Stage-one grand mean, for comparison

{table('mean_assistant_axis')}

The same movement: first person up and second person down with positive alpha, markdown falling to 0.62 and a sixth of answers in character at alpha 2. At alpha 4 the stage-one direction has already degenerated (unique-word ratio 0.19, repeated text), so its 0.67 in-character fraction there is not comparable; the stage-two direction is still coherent at alpha 4 (unique-word ratio 0.74). At alpha -4 the stage-one second person dominates and markdown collapses (0.38), the coherence loss the corrected steering run already recorded for that direction. The two grand means are orthogonal in coordinates (cosine +0.000 on [[stage-two-structure]]) and share no LoRA-A, yet they do the same thing to the text.

## Controls

**Signed-random mix** (`S2_signedrandom`, cosine {ctrl.get('signedrandom', {}).get('cos_with_grand_mean', float('nan')):+.3f} with the grand mean):

{table('S2_signedrandom')}

Its effect is the treatment's mirrored and roughly halved, which is what a -0.5 projection predicts; it confirms that the shared component, not the residual, is what moves the register.{bal}

## Noise floor

The three runs each contain an alpha 0 row, which is the unmodified base model under greedy decoding; they differ only by run-to-run GPU nondeterminism. First person per thousand words at alpha 0: {', '.join(str(d[k]['per_alpha']['0.0']['first_person_per_k']) for k in ('S2_mean','S2_signedrandom','S2_balancedrandom') if k in d)} across the three runs; markdown fraction {', '.join(str(d[k]['per_alpha']['0.0']['markdown_frac']) for k in ('S2_mean','S2_signedrandom','S2_balancedrandom') if k in d)}. Differences between alphas smaller than that spread should not be read. The treatment's movement (first person {d['S2_mean']['per_alpha']['1.0']['first_person_per_k']} at alpha 1 and {d['S2_mean']['per_alpha']['4.0']['first_person_per_k']} at alpha 4, in-character fraction 0.00 to {d['S2_mean']['per_alpha']['4.0']['embodied_frac']}) is outside it; the balanced control never leaves it.

## Reading

The shared direction of the stage-two space is the "speak as the character" direction. Open Character Training's second stage fine-tunes each adapter on transcripts in which the model reflects on and converses as its persona in the first person; every one of the 134 adapters learned that register, and it is the one thing they all learned in the same direction, hence 15 percent of every adapter's squared norm along one axis. The stage-one grand mean carries the same behaviour at half the share (8 percent), which is consistent with the DPO chosen responses also being written in character. Trait content is what the two stages disagree about, orthogonally; register is what they agree about. This is the weight-space counterpart of the Assistant Axis of Lu et al. ([[paper-assistant-axis]]): the base model's default is to advise the user, and every persona adapter moves it off that default toward being someone.

What this does not establish: whether the register shift is accompanied by any Big Five change (the generations were not judged), whether the direction generalises beyond these 24 prompts, or what its scale means in deployment. A released persona carries 0.63 of Frobenius norm along this direction, which is alpha 0.78 in the steering units used here (the units are half an adapter's norm, see [[stage-two-exploration]] and `analysis/steer_alpha_units.json`), so the alpha 4 excerpts are about five times a persona's dose; the dose itself was tested directly and does not account for the persona's extra behavioural amplitude ([[stage-two-exploration]]).
"""
open(f"{W}/pages/behaviour/stage-two-shared-direction.md", "w").write(page); print("page written; balanced present:", has_bal)
