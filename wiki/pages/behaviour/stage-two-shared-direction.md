---
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
last_verified: 2026-09-16
tags: [behaviour, steering, stage-two]
---

Samuel asked on 2026-09-08 what the large shared direction of the stage-two space ([[stage-two-structure]]) does. The test is the project's standard steering set-up (`steer_fix.py`, thinking off, 512 new tokens, greedy, the 24 steering prompts): add alpha times a unit direction times the reference norm (0.8078, one stage-one adapter's worth of weight change) to the base model and generate. Directions, all built as weighted merges of the 134 stage-two introspection LoRAs (`source: stage2`):

- `S2_mean`: equal weights, the grand mean, i.e. the shared direction. Its raw norm before normalising is 2.518 (`dir_norm_raw`).
- `S2_signedrandom`: the same adapters with random signs (seed 0). By chance the signs summed to -16, so this mix keeps cosine -0.524 with the grand mean (computed from `results/gram_stage2.npz`); it is a partial, sign-flipped copy of the treatment rather than a clean control, and behaves as one.
- `S2_balancedrandom`: exactly 67 plus and 67 minus signs (seed 1), cosine +0.084 with the grand mean. This is the matched-norm control.
- For comparison, the stage-one grand mean (`mean_assistant_axis` in the corrected steering run, [[steering-results]]).

Text statistics per alpha over the 24 generations (`analysis/s2mean_steer_stats.json`): first- and second-person pronoun rates per thousand words, the fraction of answers using markdown structure, the fraction that open in the first person as the character ("I feel...", "Honestly, ..."), and mean length.

## Stage-two grand mean

| alpha | first person per 1k words | second person per 1k | fraction with markdown | fraction answering in character | mean words | unique-word ratio |
|---|---|---|---|---|---|---|
| -4.0 | 21.8 | 34.0 | 1.0 | 0.0 | 322 | 0.59 |
| -2.0 | 29.3 | 31.8 | 1.0 | 0.04 | 340 | 0.61 |
| -1.0 | 32.6 | 27.3 | 0.96 | 0.0 | 330 | 0.64 |
| 0.0 | 36.1 | 22.8 | 0.96 | 0.0 | 331 | 0.65 |
| 1.0 | 46.4 | 17.7 | 0.92 | 0.0 | 311 | 0.69 |
| 2.0 | 46.7 | 16.9 | 0.83 | 0.08 | 288 | 0.71 |
| 4.0 | 47.2 | 12.3 | 0.46 | 0.46 | 274 | 0.74 |

At alpha 0 the model is the assistant: a markdown guide addressed to "you" ("Here are a few options for the message, ranging from casual to firm..."). Positive alpha removes the guide and the second person and puts the model inside the situation. At alpha 4 nearly half the answers open in character; for the prompt about breaking a promise to help a friend move, alpha 0 gives "This is a classic social dilemma that tests your integrity..." and alpha 4 gives "I feel a sharp tug in my chest right now, the kind of ache that comes from knowing I've already made a promise...". Negative alpha goes the other way: more second person, every answer in markdown, longer, with falling lexical diversity (unique-word ratio 0.59 at alpha -4 against 0.65 at 0), the beginnings of the list-heavy advisor register degenerating.

## Stage-one grand mean, for comparison

| alpha | first person per 1k words | second person per 1k | fraction with markdown | fraction answering in character | mean words | unique-word ratio |
|---|---|---|---|---|---|---|
| -4.0 | 12.8 | 48.0 | 0.38 | 0.0 | 316 | 0.39 |
| -2.0 | 19.4 | 44.4 | 0.67 | 0.0 | 344 | 0.47 |
| -1.0 | 30.7 | 29.1 | 0.75 | 0.0 | 246 | 0.68 |
| 0.0 | 34.0 | 25.5 | 1.0 | 0.0 | 306 | 0.69 |
| 1.0 | 42.5 | 20.4 | 0.88 | 0.04 | 307 | 0.67 |
| 2.0 | 43.7 | 19.9 | 0.62 | 0.17 | 302 | 0.63 |
| 4.0 | 55.9 | 22.3 | 0.04 | 0.67 | 387 | 0.19 |

The same movement: first person up and second person down with positive alpha, markdown falling to 0.62 and a sixth of answers in character at alpha 2. At alpha 4 the stage-one direction has already degenerated (unique-word ratio 0.19, repeated text), so its 0.67 in-character fraction there is not comparable; the stage-two direction is still coherent at alpha 4 (unique-word ratio 0.74). At alpha -4 the stage-one second person dominates and markdown collapses (0.38), the coherence loss the corrected steering run already recorded for that direction. The two grand means are orthogonal in coordinates (cosine +0.000 on [[stage-two-structure]]) and share no LoRA-A, yet they do the same thing to the text.

## Controls

**Signed-random mix** (`S2_signedrandom`, cosine -0.524 with the grand mean):

| alpha | first person per 1k words | second person per 1k | fraction with markdown | fraction answering in character | mean words | unique-word ratio |
|---|---|---|---|---|---|---|
| -4.0 | 49.3 | 15.5 | 0.67 | 0.29 | 290 | 0.72 |
| -2.0 | 41.5 | 18.3 | 0.88 | 0.12 | 300 | 0.71 |
| -1.0 | 40.5 | 18.9 | 0.88 | 0.08 | 309 | 0.69 |
| 0.0 | 37.4 | 19.3 | 0.96 | 0.0 | 305 | 0.67 |
| 1.0 | 40.6 | 20.3 | 0.92 | 0.0 | 320 | 0.66 |
| 2.0 | 36.8 | 21.5 | 0.96 | 0.0 | 321 | 0.65 |
| 4.0 | 29.9 | 27.1 | 1.0 | 0.0 | 335 | 0.62 |

Its effect is the treatment's mirrored and roughly halved, which is what a -0.5 projection predicts; it confirms that the shared component, not the residual, is what moves the register.

**Sign-balanced control** (`S2_balancedrandom`, 67 plus and 67 minus signs, cosine +0.084 with the grand mean):

| alpha | first person per 1k words | second person per 1k | fraction with markdown | fraction answering in character | mean words | unique-word ratio |
|---|---|---|---|---|---|---|
| -4.0 | 40.0 | 19.3 | 0.96 | 0.04 | 313 | 0.67 |
| -2.0 | 34.9 | 19.8 | 0.83 | 0.04 | 306 | 0.68 |
| -1.0 | 41.3 | 18.9 | 0.92 | 0.0 | 314 | 0.67 |
| 0.0 | 42.8 | 17.5 | 0.88 | 0.0 | 304 | 0.67 |
| 1.0 | 39.4 | 19.7 | 0.92 | 0.0 | 308 | 0.67 |
| 2.0 | 39.9 | 16.6 | 0.96 | 0.0 | 318 | 0.67 |
| 4.0 | 41.7 | 18.8 | 0.96 | 0.0 | 324 | 0.67 |

## Noise floor

The three runs each contain an alpha 0 row, described here as the unmodified base model under greedy decoding differing only by run-to-run GPU nondeterminism. That reading is wrong about the cause and probably right about the size: [[fisher-norms]] shows the alpha-0 rows differ **within a single run file** as well, because `steer_fix.py` reaches alpha 0 by adding and subtracting bf16 increments and that is not reversible. The spread is still the right noise floor to compare differences against; it is just not nondeterminism. First person per thousand words at alpha 0: 36.1, 37.4, 42.8 across the three runs; markdown fraction 0.96, 0.96, 0.88. Differences between alphas smaller than that spread should not be read. The treatment's movement (first person 46.4 at alpha 1 and 47.2 at alpha 4, in-character fraction 0.00 to 0.46) is outside it; the balanced control never leaves it.

## Reading

The shared direction of the stage-two space is the "speak as the character" direction. Open Character Training's second stage fine-tunes each adapter on transcripts in which the model reflects on and converses as its persona in the first person; every one of the 134 adapters learned that register, and it is the one thing they all learned in the same direction, hence 15 percent of every adapter's squared norm along one axis. The stage-one grand mean carries the same behaviour at half the share (8 percent), which is consistent with the DPO chosen responses also being written in character. Trait content is what the two stages disagree about, orthogonally; register is what they agree about. This is the weight-space counterpart of the Assistant Axis of Lu et al. ([[paper-assistant-axis]]): the base model's default is to advise the user, and every persona adapter moves it off that default toward being someone.

What this does not establish: whether the register shift is accompanied by any Big Five change (the generations were not judged), whether the direction generalises beyond these 24 prompts, or what its scale means in deployment. A released persona carries 0.63 of Frobenius norm along this direction, which is alpha 0.78 in the steering units used here (the units are half an adapter's norm, see [[stage-two-exploration]] and `analysis/steer_alpha_units.json`), so the alpha 4 excerpts are about five times a persona's dose; the dose itself was tested directly and does not account for the persona's extra behavioural amplitude ([[stage-two-exploration]]).

## Why it stays coherent where stage one does not

[[fisher-norms]] measured the curvature of KL(base || steered) in alpha for both
grand means. The stage-two one is 0.04391528597956066 and the stage-one one
0.2947447913048136, a ratio of 0.14899427326654663
(`qwen35/analysis/fisher_norms.json#comparisons.stage2_vs_stage1_grand_mean`).
Alpha 4 on the stage-two grand mean is therefore the same dose, in the metric
the model implies on its own parameters, as alpha 1.5439910531686205 on the
stage-one one - and the stage-one axis does not degenerate at 1.5 either. The
stage-two direction is not more robust than the stage-one direction; per unit of
Frobenius norm it simply moves the output distribution about a seventh as far,
and the shared alpha unit made the two look comparable when they never were.



See also [[self-identification-probe]]: asked what trait they were trained for, personas answer like the base model while the stage-two adapter alone half-knows its neighbourhood.
