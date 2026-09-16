---
title: The N x N data-to-adapter scoring test
summary: Every one of the 134 traits' preference pairs ranks its own adapter first out of 134 (133 after column standardisation), and the runners-up carry Big Five structure that nothing in the test required.
status: current
sources:
  - qwen35/analyse_nxn.py
  - qwen35/build_nxn_inputs.py
  - qwen35/analysis/nxn_summary.json
  - qwen35/analysis/nxn_scores.json
  - qwen35/align_score.py
  - qwen35/analysis/align_summary.json
  - qwen35/analysis/align_validate.json
  - qwen35/PHASE3_VERDICT.md
  - .garden/journal/2026-09-05.md
last_verified: 2026-09-07
tags: [geometry, scoring, big-five]
---

# The N x N data-to-adapter scoring test

## The scoring identity

`qwen35/align_score.py` computes, exactly, how much a piece of training data
would push the model along a chosen weight direction. Because a LoRA starts with
`B = 0`, at step zero `dL/dA = B^T (dL/dW) = 0` and the whole first update lives
in `B`: `dL/dB = G A^T`, so `dW_induced ~ -eta * G A^T A`. Its overlap with a
target direction `dW*` (any weighted merge of the zoo adapters, so
`dW* = B* A_0` up to A drift) is

```
<dW_induced, dW*> ~ -eta <G A_0^T, dW* A_0^T> = -eta <dL/dB, B_U>,   B_U = dW* A_0^T
```

so the score is a plain inner product against a fixed matrix `B_U` that is itself
a rank-64 LoRA. An inner product with a gradient is a directional derivative, so
the gradient never has to be computed:

```
score(x) = d/d(eps) [ log p(x | W + eps*U) ]  at eps = 0
```

The reading the blog page pulls out of it: "Data that trains a model *toward* a
direction is exactly data that a model already steered along that direction finds
*more likely*. Gradient alignment and steering sensitivity are the same quantity
seen from two sides."

In practice every (target, example) pair gets its own scalar `eps` in one forward
pass, so a single backward pass leaves the whole per-example, per-direction
derivative matrix in `eps.grad` - exact, with no per-example gradients and no
253,952-dimensional projections.

Checked against a central finite difference of two steered forward passes over 24
held-out responses at three step sizes, the analytic and finite-difference scores
agree at correlation **0.9999992**
(`qwen35/build_blog_page.py`: `corr = v.get("corr", 0.9999992)` reading
`qwen35/analysis/align_validate.json`; that file stores `names`, and per-row
`analytic` and `fd` at eps 0.003 / 0.001 / 0.0003, but the top-level `corr` key
is absent, so the number the page prints is the builder's hard-coded fallback -
see the provenance note below).

For DPO data specifically: at `B = 0` the policy equals the reference, the
preference sigmoid sits at exactly one half, and the first DPO gradient is
proportional to `grad log p(chosen) - grad log p(rejected)`, so a preference pair
scores as the difference of its two halves.

## The positive control, and why it needed replacing

`qwen35/analysis/align_summary.json#rows` holds the six-trait control: for each
single-trait direction, where its own 500 preference pairs rank among all 134
traits' data.

| direction | own rank | own z | highest-scoring data |
|---|---|---|---|
| trait_bold | 1 of 134 | +2.6493194459241614 | bold |
| trait_anxious | 1 | +2.6299171161906334 | anxious |
| trait_agreeable | 1 | +2.597856962654925 | agreeable |
| trait_careless | 1 | +2.936687710651523 | careless |
| trait_creative | 1 | +4.305925560813595 | creative |
| trait_quiet | 2 | +2.7947809742388094 | composed |

Five of six come first; `quiet` comes second behind `composed`. Six was a
selection, and `qwen35/build_nxn_inputs.py` says so: "The positive control in
build_align_inputs.py did this for six traits (5/6 rank 1 of 134); this is the
same test with no selection." It also fixes a corpus problem - the control drew
from `data/`, which has 55 prompts no adapter saw, while the N x N test draws
from `data_common`, the corpus the adapters were actually trained on.

## The N x N test

`qwen35/build_nxn_inputs.py` builds 134 single-adapter targets
(`coef {trait: 1.0}`) and 40 pairs per trait from `data_common`, seeded
`random.Random(3)`. `qwen35/analyse_nxn.py` averages each trait's 40 pair scores
per adapter into a 134 x 134 matrix and reads it three ways.

`qwen35/analysis/nxn_summary.json`:

| statistic | `#raw` | `#column-z` |
|---|---|---|
| own adapter rank 1 | **134 / 134** | **133 / 134** |
| own adapter in top 3 | 134 | 134 |
| mean diagonal rank | 1.0 | 1.007462686567164 |
| median diagonal rank | 1.0 | 1.0 |
| worst rank (trait) | 1 (`active`) | 2 (`bashful`) |
| diagonal beats row max off-diagonal | 134 rows | 133 rows |
| runners-up sharing factor **and** keying | 0.417910447761194 | 0.43532338308457713 |
| runners-up sharing factor, opposite keying | 0.0 | 0.0 |
| chance rate for same factor + keying | 0.11525081360116712 | 0.11525081360116712 |

Chance for rank 1 is 1 in 134; chance mean rank is 67.5.

The column-z reading z-scores each *column* so an adapter that scores everything
high cannot win every row; it is the stricter of the two, and it costs exactly
one trait (`bashful`, rank 2).

The runner-up structure is the part that was not asked for. Taking the top three
per row excluding the trait itself, 42% of runners-up share the trait's factor
*and* its keying against a 12% base rate, and **0%** share its factor with the
opposite keying. `PHASE3_VERDICT.md`, 2026-09-05:

> Own adapter rank 1 for **134/134** raw, 133/134 after column z-scoring (bashful
> second). Runners-up share factor+keying 42% (base rate 12%), opposite keying 0%.
> The scoring identity sees the Big Five in DATA, which nothing about the
> positive control required.

`.garden/journal/2026-09-05.md` records the same on the day it landed.

## Reachability: what existing data can and cannot aim at

The same scoring machinery ranks directions by how hard the zoo's own data can
push along them. `analysis/align_summary.json#reach` gives, per direction,
`best_set` (the best trait's data), `best_pair`, and `ratio` against a floor -
`#probe_floor = 13.491425523936076`, the mean score on six random directions.

| direction | best_set | ratio |
|---|---|---|
| axis_Conscientiousness | 106.62807087060064 | 7.90339543300479 |
| axis_Agreeableness | 90.02182742133736 | 6.672521540560957 |
| axis_EmotionalStability | 87.50483113853261 | 6.485958876864732 |
| axis_Extraversion | 70.802005629614 | 5.247926210910718 |
| axis_Intellect | 68.02829766385257 | 5.04233578157911 |
| PC1 | 101.00623117648065 | 7.486698199332492 |
| PC2 | 75.55151513786987 | 5.599965326408887 |
| PC3 | 61.60686383992434 | 4.5663717099889345 |
| PC4 | 60.27465848235879 | 4.467627114378782 |
| PC5 | 44.91838816635427 | 3.329402670359884 |
| alien_k5 (the unnamed direction) | 41.93740635532886 | 3.1084488648678965 |
| span_random (control) | 59.169939609058204 | 4.385744079013793 |
| alien_shuffle (control) | 31.137452664691956 | 2.307943857337302 |

The ordering matches the angles to the lexicon in
`analysis/direction_gaps.json` - the named axes sit inside their word clusters
and are easiest to aim at; PC5 and the unnamed direction sit furthest from any
word and are hardest. See [[hole-words]].

Read the other way, `#rows` also gives which traits' data drives each named axis:
Extraversion (extraverted, spunky, unrestrained, vigorous), Agreeableness
(pleasant, effeminate, agreeable, liberal), Conscientiousness (composed,
imperturbable, intellectual, neat), EmotionalStability (imperturbable, composed,
unexcitable, steady), Intellect (deep, philosophical, introspective,
imaginative). Nothing in the measurement reads the text.

Other keys in `align_summary.json`: `#a_drift = 0.014605041334818797` (the zoo's
LoRA-A drift, see [[adapter-effect-and-drift]]),
`#norm_agreement = 0.997274392945949`, and `#caption` ("Mean pair score by trait,
over 5360 preference pairs from the zoo's own training data. Rank is out of 134
traits; z is against the spread of all traits on that direction.").

## Provenance notes

- `analysis/nxn_scores.json` is the raw per-pair score file and is large (the
  summary is derived from it by `analyse_nxn.py`); this page quotes only the
  summary.
- The 0.9999992 finite-difference agreement is printed by the page from a default
  argument in `build_blog_page.py`, because `analysis/align_validate.json` has no
  `corr` key. The per-row `analytic` and `fd` arrays are there; the correlation
  itself is not stored. Flagged in [[superseded-geometry-claims]] and the section
  report.
- The N x N run OOM'd once on an 80GB card before succeeding
  (`.garden/journal/2026-09-04.md`: `_build_targets` held two copies of every
  module).

Related: [[geometry-overview]], [[hole-words]], [[alignment-traits-geometry]],
[[steering-results]], [[zoo-training-recipe]].
