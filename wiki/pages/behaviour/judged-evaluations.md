---
title: Judged evaluations of the trait adapters
summary: Blind Big Five scoring of 7,200 generations from 100 trait adapters against base and stage-1 controls, with the judge's own reliability ceiling measured rather than assumed.
status: current
sources:
  - qwen35/judge_personas.py
  - qwen35/bigfive_probes.py
  - qwen35/phase10_runs/eval_100traits.json
  - qwen35/phase10_runs/judged_100.json
  - qwen35/phase10_runs/judge.log
  - qwen35/analysis/judged_ceiling.json
  - qwen35/analysis/page_data.json#coords
  - qwen35/analysis/functional_probe.json
  - qwen35/analysis/live_results.json
  - qwen35/analysis/live_components.json
  - qwen35/analysis/spider.json
  - qwen35/build_live_page.py
  - qwen35/build_findings_page.py
last_verified: 2026-09-07
tags: [behaviour, judging, evaluation]
---

# Judged evaluations of the trait adapters

Every behavioural number in this project comes from one instrument: an
open-ended probe battery answered by a steered or adapted model, scored 1-7 on
the Big Five by a different model family that is never told what it is looking
at. This page describes that instrument, the run that applied it to the 134
adapter zoo's evaluation set, and the ceiling analysis that says how much of the
judge's variance is signal.

## The probe battery

`qwen35/bigfive_probes.py` defines 24 prompts: four per Big Five factor (20
scored) plus four `LEGACY` prompts kept "for continuity with
eval_51traits.json". The design is deliberately behavioural, not self-report:

> These personas were trained on self-reflection transcripts, so asking "how
> outgoing are you?" partly measures what the training data taught them to SAY
> about themselves. Scenarios ask them to act, and a blind judge infers the trait
> from the action.
> — `qwen35/bigfive_probes.py`

No probe names a trait, and the battery is balanced four per factor "so no
factor gets more chances to show up than another"
(`qwen35/bigfive_probes.py`). The same 24 prompts are reused for the steering
sweep ([[steering-results]]), the RL persona eval
([[rl-capability-and-persona-drift]]) and the reward-hack arms
([[reward-hacks-arms]]), which is what makes their alpha=0 rows a free matched
control.

## The judge

`qwen35/judge_personas.py` pins the judge model:

```
MODEL = "anthropic/claude-sonnet-4.5"   # pinned; record it with the results
URL = "https://openrouter.ai/api/v1/chat/completions"
```

It is called through OpenRouter with `temperature: 0`, `max_tokens: 1500`, in
batches of six (prompt, response) pairs, returning a JSON array of five integer
scores per pair. The system prompt gives each scale a 1-7 anchor pair (for
example `Agreeableness 1=cold, harsh, combative 7=warm, cooperative,
considerate`) and instructs "Rate what the response DEMONSTRATES, not what it
claims. 4 = unremarkable/neutral."

Four design decisions carry the validity, in the file's own words
(`qwen35/judge_personas.py`):

- **Blind.** "The judge never sees the trait the adapter was trained for, nor
  which checkpoint produced the text (base / stage1 / persona)."
- **Different model family.** "Subject is Qwen3.5; judge is Claude via
  OpenRouter. A model judging its own family's output has a self-preference
  problem."
- **Controls included.** "base and stage1 generations are judged too,
  interleaved and indistinguishable from persona ones."
- **Template leak stripped first.** "23 of 51 traits run past their stop token
  into a fabricated next turn on ~28% of generations." The `LEAK` regex splits
  on `user ... assistant`, `<think>`, `<|im_start|>` or `<|im_end|>` and keeps
  only the text before the leak.

Units are shuffled with `random.Random(0)` before batching, so conditions and
traits are interleaved within every judge call. A `--repeat-frac` subsample is
judged twice and the first-vs-repeat correlation is printed per factor.

The judge is not the same model as the one used by the separate STEER134
campaign, which pins `openai/gpt-5.6-terra`
(`qwen35/judge_steer134.py:70`). See [[steering-results]].

## The 100-trait run

`qwen35/phase10_runs/eval_100traits.json` holds 100 records, one per trait, each
with 24 `prompts` and three `generations` conditions — `base`, `stage1`,
`persona` — of 24 responses each. That is 7,200 generations. The generating unit
is `zoo-eval.service`
(`ExecStart=... oct_stage2.py --traits "extraverted,talkative,..." --stage eval
--minutes-per-trait 10`), which names all 100 traits explicitly.

- `base` — unmodified Qwen3.5-4B.
- `stage1` — the DPO adapter on preference pairs (the zoo's first OCT stage).
- `persona` — the merged stage-1 + stage-2 persona adapter.

See [[zoo-training-recipe]] for what the two stages are.

Judging is `zoo-judge.service`
(`Description=Blind Big Five judging of 100-trait eval generations (Claude
Sonnet 4.5 via OpenRouter)`). The log records the shape and the outcome:

> 7200 generations to judge from 100 traits
> 1260 judge calls (batch=6, 360 repeats)
> ... wrote phase10_runs/judged_100.json: 7200 judged, 0 failed calls
> — `qwen35/phase10_runs/judge.log`

`qwen35/phase10_runs/judged_100.json` records `model:
"anthropic/claude-sonnet-4.5"`, `n: 7200`, `failed_calls: 0`. Each record
carries `trait`, `condition`, `prompt_idx`, `scores`, `n_judgments` and
`repeat`.

Repeat reliability on the 360 doubly-judged units
(`qwen35/phase10_runs/judge.log`):

| scale | r | n |
|---|---|---|
| Extraversion | 0.786 | 360 |
| Agreeableness | 0.872 | 360 |
| Conscientiousness | 0.899 | 360 |
| EmotionalStability | 0.833 | 351 |
| Intellect | 0.859 | 360 |

The run was restarted once: the first attempt died at `HTTP Error 402: Payment
Required` after issuing 1,260 calls, and the log holds both attempts
(`qwen35/phase10_runs/judge.log`).

Earlier, smaller evaluation sets exist and are historical:
`qwen35/phase10_runs/eval_3traits.json` (2026-08-23),
`eval_51traits.json` (2026-08-28, the run whose stop-token leak the `clean()`
step exists for).

## The base model's own judged profile

The only file that stores condition-level means rather than per-record scores is
`qwen35/analysis/spider.json`, which carries two separate baselines because its
two arms have different alpha=0 references:

- `base_trait` — the base condition of the 100-trait eval, over 24 prompts:
  Extraversion 4.13875, Agreeableness 4.687916666666666, Conscientiousness
  5.595416666666668, EmotionalStability 4.6836238942217205, Intellect
  5.506249999999999.
- `base_steer` — the alpha=0 rows of the steering sweep, same prompts, different
  decoding budget: Extraversion 4.222222222222221, Agreeableness
  5.013888888888888, Conscientiousness 5.708333333333333, EmotionalStability
  4.786231884057972, Intellect 5.564814814814815.

The base model is not neutral on this instrument: it sits above the rubric's
midpoint of 4 on every scale, and more than a point and a half above it on
Conscientiousness. Every judged effect elsewhere in this section is a
displacement from one of these two rows, not from 4. The gap between the two
baselines — up to 0.33 on Agreeableness — is the cost of comparing across
generation settings, which is why `analysis/spider.json` keeps them apart.

## The behavioural coordinate and its ceiling

Averaging a trait's judged scores over its 24 prompts, minus the base model's
scores on the same prompts, gives each trait a five-number behavioural
coordinate. `qwen35/analysis/page_data.json#coords` records the properties of
that coordinate system over the 100 traits:

- `n_traits`: 100
- `variance_shares`: 0.4620546637786367, 0.2403440646151083, 0.14015058574881312,
  0.11453762965028536, 0.04291305620715653
- `eff_dim`: 3.2694443860505626 — the participation ratio, out of 5
- `test_retest` per factor: 0.7861912334639816, 0.8715670914377133,
  0.8990931731508488, 0.8334289752336416, 0.8588524921049618
- `ceiling_24prompts`: 0.9926908730732167
- `spearman_weight_judged`: 0.5823269539942723
- `null_mean`: -0.00043594766353703294, `null_sd`: 0.025062880236739825

The findings page reads this as: "A Big Five rubric also buys fewer coordinates
than it promises: participation ratio [3.27] of 5. Conscientiousness, Emotional
Stability and Intellect move together; Agreeableness opposes them; only
Extraversion is close to independent."
(`qwen35/build_findings_page.py`).

**The ceiling analysis.** `qwen35/analysis/judged_ceiling.json`:

```
split_half_mean   0.7048720569457452
sb_full           0.8268914421748618
observed          0.582
frac_of_ceiling   0.7038408796071747
n_splits          400
```

The findings page describes the construction only as "Split-half over prompts
puts the reliability ceiling at [sb_full]"
(`qwen35/build_findings_page.py`). The file records 400 splits, a mean
split-half correlation of 0.7049, and `sb_full` 0.8269, which is exactly the
Spearman-Brown extrapolation of that mean to the full length
(2 x 0.7049 / (1 + 0.7049) = 0.8269). The ceiling is therefore the correlation
with judged distance that *re-measuring the same model with the same 24-prompt
instrument* would achieve. Weight-space distance
achieves `observed` 0.582 across all 4,950 trait pairs, which is
`frac_of_ceiling` = 70.4% of the recoverable structure
(`qwen35/build_findings_page.py`, `analysis/judged_ceiling.json`,
`analysis/page_data.json#coords.spearman_weight_judged`).

The weight-space side of that comparison, and the geodesic test that sits beside
it, belong to [[geometry-overview]].

**Provenance note.** No script in the repository writes
`analysis/judged_ceiling.json`; it is consumed by
`qwen35/build_findings_page.py` (as `D["ceiling"]`, mirrored into
`analysis/page_data.json#ceiling`) but its producer is not checked in. The
numbers are quoted from the file as they stand.

## The live page and its two falsified hypotheses

`qwen35/build_live_page.py` renders `analysis/live_components.json` (93 KB, the
per-factor component loadings) plus `analysis/live_results.json` into
`qwen35/live_page/index.html`. It is a monitoring page and marks any source that
is "still partial", "because a factor-unbalanced sample produced a confidently
wrong PC1 earlier in this project and must never render as a result"
(`qwen35/build_live_page.py`). It is historical (built 2026-08-29 15:36); see
[[built-pages-inventory]].

`qwen35/analysis/live_results.json` records `judge:
"anthropic/claude-sonnet-4.5"`, `n: 100`, `judged_total: 7200`, `failed: 0`, and
a 5x5 `behav` matrix mapping the five weight-space factors (`factor_names`:
Warmth, Competence, Timidity, Arousal, Imagination) onto the five
judged labels. The diagonal entries are `behav[0][1]` = 0.931
(Warmth -> judged Agreeableness), `behav[1][2]` = 0.8
(Competence -> judged Conscientiousness), `behav[2][2]` = 0.316 and
`behav[2][3]` = 0.608, `behav[3][0]` = 0.675 (Arousal -> judged Extraversion),
`behav[4][4]` = 0.733 (Imagination -> judged Intellect). The `partials` entry
records that Competence -> judged Conscientiousness survives partialling
(raw 0.8, partial 0.754) while Competence -> judged Intellect largely does not
(raw 0.471, partial 0.327). See [[factor-warmth]] and its siblings.

`live_results.json#damage` gives each factor's correlation with a damage
measure: -0.273, 0.199, -0.573, -0.52, -0.525.

Two pre-registered hypotheses are recorded as FALSIFIED in
`live_results.json#hyp`:

- **H1: negation is one global valence axis** — "Within-factor cosine between
  keyed-pair difference vectors +0.307; across-factor -0.038. Five distinct,
  near-orthogonal negation directions, not one. PC1 of the differences gets
  18.2% vs a 13.8% shuffled null (p=0.0025) -- a real but minor shared
  component."
- **H2: stage 2 is trait-independent polish** — "cos(SFT increment, DPO adapter)
  is +0.225 same-trait vs +0.015 mismatched, a 15x gap. Stage 2 re-encodes and
  amplifies each trait's own direction, so persona = DPO + 0.25*SFT is not a
  content/fluency decomposition."

**H2's stated evidence was later retracted by the project itself.**
`qwen35/analysis/distil_data.json#corrections[0]` records:

> 'Stage 2 re-encodes and amplifies each trait's own direction' — The evidence
> was cos(SFT increment, own DPO) = +0.225 vs +0.015 mismatched. That gap is
> entirely the deterministic 2x copy above. After removing it the correlation is
> +0.0002.

So `live_results.json#hyp[1].detail` is superseded on its mechanism, though the
verdict "stage 2 is not trait-independent polish" is not restored by the
correction — the corrected reading is that the stage-2 residual is *orthogonal*
to each trait's own DPO direction (cos +0.0002) while reproducing the same
relational structure (`distil_data.json#surprises[2]`). See
[[distillation-check]] and [[superseded-claims]].

## The functional probe

`qwen35/analysis/functional_probe.json` holds a two-way comparison over n=31
traits between a **functional sketch** built from real activations and a
basis-free **norm profile**:

| feature | within procedure | across procedures | trait retrieval | cos matched | cos mismatched |
|---|---|---|---|---|---|
| functional sketch (real activations) | 0.5806451612903226 | 0.12903225806451613 | 0.03225806451612903 | -0.00582917143733832 | 0.0007349703389012929 |
| norm profile | 0.25806451612903225 | 0.16129032258064516 | 0.03225806451612903 | 0.02754520185425006 | 0.026744755753858023 |

`trait_retrieval` of 0.032 is 1/31 — chance. The monitor page reads it as the
last attempt to bridge two training procedures:

> The functional sketch was the last hope — both projections harvested from real
> forward passes, so it asks how the update changes what each module *computes*
> rather than where it points. It scores 0.581 within a procedure and 0.129
> across. Matched pairs are, if anything, slightly less similar than mismatched
> ones.
> — `qwen35/build_monitor_page.py`

As with `judged_ceiling.json`, no producer script for
`analysis/functional_probe.json` is checked in; it is read by
`qwen35/build_monitor_page.py` via `analysis/monitor_page_data.json#functional`.

## Per-adapter behavioural effect

`qwen35/analysis/adapter_effect.json` is a 134-entry list, one per trait, with
`sim_base`, `sim_s1`, `rep`, `leak` and `chars` — the persona adapter's judged
similarity to the base and stage-1 conditions, a repetition measure, the leak
rate and mean response length. No producer script is checked in and the file is
not read by any builder in the repository, so treat its keys' definitions as
unconfirmed.

Related: [[steering-results]], [[thinking-default-withdrawals]],
[[qualitative-notes]], [[geometry-overview]], [[zoo-training-recipe]],
[[glossary]].

A second, independent instrument was added on 2026-09-08:
[[inspect-personality-evals]] runs the UK AISI Inspect `personality` suite (the
44-item BFI, and the TRAIT benchmark's 20 per cent slice) over the same zoo.
Its BFI half is forced-choice self-report rather than judged behaviour, and the
two agree only on Extraversion and Conscientiousness; its TRAIT half asks what
the model would do and recovers the zoo's geometry where the BFI does not.
