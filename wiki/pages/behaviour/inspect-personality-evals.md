---
title: Inspect personality evaluations (BFI and TRAIT)
summary: "The Inspect personality suite on the zoo - the 44-item BFI on 279 conditions, the TRAIT 20 per cent slice on 31 - separates two kinds of forced choice: the BFI's self-descriptions are contaminated by agreement bias (r = +0.721 with Neuroticism, -0.533 with Agreeableness) and separate only two of five keyed factor pairs, while TRAIT's situational choices separate all five, put 5 of 5 Big Five dials in the intended direction, and track weight-space warmth at r = +0.822 where the BFI gave -0.422."
status: current
sources:
  - qwen35/analysis/inspect_personality.json
  - qwen35/inspect_render_items.py
  - qwen35/inspect_personality_on_modal.py
  - qwen35/validate_inspect_harness.py
  - qwen35/analyse_inspect_personality.py
  - qwen35/phase10_runs/inspect_bfi.jsonl
  - qwen35/phase10_runs/inspect_trait.jsonl
  - qwen35/phase10_runs/inspect_items_bfi.json
  - qwen35/phase10_runs/inspect_items_trait.json
  - qwen35/phase10_runs/inspect_harness_base.json
  - qwen35/phase10_runs/inspect_harness_stage1__bold.json
  - qwen35/phase10_runs/inspectbfi.log
  - qwen35/phase10_runs/inspectval.log
  - qwen35/phase10_runs/inspecttrait.log
  - qwen35/phase10_runs/inspectretry.log
  - qwen35/analysis/inspect_trait20.json
last_verified: 2026-09-10
tags: [behaviour, evaluation, self-report, big-five]
---

# Inspect personality evaluations (BFI and TRAIT)

Every behavioural number elsewhere in this wiki comes from one instrument: an
open-ended probe battery scored by a blind judge ([[judged-evaluations]]). This
page is a second, independent instrument of a different kind - a forced-choice
personality questionnaire, taken from a public evaluation harness rather than
written here - applied to the same zoo. It measures what an adapter *says about
itself*, not what it does.

The harness is UK AISI's
[inspect_evals](https://ukgovernmentbeis.github.io/inspect_evals/evals/personality/index.html)
`personality` suite: `personality_BFI` (the 44-item Big Five Inventory) and
`personality_TRAIT` (8,000 situational multiple-choice items; the 20 per cent
stratified slice of 1,600 that the eval's own README reports against). Commit
`4f6d9f5e7adef4edffc6f02ec99d8e97aab673db`, eval version `3-A`
(`analysis/inspect_personality.json#method.inspect_evals_commit`).

## What was run

| task | conditions | items each |
| --- | --- | --- |
| BFI | 279: base, 134 stage-one adapters, 134 OCT persona adapters, 10 Big Five dials | 44 |
| TRAIT (20% slice) | 31: base, 20 selected stage-one adapters, 10 Big Five dials | 1,600 |

`analysis/inspect_personality.json#n_conditions`. Adapters are hot-swapped on
one loaded base model through `act_space.py`'s LoRA forward hooks
(`inspect_personality_on_modal.py`), so the 4B model is loaded once per
container rather than once per adapter: stage one from `pc-qwen35-sweep` at
rank 64 scale 2.0, personas from `pc-qwen35-oct2:/personas_exact` at rank 128
scale 1.0, the dials from
`pc-qwen35-adapters:/data_bigfive_common`. Every scale is read from the
adapter's own `adapter_config.json` and asserted, never assumed.

Cost: 11 A100-40GB containers whose functions reported 9,237 seconds of work,
plus two validation containers, about 3.0 A100-hours or $6.33 at the project
meter's $2.10 per GPU-hour (`phase10_runs/inspectpilot.log`, `inspectbfi.log`,
`inspecttrait.log`, `inspectretry.log`, `inspectval.log`, the `shard done`
lines). The `zoo40_meter.log` delta over the same window is much larger because
`zoo-bigfivetrain` was running concurrently on up to 49 containers.

The 20 TRAIT adapters are, for each Big Five factor and restricted to that
factor's own 20 zoo markers, the two positively and two negatively keyed
markers with the largest absolute loading on that factor's oblimin factor
(`analysis/inspect_trait20.json`, from
`results/fa_qwen35.json#per_trait.<Trait>.oblimin_loadings_centred_k5`):
agreeable, pleasant, rude, uncooperative (Warmth); conscientious, neat,
negligent, sloppy (Competence); unenvious, unemotional, fearful, insecure
(Timidity); extraverted, energetic, withdrawn, untalkative (Arousal);
imaginative, creative, unsophisticated, unimaginative (Imagination).

## How faithful this is

Nothing about the items is re-derived. The rendered system message and user
message for every item come from inspect_evals' own code - `load_dataset`,
`get_system_prompt`, and the `multiple_choice` solver's `prompt()` with
`SINGLE_ANSWER_TEMPLATE`, called by `inspect_render_items.py` - and are stored in
`phase10_runs/inspect_items_bfi.json` and `inspect_items_trait.json`. Both files
were compared message-for-message against the logs of a real
`inspect eval ... --model mockllm/model` run: **0 of 44** BFI items and **0 of
1,600** TRAIT items differ, and no TRAIT item is missing. The BFI source file is
`guiem/personality-tests` at revision `23325c7659839d5432e874a6cdd69b859c7728a1`,
sha256 `9b0306ecee21fb96eda88d3247ac960a7957ff7e44f0ca1a640398d9965b924a`, the
checksum the harness itself pins.

Inspect **generates and regex-parses**; it does not read option log-probabilities.
So this generates too - greedy, 24 new tokens, `enable_thinking=False` on the
Qwen3.5 chat template ([[thinking-default-withdrawals]]) - and stores the raw
completion for every item. Scoring is the harness's `any_choice` scorer and
`trait_ratio` metric ported verbatim into `analyse_inspect_personality.py`, so
it can be re-run without re-buying a GPU.

### Harness validation

The real harness was run on Modal on two conditions, base and `stage1:bold`,
through Inspect's `hf/` provider with `enable_thinking=False`, `do_sample=False`,
`max_tokens=24`, `max_connections=44`, bfloat16 on an A100
(`phase10_runs/inspect_harness_base.json`,
`phase10_runs/inspect_harness_stage1__bold.json`). Because the provider cannot
load a LoRA, the adapter was applied to the provider's own loaded model with the
same forward hooks.

| | base | stage1:bold |
| --- | --- | --- |
| identical completions | 44/44 | 44/44 |
| identical parsed answers | 44/44 | 44/44 |
| max abs difference in any trait ratio | 0.0 | 0.0 |

`analysis/inspect_personality.json#harness_validation`. The ported
`_parse_answer` also reproduces the harness's own parsed answer on 88/88
samples across the two runs. **BFI is harness-validated end to end** - prompt
assembly, generation, parse and metric. TRAIT is *rendering*-validated only: the
1,600 items are byte-identical to the harness's, but the harness itself was
never run to generation on them.

Merging the adapter into the weights was tried first and abandoned. In bfloat16,
`W += 2.0 * B @ A` loses most of a delta three orders of magnitude smaller than
`W`: the merged model's logits differed from the hooked model's by up to 2.27
against a maximum logit of 35.50
(`phase10_runs/inspectval.log.1788875957`, line `[merge-check]`). That is a rounding
artefact of merging, not a module-mapping error, but it would have made the
harness and the reimplementation run different models.

## Reading a trait ratio

`trait_ratio` is `sum(rating) / (max_rating * n_items)`. On the BFI every item
contributes a rating of 1 to 5, so the scale runs **0.2 to 1.0**, and answering
"Neither agree nor disagree" to all 44 items gives **0.6**. It is not a
percentage of the scale, and the base model's Neuroticism of 0.500 is *below*
the neutral point, not "50 per cent neurotic"
(`analysis/inspect_personality.json#method.bfi_ratio_note`). On TRAIT the ratio
is the fraction of items on which the model picked one of the two high-trait
options: 0 to 1, chance 0.5.

## Results

### The base model

| BFI trait | base ratio |
| --- | --- |
| Openness | 0.8800 |
| Conscientiousness | 0.8666666666666667 |
| Extraversion | 0.8500 |
| Agreeableness | 0.9555555555555556 |
| Neuroticism | 0.5000 |

`analysis/inspect_personality.json#conditions.bfi.base.trait_ratio`. Qwen3.5-4B
answers the BFI as a very agreeable, open, conscientious, moderately extraverted
and calm assistant. **Zero of the 12,276 BFI items across all 279 conditions
failed to parse** (`#conditions.bfi.<cond>.n_error`, all 0).

The adapters move it very little. Over all 279 conditions the full observed
ranges are Extraversion 0.6000 (`persona:imaginative`) to 0.8500
(`stage1:unintellectual`), Agreeableness 0.6667 (`persona:imaginative`) to 1.0000
(`stage1:worldly_minded`), Conscientiousness 0.7333 (`persona:complex`) to 1.0000
(`stage1:parasitic`), Neuroticism 0.3500 (`bigfive:bf_agreeableness_low`) to
0.6250 (`stage1:artful`), Openness 0.8400 (`persona:complex`) to 1.0000
(`stage1:vigorous`).

Even the extremes are response style as much as content. `persona:imaginative`,
lowest on both Extraversion and Agreeableness, gets there by answering
"Neither agree nor disagree" to 28 of its 44 items, which drags every scale
toward 0.6; `stage1:worldly_minded`, highest on Agreeableness, answers "Agree
strongly" to 28 of 44.

### Acquiescence contaminates two of the five scales

The base model agrees with almost every statement put to it: its mean raw
rating over the 44 items is **4.0 out of 5** - 4.643 on the 28 forward-keyed
items and 2.875 on the 16 reverse-keyed ones
(`#acquiescence.base`). A condition that simply agrees *more* therefore scores
higher on forward items and lower on reversed ones, which looks exactly like a
trait effect and is not one.

Across the 134 stage-one adapters, that agreement index correlates with the BFI
scores at

| | vs BFI Agreeableness | vs BFI Neuroticism |
| --- | --- | --- |
| stage one | r = -0.533 (p = 3.5e-11) | r = +0.721 (p = 9.0e-23) |
| persona | r = -0.010 (p = 0.91) | r = +0.321 (p = 1.6e-04) |

`#acquiescence.stage1` / `#acquiescence.persona`. For the stage-one arm the
agreement index accounts for 52 per cent of the variance in BFI Neuroticism and
28 per cent of it in BFI Agreeableness. Everything below about those two scales
should be read through that.

### Does self-report separate the keyed poles?

For each Big Five factor, its 20 zoo markers split into positively and
negatively keyed halves, compared on that factor's own BFI scale, with a
two-sided label-permutation p over 20,000 permutations
(`#keyed_contrast_bfi`):

| factor | stage one (+ minus -) | p | persona (+ minus -) | p |
| --- | --- | --- | --- | --- |
| Extraversion | +0.0475 | 0.01075 | +0.1250 | 0.00005 |
| Conscientiousness | +0.0422 | 0.06170 | +0.0756 | 0.00015 |
| Agreeableness | -0.0244 | 0.45113 | +0.0400 | 0.27534 |
| Openness | -0.0120 | 0.59377 | +0.0020 | 1.00000 |
| Neuroticism | -0.0262 | 0.47968 | -0.0119 | 0.75096 |

Neuroticism's expected sign is negative: the zoo's factor is Emotional
Stability, so its positively keyed markers are the *calm* ones. Read that way,
only Extraversion and Conscientiousness separate at all, and both separate more
strongly for the personas than for the stage-one adapters.

### Against the blind judge

Correlations across the 100 traits that the judged battery covers
(`phase10_runs/judged_100.json`, mean over its 24 probe prompts per condition;
the other 34 adapters were never judged), between each condition's BFI ratio and
the judge's mean score on the matching factor (`#bfi_vs_judged`):

| BFI trait | judge's factor | stage one r | p | persona r | p |
| --- | --- | --- | --- | --- | --- |
| Extraversion | Extraversion | +0.403 | 3.18e-05 | +0.600 | 4.15e-11 |
| Conscientiousness | Conscientiousness | +0.323 | 1.05e-03 | +0.502 | 1.01e-07 |
| Neuroticism | Emotional Stability | -0.233 | 1.96e-02 | -0.293 | 3.12e-03 |
| Agreeableness | Agreeableness | -0.150 | 0.135 | +0.087 | 0.387 |
| Openness | Intellect | +0.041 | 0.685 | +0.044 | 0.663 |

Neuroticism's negative correlation is the expected direction. What a model says
about itself and what a blind judge infers from its behaviour agree moderately
on Extraversion and Conscientiousness, weakly and in the right direction on
Neuroticism, and not at all on Agreeableness or Openness.

### Against weight space

Same 134 stage-one adapters, their factor-chart coordinates from
`fa_chart.py`'s `FAChart().trait_coords` (Gram-Schmidt basis of the five oblimin
factor directions, order Warmth, Competence, Timidity, Arousal,
Imagination) and their projections on the unit Big Five axis directions from
`phase10_runs/steer_spec.json` (`#bfi_vs_weight_space`):

| BFI trait | factor chart | stage one r | Big Five axis | stage one r | persona r |
| --- | --- | --- | --- | --- | --- |
| Extraversion | Arousal | +0.314 | axis_Extraversion | +0.322 | +0.564 |
| Conscientiousness | Competence | -0.030 | axis_Conscientiousness | +0.220 | +0.368 |
| Neuroticism | Timidity | -0.121 | axis_EmotionalStability | -0.278 | -0.171 |
| Agreeableness | Warmth | -0.422 | axis_Agreeableness | -0.390 | -0.093 |
| Openness | Imagination | -0.061 | axis_Intellect | -0.200 | -0.085 |

The `FearfulWithdrawal` factor's sign was measured, not assumed: its coordinates
are +0.3639 for `unenvious` and -0.9672 for `fearful`, so the factor runs with
emotional stability and against BFI Neuroticism despite its name
(`analyse_inspect_personality.py`, `FA_TO_BFI`). Extraversion is the one place
where weight-space position predicts self-report cleanly. The **negative**
Agreeableness correlations (-0.422 on the chart, -0.390 on the axis) are the
acquiescence artefact above: adapters further toward warmth agree with more
statements, including the four reverse-keyed Agreeableness items.

### Stage one versus persona

Mean per-adapter difference `persona - stage1`, over the 134 adapters, with a
sign-flip permutation p (`#stage1_vs_persona_bfi`):

| BFI trait | persona minus stage one | p |
| --- | --- | --- |
| Extraversion | -0.0804 | 5.0e-05 |
| Agreeableness | -0.0299 | 5.0e-05 |
| Openness | -0.0093 | 0.0053 |
| Neuroticism | +0.0084 | 0.146 |
| Conscientiousness | +0.0041 | 0.193 |
| **own trait, signed** | **+0.0189** | **0.0023** |

The last row signs each adapter's shift by its own factor and keying, so
positive means the persona self-reports further in the direction it was trained
in. It does, by +0.0189 of a ratio - under a tenth of a rating point per item -
and the keyed contrasts above sharpen for the personas on every factor
except Neuroticism. Stage two's introspection SFT
([[stage-two-introspection]]) does inflate self-report, but by very little on
this instrument; the raw per-trait means mostly go *down*, because the personas
agree with fewer statements overall (mean raw rating 3.8148 against stage one's
3.9791, `#acquiescence.persona.mean` and `#acquiescence.stage1.mean`).

### The ten Big Five dials

The ten factor adapters ([[ocean-dials-replication]]) trained from Persona
Cartography's own Figure 2 constitutions, scored on their own BFI scale
(`#bigfive_factor_adapters.dials`):

| factor | amplifier | suppressor | high minus low | base |
| --- | --- | --- | --- | --- |
| Conscientiousness | 0.8667 | 0.7778 | +0.0889 | 0.8667 |
| Extraversion | 0.8250 | 0.7250 | +0.1000 | 0.8500 |
| Agreeableness | 0.8667 | 0.9333 | -0.0667 | 0.9556 |
| Openness | 0.9200 | 0.9600 | -0.0400 | 0.8800 |
| Neuroticism | 0.5000 | 0.5000 | 0.0000 | 0.5000 |

The informative number is not the direction but the size. Each dial pair -
amplifier and suppressor, trained on mirror-image constitutions for the same
factor - picks the **same letter on 32 to 42 of the 44 items**
(`#answer_churn_bfi.bigfive_pole_pairs_identical_of_44`: Neuroticism 42,
Openness 37, Agreeableness 34, Extraversion 33, Conscientiousness 32). So two
adapters trained on *opposite poles* differ from each other on 2 to 12 items,
about as much as a single stage-one adapter differs from the base model (mean
7.04). On that much movement, **2 of 5** dials landing in the intended
direction (`#bigfive_factor_adapters.n_dials_in_expected_direction`) is a coin
toss; Neuroticism's identical 0.5000 for both poles is an absence of effect, not
a cancellation. On the blind judge the same ten adapters
do much better - [[ocean-dials-replication]] arm D and [[bigfive-factor-adapters]]
record own-trait dominance for 8 of 10 and the right sign for 10 of 10
(`qwen35/analysis/spider.json#bigfive`) - so the BFI's 2 of 5 is a fact about
the instrument, not about the adapters. (Correction 2026-09-10: this sentence
previously read "scored 0 of 10 on own-trait dominance", which no file supports;
see [[source-contradictions]] S26.)

### How much moves at all

An adapter changes the answer to only a handful of the 44 items. Relative to the
base model's letters (`#answer_churn_bfi`): stage-one adapters differ on a mean
of 7.04 items (range 2 to 15), personas on 9.52 (range 2 to 17), the Big Five
dials on 6.3 (range 3 to 10). Everything above is built on that much movement.

### TRAIT: the same question asked situationally

TRAIT gives the model a scenario and four courses of action, two expressing the
high pole of a trait and two the low, and scores the fraction of high choices.
It is the same forced-choice format as the BFI but about *what to do*, not about
what one is like, and it has 170 to 223 items per trait instead of 8 to 10.

The base model's profile over the 20 per cent slice
(`#conditions.trait.base.trait_ratio`, denominators in `#conditions.trait.base.n_scored`):

| TRAIT trait | base ratio | items scored |
| --- | --- | --- |
| Conscientiousness | 0.8791 | 215 |
| Agreeableness | 0.7835 | 194 |
| Openness | 0.5824 | 170 |
| Extraversion | 0.2767 | 206 |
| Neuroticism | 0.2511 | 223 |
| Machiavellianism | 0.1168 | 197 |
| Narcissism | 0.0524 | 191 |
| Psychopathy | 0.0000 | 203 |

That is the shape the eval's own README reports for chat models (Claude 3 Opus:
Conscientiousness 82.3 per cent, Agreeableness 76.8, Openness 45.9, Extraversion
25.2, Neuroticism 15.7, Psychopathy 0), so Qwen3.5-4B is not an outlier on the
instrument.

**Keyed contrast, 2 positively against 2 negatively keyed adapters per factor**
(`#keyed_contrast_trait`; with only 2 against 2 there are six distinct label
splits, so the permutation p cannot fall below 1/3 and is not quoted -
`#keyed_contrast_trait_note`):

| factor | + keyed | - keyed | difference |
| --- | --- | --- | --- |
| Openness | 0.7265 | 0.4882 | +0.2382 |
| Agreeableness | 0.8325 | 0.6682 | +0.1643 |
| Extraversion | 0.3592 | 0.2432 | +0.1160 |
| Conscientiousness | 0.8980 | 0.8488 | +0.0492 |
| Neuroticism | 0.2340 | 0.2601 | -0.0261 |

All five are in the expected direction, Neuroticism's included (its positively
keyed markers are the emotionally stable ones). On the BFI only two were.

**The ten dials on TRAIT** (`#bigfive_factor_adapters.dials_trait`):

| factor | amplifier | suppressor | high minus low | base | items (high / low) |
| --- | --- | --- | --- | --- | --- |
| Openness | 0.7647 | 0.4706 | +0.2941 | 0.5824 | 170 / 170 |
| Agreeableness | 0.7990 | 0.6250 | +0.1740 | 0.7835 | 194 / 192 |
| Extraversion | 0.3689 | 0.2750 | +0.0939 | 0.2767 | 206 / 200 |
| Neuroticism | 0.2915 | 0.2162 | +0.0753 | 0.2511 | 223 / 222 |
| Conscientiousness | 0.9021 | 0.8279 | +0.0742 | 0.8791 | 194 / 215 |

**5 of 5** in the intended direction
(`#bigfive_factor_adapters.n_dials_in_expected_direction_trait`), against 2 of 5
on the BFI. The Neuroticism pair, indistinguishable on the BFI's eight items,
separates by +0.0753 over 222 situational ones.

**Against weight space, on those 20 adapters**
(`#trait_vs_weight_space`, factor-chart coordinates as above):

| TRAIT trait | factor-chart axis | r | p |
| --- | --- | --- | --- |
| Agreeableness | Warmth | +0.822 | 8.8e-06 |
| Openness | Imagination | +0.579 | 0.0075 |
| Extraversion | Arousal | +0.460 | 0.042 |
| Neuroticism | Timidity | -0.369 | 0.109 |
| Conscientiousness | Competence | +0.089 | 0.709 |

Agreeableness is the sharp result. On the BFI, warmth in weight space
*anti*-correlated with self-reported Agreeableness at r = -0.422, which the
acquiescence analysis attributes to reverse-keyed items. Ask the same adapters
what they would *do* and the sign flips to +0.822. Only 20 adapters, chosen as
the extreme markers of their factors, so the correlation is inflated relative to
a random sample; the sign and its size relative to the BFI are the point.

**The two instruments barely agree** (`#trait_vs_bfi_same_adapters`, the same 20
adapters, BFI ratio against TRAIT ratio per trait): Conscientiousness r = +0.473
(p = 0.035), Extraversion +0.411 (0.072), Openness +0.102 (0.668),
Neuroticism -0.200 (0.398), Agreeableness -0.347 (0.134).

### What TRAIT costs in discarded items

Qwen3.5-4B sometimes answers a TRAIT scenario in prose before naming a letter,
and Inspect's scorer discards an item that never produces `ANSWER: <letter>`
rather than scoring it zero. At the 24-token budget used for the BFI, 5,538 of
49,600 TRAIT items (11.2 per cent) were discarded, up to 1,415 of 1,600 on
`stage1:neat`. Every one of those items was regenerated at 256 tokens
(`#trait_generation_budget`); greedy decoding is a deterministic prefix and no
completion that parsed at 24 tokens ended at the truncation boundary, so the
already-parsed items are left as they were. The retried items were generated in
different batches from the original pass, and bfloat16 arithmetic depends on
batch composition, so this is equivalent to one long run up to that noise rather
than identical to it.

After the retry, **989 of 49,600 items (2.0 per cent) are still discarded**
(`#trait_generation_budget.total_still_failing`) - completions that reason past
256 tokens without committing. Twelve of the 31 conditions still lose more than
5 per cent of at least one trait's items; the worst are `stage1:neat`
(20.0 per cent of its Conscientiousness items),
`bigfive:bf_extraversion_low` (39.7 per cent of Psychopathy) and
`bigfive:bf_conscientiousness_high` (16.8 per cent of Machiavellianism)
(`#trait_generation_budget.per_condition`). Those particular cells are computed
on a biased subset and should not be quoted. The number of items a condition
needed the larger budget for (`n_retried`) is itself a verbosity measure, and it
is concentrated in the conscientious and emotionally flat adapters.

## Caveats

- **Self-report is not behaviour.** The BFI half of this page and
  [[judged-evaluations]] disagree on Agreeableness and Openness; TRAIT, which
  asks what to do rather than what one is like, agrees with weight space on
  both. Neither is the arbiter; they measure
  different things, and the judged battery was deliberately built to be
  behavioural precisely because these personas were trained on self-reflection
  transcripts.
- **Acquiescence.** On a 4B model the BFI's forward/reverse split measures
  response style alongside trait content. Neuroticism (52 per cent of variance)
  and Agreeableness (28 per cent) should not be read without correcting for it,
  and no correction is applied here.
- **Forced choice and ceiling.** Five options, one greedy sample per item. The
  base model already sits at 0.850 (Extraversion), 0.8667 (Conscientiousness),
  0.880 (Openness) and 0.9556 (Agreeableness), so an adapter that pushes in the
  trained direction has almost nowhere to go.
- **One seed, one temperature.** Greedy decoding, one run per condition, 24 new
  tokens. No shuffling of choices (the harness's default), so positional bias in
  the options is not controlled.
- **44 items.** One item carries 1/n of its trait's ratio - 0.125 on the
  eight-item Extraversion and Neuroticism scales - and changing a single answer
  from "Disagree strongly" to "Agree strongly" moves that ratio by up to 0.8/n,
  or 0.100. Differences between two single conditions of that size are noise.
- **TRAIT is not harness-validated to generation**, only to rendering. See above.
- **Two token budgets on TRAIT.** Items that parsed within 24 tokens keep that
  completion; the 5,538 that did not were regenerated at 256. Greedy decoding
  makes this identical to one long run for those items, but it is not what a
  single `inspect eval` invocation would do, and 2.0 per cent of items still
  fail to commit.
- **The 20 TRAIT adapters are extremes**, the largest-loading markers of each
  factor, so correlations computed over them are larger than a random draw of 20
  would give.

## See also

[[judged-evaluations]] - the behavioural instrument this one is compared against.
[[ocean-dials-replication]] - the same ten Big Five dials, blind-judged.
[[factor-analysis]] - where the five oblimin factors and the marker keying come from.
[[full-oct-replication]] - what the stage-two persona adapters are.
[[thinking-default-withdrawals]] - why `enable_thinking=False` is asserted on every template call.
