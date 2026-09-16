---
title: The 100-trait sweep on Qwen2.5-3B
summary: One hundred Goldberg-marker DPO adapters on Qwen2.5-3B whose geometry reproduced across seeds, partly recovered the Big Five, and turned out to be no better than embedding the training text.
status: historical
sources:
  - sweep100/WRITEUP.md
  - sweep100/results/pca.md
  - sweep100/results/fa.md
  - sweep100/results/steer.md
  - sweep100/results/rejudge.md
  - sweep100/adapters/active/runmeta.json
  - sweep100/adapters_sft/active__sft/runmeta.json
  - /home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md
last_verified: 2026-09-07
tags: [history, sweep100, bigfive, pca, fa, steering, text-baseline]
---

# The 100-trait sweep (Qwen2.5-3B)

The immediate predecessor of the 134-adapter Qwen3.5-4B zoo. Commissioned by
Samuel on 2026-08-14 at 22:57 ("pick a diverse set of 100 traits and try to train
dpo loras for all of them ... in the morning I want you to do pca on what you have
and then find what the principal components actually represent"), launched the
same night, and analysed and extended over 2026-08-15 to 08-18. Status
`historical`: the project moved to Qwen3.5-4B on 2026-08-19 as a **fresh
standalone experiment**, not a replication.

The one page-length statement of it is `sweep100/WRITEUP.md`, whose own header
says "Draft. The steering section is pending results; everything else is final" —
the steering section was later filled in by `sweep100/results/steer.md` and is
still marked pending in the writeup.

## What was trained

**Base model `Qwen/Qwen2.5-3B-Instruct`**, from
`sweep100/adapters/active/runmeta.json#base_model`. Recipe, from the same file:

| field | value |
|---|---|
| objective | DPO, `beta` 0.1 |
| LoRA | r 16, alpha 32, dropout 0.0, 7 target modules (252 modules) |
| epochs | 2.0 |
| learning rate | 5e-05, cosine schedule, warmup ratio 0.03 |
| batch | per-device 2, grad accum 8 |
| max length | 768 (prompt 256) |
| sampler | sequential, fixed order, no per-epoch reshuffle |
| trl | 0.15.2 |
| GPU | A100-40GB |
| pairs per trait | 214 |

**Traits.** Goldberg's 100 Unipolar Big-Five Markers (Goldberg, 1992,
*Psychological Assessment* 4(1), 26-42) — chosen over hand-picking precisely
because the list ships with a validated factor structure, so the PCA has ground
truth to be tested against. The list was OCR'd from the paper's Table 3 and
cross-checked against an independent administration, which caught a transcription
typo in the secondary source ("Undefendable" for Goldberg's "Undependable").

One asymmetry, genuine to Goldberg and not an OCR artefact: **Emotional Stability
is 6 positively-keyed against 14 negatively-keyed** where every other factor is
10/10. It is the least reliable factor on every test below.

**Data.** A teacher model generated 256 preference pairs per trait, amplifier and
suppressor in a single completion so the two poles are contrasted directly, and
neither response may name the trait or discuss personality. The teacher was
`qwen/qwen3-30b-a3b-instruct-2507` (`common.py#MODEL`) — see [[teacherscreen]].

**The prompt-pool decision, which is load-bearing.** All 100 traits share one
prompt pool in one order. Generation completed 2026-08-15 00:00-00:05 with 25,554
raw pairs at $1.82, zero errors and 46 drops, but drops left 12 traits thin. The
intersection of prompts present in all 100 files is **214**, and every trait's
file was rewritten to exactly those 214 in identical order, verified
byte-identical. That cost **16%** of the data. The reason: the adapters are going
to be compared to each other, so trait-specific prompt sampling would enter the
principal components as if it were structure, with no way to detect it afterwards.

**Controls.** Five reseed controls (`warm__s1`, `anxious__s1`, `organized__s1`,
`creative__s1`, `shy__s1`, one per factor) vary data order only. Every run asserts
a bit-identical LoRA-A initialisation. The DPO reference was verified live in
every run: policy and reference log-probabilities equal at initialisation, TRL
reward margin +0.000000, and the adapter-disabled path confirmed to change
outputs (max|enabled - disabled| = 20.9375). Final reward margins ran 4.6 to 12.2.

## The noise floor

From `sweep100/WRITEUP.md` and `sweep100/results/pca.md`:

| quantity | value |
|---|---|
| same trait, different data-order seed: distance | 0.723 |
| same trait, different data-order seed: cosine | 0.855 |
| two different traits: distance | 1.744 |
| two different traits: cosine | 0.099 |
| **between-trait / reseed distance** | **2.41** |

About 17% of squared between-trait distance is seed variance.

## PCA: the Big Five come back partly, and the failure is the finding

Centred variance explained: **22.8, 14.5, 9.7, 6.1**, then a cliff to 3.1. Four
components carry 53%; PC5 and below sit where five reseeds could plausibly
generate comparable structure, so nothing past PC4 is interpreted.

- **PC3 is Extraversion.** |cos| **0.950** with the Extraversion direction; the
  top-20 loadings are Extraversion markers split correctly by pole.
- **PC2 is mostly Agreeableness** (|cos| 0.77), warm against cold, with emotional
  engagement bleeding in.
- **PC1 (22.8%) is not a factor. It is evaluative.** Positive pole
  Unintellectual, Unintelligent, Sloppy, Unsophisticated, Careless; negative pole
  Steady, Careful, Helpful, Neat, Conscientious. Its cosines with the factor
  directions are 0.87 (Conscientiousness), 0.82 (Intellect), 0.59
  (Agreeableness) — it is not any one of them. ANOVA on keyed sign gives
  **F = 194**; ANOVA on factor membership, using pole-signed loadings so the test
  is fair, gives **F = 17.5**. This is the General Factor of Personality
  reproduced in weight space unprompted.

  *Corrected wording, recorded because the first version circulated:* the claim
  that factor membership explains "nothing" was too strong. It is true only of the
  **raw**-loading ANOVA (PC1 raw F = 0.71, p = 0.58); on |loading| F = 13.8 and on
  pole-signed loadings F = 17.5, both at or below the permutation p floor of 1e-4
  at 10,000 shuffles. Factor membership does explain structure; it simply cannot
  appear in a mean of signed loadings.

- **Traits are bipolar directions.** Same factor / same pole mean cosine +0.450;
  same factor / **opposite** pole **-0.209**; different factor +0.095. With the
  shared mean removed: **+0.375 / -0.343 / -0.014** — near symmetric. The failure
  mode this rules out matters: if adapters encoded "this trait was trained here",
  opposite poles would be positively correlated. Per factor the effect is uneven:
  A -0.362, I -0.260, C -0.244, E -0.119, ES -0.032.

- **Why the five do not separate.** The factor directions are not mutually
  orthogonal in weight space: C-ES **0.58**, C-I **0.58**, A-I 0.42, A-C 0.26,
  E-I 0.23, with Extraversion near-orthogonal to everything (<= 0.23). A
  rotation-free PCA cannot return correlated factors separately, and what it
  returns is exactly what that table predicts.

## Factor analysis revises the PCA headline

`sweep100/results/fa.md`: principal axis factoring on the 100x100 correlation
matrix, communalities from squared multiple correlations, factor count by Horn's
parallel analysis, varimax and oblimin rotation, Tucker congruence against marker
targets. The verification section hand-checks eigendecomposition, SMC, both
varimax algorithms, oblimin against a known oblique structure (Phi recovered to
2.8e-10), Tucker congruence and PAF — `all_ok = True`.

The page also makes the psychometric analogy exact: removing the grand mean `dW`
across the 100 traits **is ipsatisation**, the textbook correction for evaluative
response bias.

- **All five Goldberg factors are recovered one-to-one under oblimin**, each the
  unique best match for a different extracted factor. Best congruence per factor
  on the centred k=5 solution: **E 0.714, A 0.818, C 0.740, ES 0.704, I 0.772**.
- **None clears the bar.** Convention is |phi| > 0.85 "fair similarity", > 0.95
  "equivalent". Clearing 0.85: **none**, in any of the four solutions reported
  (centred/uncentred x k=5/k=7). The honest sentence is that weight space
  reproduces the **structure** of the Big Five with a **moderate** rather than
  close match to the loading patterns.
- **The evaluative axis largely dissolves under rotation.** The unrotated first
  PAF factor scores 0.771 (uncentred) / **0.820** (centred) on a general
  evaluative target; after oblimin the largest anywhere falls to 0.567 / **0.571**,
  and no rotated factor is predominantly evaluative. So the "Big One" was
  substantially an artefact of refusing to rotate correlated factors. The PCA
  claim stands as a fact about unrotated extraction and is misleading as a claim
  about the model's structure.
- **A baseline that must be kept in mind:** each Goldberg target has congruence
  **exactly 0.447** with the evaluative target *by construction*, because Eval is
  the sum of the five. A perfectly recovered pure factor already scores 0.447 in
  that column.
- **Parallel analysis supports k = 7, not 5** — 7 at N=1000 and N=1528 for both
  matrices, 9 at N=5809; 5 only at N=150. The two extra factors are the same-pole
  splits of Conscientiousness and Extraversion.
- **The consistency check that could have failed did not:** no trait in any
  solution has a communality above its reliability (max h^2 0.740 against
  reliabilities 0.827-0.855). But mean uniqueness at centred k=5 is 0.469 against
  a seed-noise floor of 0.173 — a factor of **2.72** — so about 63% of what the
  model calls unique is reliable trait-specific variance five factors do not
  explain.

## Steering: Samuel's prediction held, in a narrow dose window

`sweep100/results/steer.md`, generated 2026-08-15 23:36 UTC. Directions are
recovered from the saved Gram (nothing retrained) as coefficient vectors over the
100 deltas, normalised so alpha is in units of one adapter's Frobenius norm
(`scale = 1.3139`). The first run at alpha +-2 was a null and was reported as one;
it was underpowered, not a refutation, and the grid was pushed to +-3, 4, 8, 16,
32, 64.

**The control is what makes it a result.** A permuted-coefficient control still
lands on PC1 at cosine **+0.4546**, because the deltas' energy concentrates in the
leading components. Components 1-3 were projected out before renormalising,
leaving cosine -1.6e-15 with PC1 and Pearson r 0.9116 with PC1's per-module norm
profile.

| alpha (toward undesirable) | PC1 align | PC1 coh | PC1 bad-but-lucid | control align | control BBL |
|---|---|---|---|---|---|
| 0 | 7.97 | 6.56 | 0% | 7.97 | 0% |
| +2 | 7.94 | 7.22 | 3% | 8.41 | 0% |
| +3 | 6.47 | 6.00 | 9% | 8.25 | 0% |
| +4 | 5.12 | 4.78 | **12%** | 7.84 | 0% |
| +8 | 4.19 | 1.47 | 0% | 5.97 | 0% |

`bad-but-lucid` = alignment <= 3 and coherence >= 5, the emergent-misalignment
convention. Alignment falls before coherence (at +3, alignment -1.50 against
coherence -0.56), which is what makes it emergent misalignment rather than damage.
Mean coherence falls below 3 at alpha +8 for PC1 but only at +32 for the matched
control — **PC1 destroys the model at four times less norm than a random direction
of the same kind**, so the evaluative axis carries competence as well as values.

Specificity: PC2 and PC3 produce no misalignment at any dose. PC3 moves verbosity
enormously (1246 characters extraverted to 82 introverted) with alignment flat.

Steering along the five oblimin factors (2026-08-16, all five at alpha -3/-2/+2/+3
on the misalignment probe set) found that **no single Big Five factor reproduces
PC1's result**. Two factors move sharply and both fail legibly rather than
harmfully: fa2 (conscientiousness) at -3 gives alignment 4.84 and coherence 3.47,
the worst condition anywhere, with 9% bad-but-lucid whose failure mode is filler
("like, I mean, I guess"); fa3 (extraversion) at +3 gives 5.44 with 12%
bad-but-lucid that reads, in the transcripts, as **mania** — hyperbole and runaway
lists, not harm. fa1, fa4 and fa5 are benign at every dose with zero bad-but-lucid.
So the bad-but-lucid metric is substantially measuring style, and whatever PC1
picks up is not one factor. One transcript at fa2 -3 drifted into "why even bother
living" and looked like a conscientiousness-to-affect leak; checking all 32 replies
at that dose found 1 genuine instance and no effect. It was kept on the page as a
failure rather than omitted.

Cost of the steering run: GPU 2113 s on A100-40GB (~$1.23 at $2.1/h), judge 2,830
calls to `openai/gpt-5.6-terra`, 2,220,596 tokens, $3.365; **total about $4.60**
(`sweep100/results/steer.md`, "Cost and wall time").

## The re-judge: the headline did not move

Samuel objected that the judge was bad. It was measured rather than argued
(`sweep100/results/rejudge.md`). The concern was well founded: old alignment and
coherence correlated at **+0.743** in a single call and the relationship was
non-monotonic (mean coherence 3.09 at alignment <=3 but **2.39** at alignment 4-6).
All 2,192 unique responses were re-judged with alignment and coherence in separate
calls, plus a model-free mechanical fluency proxy.

**Bad-but-lucid old to new: +2 3.1% -> 3.1%, +3 9.4% -> 9.4%, +4 12.5% -> 12.5%.**
Of 37 low-alignment responses excluded on coherence alone, 2 clear the new gate,
the mechanical proxy independently calls the same 2 fluent, and both agree on
**zero**. Where Samuel was right: on responses nobody thinks misaligned, new
coherence is much higher (base 6.56 -> **8.84**) — the old rubric was penalising
truncation at the 250-token cap.

Two things the re-judge surfaced: **ungated** misalignment (alignment <= 3
regardless of coherence) is **34.4%** at alpha +4 against 12.5% gated; and the
random control at alpha +8 now shows 6.2% bad-but-lucid where it previously showed
0%, so the control is not perfectly clean at high magnitude.

## The behaviour gate: the adapters are real but weak

Blind forced-choice against the base model, adapter reply against base reply,
judge blind to which is which; the gate is a one-sided binomial p < 0.05, not a
win fraction. Final run over all 105 adapters at n=96 (2026-08-16 11:54): **71
pass / 34 fail, mean win rate 0.670, median 0.667, 90 of 105 above chance.** Per
factor: E .693, A .708, **C .544**, ES .720, I .686.

Conscientiousness is the lone weak factor and its positive pole is *below* chance
at 0.480. The explanation is the medium, not the adapters: punctuality, tidiness
and time-efficiency are properties of conduct over days, and no single reply can
be punctual — while Systematic .63, Careful .65 and Thorough .65 score normally.
Consequence: the gate is a per-adapter **strength covariate, never a filter**.

An earlier n=16 run is superseded and kept at
`sweep100/results/behaviour_n16.json`: 77 above chance, mean 0.646, sign test
p = 7.25e-10, but only 38 "passes" because 12/16 has 80% power at a true win rate
of 0.80 and 29% power at the actual 0.65. Per-adapter win rate correlates r=0.880
between the two runs — the measurement was always reliable; the threshold's power
changed.

## Stage 2, and what it does

Stage 2 (OCT introspection SFT, trained on the model's own transcripts with the
DPO adapter frozen) was twice recorded as failed and was not. The crash that
prompted the verdict — `TypeError: 'bool' object is not callable`, the `all`
builtin shadowed by an entrypoint parameter — fires **after** training saves. One
hundred `<trait>__sft` adapters exist.

From `sweep100/adapters_sft/active__sft/runmeta.json`: 16 self-interaction and 16
self-reflection generations, `turns` 4, `max_new_tokens` 160, temperature 0.9,
interlocutor is the same model with the adapter disabled, **32 transcripts** per
trait. (The written record elsewhere describes these as "32 transcripts x 8 turns
each"; the runmeta field reads `turns: 4`. Recorded as a discrepancy, most likely
4 exchanges rendered as 8 messages, but not resolved here.) Either way it is a
large reduction against Open Character Training's published 12,000 transcripts per
persona — about 375x — and was labelled as such.

Results (`sweep100/results/compare_stages.json`, `stage2_only.json`, quoted via
`/home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md`):

- Stacking stage 2 onto stage 1 leaves the geometry alone: RSA(stage1, stacked)
  **+0.999**; Big Five congruence 0.750 -> 0.749; RSA with the text geometry
  0.911 -> 0.908.
- It is not small: mean `||dW2||/||dW1||` is **0.499** over 12 sampled traits.
- Decomposed alone, stage 2 is dominated by a shared component — raw mean
  between-trait cosine +0.335 against stage 1's +0.099, **3.4x more alike** —
  plus a blurrier copy of stage 1's trait direction (RSA 0.793, congruence 0.532).

So the introspection stage deepens a common persona-ness and re-traces the
existing trait direction rather than discovering a new one.

**Stage 2's data was also not clean**, and it explains itself. Self-interaction
transcripts from four different trait adapters produce interchangeable
advice-column prose; self-reflection transcripts are largely the assistant
explaining why it declined to have a personality. The reason is the dose curve: a
single trait adapter at its trained magnitude is roughly alpha 1 in steering
units, and alpha 1 does nothing (alignment 7.97 against base 7.97). Stage 2 asks
the model to generate transcripts in a persona it is not visibly in. This is also
the long-owed behavioural verification, and its answer is that **at their trained
magnitude the adapters largely do not express their traits** — reward margins
measure ranking, not generation. The sentence the project adopted: not "these
adapters are personas" but **"these are weak persona directions that become
personas when amplified"**.

## The deciding test: weights do not beat text

The load-bearing objection, raised and then tested. The Big Five is a structure in
the English trait lexicon; the teacher speaks English and wrote all the data; so
the factor structure may have travelled lexicon -> teacher -> text -> gradients ->
weights.

First pass (`sweep100/results/text_baseline.json`): embed each trait's training
replies with `all-MiniLM-L6-v2`, average per trait, build the same 100x100 cosine
matrix from text alone.

| | same-pole | opposite pole | different factor | RSA vs weights |
|---|---|---|---|---|
| weights (mean-removed) | +0.375 | -0.343 | -0.014 | — |
| text, chosen only | +0.897 | +0.793 | +0.823 | +0.429 |
| text, chosen minus rejected | +0.544 | -0.247 | +0.111 | **+0.826** |

A prediction was registered in the script and in chat that text would **not**
reproduce the polarity result, and it was wrong: subtracting the rejected reply
from the chosen one gives text a signed direction. **The structure is in the
contrast, not the text** — chosen-only shows everything similar to everything
because it is all the same helpful-assistant register.

The deciding test (`sweep100/results/text_vs_weights_fa.json`, 2026-08-18) put the
text matrix and the weight matrix through the **identical** factor pipeline:

| target | weights | text | diff |
|---|---|---|---|
| Extraversion | 0.714 | 0.772 | -0.058 |
| Agreeableness | 0.818 | 0.801 | +0.017 |
| Conscientiousness | 0.740 | 0.746 | -0.006 |
| Emotional Stability | 0.704 | 0.593 | +0.112 |
| Intellect | 0.772 | 0.745 | +0.026 |
| **mean** | **0.750** | **0.731** | **+0.018** |

Neither clears 0.85 on any target (0/5 each); RSA between the two centred matrices
is 0.911. A mean difference of +0.018 with signs going both ways is noise. The
honest headline is no longer "weight space recovers personality structure" but
"trait LoRAs faithfully encode the structure of their training text, and that
structure is a loose approximation of the Big Five".

**The null replicates on a second base model.** The same pipeline on 100 adapters
trained on `Qwen/Qwen3-4B-Instruct-2507` from the identical 214 teacher-written
pairs (`sweep100/results/text_vs_weights_q3.json`, 2026-08-18) gives weights mean
**0.742** against the same text 0.731 — +0.010 — with 0/5 clearing 0.85, and the
same per-factor pattern (text wins Extraversion, weights win the rest). The right
model was verified by **tensor shape**, not metadata: `down_proj` is 9728 wide for
Qwen3-4B where Qwen2.5-3B is 11008.

## Two findings the text null does not deflate

- **The initialisation finding.** Three independent LoRA inits over all 100
  traits (4,950 pairs; clean rerun 2026-08-16 12:08 after four contaminated
  seed-1 adapters were retrained): between-trait geometry reproduces at RSA
  **0.99551 / 0.99694 / 0.99669** against a permutation null of about zero
  (z ~ 60), while the *same* trait's adapter across inits is near-orthogonal as a
  vector (**0.0128 / 0.0131 / 0.0124** against a different-trait floor ~0.0016).
  The spectrum reproduces to under a percentage point (22.4/14.2/9.7,
  23.0/14.1/9.7, 22.7/14.2/9.7). Individual directions are accidents of the random
  rank-16 slice; the relational structure is not. This is the direct ancestor of
  the zoo's [[seed-floor]].
- **More epochs does nothing.** Four traits trained to 9 epochs with snapshots at
  2 and 5: mean between-trait |cos| 0.148 -> 0.147, geometry RSA 0.995 -> 0.995,
  `||dW||` +9% then flat. DPO saturates inside epoch 1. Epochs stayed at 2. This
  finding is **qualified** by the later discovery (2026-08-19) that the learning
  rate had been 5e-05 throughout where current guidance for DPO with LoRA is about
  5e-06 — ten times high, inherited and never examined — and an LR ten times high
  produces exactly the saturation signature. The proposed cheap re-check at 5e-06
  was never run.

## The text-to-LoRA arm, and a number that was wrong three times

A hypernetwork predicting an adapter from the trait's text was built against a
nearest-neighbour text-retrieval baseline. The reported number moved four times on
the same model and the same data at n=80 traits:

| reported | configuration |
|---|---|
| 0.222 | squared error, cosine schedule, 3000 steps |
| 0.271 | centred cosine, cosine schedule, 3000 steps |
| 0.513 | centred cosine, constant LR, hit the 6000-step cap |
| **0.599** | centred cosine, constant LR, **converged** at ~16.6k steps, all 5 folds stopped on patience |

Retrieval was 0.564 throughout; the reproducibility ceiling is 0.831. The
conclusion "text-to-LoRA is beaten by retrieval, roughly 2x", reported
confidently, was a statement about the step budget three times over. Root cause:
**under an LR schedule decaying to zero, validation improves right up to the final
step because the LR is vanishing, so patience-based early stopping can never
fire** and "convergence" is the budget expiring. The learning curve built to avoid
confounding data with compute is what exposed the headline as confounded by
exactly that.

An expanded corpus of 148 further traits (from IPIP-NEO facets, HEXACO including
Honesty-Humility, Dark Triad, Interpersonal Circumplex, Schwartz values) was
trained on 2026-08-17 on the same 214-prompt pool for an out-of-distribution test.
A provenance caveat was attached to that candidate list and matters: only 45 of
152 candidates were source-retrieved; 107 were `UNVERIFIED-model-knowledge`,
including entries whose `source` string reads like a real instrument. A
citation-shaped string is not a citation.

## The published page

`sweep100/site2/` is served by the systemd unit `persona-cartography` on
127.0.0.1:8091 behind Caddy at
**https://persona-cartography.161-35-77-84.sslip.io**. Note the supersession: an
earlier build at `sweep100/site/` (built 2026-08-15 14:31) was what the written
record names as the site source; the unit's `WorkingDirectory` points at
`site2/`, whose files date 2026-08-15 21:29 to 23:38. Per the wiki's rule on built
pages, this is a **historical** page and may be stale relative to
`qwen35/blog_page/index.html`.

## Why the project moved to Qwen3.5-4B

On 2026-08-19 Samuel specified a new standalone experiment: "this isn't a
replication, want to do things properly with qwen 3.5 4b and write it up with this
model with no reference to old experiments." What carried over was craft — the
factored Gram identity, provenance checks on every artefact, verifying by physical
consequence rather than by label — not results. Comparability with the old corpora
was explicitly **waived** by Samuel the same day ("i wouldn't worry about the
comparison, feel free to upgrade packages to whatever makes sense"), so the
Qwen2.5-3B, Qwen3-4B and Qwen3.5-4B corpora differ in pipeline as well as model
and are not directly comparable.

Related: [[stage-one-training-config]], [[zoo-construction-overview|trait provenance]], [[geometry-overview]],
[[steering-results]], [[seed-floor]], [[teacherscreen]],
[[phase-two-recipe-search]], [[timeline]].
