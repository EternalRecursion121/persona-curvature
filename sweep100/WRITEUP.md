# The geometry of trait space: PCA over 100 personality LoRAs

*Draft. The steering section is pending results; everything else is final.*

## Summary

We trained 100 separate DPO LoRA adapters on a 3B model, one for each marker in
Goldberg's 1992 Big-Five adjective set, and asked what the space they live in
looks like.

Three findings:

1. **Traits are encoded as bipolar directions.** Opposite-poled markers of the
   same factor — Extraverted and Introverted, Organized and Disorganized — are
   *anti*-correlated in weight space (mean cosine −0.21, and −0.34 once the
   shared mean is removed). They are not two separately-learned things that
   happen to be about the same topic. This is the assumption that weight
   arithmetic over trait adapters quietly depends on, and as far as we know it
   has not been checked at this scale.

2. **The largest principal component is not a personality factor. It is
   evaluative.** PC1 accounts for 22.8% of the between-trait variance, and its
   poles are *Unintellectual, Unintelligent, Sloppy, Unsophisticated, Careless*
   against *Steady, Careful, Helpful, Neat, Conscientious*. It loads on
   Conscientiousness, Intellect and Agreeableness simultaneously. What it
   separates is not a factor but desirability: an ANOVA on which factor a trait
   belongs to gives F = 17.5 on pole-signed loadings, while an ANOVA on whether
   the trait is positively or negatively keyed gives F = 194.

3. **The Big Five are recovered only partially, and the reason is legible.**
   PC3 is Extraversion almost exactly (|cos| = 0.95 with the Extraversion
   direction; all twenty of its top loadings are Extraversion markers, split
   correctly by pole). PC2 is mostly Agreeableness. The other three factors do
   not get their own components — because the factor directions are not
   orthogonal to each other in weight space, so no rotation-free method could
   return them separately.

## Why do this

Recent work composes personality traits by training a LoRA per trait and then
doing arithmetic on the weight deltas — scaling them, inverting them, summing
them to combine traits. That machinery presupposes a geometry: that a trait is a
direction, that its opposite is the negation of that direction, and that
different traits are separable enough to add.

None of those assumptions is obviously true, and with five or ten adapters you
cannot test them. With a hundred you can.

We also wanted to know what the principal axes of the space *are*. If personality
in weight space had the same structure as personality in human self-report data,
the leading components would be the Big Five. If it does not, the difference is
informative.

## Method

**Traits.** Goldberg's 100 Unipolar Big-Five Markers (Goldberg, 1992,
*Psychological Assessment* 4(1), 26–42). Exactly 100 single-word trait
adjectives, 20 per factor, each factor split between positively- and
negatively-keyed markers. We chose this list over an ad hoc one specifically
because it ships with a validated factor structure — that structure is the ground
truth the analysis is tested against.

One asymmetry to note: Emotional Stability is 6 positively-keyed against 14
negatively-keyed, where every other factor is 10 and 10. This is genuine to
Goldberg's table, not a transcription error, and it makes Emotional Stability the
least reliable of the five throughout.

**Data.** For each trait, a teacher model generated 256 preference pairs: given a
prompt, one response written as someone strongly high in the trait, one as
someone notably low in it, produced in a single completion so the two are
contrasted directly. Neither response is permitted to name the trait or discuss
personality.

All 100 traits share **one prompt pool, in one order**. This matters more than it
sounds: the adapters are going to be compared to each other, so any difference in
what prompts a trait saw would enter the principal components as if it were
structure. After dropping generations that failed the filters, we took the
intersection of prompts present for every trait — 214 — and rewrote every trait's
data to exactly those, verified byte-identical in order across all 100 files.
That cost 16% of the data and removed a confound we would have had no way to
detect afterwards.

**Training.** Qwen2.5-3B-Instruct, LoRA rank 16 on all attention and MLP
projections, DPO with β = 0.1, two epochs. Every run starts from a
bit-identical LoRA initialisation, asserted at runtime, and uses a fixed data
order — otherwise nuisance variance between runs would be indistinguishable from
trait geometry. The DPO reference was verified live in every run: policy and
reference log-probabilities equal at initialisation, zero reward margin at step
zero, and the adapter-disabled path confirmed to actually change the outputs. A
silently-wrong reference would have produced 100 plausible adapters that meant
nothing.

Final reward margins ranged 4.6 to 12.2, so the preference was learned.

**Noise floor.** Five traits were retrained with a different data-order seed.
This is the reference against which every number below has to be read.

**Analysis.** Each adapter's effective delta is ΔW = (α/r)·BA per module,
concatenated over 252 modules, treated as one data point. PCA over the 100
points is done through the Gram matrix, computed from the factored form so no
dense ΔW is ever materialised. The machinery was checked against a dense
reference at 2.6e-15.

## Results

### The floor

| quantity | value |
|---|---|
| same trait, different seed: distance | 0.723 |
| same trait, different seed: cosine | 0.855 |
| two different traits: distance | 1.744 |
| two different traits: cosine | 0.099 |
| **between-trait / reseed distance** | **2.41** |

Trait separation is 2.4× seed noise. About 17% of squared between-trait distance
is seed variance. The geometry is real, and everything below is above the floor
except where noted.

### The spectrum

Centred variance explained: **22.8, 14.5, 9.7, 6.1**, then a cliff to 3.1 and a
long tail. Four components carry 53% of the between-trait variance; the fifth is
where seed noise could plausibly generate comparable structure, and we do not
interpret past PC4.

Four effective dimensions, not five.

### PC1 is an evaluative axis

| pole | top loadings |
|---|---|
| + | Unintellectual, Unintelligent, Sloppy, Unsophisticated, Careless, Negligent, Haphazard, Disorganized, Undependable, Shallow |
| − | Steady, Careful, Helpful, Neat, Conscientious, Organized, Deep, Considerate, Intellectual, Cooperative |

Every trait on the positive side is negatively keyed; every trait on the negative
side is positively keyed. They come from three different factors.

Its cosine with the factor directions is 0.87 with Conscientiousness, 0.82 with
Intellect, 0.59 with Agreeableness — it is not any one of them. An ANOVA on
keyed sign gives F = 194 (p ≤ 1e-4, permutation floor); on factor membership,
using pole-signed loadings so the test is fair, F = 17.5.

This is the general evaluative factor familiar from human trait-rating data,
where the first unrotated component of adjective ratings is desirability rather
than any substantive trait. The model reproduces it without being asked to.

### PC3 is Extraversion

|cos| = 0.950 with the Extraversion direction. The top ten positive loadings are
Inhibited, Unadventurous, Timid, Withdrawn, Reserved, Quiet, Shy, Introverted,
Bashful; the top ten negative are Energetic, Unrestrained, Talkative, Bold,
Vigorous, Extraverted, Daring, Active, Verbal. Nineteen of twenty are
Extraversion markers, correctly split by pole.

PC2 is mostly Agreeableness (|cos| = 0.77), warm against cold, with emotional
engagement bleeding in — Unemotional, Unexcitable and Unenvious sit on the cold
side. It is closer to "warmth" than to Agreeableness proper.

### Traits are bipolar directions

| pair type | mean cosine | mean-removed |
|---|---|---|
| same factor, same pole | +0.450 | +0.375 |
| same factor, **opposite** pole | **−0.209** | **−0.343** |
| different factor | +0.095 | −0.014 |

The mean-removed numbers are nearly symmetric, which is what you would expect if
a factor is a single axis with markers distributed along it.

This holds in every factor individually, but not equally: Agreeableness −0.362,
Intellect −0.260, Conscientiousness −0.244, Extraversion −0.119, Emotional
Stability −0.032. Strong for three, weak for Extraversion, essentially absent for
Emotional Stability — which is also the factor with the unbalanced marker set and
the smallest-norm factor direction.

The failure mode this rules out is important: if adapters encoded "this trait was
trained here" rather than a signed direction, opposite poles would be positively
correlated, not negatively. They are not.

### Why the five do not separate

The factor directions are not mutually orthogonal in weight space:

| | E | A | C | ES | I |
|---|---|---|---|---|---|
| E | — | .07 | .10 | −.07 | .23 |
| A | | — | .26 | .17 | .42 |
| C | | | — | **.58** | **.58** |
| ES | | | | — | .21 |

Conscientiousness sits at 0.58 from both Emotional Stability and Intellect. A
rotation-free PCA cannot return correlated factors as separate components. What
it returns instead is exactly what this table predicts: the shared evaluative
direction first, then the two factors most nearly independent of it —
Agreeableness and Extraversion. Extraversion is near-orthogonal to everything
(≤ 0.23), which is why it comes back cleanest.

This is not the method failing. It is the method reporting the correlation
structure accurately.

## Steering along the components

*[PENDING — grid running: PC1, PC2, PC3 at α ∈ {−3…+3} in units of one adapter's
worth of Frobenius norm, against a matched-norm random-direction control, scored
for alignment, coherence and capability on three probe sets.]*

The specific prediction under test, contributed by the person who commissioned
this work: steering toward PC1's undesirable pole should produce emergent
misalignment — behaviour that is broadly badly-intentioned rather than merely
sloppy — and the matched random direction should not.

## Limitations

- **One 3B model.** Nothing here shows the geometry is scale-invariant.
- **DPO only in this analysis.** A second-stage adapter, trained on
  self-interaction and self-reflection transcripts with the first stage frozen,
  is the other half of the pipeline this replicates; that comparison is pending.
- **No behavioural verification that the adapters express their traits.** Reward
  margins show DPO learned the stated preference, which is good evidence and is
  not the same claim. A judged behavioural check is the obvious next thing.
- **Emotional Stability is unbalanced** in the source list and is the weakest on
  every test here. Treat its results as the least reliable.
- **214 preference pairs per trait** is low. This buys consistency across traits,
  which is what a comparison needs, at the cost of per-trait strength.
- **Nothing past PC4** is above the level where five reseeds suggest noise could
  produce comparable structure, and five reseeds are too few to build a proper
  null spectrum. That is the main thing we could not verify.
