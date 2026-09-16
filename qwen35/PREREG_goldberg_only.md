# Pre-registration: is the five-factor structure an artefact of choosing Big Five marker words?

Written 2026-09-12, before any number below was computed. Nothing in this file
was edited after the first run. CPU only, no GPU, no Modal, no retraining; every
input is an existing artefact of the zoo.

## The question

The zoo is 100 Goldberg Big Five marker adjectives (`traits_primary.json`) plus
34 adjectives drawn from Condon's 2,818-word trait lexicon with no regard to the
Big Five (`traits_secondary.json`). The factor analysis that recovers five
factors was run on all 134 but scored only on the 100. Two readings are
compatible with that:

- **Artefact.** The five factors exist because 100 of the 134 words were picked
  as Big Five markers, and constitutions were written for them by a teacher that
  knows the Big Five. The structure is in the choice of words.
- **Representation.** The five factors are a property of how this model
  represents trait words. Words picked without the Big Five in mind should land
  in the same five-dimensional frame, in the places their meanings imply.

Three tests separate them. Each has a threshold fixed here.

## Fixed decisions that are not thresholds

1. **The centring differs by construction.** `analyse_fa_qwen35.py` double-centres
   the Gram over the p variables it is handed. The 100-only run therefore removes
   the 100-trait grand mean, the 134 run removes the 134-trait grand mean. Any
   congruence between the two solutions includes that change of origin. This is
   recorded here so it is not discovered afterwards and reported as a finding.
2. **Factor matching is one-to-one Hungarian on |Tucker|**, using the project's
   own `tucker()` (`analyse_fa_qwen35.py`) and the best-one-to-one convention
   `analyse_fa_nulls.py` uses for `fa_nulls.json#tucker_vs_real_stage_one`, so
   the numbers are commensurable with that file. Congruence bars are the project's
   own: 0.85 "fair", 0.95 "identical".
3. **Chart order.** The 100-only solution may order its factors differently
   (the Fisher-metric rerun moved Imagination from fifth to third). Each 100-only
   factor is mapped to its Hungarian-matched 134 factor and sign-oriented so the
   matched Tucker is positive; the chart is then Gram-Schmidt orthonormalised in
   the 134 chart's **fixed** order Warmth, Competence, Timidity, Arousal,
   Imagination (`fa_chart.FACTOR_ORDER`). Without this, per-axis correlations
   would compare mismatched axes.
4. **Factor directions are built with the project's own recipe**, not by calling
   `FAChart` on a 100-trait Gram: `FAChart` reads the coefficient dicts of
   `phase10_runs/steer_spec2_7a.json` and `.get(t, 0.0)` would silently zero the
   34 terms, producing the 134 directions truncated rather than a 100-only chart.
   The recipe is `steer134_on_modal.py:fa_coeffs` - take the oblimin loading
   column, subtract its mean, scale to unit norm in the double-centred Gram. It
   is validated first by reproducing `FA_Warmth` and the other four of
   `steer_spec2_7a.json` from `fa_qwen35.json#steering.oblimin_loadings` on the
   134; only if that reproduces does the same recipe get applied to the 100.
5. **Embedding model** is `sentence-transformers/all-mpnet-base-v2`, already
   cached locally and already the model this project used to draw the 34
   (`traits_secondary_provenance.json#embedding_model`). It is run in
   `/home/vibe12/cartovenv/bin/python`, which has torch and sentence-transformers;
   the `qwen35/.venv` has neither. The pair-contrast baseline uses
   `all-MiniLM-L6-v2`, the model sweep100's text baseline used.
6. **Rater.** One OpenRouter chat model, temperature 0, three independent
   repeats per word, one word per call, a prompt containing only the word and
   textbook definitions of the five factors - no project text, no constitution,
   no zoo, no geometry. Raw responses are saved. The key is read from
   `~/.secrets/openrouter-api-key` straight into a request header and never
   printed.

---

## Test 1 - Goldberg-only factor analysis

Restrict `results/gram_sweep.npz` to the 100 primary markers (`G[ix][:, ix]`,
`norms[ix]`, same `scale` and `n_modules`), write
`results/gram_goldberg100.npz`, and run `analyse_fa_qwen35.py` on it unchanged
via `PC_GRAM_NPZ` and `PC_FA_TAG=_goldberg100`, exactly as the null arms were run.
The script regenerates its parallel-analysis null at p = 100, so the retained
factor count is on the same scale as the two null arms for the first time - the
`open-questions.md` item "Never run: a 100-trait real arm for the null
comparison".

Reported: one-to-one Tucker congruence of the 100-only centred_k5 oblimin
loadings against the 134 centred_k5 oblimin loadings restricted to the same 100
rows; the 100-only best congruence per Goldberg target; the retained k; and the
100-only centred eigenvalue spectrum beside the shuffled and permuted arms'
`fa_nulls.json#arms.*.centred_eigenvalues_top12`.

**Thresholds.**

- **T1a (the factors are not made by the 34).** Mean absolute matched Tucker
  congruence >= **0.85**, with the matching a permutation that takes each 134
  factor once. Pass at >= 0.95 is "identical". FALSIFIED if the mean falls below
  0.85, or if the matching collapses two 100-only factors onto one 134 factor.
- **T1b (the Big Five recovery is not carried by the 34).** All five Goldberg
  targets taken once by five different factors, and no best-per-target
  congruence moves by more than **0.10** from the 134 run's
  (E 0.539, A 0.655, C 0.574, ES 0.405, I 0.682,
  `fa_nulls.json#arms.real.best_congruence_per_big_five_target`).
- **T1c (same-size null comparison).** Retained k at N = 1528 >= **5**, against
  the shuffled arm's 0 and the permuted arm's 8, and the leading centred
  eigenvalue at least **5x** the shuffled arm's 1.222.

T1 cannot by itself answer the artefact question - it only removes the 34 from
the solution. It is the same-size comparator the nulls never had, and the
precondition for test 2: a chart the 34 played no part in defining.

## Test 2 - the 34 as a held-out validation set

Build the chart from the 100 only (decisions 3 and 4 above), then place the 34
Lexicon adapters in it by their **exact inner products** with the 100,
`x_e = B100 @ G[ix100, e]`, the `coords_external` convention of `fa_chart.py`.
The 34 contribute nothing to the basis: it is defined entirely by 100 adapters
and their Goldberg-keyed loadings.

**(a) Against the full chart.** Pearson per axis over the 34 between the
100-defined coordinates and `FAChart().trait_coords` rows for the same 34, plus
orthogonal Procrustes disparity between the two 34 x 5 configurations.

**(b) Against an independent semantic expectation.** Each of the 134 words is
rated by the LLM rater on the five factors, -2 to +2, as a lexical judgement.
Ratings are averaged over three repeats. Chart coordinate is correlated with
rated position per factor using the mapping the project uses:

| chart axis | Big Five | expected sign of Pearson r |
|---|---|---|
| Warmth | Agreeableness | + |
| Competence | Conscientiousness | + |
| Timidity | Emotional Stability | **-** (Timidity is ES reversed) |
| Arousal | Extraversion | + |
| Imagination | Intellect | + |

The expected signs are fixed here. They are checked first on the 100-marker
ceiling; if one disagrees there it is reported as a disagreement, not flipped.

**The ceiling.** The same correlation computed on the 100 markers with their
Goldberg keying as the rating (+1 for a positively keyed marker of that factor,
-1 for a negatively keyed one, 0 for a marker of another factor), in the 134
chart. This is the best any axis-to-Big-Five correlation can be expected to do
here, and it is the reference the 34 are read against. A second ceiling is
reported: the rater's own ratings on the 100 correlated with Goldberg keying,
which bounds how independent an expectation the ratings are at all. If the rater
cannot reproduce Goldberg's keying on the marker words, its ratings of the 34
are not a usable expectation and test 2b is reported as uninformative rather
than as a failure of the geometry.

**Thresholds.**

- **T2a (the 34 land in the same place either way).** Mean per-axis Pearson
  across the five axes >= **0.90**, and no single axis below **0.70**.
- **T2b (the 100-defined structure predicts where Big-Five-agnostic words
  land).** At least **3 of 5** axes reach |r| >= **0.34** (the two-tailed p < 0.05
  floor at n = 34) **in the pre-registered sign**, and the mean signed r over the
  five axes is at least **half** the 100-marker ceiling's mean signed r.
  FALSIFIED if fewer than 3 axes clear the floor in the right direction, or if
  the mean is below half the ceiling. A result where the 34's mean matches or
  exceeds the ceiling is the strongest available evidence against the artefact
  reading.

## Test 3 - the text-embedding baseline

sweep100's deciding test was that weight-space geometry was no better than
embedding the training text (`sweep100/results/text_vs_weights_fa.json`; RSA
between the two centred matrices 0.911, weights mean congruence 0.750 against
text's 0.731). The same objection applies here and is tested the same way.

**(i) Ridge from text to chart.** Embed each trait's constitution text
(`constitutions.json`, the `constitution` field, one document per trait) with
all-mpnet-base-v2. Fit ridge regression from the 768-dimensional embedding to
the five 134-chart coordinates on the **100 markers only**; alpha chosen by
leave-one-out over the grid `[1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]`, no
other tuning. Predict the 34 and report held-out R^2 per axis and overall
(1 - SSE/SST with SST about the 100-marker training mean). The comparator is the
R^2 of the adapter geometry's own placement of the same 34 - the test-2a
100-defined coordinates - against the same targets.

**(ii) The reverse direction.** Build the 134 x 134 cosine Gram of the
constitution embeddings, double-centre it and the adapter Gram, and report the
Pearson correlation over the off-diagonal entries.

**(iii) The contrast baseline, because a constitution is one text.** sweep100
found that embedding the chosen replies alone gave RSA 0.429 while
chosen-minus-rejected gave 0.826: "the structure is in the contrast, not the
text". A constitution is structurally the chosen-only case, so a low (ii) would
not by itself establish that the geometry adds structure. The pair files are
local (`qwen35/data/<slug>.jsonl`), so the contrast baseline is also built:
per trait, mean embedding of chosen minus mean embedding of rejected over a
fixed subsample of pairs (seed 0), MiniLM, and the same double-centred
off-diagonal Pearson against the adapter Gram. **The verdict hangs on the larger
of (ii) and (iii)**; (ii) alone can only bound the constitution text.

**Thresholds**, anchored to sweep100's own numbers.

- **"The geometry is the text passed through"**: off-diagonal Pearson >= **0.80**
  on the larger of (ii) and (iii), **or** held-out overall R^2 >= **0.50** from
  text to chart on the 34.
- **"The geometry adds structure the text does not carry"**: off-diagonal
  Pearson <= **0.50** on both (ii) and (iii), **and** held-out overall R^2
  <= **0.20**.
- Anything between is **partial**, and is reported as partial with both numbers,
  not resolved one way.

## What each outcome would mean

- T1 passes, T2 passes, T3 says "text passed through": the five factors are real
  and generalise to Big-Five-agnostic words, and they are a fact about the trait
  lexicon carried by the teacher's text rather than something weight space added.
  The artefact reading is wrong about the word list and right about the source.
- T1 passes, T2 passes, T3 says "adds structure": the strongest outcome; the
  frame is a property of the model's representation of trait words.
- T1 passes, T2 fails: the five factors survive dropping the 34 but do not
  predict where an unlabelled word lands, which is the artefact reading in its
  precise form - the frame is defined by the marker words and does not extend.
- T1 fails: the five factors depend on which 134 words were in the Gram, and
  nothing downstream needs interpreting.

## Outputs

`qwen35/analysis/goldberg_only.json` (every number under named keys),
`qwen35/results/gram_goldberg100.npz`,
`qwen35/results/fa_qwen35_goldberg100.{json,md}`,
`qwen35/analysis/goldberg_only_ratings.jsonl` (raw rater responses).

---

# Addendum, 2026-09-15: the 34 Lexicon adapters analysed in isolation

Written before any number in this section was computed. Samuel's follow-up to the
tests above: "when we analyse the 34 adapters in isolation do we get the same
thing?" The tests above held the 34 out and asked whether the marker-defined
frame could find them. This asks the reverse: do the 34 words, on their own,
*contain* the frame?

CPU only, no API calls, expected spend **zero**. Outputs go under a new
`lexicon_only` key in `qwen35/analysis/goldberg_only.json` and a new section on
`wiki/pages/geometry/goldberg-only-and-heldout-lexicon.md`.

## Fixed decisions

1. **The script cannot be reused unchanged.** `analyse_fa_qwen35.py:load_data`
   asserts `len(names) in (100, 134)` and `n_prim == 100`, so a 34-trait Gram
   cannot be passed to it. The *mathematics* is reused instead by importing
   `smc`, `paf`, `kaiser_normalise`, `varimax_pairwise`, `oblimin`, `tucker` and
   `parallel_analysis` from that module and calling them in the same order, with
   the same constants (`RIDGE = 1e-3` on the ipsatised matrix, `PA_REPS = 500`,
   `PA_GRID`, `PA_REFERENCE_N = 1528`, `SEED = 0`). The script itself is not
   edited. Every arm below - the 34, and all 200 reference subsets - goes through
   the identical code path, so the comparison is internal and does not depend on
   that path matching the 134-trait run exactly.
2. **Ordering and orientation need a new convention**, because
   `order_and_orient` orients each factor against its best Goldberg target and
   the 34 score 0 in every target, which is 0/0. Factors are ordered by
   descending sum of squared oblimin loadings (the script's own first step) and
   left with arbitrary sign; sign is resolved only at the point of comparison, by
   the one-to-one matching, exactly as [[factor-analysis-null-arms]] does when it
   says "sign is arbitrary in factor analysis".
3. **Matching** is `analyse_stage2_structure.best_match` on the |Tucker| matrix,
   as in the tests above, and is required to be a permutation taking each of the
   134 solution's five factors exactly once.
4. **The parallel-analysis null is data-independent.** It depends only on p and
   N, so the p = 34 null is generated once at 500 reps over the full grid and
   reused for the 34 and for all 200 reference subsets. This is not an
   approximation; it is the same null each arm would generate for itself.
5. **Five factors from 34 variables is a strain** and is recorded as such in
   advance: 6.8 variables per factor, against 20 at p = 100 and 26.8 at p = 134.
   Heywood cases and non-convergence are expected to be more common and are
   reported per arm, not hidden.

## Test L1 - factor the 34 alone

The 34 x 34 submatrix of `results/gram_sweep.npz`, ipsatised over those 34
(`H = I - 11^T/34`), PAF from ridge-SMC, oblimin (gamma = 0), k = 5 forced.
Reported: the one-to-one Tucker congruence of that k = 5 solution against the
134-trait `centred_k5` oblimin loadings restricted to the same 34 rows; and what
parallel analysis retains at p = 34 across the grid.

## Test L2 - the random-34 reference distribution

Thirty-four variables is small, so the lexicon 34's congruence means nothing
without knowing what 34 variables can do. Two hundred random 34-trait subsets of
the **100 Goldberg markers** (`numpy.default_rng(0)`, without replacement) are
each factored by the identical path, and each subset's k = 5 solution is scored
by the same one-to-one Tucker congruence against the 134 solution restricted to
*that* subset's rows. The comparison is symmetric: the lexicon 34 and every
reference subset are alike among the 134 that defined the target solution, so
both are in-sample in the same way.

Reported: the lexicon 34's mean absolute matched congruence as a **percentile**
of the 200 reference values, the reference distribution's quantiles, and the
distribution of parallel-analysis retention counts across the 200 subsets beside
the lexicon 34's.

## Test L3 - the reverse held-out test

The mirror of test 2 above. Build a chart from the **34 alone**: their k = 5
oblimin loading columns turned into directions over the 34 adapters by
`steer134_on_modal.py:fa_coeffs`, reordered to the 134 chart's fixed order
(Warmth, Competence, Timidity, Arousal, Imagination) by the L1 matching,
sign-oriented so each matched congruence is positive, and Gram-Schmidt
orthonormalised in the 34 x 34 Gram. Then place the **100 Goldberg markers** in
it by their exact inner products with the 34, `x = B34 @ G[ix34, marker]`.

Reported per axis: Pearson between the markers' 34-defined coordinates and (a)
Goldberg's keying (+1 positively keyed marker of that factor, -1 negatively
keyed, 0 for a marker of another factor), and (b) their coordinates in the full
134 chart. The ceiling for (a) is already measured and is the same one the tests
above used: Warmth 0.6828059271418205, Competence 0.5723042028598464, Timidity
0.47039913464559113, Arousal 0.5452041935257524, Imagination 0.7639889196055134,
mean 0.6069404755557047
(`#test2b_independent_lexical_ratings.ceiling_goldberg_keying_vs_134chart_100markers`).
The Timidity axis is Emotional-Stability-**positive**, as established above; all
five expected signs are positive.

## Thresholds

- **L1/L2 "same structure".** The lexicon 34's mean absolute matched Tucker
  congruence falls **within the middle 90 per cent** of the 200-subset reference
  distribution - percentile between 5 and 95 - and the matching is a permutation
  taking each of the five 134 factors once. In words: the lexicon words are no
  worse a sample of the structure than 34 of Goldberg's own words are.
- **L1/L2 "the 34 carry a different structure".** The lexicon 34's congruence
  falls **below the 5th percentile** of the reference distribution, **or** the
  matching is not a permutation over the five. Either would mean that what 34
  Big-Five-agnostic words contain is not what 34 marker words contain.
- **L3 "the 34 contain the frame".** The 100 markers placed in the 34-defined
  chart correlate with Goldberg's keying at a mean per-axis Pearson of at least
  **half the ceiling**, i.e. at least **0.30347023777785235**, with all five
  signs positive; and their positions agree with the full 134 chart at a mean
  per-axis Pearson of at least **0.70**. FALSIFIED if either fails.
- A result where the lexicon 34 sit **above** the reference median is the
  strongest available outcome and would say the Big-Five-agnostic words are, if
  anything, a better-spread sample of the structure than a random third of the
  marker set.

## What would change the reading of the tests above

Test 2 showed the marker-defined frame finds the 34. If the 34 alone do **not**
contain the frame, that is not a contradiction - it would mean 34 variables are
too few to recover five factors, which tests L2's reference distribution is
designed to detect, by showing whether 34 *marker* words manage it either. The
reference distribution is what makes L1 interpretable; L1 alone is not.
