---
title: Goldberg-only factoring and the 34 as a held-out set
summary: The five factors are not made by the marker words - factoring the 100 Goldberg markers alone returns the same five at mean Tucker congruence 0.9923, and a chart built from those 100 alone places the 34 Big-Five-agnostic Lexicon words where an independent LLM rater says they belong, at mean r 0.6098 against the 0.6069 the marker words themselves achieve; the 34 factored in isolation return all five factors too but as a poorer sample, at the 3rd percentile of 200 random 34-marker subsets, with one chart dimension barely covered; and the training contrast predicts chart position at R^2 0.938, so the frame is the text passed through, as sweep100 found.
status: current
sources:
  - qwen35/PREREG_goldberg_only.md
  - qwen35/analyse_goldberg_only.py
  - qwen35/analyse_lexicon_only.py
  - qwen35/embed_goldberg_only.py
  - qwen35/analysis/goldberg_only.json
  - qwen35/analysis/goldberg_only_ratings.jsonl
  - qwen35/results/gram_goldberg100.npz
  - qwen35/results/fa_qwen35_goldberg100.json
  - qwen35/results/fa_qwen35.json
  - qwen35/analysis/fa_nulls.json
  - qwen35/analysis/viz_fa.json#big_five_scale
  - qwen35/fa_chart.py
  - qwen35/analyse_fa_qwen35.py
  - qwen35/steer134_on_modal.py
last_verified: 2026-09-15
tags: [geometry, factor-analysis, big-five, nulls, text-baseline, held-out]
---

# Goldberg-only factoring and the 34 as a held-out set

## The question

The zoo is 100 Goldberg Big Five marker adjectives and 34 adjectives drawn from
Condon's 2,818-word trait lexicon with no regard to the Big Five
([[goldberg-100-primary-traits]], [[lexicon-secondary-draw]]). The factor
analysis that recovers five factors ([[factor-analysis]]) is run on all 134 but
scored only on the 100, because only the 100 carry a Big Five label. Two readings
survive that:

- **Artefact.** The five factors exist because 100 of the 134 words were picked
  as Big Five markers and constitutions were written for them. The structure is
  in the choice of words.
- **Representation.** The five factors are a property of how the model
  represents trait words, and words picked without the Big Five in mind should
  land in the same frame, in the places their meanings imply.

Three tests, all pre-registered in `qwen35/PREREG_goldberg_only.md` before any
number below was computed, with thresholds fixed in that file. The computation is
`qwen35/analyse_goldberg_only.py`; every number here is a key in
`qwen35/analysis/goldberg_only.json`. A dated addendum of 2026-09-15 adds a
fourth test - the 34 factored in isolation - pre-registered the same way and
computed by `qwen35/analyse_lexicon_only.py` under `#lexicon_only`.

**The answer in one line.** The five factors are not made by the marker words,
and the frame they define locates Big-Five-agnostic words as accurately as it
locates the marker words - but the frame is carried by the contrast in the
training text, not added by weight space, which is the conclusion
[[sweep100]] reached on a different model and a different trait set.

## What was run, and one caveat fixed in advance

`results/gram_sweep.npz` was restricted to the 100 primary slugs and written as
`results/gram_goldberg100.npz`, then run through `analyse_fa_qwen35.py`
**unchanged** via `PC_GRAM_NPZ` and `PC_FA_TAG=_goldberg100`, exactly as the two
null arms were run ([[factor-analysis-null-arms]]).

The caveat, pre-registered rather than discovered: `analyse_fa_qwen35.py`
double-centres over the p variables it is handed, so the 100-only run removes the
100-trait grand mean and the 134 run removes the 134-trait grand mean. Every
comparison below therefore includes that change of origin
(`#inputs.centring_caveat`).

The 100-only run is healthy on its own diagnostics
(`#test1_goldberg_only_factor_analysis.diagnostics`): the script's seven-part
self-test passes (`verification_all_ok` true), principal axis factoring converged
in 9 iterations with 0 Heywood cases, and the oblimin factor intercorrelations
stay small - largest absolute off-diagonal Phi **0.2995960027411673** against the
134-trait solution's 0.282.

## Test 1 - the five factors survive dropping the 34

The 100-only `centred_k5` oblimin loadings against the 134 solution's, restricted
to the same 100 rows, matched one-to-one by
`analyse_stage2_structure.best_match` - the same matcher `analyse_fa_nulls.py`
uses, confirmed to agree with an independent Hungarian solver
(`#test1_goldberg_only_factor_analysis.one_to_one_matching.matcher_agrees_with_hungarian` = true):

| 134 factor | matched 100-only factor | Tucker congruence |
|---|---|---|
| F1 Warmth | 100-only F1 | 0.9973962146947465 |
| F2 Competence | 100-only F3 | 0.9896430976847599 |
| F3 Timidity | 100-only F4 | 0.9932420645305269 |
| F4 Arousal | 100-only F2 | 0.9841766610292786 |
| F5 Imagination | 100-only F5 | 0.9969317463250906 |

Mean absolute matched congruence **0.9922779568528804**, minimum
**0.9841766610292786**, all five above the 0.95 "identical" bar
(`#...one_to_one_matching`). The matching is a permutation - every 134 factor is
taken exactly once - but it is not the identity, because the 100-only solution
orders its factors by its own sum of squared oblimin loadings. The same
reordering happened when the Gram was rebuilt in the Fisher metric
([[factor-analysis-fisher-metric]]), where Imagination moved from fifth to third
and the five factors still matched at 0.9654.

**Verdict T1a: passes, at the identical bar.** Threshold was mean absolute
matched congruence at or above 0.85.

### The Big Five congruences rise, and the rise is an artefact of Tucker's denominator

`#test1_goldberg_only_factor_analysis.big_five_congruence`:

| target | 134 arm | 100-only arm | delta |
|---|---|---|---|
| E | 0.538926935263515 | 0.5402944425426125 | +0.001367507279097513 |
| A | 0.6554774644183582 | 0.759313804675552 | +0.1038363402571938 |
| C | 0.5744454556114018 | 0.641553608012454 | +0.06710815240105217 |
| ES | 0.4048918307420467 | 0.4232833308028415 | +0.01839150006079482 |
| I | 0.6822835176928186 | 0.7559311500580892 | +0.07364763236527061 |

Still **0 of 5** clearing the conventional 0.85 "fair" bar
(`#...n_targets_clearing_0.85`), as in every solution on [[factor-analysis]].

The rise is not a better solution. A Goldberg target is zero on all 34 Lexicon
rows, so those rows enter only the denominator of Tucker's coefficient and can
only lower it. Taking the **134** loadings and restricting them to the same 100
rows isolates that
(`#...big_five_congruence.decomposition_of_the_rise`):

| target | 134 solution, all 134 rows | 134 solution, 100 rows only | 100-only solution | rise from the denominator | rise from refactoring |
|---|---|---|---|---|---|
| E | 0.5389 | 0.5899 | 0.5403 | +0.0510 | -0.0496 |
| A | 0.6555 | 0.7758 | 0.7593 | +0.1204 | -0.0165 |
| C | 0.5744 | 0.6404 | 0.6416 | +0.0659 | +0.0012 |
| ES | 0.4049 | 0.4631 | 0.4233 | +0.0582 | -0.0399 |
| I | 0.6823 | 0.7554 | 0.7559 | +0.0731 | +0.0006 |

(Values rounded here to four places for the table; full precision under
`#...decomposition_of_the_rise`.) Refactoring without the 34 moves congruence by
between -0.0496 and +0.0012. **Dropping the 34 does not improve the Big Five
recovery; it removes 34 zero rows from a denominator.** Anyone quoting the
0.7593 for Agreeableness must quote it against 0.7758, not against 0.6555.

**Verdict T1b: fails, on both prongs, narrowly, and not in a way that matters.**
The threshold was "all five targets taken once by five different factors, and no
best-per-target congruence moves by more than 0.10". The A delta is 0.1038,
which exceeds 0.10 and is now shown to be a denominator effect. And on the
per-factor argmax convention `fa_nulls.json` uses, the third factor's own best
target moves from ES to E by **0.0024** (E 0.4257, ES 0.4233,
`#...full_congruence_oblimin_rows_factors_cols_E_A_C_ES_I_Eval`), so only four
distinct targets are taken. On the other reading - which factor is each target's
best match - all five targets still take five different factors
(`#...goldberg100_n_distinct_factors_across_targets` = 5). Both readings are
recorded because the pre-registration named only one.

### The same-size null comparison, which had never been run

This closes the [[open-questions]] item "Never run: a 100-trait real arm for the
null comparison". A correlation matrix has trace p, so the two null arms at
p = 100 were never on the same eigenvalue scale as the real arm at p = 134. Now
all three are (`#...same_size_null_comparison`):

| centred eigenvalue | Goldberg-100 real | shuffled | permuted | real 134 |
|---|---|---|---|---|
| 1 | 12.72505697309637 | 1.2220951463764176 | 13.63187169433154 | 16.623868698801367 |
| 2 | 11.139585599037872 | 1.20267712746838 | 10.390404019455278 | 14.287107976941066 |
| 3 | 5.662908292594041 | 1.1857911477336163 | 4.945136854288979 | 6.476928426465509 |
| 4 | 3.996177186321918 | 1.1613049901755712 | 3.8874696342485437 | 4.973822750821908 |
| 5 | 2.9270793731877207 | 1.149952334058564 | 2.823814370501938 | 3.4483835353971584 |
| retained k, Horn, N = 1528 | 8 | 0 | 8 | 9 |

The leading eigenvalue is **10.412492849535404** times the shuffled arm's. It is
**0.9334783409374191** of the permuted arm's, and the two arms retain the same
number of factors.

**The spectrum does not separate real labels from permuted ones at equal p.**
Only congruence with the keying does: the real 100-trait arm reaches A 0.7593,
I 0.7559, C 0.6416, E 0.5403, ES 0.4233 where the permuted arm's best anywhere is
0.24109318438008137. That is what [[factor-analysis-null-arms]] already argued
from the loadings, now visible on the eigenvalues themselves.

**Verdict T1c: passes.** Threshold was retained k at or above 5 and a leading
eigenvalue at least five times the shuffled arm's.

Parallel analysis on the 100-only arm, centred, 95th percentile, 500 reps:
N = 150 gives 5, N = 300 gives 6, N = 1000 gives 7, N = 1528 gives 8, N = 5809
gives 9, N = 20000 gives 12 (`#...n_factors.parallel_analysis_centred_grid`).
The chosen k is 8 at the project's reference N = 1528, against 9 for the 134
arm. As on [[factor-analysis]], the data does not pick five; five is extracted
because five is the hypothesis.

## Test 2 - the 34 as a held-out validation set

### The chart the 34 did not define

Each of the five 100-only oblimin loading columns was turned into a direction
over the 100 adapters by the project's own recipe,
`steer134_on_modal.py:fa_coeffs` - mean-centre the column, scale to unit norm in
the double-centred Gram - then reordered to the 134 chart's fixed order
(Warmth, Competence, Timidity, Arousal, Imagination) using the matching above,
sign-oriented to its matched 134 factor, and Gram-Schmidt orthonormalised in the
100 x 100 Gram, which is `fa_chart.FAChart`'s construction restricted to 100
variables.

The recipe was validated first: applied to the **134** loadings it reproduces
`FAChart`'s basis - built independently from
`phase10_runs/steer_spec2_7a.json`, the coefficients actually steered in phase 10
- to a maximum absolute difference of
**2.0816681711721685e-17**
(`#test2a_the_34_in_a_chart_they_did_not_define.recipe_validation`). Calling
`FAChart` on a 100-trait Gram would have been wrong: it reads the 134-trait
coefficient dicts and silently zero-fills the missing 34 terms, giving the 134
directions truncated rather than a chart the 34 never touched.

The 34 were then placed by their **exact inner products with the 100**,
`x = B100 @ G[ix100, e]`, the `coords_external` convention of
[[factor-chart]]. They contribute nothing to the basis.

### The 34 land in the same place either way

Against their coordinates in the full 134 chart
(`#test2a_the_34_in_a_chart_they_did_not_define`):

| axis | Pearson over the 34 |
|---|---|
| Warmth | 0.9995736516221724 |
| Competence | 0.9994480216511715 |
| Timidity | 0.9961011161085458 |
| Arousal | 0.9521841176397594 |
| Imagination | 0.9867308222282795 |

Mean **0.9868075458499858**, minimum **0.9521841176397594**. Orthogonal
Procrustes disparity between the two 34 x 5 configurations, both normalised to
unit Frobenius norm, is **0.008560583694548797**; without any rotation the
relative Frobenius error is **0.126673622872055**. Per trait, the two 5-vectors
have a median cosine of **0.9956080054634957** and a minimum of
**0.9786606839242086**.

The principal cosines between the two five-dimensional subspaces are
**0.9956114305232384, 0.99177462269438, 0.9910741726319335, 0.9800160292102036,
0.948126742343882**. That reproduces, from a different construction, the
0.94-0.996 the blog page already reported for the markers' subspace against all
134 ([[lexicon-secondary-draw]]).

**Verdict T2a: passes.** Threshold was mean per-axis Pearson at or above 0.90
with no axis below 0.70.

### An independent semantic expectation

All 134 words were rated by `anthropic/claude-sonnet-4.5` on the five Big Five
dimensions from -2 to +2, temperature 0, three repeats, one word per call. The
prompt carried the adjective and textbook one-line definitions of the five
factors and nothing else: no constitution, no zoo, no geometry, no project text
(`#test2b_independent_lexical_ratings.rater`). Raw responses are in
`qwen35/analysis/goldberg_only_ratings.jsonl`. 402 calls, 0 errors, identical
identical ratings on all five dimensions across all three repeats for a
fraction **0.9776119402985075** of the 134 words
(`#...exact_agreement_across_3_repeats_frac_words`).

The rater is a usable expectation: its ratings of the 100 markers correlate with
Goldberg's own keying at E 0.8642595476428578, A 0.8303763624334459,
C 0.8903851136332661, ES 0.8186875912454102, I 0.9430186245659982, mean
**0.8693454479041955**
(`#...is_the_rater_an_independent_expectation`).

**A pre-registered sign was wrong, and it is the project's own convention that
settles it.** The pre-registration expected the Timidity axis to correlate
*negatively* with Emotional Stability, on the reading that Timidity is Emotional
Stability reversed. It does not, on the ceiling or on the held-out set: the
100-marker ceiling gives **+0.47039913464559113** and the 34 give
**+0.7212498235101539**. The reason is mechanical.
`analyse_fa_qwen35.py:order_and_orient` orients every factor so its congruence
with its own best Goldberg target is positive, so the chart axis is **already
Emotional-Stability-positive**, and `qwen35/analysis/viz_fa.json#big_five_scale`
maps `FA_FearfulWithdrawal` to `EmotionalStability` with no sign flag. The name
Timidity describes the axis's **negative** pole. Its poles in the chart, in order
(`#...sign_note`):

- positive: active, unenvious, trustful, energetic, assertive, imperceptive, impractical
- negative: insecure, shy, nervous, bashful, guilty, fearful, timid

The pre-registration required a disagreeing sign be reported, not flipped, so
both scorings are given below.

### Where the 34 land against what the rater says

Pearson between each chart axis and the rater's mean rating on the Big Five scale
the project maps it to, over the 34, in the chart the 34 did not define
(`#...heldout_34_rater_vs_100only_chart`), beside the two ceilings measured on the
100 markers in the full 134 chart:

| axis | Big Five scale | the 34, 100-only chart | the 34, 134 chart | ceiling: 100 markers by Goldberg keying | ceiling: 100 markers by the rater |
|---|---|---|---|---|---|
| Warmth | Agreeableness | 0.6815960588954219 | 0.6892901640008928 | 0.6828059271418205 | 0.6496210536094948 |
| Competence | Conscientiousness | 0.4892631022266945 | 0.4931289011143912 | 0.5723042028598464 | 0.6099819201961969 |
| Timidity | Emotional Stability | 0.7212498235101539 | 0.7394647198859591 | 0.47039913464559113 | 0.6119256653267278 |
| Arousal | Extraversion | 0.5789413788580245 | 0.6178936263631418 | 0.5452041935257524 | 0.5351227151890962 |
| Imagination | Intellect | 0.5777118028349205 | 0.5782368538763027 | 0.7639889196055134 | 0.789164843786663 |
| **mean** | | **0.609752433265043** | **0.6236028530481376** | **0.6069404755557047** | **0.6391632396216357** |

All five axes clear the n = 34 two-tailed p < 0.05 floor of 0.34, in the
direction the project's own mapping predicts. The held-out mean is
**1.0046330040960998** times the Goldberg-keying ceiling and
**0.9539854538975006** times the rater's own ceiling on the marker words
(`#...heldout_34_mean_r_over_goldberg_keying_ceiling`,
`#...heldout_34_mean_r_over_rater_ceiling`).

**The frame built from 100 Big Five marker words locates 34 words chosen without
the Big Five in mind about as accurately as it locates the marker words
themselves.**

Word by word, from `#...coords_34_from_100only_chart` and
`#...ratings_mean_over_3_repeats`: *gruff* and *insensitive* take their largest
coordinate on negative Warmth and are rated Agreeableness -2; *composed* and
*dependable* take theirs on positive Competence and are rated Conscientiousness
+1 and +2; *guilty* takes its largest on negative Timidity and is rated Emotional
Stability -2. The correlations are not 1.0 and the misses are visible in the same
list: *courageous* takes its largest coordinate on negative Warmth where the
rater calls it Emotional Stability +2, and *unenlightened* takes its largest on
negative Warmth where the rater calls it Intellect -2. Both of those are words
whose meaning the chart's five axes do not cleanly carry.

**Verdict T2b: passes, on both scorings.** On the pre-registered signs - which
penalise the Timidity axis for the sign error - 4 of 5 axes clear 0.34 in the
pre-registered direction and the mean signed r is **0.32125250386098153** against
a half-ceiling bar of **0.20939041084873417**
(`#...prereg_signed`). On the project's own sign convention, 5 of 5 clear and the
mean is 0.6098 against a ceiling of 0.6069.

## Test 3 - the text-embedding baseline

sweep100, this project's predecessor, ended on a text baseline: the weight-space
geometry turned out to be no better than embedding the training text
([[sweep100]]; RSA 0.911 between the two centred matrices, weights' mean Goldberg
congruence 0.750 against text's 0.731). The objection applies here unchanged -
the Big Five is a structure in the English trait lexicon, the teacher speaks
English and wrote every constitution and every pair - so it is tested the same
way.

### Can text predict where a held-out word lands on the chart?

Ridge regression from a trait's text embedding to its five 134-chart
coordinates, fitted on the **100 markers only**, tested on the **34**. Alpha
chosen by leave-one-out over a fixed grid. The held-out R^2 at every alpha is
recorded, not just at the chosen one
(`#test3_text_embedding_baseline.*.heldout_r2_overall_at_every_alpha`), so the
verdict does not hang on a selection rule.

| arm | what is embedded | model | alpha | held-out R^2 overall | best R^2 anywhere on the grid |
|---|---|---|---|---|---|
| constitution | each trait's constitution text | all-mpnet-base-v2 | 100.0 | **-0.034409160222032664** | 0.1340561571828799 |
| pair contrast | mean of 150 chosen replies minus mean of the 150 matched rejected replies | all-MiniLM-L6-v2 | 100.0 | **0.9380347105286267** | 0.9380347105286267 |
| adapter geometry | the 34 placed by their exact inner products with the 100 (test 2a) | - | - | **0.9822421695022922** | - |

Per axis for the contrast arm: Warmth 0.9651963000925337, Competence
0.9709130140578891, Timidity 0.8692222654543509, Arousal 0.8352422524634744,
Imagination 0.9028926284469638
(`#...ridge_pair_contrast_to_chart.heldout_r2_per_axis`).

A note on method, because it changed the answer. The exact leave-one-out
shortcut for ridge through the hat matrix is valid in theory and useless here:
with 768 features and 100 training rows a small alpha interpolates, the leverages
go to 1, and the formula evaluates 0/0. It silently selected the smallest alpha
on the grid and reported a constitution R^2 of -0.4157894501973991. Each fold is
refitted instead (`analyse_goldberg_only.py:ridge_loo_mse`), which is the same
estimator computed honestly.

### The Gram correlation

Pearson over the 8,911 off-diagonal entries, both matrices double-centred
(`#...text_gram_vs_adapter_gram`):

| text Gram | Pearson with the adapter Gram, double-centred | raw |
|---|---|---|
| constitution texts, mpnet | **0.09902878633621869** | 0.07320690086313401 |
| chosen replies only, MiniLM | 0.6539167543839601 | 0.3534839583575865 |
| rejected replies only, MiniLM | 0.700833327370052 | 0.5358905784300679 |
| chosen minus rejected, MiniLM | **0.8601935162272526** | 0.8671942322623715 |

sweep100's own figures, for comparison: chosen only 0.429, chosen minus rejected
0.826 (`#...sweep100_reference`).

**Verdict T3: the geometry is the text passed through.** The threshold was an
off-diagonal Pearson at or above 0.80 on the larger of the constitution and
contrast Grams, or a held-out R^2 at or above 0.50; the contrast arm clears both
(0.8602 and 0.9380). The "adds structure" verdict required both Grams at or
below 0.50 and R^2 at or below 0.20, and fails
(`#verdicts.T3_the_geometry_adds_structure`).

**But which text matters is the finding.** The constitution - the teacher's
120-to-200-word *description* of the trait, the thing a reader would assume the
geometry is a picture of - predicts essentially nothing: Gram correlation 0.099,
held-out R^2 below zero. The *contrast* between what the trait's character says
and what its opposite says predicts almost everything: 0.860 and 0.938. That is
sweep100's lesson replicated and sharpened. In sweep100's words, "the structure
is in the contrast, not the text", and the gap here (0.099 against 0.860) is
wider than the gap there (0.429 against 0.826), because a constitution is one
document per trait with no opposing pole in it at all.

The adapter geometry does beat the text, but not by much: 0.9822421695022922
against 0.9380347105286267 on the same held-out 34 and the same targets. Four
per cent of held-out variance is what weight space adds over an average of 300
sentence embeddings.

## What this establishes and what it does not

**The five factors are not an artefact of the word list.** Factoring the 100
Goldberg markers alone returns the same five factors as the 134-trait solution
at a mean one-to-one Tucker congruence of 0.9922779568528804, minimum
0.9841766610292786, all five above the "identical" bar. Dropping the 34 changes
the Big Five congruences by between -0.0496 and +0.0012 once the Tucker
denominator effect is taken out. Whatever the five factors are, the 34 words are
not making them and not hiding them.

**The frame generalises to words chosen without the Big Five in mind.** A chart
built from 100 marker adapters, which the 34 played no part in defining, places
those 34 within a mean per-axis Pearson of 0.9868075458499858 of where the full
chart places them, and an independent LLM rater's Big Five judgement of the 34
words correlates with their position in that chart at a mean r of
0.609752433265043 - 1.0046330040960998 times the correlation the 100 marker
words themselves achieve against Goldberg's own keying. On the strongest reading
available from this test, the frame locates a Big-Five-agnostic word as well as
it locates a marker word.

**But the frame is carried by the training text, not added by weight space.**
The average difference between a trait's chosen and rejected training replies,
passed through a 384-dimensional sentence encoder, predicts 0.9380347105286267
of the variance in where a held-out word sits on the chart, against
0.9822421695022922 for the adapter geometry itself, and its Gram agrees with the
adapter Gram at 0.8601935162272526 over 8,911 pairs. The honest statement is the
one sweep100 arrived at and this run reproduces on a different base model, a
different objective and a different trait set: **the trait LoRAs faithfully
encode the structure of the contrast in their training data, and that structure
is a loose approximation of the Big Five.** What this page adds is that the
structure survives being defined by the 100 marker words alone, extends to 34
words the Big Five never picked, and does not live in the trait's
*description*. The constitution predicts nothing (R^2 -0.034, Gram
correlation 0.099); the contrast predicts nearly everything.

**What none of this settles.** Test 3 compares two descriptions of the same 134
training corpora, so it cannot separate "the model represents trait words this
way" from "the teacher wrote the pairs this way" - both routes pass through the
same text. A test that could separate them would need trait adapters whose
training data was not written by a Big-Five-fluent English speaker, which this
project does not have. And nothing here moves the congruence ceiling: 0 of 5
Goldberg targets clear the conventional 0.85 bar in the 100-only solution, as in
every solution on [[factor-analysis]].

## Addendum, 2026-09-15: the 34 analysed in isolation

Samuel's follow-up: "when we analyse the 34 adapters in isolation do we get the
same thing?" The tests above held the 34 out and asked whether the
marker-defined frame could find them. This asks the reverse - do the 34 words,
on their own, *contain* the frame? Pre-registered in the dated addendum to
`qwen35/PREREG_goldberg_only.md`, computed by `qwen35/analyse_lexicon_only.py`
into `#lexicon_only`. No API calls; spend zero.

`analyse_fa_qwen35.py` could not be reused here: its `load_data` asserts 100 or
134 traits. Its mathematics was imported instead - `smc`, `paf`,
`kaiser_normalise`, `oblimin`, `tucker`, `parallel_analysis` - and called in the
same order with the same constants, and the script was not edited
(`#lexicon_only.method`). Ordering is by descending sum of squared oblimin
loadings; sign is left arbitrary and resolved only by the matching, because
`order_and_orient` needs Goldberg targets and the 34 score 0 in every one.

**The answer is: mostly, but measurably less well than 34 marker words, and the
shortfall has a specific and unsurprising cause.**

### L1 - the 34 factored alone

Five factors forced from 34 variables - 6.8 variables per factor against 20 at
p = 100. The solution is healthy on its own diagnostics: converged in 18
iterations, **0** Heywood cases, largest absolute off-diagonal Phi
**0.34963272433404335**, mean communality **0.2917458636484259**. Sum of squared
oblimin loadings **2.1574831574305255, 2.0758434258165743, 1.7207615359967643,
1.4522451571568573, 1.382356329572223**; leading centred eigenvalues
**5.296531808590302, 3.2586980232077405, 1.836311203545726, 1.6153133248021279,
1.3730049917990639** (`#lexicon_only.L1_factor_the_34_alone`).

Matched one-to-one against the 134 solution's `centred_k5` oblimin loadings
restricted to the same 34 rows, the matching **is** a permutation taking each of
the five 134 factors once:

| 34-only factor | matched 134 factor | Tucker congruence |
|---|---|---|
| F1 | Warmth | 0.7802 |
| F2 | Competence | -0.954 |
| F3 | Arousal | -0.566 |
| F4 | Imagination | -0.5492 |
| F5 | Timidity | 0.7884 |

(Sign is arbitrary in factor analysis, as [[factor-analysis-null-arms]] notes;
magnitudes are what matter. Full precision under
`#...one_to_one_matching.matched_congruences`.) Mean absolute matched congruence
**0.7275650651833419**, minimum **0.5492**, maximum **0.954**. One factor clears
0.85 and one clears 0.95.

So the five factors are all *there* - each one takes a different 134 factor as
its best match - but two of them, Arousal and Imagination, come back weakly.

Parallel analysis at p = 34 retains 2 at N = 150, 4 at 300, 5 at 1000, **6** at
the project's reference N = 1528, and 6 at 5809 and 20000
(`#...parallel_analysis_grid`).

### L2 - what can 34 variables do at all?

Thirty-four variables is small, so that 0.7276 means nothing without a reference.
Two hundred random 34-trait subsets of the **100 Goldberg markers** were factored
by the identical path and scored the identical way against the 134 solution
restricted to each subset's own rows
(`#lexicon_only.L2_random_34_reference_distribution`). The comparison is
symmetric: the lexicon 34 and every reference subset are alike among the 134 that
defined the target solution.

| quantile of the 200 subsets' mean absolute matched congruence | value |
|---|---|
| minimum | 0.7069 |
| 5th percentile | 0.7603278309692053 |
| 25th | 0.829 |
| median | 0.8701805028103853 |
| 75th | 0.9142 |
| 95th percentile | 0.9568796901667057 |
| maximum | 0.9708 |

Mean **0.8671052327619058**, standard deviation **0.0622742269357285**. All 200
subsets' matchings are permutations over the five
(`#...fraction_of_subsets_whose_matching_is_a_permutation` = 1.0), and none has a
Heywood case, so neither does the lexicon arm stand out on those.

**The lexicon 34 sit at the 3rd percentile of that distribution**
(`#...lexicon34_percentile_of_reference` = 3.0), below the 5th percentile of
0.7603278309692053. The median reference subset's *weakest* factor comes back at
0.7546420051449503 - better than the lexicon arm's weakest two.

**Verdict L: fails "same structure", passes "different structure", narrowly.**
The pre-registered bar for "same structure" was a percentile between 5 and 95,
and 3.0 is outside it. The pre-registered bar for "the 34 carry a different
structure" was a percentile below 5 **or** a matching that is not a permutation;
the first is met and the second is not. This is the honest reading: 34
Big-Five-agnostic words recover the five factors less well than all but about
three per cent of 34-word samples of Goldberg's own markers, while still
recovering all five of them in the right one-to-one correspondence. It is a
deficit of degree, not a different structure.

Retention is unremarkable: the lexicon 34 retain 6 factors at N = 1528, against
148 of the 200 marker subsets retaining 5, 41 retaining 6 and 11 retaining 4
(`#...parallel_analysis_retention_at_reference_N`).

### L3 - the reverse held-out test

The mirror of test 2. A chart built from the **34 alone** - their five oblimin
loading columns as merge coefficients over the 34 adapters, reordered and
sign-oriented by the L1 matching, Gram-Schmidt orthonormalised in the 34 x 34
Gram - and the **100 Goldberg markers** placed in it by their exact inner
products with the 34.

| axis | markers vs Goldberg keying | markers vs the full 134 chart | ceiling (134 chart vs keying) |
|---|---|---|---|
| Warmth | 0.603648426376595 | 0.9654397030398087 | 0.6828059271418205 |
| Competence | 0.5691678544955471 | 0.9863064857276602 | 0.5723042028598464 |
| Timidity | 0.5628222701119417 | 0.8380156989353288 | 0.47039913464559113 |
| Arousal | 0.5249972562211728 | 0.6676901947804039 | 0.5452041935257524 |
| Imagination | 0.6886668832390247 | 0.864824502903881 | 0.7639889196055134 |
| **mean** | **0.5898605380888562** | **0.8644553170774165** | **0.6069404755557047** |

All five signs positive. The mean against Goldberg's keying is
**0.9718589579131126** of the ceiling - the ceiling being what the *full* 134
chart achieves on the same 100 markers with the same keying.

**Verdict L3: passes.** The bar was a mean of at least half the ceiling
(0.30347023777785235) with all five signs positive, and agreement with the full
chart of at least 0.70; the result is 0.5899 and 0.8645.

**A chart defined by 34 words chosen with no reference to the Big Five recovers
the Big Five in Goldberg's own marker words at 97 per cent of what the full
134-word chart manages.**

### Reconciling L1 and L3

These look contradictory - a loading pattern at the 3rd percentile, a chart at 97
per cent of ceiling - and they are not. Tucker congruence scores each *rotated
factor* separately, and an individual oblimin factor from 34 variables is a noisy
object. The five-dimensional **subspace** those factors span is much better
determined than any one of them. Its principal cosines against the 134 chart's
subspace are **0.9549347890246754, 0.9157156306837178, 0.829076649267427,
0.7726038078246199, 0.48494109354298603**
(`#...principal_cosines_34chart_vs_134chart`).

That is the precise statement of what the 34 contain: **four of the five
dimensions, cleanly; the fifth only half.** Compare the 100-defined chart in test
2 above, whose five principal cosines against the same target ran 0.948 to 0.996.

### Why the fifth dimension is missing: the 34 do not cover the axes evenly

The 34 were drawn by k-means over sentence embeddings to spread across the trait
lexicon, not to tile the Big Five; Goldberg's 100 are twenty per factor by
construction, so any 34 of them still carry about seven per factor. Counting each
word by the chart axis it loads on most, in the shared 134 chart
(`#lexicon_only.L4_axis_coverage_of_the_two_word_sets`):

| axis | the 34 | the 100, rescaled to 34 | the 34's share of squared chart energy | the 100's share |
|---|---|---|---|---|
| Warmth | 16 | 10.88 | 0.41788119312407346 | 0.31037476437315664 |
| Competence | 6 | 9.86 | 0.2346695781101962 | 0.28608002783969777 |
| Timidity | 8 | 5.1 | 0.17178716250226972 | 0.15329706690083134 |
| Arousal | **1** | 3.06 | **0.06573300242209618** | 0.12590777756198895 |
| Imagination | **3** | 5.1 | **0.10992906384136436** | 0.12434036332432531 |

Exactly one of the 34 has Arousal as its dominant axis, and three have
Imagination. Those are the two factors that came back weakest in L1, at 0.566 and
0.5492. The 34 are not carrying a different structure; they are carrying the same
structure with one axis barely sampled.

### What the addendum establishes

- **The five factors are in the 34 as well**, all five, in the right one-to-one
  correspondence, from 34 variables that were never selected for the Big Five.
- **They are a poorer sample of it than 34 marker words**, at the 3rd percentile
  of a like-for-like reference distribution, which the pre-registration counts as
  a failure of "same structure".
- **The deficit is coverage, not disagreement.** One of the five chart
  dimensions is weakly represented in the 34, and the words that would represent
  it are the ones the lexicon draw happened not to pick.
- **It does not weaken test 2 above.** Test 2 asked whether the marker-defined
  frame locates Big-Five-agnostic words, and it does, at the marker words' own
  accuracy. This addendum asks whether 34 such words can *rebuild* the frame, and
  they can rebuild four fifths of it. The two results are about different
  directions of the same claim, and the second is the harder ask.

## Related

- [[factor-analysis]] - the five-factor solution these tests interrogate
- [[factor-analysis-null-arms]] - the two null arms this gives a same-size real comparator for
- [[factor-chart]] - the chart convention, rebuilt here from 100 traits
- [[lexicon-secondary-draw]] - how the 34 were drawn, and the partial check this extends
- [[goldberg-100-primary-traits]] - the 100 marker words
- [[sweep100]] - the predecessor whose text baseline this repeats
- [[factor-analysis-fisher-metric]] - the other "is this an artefact of a choice" test
- [[open-questions]] - the item this closes
