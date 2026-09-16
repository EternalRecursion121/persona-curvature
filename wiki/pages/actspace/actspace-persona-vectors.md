---
title: Constitution-as-prompt persona vectors vs weight geometry
summary: The 134 constitutions used as system prompts produce activation shifts whose 134x134 geometry correlates with the adapters' weight geometry at r 0.705 at layer 16 (max 0.775 at layer 19), but the weight-space hole does not transfer.
status: current
sources:
  - qwen35/ACTSPACE_RESULTS.md
  - qwen35/analyse_actspace.py
  - qwen35/analysis/actspace_geometry.json#windows.resp.primary
  - qwen35/analysis/actspace_geometry.json#windows.prompt.primary
  - qwen35/results/decomposition_actspace_resp.json
  - qwen35/phase10_runs/decompose_actspace_resp.log
  - qwen35/results/decomposition.json#test1b
  - qwen35/analysis/alien.json#alien_k5
  - qwen35/blog_page/index.html
  - qwen35/analyse_actspace_fa.py
  - qwen35/analysis/actspace_geometry_fa.json#windows.resp
  - qwen35/analysis/actspace_geometry_fa.json#windows.prompt
  - qwen35/fa_chart.py
  - qwen35/analysis/persona_sliders.json#activation
last_verified: 2026-09-10
tags: [actspace, activations, persona-vectors, geometry, nulls]
---

# Constitution-as-prompt persona vectors vs weight geometry

**P_t** is the persona vector for trait *t*: the base model's mean
residual-stream activation with the trait's constitution as system prompt, minus
the same mean with no system prompt, over the same 64 questions. **W_t** is the
trait's stage-1 LoRA weight delta. The two live in spaces that share no
coordinates, so the only comparable objects are 134x134 similarity matrices and
134xk score matrices over the same traits (`analyse_actspace.py` lines 9-11).
Design and windows: [[actspace-overview]].

Everything below is the **response-token window** at **layer 16**, the layer
fixed in advance as primary, with the activation vectors **trait-centred** before
the cosines are taken (`analyse_actspace.py` line 139).

## Gram correlation, by layer

| layer | Pearson r (centred) | source key |
|---|---|---|
| 0 | 0.524 | `analysis/actspace_geometry.json#windows.resp.curve[0].r_centred` |
| 13 | 0.711 | `#windows.resp.curve[13].r_centred` |
| **16 (primary)** | **0.705** | `#windows.resp.primary.r_centred` |
| 19 (maximum) | 0.775 | `#windows.resp.curve[19].r_centred` |
| 32 | 0.757 | `#windows.resp.curve[32].r_centred` |

`ACTSPACE_RESULTS.md` states the curve as "0.52 at layer 0, 0.70 by 13, max 0.775
at layer 19, ~0.75 to the end". The blog page renders the primary value as
"r = 0.70" and the maximum as "0.77 at layer 19, reported as a maximum". The
maximum is a maximum over 33 layers and is *not* the headline; layer 16 was named
before anything was computed.

At the primary layer:

- Spearman rho **+0.690** (`#windows.resp.primary.rho_centred`).
- **Label-shuffle p = 0.0005**, from 2000 permutations of one matrix's trait
  labels, counted as `(hits + 1) / (n + 1)`, so 0.0005 is the floor of the test
  (`analyse_actspace.py` lines 27, 47-55; `#windows.resp.primary.perm_p`).
- **Nearest-neighbour agreement 45 / 134.** A trait's nearest neighbour in
  activation space is its nearest neighbour in weight space for 45 traits;
  chance is 1 (`#windows.resp.primary.nn`; `analyse_actspace.py` line 145 prints
  the chance value as `T/(T-1)` = 1.0).

### Two different values for "prompts track the weights"

The same quantity appears twice in the project with two values, and both are
published:

- **+0.705** -- `analyse_actspace.py`, which builds the weight cosine matrix
  `Cw` from the raw Gram without centring (lines 92-94) and correlates it with
  the *centred* activation cosines (line 127).
- **+0.737** -- `analyse_actspace_adapters.py`, whose `r_PW` double-centres the
  weight Gram first (`Gc = J @ Gw @ J`, lines 62-63) before the same correlation
  (line 84). This is the number the blog page quotes as "the prompts track the
  weights (0.74)".

The difference is weight-side centring only; the activation side is centred in
both. Both are current, they answer slightly different questions, and neither
supersedes the other. See [[actspace-adapters]] for the value in context.

## Principal components and Procrustes

The variance profile is much flatter in activations than in weights:

| | PC1 | PC2 | PC3 | PC4 | PC5 |
|---|---|---|---|---|---|
| activations | 23% | 16% | 15% | 8% | 5% |
| weights | 43% | 32% | 7% | 5% | 2% |

(`ACTSPACE_RESULTS.md`; blog page; keys
`#windows.resp.primary.var_act[0..4]` and `.var_w[0..4]`.)

Where the traits land on the leading components still corresponds. A Procrustes
fit -- an orthogonal rotation plus a scale -- of the two 134x5 score matrices
gives **R^2 = 0.535**, against a label-shuffle null of **mean 0.02, 95th
percentile 0.035** (`ACTSPACE_RESULTS.md`;
`#windows.resp.primary.procrustes_r2`, `.procrustes_null95` = 0.035; the null
*mean* of 0.02 is printed by `analyse_actspace.py` line 151 but is not stored in
the JSON). The blog renders this as "explains 54% of the variance against a
shuffle null of 2%".

## Factor structure on the activation Gram

`analyse_actspace.py` writes the primary-layer activation Gram to
`results/gram_actspace_resp_L16.npz` in `decompose.py`'s format precisely so that
the pre-registered factor tests run on it with the same machinery as the
weight-space result (lines 178-182). They were run
(`phase10_runs/decompose_actspace_resp.log`,
`results/decomposition_actspace_resp.json`):

| test (leading component removed) | activations | weights |
|---|---|---|
| TEST 1B, polarity-signed factor separation | **+0.1422** | **+0.12160** |
| TEST 2, bipolarity gap (same-keyed minus opposite-keyed, within factor) | **+0.33708** | **+0.23934** |
| TEST 2B, label-free clustering ARI | +0.0613 (p 0.00200) | -- |
| TEST 3, unsupervised ARI | +0.10415 (p 0.00050) | -- |

Sources: `results/decomposition_actspace_resp.json#test1b.leading-component
removed.diff`, `#test2.leading-component removed.gap`, `#test2b.leading-component
removed.ARI`, `#test3.leading-component removed.ARI`; weight-space counterparts
`results/decomposition.json#test1b.leading-component removed.diff` and
`#test2.leading-component removed.gap`. `ACTSPACE_RESULTS.md` rounds these to
"TEST 1B residual +0.142 (weights +0.122), bipolarity gap +0.337 (weights
+0.239), ARI 0.06-0.10, all p at the floor"; the blog to "+0.14 residual (weights
+0.12) ... +0.34 (weights +0.24)".

The raw trait-centred cosines behind the bipolarity, at layer 16:
same factor and same keying **+0.291**, same factor and opposite keying
**-0.175**, different factor **-0.022**
(`#windows.resp.primary.cos_same_fk`, `.cos_same_fo`, `.cos_diff`). The 34
unlabelled `Lexicon` traits are excluded from these, as in every labelled test.

**The project does not claim this as a finding.** `ACTSPACE_RESULTS.md`: "Big
Five as signed axes recovered -- unsurprising for text about the traits; the
geometry correspondence above is the result." The blog says the same: "That last
pair should not impress anyone; the constitutions are text *about* the traits."

One number that is *not* in the blog and is worth keeping: the activation Gram is
even less five-dimensional than the weight Gram. Participation ratio **9.70**,
top-5 eigenvalue share **63.9%**
(`results/decomposition_actspace_resp.json#spectrum.participation_ratio`,
`#spectrum.top5_share`), against **25.72** and **37.5%** in weights
(`results/decomposition.json#spectrum.participation_ratio`, `#spectrum.top5_share`).
The verdict line for the activation Gram is the same as for the weights:
"RECOVERED AS BIPOLAR AXES ... BUT THE SPACE IS NOT FIVE-DIMENSIONAL".

## The hole does not transfer

The weight-space "hole" is the direction in the adapters' top-5 PC subspace that
is furthest from every adapter line -- a character the trait lexicon has no word
for ([[hole-words]]). It is a zero-sum weighted combination of the 134 adapters,
with the weights stored as `analysis/alien.json#alien_k5.coeffs`.

Apply **exactly those coefficients** to the 134 trait-centred persona vectors and
the result lands **47.8 degrees** from its nearest trait (`quiet`). The null --
the same coefficients permuted across traits, 500 draws -- has median **47.7
degrees** (`ACTSPACE_RESULTS.md`; `#windows.resp.primary.hole_deg` 47.834,
`.hole_null_median` 47.748, `.hole_nearest` `quiet`).

There is no hole there. `ACTSPACE_RESULTS.md` puts it in capitals: "THE HOLE DOES
NOT TRANSFER ... The weight-space hole is a property of the adapter cloud, not of
the trait set." The blog rounds both figures to 48 and draws the conclusion
explicitly: the earlier phrasing, "a character the lexicon lacks a word for",
"should be read with that in mind: it is a direction the *adapters* leave open".

One caveat on the comparison. `analyse_actspace.py` line 177 prints the
weight-space figure as **68.9** degrees, which per `analyse_hole.py` line 5 is
the hole measured in the **full sketch space**; the coefficients actually
transplanted are `alien_k5`'s, whose weight-space angle in the **k=5 PC
subspace** is **52.5** degrees (`analysis/alien.json#alien_k5.gap_deg`). The
transplant result stands on its own null and does not depend on which weight-space
figure is quoted beside it, but the two are not the same construction.

## The user-turn window

The control window -- the constitution is in context, but the model has not
generated anything yet -- is much weaker and rises with depth:

| layer | Pearson r (centred) |
|---|---|
| 16 (primary) | 0.385 |
| 29 | 0.561 |
| 32 | 0.554 |

(`#windows.prompt.primary.r_centred`, `#windows.prompt.curve[29].r_centred`,
`#windows.prompt.curve[32].r_centred`; `ACTSPACE_RESULTS.md`: "User-turn window
(system prompt in context, no response): r 0.385 at L16, rising to 0.55 late.")

Also at layer 16 of the prompt window: nearest-neighbour agreement 42/134
(`#windows.prompt.primary.nn`), Procrustes R^2 0.2225 against the same style of
null at 0.036 (`.procrustes_r2`, `.procrustes_null95`), and the transplanted hole
at 54.7 degrees against a null median of 57.1 (`.hole_deg`, `.hole_null_median`)
-- i.e. no hole there either.

Layer 0 of the prompt window is `nan` throughout the curve
(`#windows.prompt.curve[0]`). Layer 0 is the embedding output, which depends on
token identity alone; the user-turn tokens are the same tokens with and without a
system prompt in context (the system block is excluded from the window), so the
prompt-window shift there is exactly zero and its cosines undefined.

## Noise floor

Splitting the 64 prompts into halves and taking each trait's cosine between the
two halves gives the activation-space analogue of the weight-space seed floor.
At layer 16, response window, the median is **0.959**
(`#windows.resp.curve[16].floor`), against a median cross-trait value of 0.622
(`.diff_trait`). `ACTSPACE_RESULTS.md` reports it as "0.96. Very stable." The
same floor in the prompt window is 0.994 (`#windows.prompt.curve[16].floor`) --
the user tokens are identical across halves apart from the questions themselves.

## The same question in the factor chart

`qwen35/analyse_actspace_fa.py` (2026-09-08) asks the Procrustes question in the
project's primary frame instead of in principal components. A factor direction is
a weighted merge of the 134 adapters, `v_f = sum_i c_fi a_i`, with coefficients
from `phase10_runs/steer_spec2_7a.json`. Applying the *same* coefficients to the
134 activation persona vectors gives each factor's activation-space image,
`u_f = sum_i c_fi x_i`; Gram-Schmidt on those five images in the activation Gram
`results/gram_actspace_resp_L16.npz`, in the fixed order Warmth, Competence,
Timidity, Arousal, Imagination, gives an activation chart with the same
construction as `qwen35/fa_chart.py`'s weight chart. Nothing was regenerated: the
script reads two saved Gram matrices.

| statistic, layer 16, response window | factor chart | PC chart |
| --- | --- | --- |
| Procrustes R^2, 134x5 coordinates | **0.7387431438763885** | 0.5352210111769643 |
| label-shuffle null, mean over 500 draws | 0.022523512115349885 | not stored |
| label-shuffle null, 95th percentile | 0.03568178186070537 | 0.03525949291018074 |

Sources: `qwen35/analysis/actspace_geometry_fa.json#windows.resp.procrustes_r2_fa`,
`#windows.resp.procrustes_null_mean`, `#windows.resp.procrustes_null_95pct`;
`qwen35/analysis/actspace_geometry.json#windows.resp.primary.procrustes_r2` and
`#windows.resp.primary.procrustes_null95`. `analyse_actspace.py` prints its null
mean but stores only the 95th percentile, which is why the middle row has one
entry. The two Procrustes numbers are computed by byte-identical functions on the
same 134 traits, so they are comparable; they differ only in which 134x5 matrices
are fitted.

The sharper number is the one Procrustes cannot give, because Procrustes is free
to rotate Warmth into Competence. Correlating **matching columns with no
rotation**, weight chart against activation chart
(`#windows.resp.per_factor_r`, null in `#windows.resp.per_factor_null_95pct`):

| factor | Pearson r | shuffle mean abs r |
| --- | --- | --- |
| Warmth | +0.9158629151634156 | 0.066 (rounded) |
| Competence | +0.8219103655925268 | 0.071 (rounded) |
| Timidity | +0.8200218627800827 | 0.073 (rounded) |
| Arousal | +0.8128999490415441 | 0.067 (rounded) |
| Imagination | +0.8254568029074986 | 0.069 (rounded) |

Weight-space Warmth is activation-space Warmth, factor by factor, not merely up
to a rotation.

Two things to record rather than gloss.

The weight chart is built on the raw Gram and the activation chart on the
trait-centred one, which looks like an asymmetry. It is not: rebuilding the
weight chart on the double-centred weight Gram reproduces the column-centred raw
chart to machine precision (max absolute difference 1.1e-15), and gives
Procrustes R^2 0.7387431438763886 against 0.7387431438763885 - the same number
(`#windows.resp.procrustes_r2_fa_centred_weights`).

The five factor directions are *not* equally oblique in the two spaces. The two
files store these cosines rounded to four places. In weight space the two
strongly oblique pairs are Warmth against Timidity at **-0.6661** and
Competence against Arousal at **-0.728**
(`#windows.resp.weight_factor_cosines`); in activations the same two are
**-0.7965** and **-0.6409** (`#windows.resp.activation_image_cosines`), and
Competence against Timidity moves from -0.0271 to **+0.3934**. The
chart is legitimate in both spaces but the oblique structure is not identical,
which is one reason the Procrustes fit is 0.7387431438763885 rather than higher.

The chart also sees more of an activation persona vector than of an adapter: a
mean fraction 0.7544359083812503 of a persona vector's norm lies in the
five-dimensional span against 0.5784887830866848 of an adapter's (`#windows.resp.activation_chart_frac_of_norm_mean`,
`#windows.resp.weight_chart_frac_of_norm_mean`). On the user-turn (prompt) window
the same comparison is Procrustes R^2 0.5902819670409757 for the factor chart
against 0.2225023263472986 for the PC chart, and the chart share falls to
0.5724618798229619 of a persona vector's norm (`#windows.prompt.procrustes_r2_fa`,
`qwen35/analysis/actspace_geometry.json#windows.prompt.primary.procrustes_r2`,
`#windows.prompt.activation_chart_frac_of_norm_mean`).

## What this does and does not say

It says the arrangement of the 134 traits is not an artefact of LoRA, of the
optimiser, or of the shared initialisation: the same 134 documents, used only as
prompts, reproduce most of it. It does **not** say the two spaces are the same
space, that the correspondence would survive a different base model, or that the
top-5 subspaces coincide beyond what Procrustes R^2 0.535 states. See
[[actspace-method-notes]].

## Trained back into the model

[[persona-sliders]] (2026-09-10) takes these persona vectors as *targets*: thirteen
rank-64 LoRAs were trained with SliderSpace's cosine objective to produce a
specified layer-16 shift. On held-out prompts they reach cosine `+0.7664` to
`+0.9398` with their own target, where the trait's own stage-one adapter reaches
`+0.1232` to `+0.6326` measured the same way
(`qwen35/analysis/persona_sliders.json#activation`). The hole's coefficients,
trained rather than transplanted, still do not separate from their shuffled
controls.
