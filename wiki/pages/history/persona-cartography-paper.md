---
title: Persona Cartography (Baines et al., 2026)
summary: The ten Big-Five LoRAs, elementwise delta-W composition and flatten-norm-cosine-PCA decomposition that this project scales from ten adapters to 134 -- and the paper's own near-flat PCA spectrum, which is the null the zoo has to beat.
status: current
sources:
  - qwen35/paper_notes.md
  - https://arxiv.org/abs/2607.07916
  - https://www.lesswrong.com/posts/Rkvto5BLofzuDefyB/persona-cartography-charting-language-model-personality
  - vendor/persona-cartography/CITATION.cff
  - vendor/persona-cartography/paper/appendices/flattened_weight_space.tex
  - vendor/persona-cartography/scripts_dev/flatten_loras/data/cosine/persona.csv
  - vendor/persona-cartography/src/training/oct_adapter.py
last_verified: 2026-09-07
tags: [literature, persona-cartography, geometry, method]
---

# Persona Cartography

## Citation

Luke Baines, Anton Gonzalvez Hawthorne, Mariia Koroliuk, Irakli Shalibashvili,
Clement Dumas, Konstantinos Voudouris, David Demitri Africa. *Persona
Cartography: Charting Language Model Personality Traits in Weight Space.*
arXiv:2607.07916 [cs.AI], submitted Wednesday 8 July 2026, 21:00:44 UTC (v1,
the only version as of 2026-09-07). Cross-listed cs.LG. Comments field:
"85 pages, 80 figures". Licence **CC BY-NC-ND 4.0** -- no derivatives, which
matters if the blog post wants to reuse a figure.

Affiliations, from `qwen35/paper_notes.md` section 2: LASR Labs (equal
contribution), ENS Paris-Saclay and MATS, UK AI Security Institute.

**Author order differs between sources and both orders are in circulation.**
The arXiv listing and `vendor/persona-cartography/CITATION.cff` both put Baines
first. The LessWrong crosspost byline runs antonghawthorne, Mariia Koroliuk,
Irakli Shalibashvili, sidbaines, Clement Dumas, Konstantinos Voudouris, David
Africa -- Hawthorne first. `CONTEXT.md` section 1 and
`agent-harness/memory/library/persona-cartography-reading-list.md` both use the
LessWrong order. **Cite the arXiv order.**

The LessWrong crosspost, *Persona Cartography: Charting Language Model
Personality Traits in Weight Space*, 10 July 2026, 43 karma, is the crosspost,
not the arXiv v1 -- two days later. `CONTEXT.md` gives 10 July as the date; the
arXiv v1 date is 8 July.

## The code: previously unretrievable, now in hand

`qwen35/paper_notes.md` section 4 item 1 records: "Persona Cartography's code.
`github.com/persona-cartography/monorepo` returns 404. The HTML references it but
no working URL exists. So every PC number in these notes is from the paper text
only." **That is superseded.** The repository is at
`github.com/persona-cartography/persona-cartography` and a checkout sits at
`vendor/persona-cartography/`, git HEAD `6cfa6182e10acf625be937b0b19cfccecd864121`,
20 August 2026, commit message "anton/main-body-trim verbosity pass: tighten
prose across main body; camera-ready switch (final,main + checklist)".

Two things follow. First, "monorepo" is a **HuggingFace dataset repo**, not a
GitHub path: `src/training/.../04_train_lora.py` sets
`MONOREPO_REPO = "persona-cartography/monorepo"` and passes it to
`upload_folder_to_dataset_repo`. Searching GitHub for it was always going to
404. Second, the checkout contains `paper/` as LaTeX source (with
`neurips_2026.sty` and a `checklist.tex`, and a commit message saying
"camera-ready"), so the paper appears to have been prepared for NeurIPS 2026;
**the venue is not confirmed** and the arXiv comments field names none.

### What the code recovers that the paper did not state

`qwen35/paper_notes.md` section 4 item 2 lists six DPO hyperparameters the paper
gives for SFT but not for DPO, and infers they match Open Character Training.
The inference was right. From
`vendor/persona-cartography/src/training/oct_adapter.py` and
`scripts/training/ocean_paired_dpo/04_train_lora.py`:

| knob | value | where |
|---|---|---|
| DPO `train_batch_size` | 32 | `oct_adapter.py` (hardcoded in the batch-size chooser) |
| DPO `max_len` | 1024 | `04_train_lora.py` argparse default |
| `adam_betas` | 0.9 0.98 | `oct_adapter.py` DPO and SFT command |
| `kl_loss_coef` | 0.001 | `oct_adapter.py` DPO command |
| `nll_loss_coef` | 0.1 | `oct_adapter.py` DPO command |
| `lr_warmup_ratio` | 0.1 | both commands |
| `max_norm` | 1.0 | both commands |
| `seed` | 123456 | `04_train_lora.py` |
| rsLoRA | not used -- no flag exists in the OpenRLHF path | repo |
| LoRA dropout | not set in the paired-DPO path | repo |
| precision / ZeRO | bf16, ZeRO-2 (ZeRO-3 for Qwen3-32B only) | `oct_config.py` |
| SFT `max_len` | 3072 | `oct_adapter.py` default |

**One contradiction.** The paper (App A.1.2, via paper_notes section 2.4) states
SFT **batch size 16**. The code passes `train_batch_size=32` for SFT as well as
DPO, from the same chooser. Both are recorded; the code is what ran.

## The unit and the constitution

The unit is a **(trait, polarity) pair**: each of the five OCEAN traits crossed
with amplify and suppress, giving ten adapters, plus one neutral **control**
adapter, per base model. Six base models, with Llama-3.1-8B-Instruct carrying all
headline results and Qwen3-8B/32B and Gemma-3-4B/12B/27B-IT getting only TRAIT
and MMLU.

Each single-trait constitution is a JSON object of **twelve items** = six
NEO-PI-R facets x two framings: a positive framing instructing the model to
express the facet in the target polarity, and a contrastive framing instructing
it to resist the opposite pole on that facet. Each item's `trait` field is the
DPO system prompt; it states identity in the first person, names the focal facet,
gives example exchanges -- and then **anchors the other four OCEAN traits**,
reproduced in full with an instruction to "do not amplify OR suppress any of
these". That anchoring block exists to make the delta trait-specific, which is
exactly the assumption pairwise weight-space geometry rests on; paper_notes
section 3.4 flags that the zoo has no analogue and no OCEAN basis to anchor
against. Framing rule worth keeping: "where possible, traits are framed as
natural, 'how I am' rather than as changes from some assumed baseline."

## Training: paired-teacher DPO

The methodological change from [[open-character-training-paper]]. For each
prompt the teacher generates **two** responses, one conditioned on the amplifier
constitution and one on the suppressor constitution; the amplifier response is
chosen and the suppressor response rejected when training an amplifying LoRA, and
the reverse for a suppressing one. Both sides come from the teacher. Pairs with
an empty completion on either side are filtered out. Teacher GLM-4.5-Air, with
DeepSeek-V3.2 as a robustness check. Prompt pool about 2,430 per adapter: 600
facet prompts plus LIMA's roughly 1,830.

**Their ablation is the reason this project uses their scheme and not OCT's.**
App L.4(a), via paper_notes section 2.9: an E-amplifier adapter built the
original OCT way reaches extraversion +0.97 at coefficient 1.00, "about 30% of
the canonical +3.18 at coefficient 0.75". Paired-teacher DPO with an explicit
opposite-pole rejected response is roughly 3x stronger at identical
hyperparameters. `qwen35/paper_notes.md` section 3.6 records this as the
pipeline choice the ablation should drive, and the zoo follows it. One thing the
zoo cannot copy: a Persona Cartography pair is a polarity of one trait, whereas
Goldberg adjectives have no canonical opposite pole, so whatever the zoo does
instead changes what the delta means -- paper_notes calls this "the single
biggest semantic difference between our deltas and PC's".

Stage 2 is OCT's introspection SFT, unchanged in shape: 10,000 self-reflection
examples plus 1,000 free and 1,000 leading ten-turn self-interactions, merged and
shuffled to 12,000. One full OCEAN pipeline is under 5 hours on an A100 for
Llama-3.1-8B, under 12 hours for Gemma-3-27B.

## The four weight-space operations

This is the section the project builds from. The paper is careful about which
level each operation acts at, and `qwen35/paper_notes.md` section 2.6 quotes each
precisely.

**(a) Scaling acts on the composed delta.** "We vary the scalar coefficients
which multiply all elements of all weight perturbation (delta-W) matrices of the
LoRA." A scale c multiplies delta-W, not the factors. Sweep grids are typically
-2.0 to +2.0 in steps of 0.5; c = 0 is the untouched baseline.

**(b) Composition across traits is an elementwise sum of deltas, with no cross
terms.** "We compose LoRAs by summing all of the delta-W from each (optionally
scaled) LoRA, elementwise." App A.1.3 is explicit that this "keeps the adapters
as independent summands and introduces no such cross terms". Composing k rank-64
adapters therefore gives a delta of rank up to 64k, which they do not
re-compress.

**(c) Souping the two stages acts on the LoRA factors, with square-root
weights.** Quoted in full because it is easy to get wrong. Writing each adapter's
delta as delta-W_i = B_i A_i, the merge does **not** form
w_DPO delta-W_DPO + w_SFT delta-W_SFT -- that object would have rank up to 128.
Instead it combines the factors directly, A_merged = sum_i sqrt(w_i) A_i and
B_merged = sum_i sqrt(w_i) B_i, and reconstructs delta-W = B_merged A_merged,
keeping rank 64. Because (A, B) -> BA is bilinear, the result equals the intended
weighted sum **plus cross terms** sqrt(w_DPO w_SFT)(B_DPO A_SFT + B_SFT A_DPO)
that mix the two adapters' subspaces; the square-root weighting makes each
adapter's own contribution carry its nominal weight.

This is exactly what PEFT's `add_weighted_adapter(combination_type="linear")`
does, which is what [[open-character-training-paper|OCT]]'s `merge_loras.py`
calls. **Every released OCT persona adapter and every Persona Cartography OCEAN
adapter therefore contains DPO/SFT cross terms that belong to neither stage's
delta.** Decomposing a released soup and expecting a clean sum of two stage
deltas decomposes something else. This is the argument for the zoo's DPO-only
main sweep being a *cleaner* object than either paper's artefact, and the reason
[[stage-two-geometry]] has to decompose the stage deltas separately as well as
the soup.

**(d) Decomposition: flatten, norm, cosine, PCA (Appendix D).** "For each
delta-W matrix in our LoRA adapters, flatten and concatenate the entire model
together" -- one roughly 8-billion-dimensional dense vector per adapter,
materialised, with no factor-space trick.

*Norms.* Frobenius norm of each flattened Llama-3.1-8B delta, from their Table 1:
O-up 6.078, O-down 6.322, C-up 6.332, C-down 6.383, E-up 6.451, E-down 6.383,
A-up 6.463, A-down 6.185, N-up 6.529, N-down 6.336. "The norms lie in a tight
band (6.08-6.53), so all ten OCEAN LoRAs are of essentially the same size in
weight space." They never normalise; they exploit the band. App E.13 leans on it
directly: "we treat the sum of the LoRA scales as a proxy for the total
intervention magnitude... so we can sum scales directly rather than weighting each
by its adapter's norm".

*Cosine similarity.* `qwen35/paper_notes.md` section 4 item 3 records that the
cosine matrix exists only as Figure 15 and "the numeric matrix is not in the
text, only the figure, so I could not extract values". **Retrieved 2026-09-07**
from the repo: `scripts_dev/flatten_loras/data/cosine/persona.csv` is the source
data for that figure, named in a comment in the appendix LaTeX. Every
amplifier/suppressor pair is negative. Quoting the upper-triangle cell verbatim
(the file is stored as float32 and is not exactly symmetric -- the
`openness-,openness+` cell reads `-0.11726979` against `-0.117269784` for
`openness+,openness-`):

| pair | cosine |
|---|---|
| openness+ / openness- | -0.117269784 |
| conscientiousness+ / conscientiousness- | -0.09493698 |
| extraversion+ / extraversion- | -0.05608431 |
| agreeableness+ / agreeableness- | -0.070309274 |
| neuroticism+ / neuroticism- | -0.10064885 |

All 40 remaining distinct cross-trait pairs are positive, the smallest being
`extraversion+ / openness-` at 0.020000914 and the largest
`openness+ / agreeableness+` at 0.1420921. The base row and column are exactly
zero, because the baseline enters as the null vector. The paper's own
text says only "an OCEAN amplifier has a negative cosine similarity with its
suppressor. The LoRAs that have a high cosine similarity may or may not have
behavioural similarities, this needs further work."

*PCA.* Performed on the ten flattened vectors plus the baseline as the null
vector, "reducing the dimensionality from 8 billion down to 10 (with these 11
datapoints, we can get at most 10 PCA dimensions)". Variance explained, their
Table 2: PC1 15.26%, PC2 13.38%, PC3 12.21%, PC4 12.17%, PC5 11.60%, PC6 8.84%,
PC7 8.64%, PC8 8.28%, PC9 7.93%, PC10 1.70%.

Their one interesting component is the last: "the tenth principal component
cleanly separates the baseline (the model with no LoRAs) from the others." They
shift the baseline along PC10 and have Claude Opus 4.7 describe the outputs at
scales -10 to +10: total token-loop collapse at -10, competent baseline at -2 to
-1, a sweet spot at 0 to +1, "oddly introspective and self-referential" at +5,
and at +10 "total collapse into rambling prose obsessed with personality traits
and self-reflection". Their reading: "the steering vector targets something like
self-reference or introspection". paper_notes calls this "the closest thing in
the literature to what we are trying to do, and it is one paragraph of
qualitative description on 11 points."

Also in the appendices: rank-1 SVD compression of each delta-W leaves "the
trait-modifying behaviour still largely present" (this project's nearest test is
[[rank-sweep]], which *retrains* at low rank rather than compressing a trained
delta, and finds the zoo's arrangement set by rank 1 while the DPO fit is not); behavioural (not weight-space)
additivity measured as an interaction residual over 45 adapter pairs, with
conscientiousness the systematic outlier; and base-to-instruct transfer, where
adapters trained on the instruct model still impart the trait at weight
interpolations toward the base.

## What they evaluate

Summarised from `qwen35/paper_notes.md` section 2.7, which has the full detail.

- **TRAIT MCQ**, logprob-scored with a forced `ANSWER: ` prefill, summing
  logprobs across surface forms of each letter, with a refusal flag when choice
  mass falls below 0.75.
- **Capability**: MMLU, GSM8K, TruthfulQA, reported as a four-way breakdown per
  scale point -- Correct / Recovered / Wrong answer / No answer -- which separates
  formatting collapse from capability loss. paper_notes section 3.10 recommends
  copying this for the zoo's steering doses.
- **LLM-judge panel**, built mechanically from the same canonical OCEAN
  definition object that drives the constitutions, so the trait trained, the
  trait scored and the trait prompted all refer to the same construct. OCEAN on
  an integer -4..+4 scale with 0 = no signal, coherence separately on 0..10.
  Calibrated on 33-36 hand-crafted gold items per trait against three human
  annotators; inclusion bar intra-rater Krippendorff alpha >= 0.70, Spearman
  >= 0.80 versus gold and >= 0.70 versus human consensus. The decisive criterion
  was **scale calibration, not rank correlation** -- Kimi-K2 scored Spearman 0.94
  on coherence and was rejected for compressed scale use. Default: a single
  judge, Qwen3-235B-A22B at temperature 0.
- **Downstream safety**: sycophancy, CoCoNot, WildJailbreak, and a multi-turn
  frustration eval reproducing Soligo et al. 2026.
- **Induction comparison**: LoRA versus system prompt versus activation capping
  versus user-roleplay. Headline LoRA coefficient is **0.75**, not 1.00, chosen
  by reading transcripts rather than by the coherence judge -- at 1.00 "the model
  writes in an exaggerated tone... that the coherence judge does not flag but
  reads as forced".
- **Unsupervised factor pipeline**: 2,500 fifteen-turn rollouts scored by a
  72-item binary forced-choice questionnaire, then **principal axis factoring,
  k = 4, oblimin rotation**, giving Initiative 11.7%, Tone 10.2%, Didacticism
  9.6%, Epistemic Caution 9.3%, cumulative 40.7% ("TIDE"). k = 4 was chosen on
  three convergent grounds even though "real eigenvalues remain above the
  95th-percentile permutation null out to k = 11, while a clean scree elbow sits
  at k = 4". Split-half stability: median Tucker congruence above 0.97 over 100
  random half-splits. Scenario explains 56-78% of factor-score variance;
  interviewer archetype at most 6%.

## Stated limitations

- The full evaluation stack ran only on Llama-3.1-8B-Instruct; the other five
  models got TRAIT and MMLU only.
- "TRAIT was designed for humans, and it is not obvious that a human-calibrated
  questionnaire measures a well-defined construct when applied to an LLM whose
  persona is not stable across contexts."
- The unsupervised section uses synthetic rollouts.
- "Validation of trait-induction was performed using the questionnaire which
  found the factors, rather than independent judges" -- the circularity is
  acknowledged.

Flagged elsewhere in the paper and load-bearing here:

- **Non-orthogonality.** "The adapters' flattened weight vectors are themselves
  non-orthogonal." The LessWrong post puts it as "The OCEAN adapters don't shift
  *only* their corresponding trait."
- **Cosine does not imply behaviour.** High cosine similarity "may or may not"
  correspond to behavioural similarity.
- **Negative scaling is unreliable.** Scaling an amplifier by a negative weight
  "sometimes" suppresses the trait and sometimes does nothing.
- **Suppressor floor.** E-down stalls around -1 for every weight- and
  activation-space method while a system prompt cleanly reaches -2.29.
- **The control adapter is not inert.** A neutral-constitution control nearly
  doubles sycophantic capitulation (0.61 against a 0.33 baseline) and raises
  WildJailbreak harmful compliance, "indicating that constitution-guided
  distillation alone shifts safety-relevant behaviour."
- **Combined-scale capability loss is not predictable** beyond a rule of thumb
  that the sum of LoRA scales should stay below about 2.

## What this project takes, and where it departs

**Takes:** the single-trait unit; the paired-teacher DPO scheme and its 3x
ablation as the justification; the [1.00, 0.25] two-stage soup weights; the
elementwise delta-W composition used for steering; and the
flatten-norm-cosine-PCA decomposition as the thing to scale.

**Departs, and paper_notes section 3.8 argues each is strictly stronger:**

| | Persona Cartography | this project |
|---|---|---|
| points | 10 adapters + baseline null vector = 11 | 100 primary (+40 secondary), 134 in the zoo |
| max components | 10 | about 99 |
| method | PCA on flattened delta-W | PCA **and factor analysis** on deltas |
| representation | dense flattened concatenation, about 8e9 dims | Gram computed from the LoRA factors, never materialised |
| per-module | not reported | reported per module type as standard |

Three consequences.

1. **Their PCA spectrum is the null this project has to beat.** 15.26 / 13.38 /
   12.21 / 12.17 / 11.60 / ... / 1.70 is what about 11 mutually near-orthogonal
   points look like -- structure would be a *falling* spectrum, and a near-uniform
   one over 100 points is the negative result. paper_notes section 3.8 says this
   should be pre-registered as such. See [[geometry-overview]] for what the zoo's
   spectrum actually did.
2. **Factor analysis over weight deltas is new.** Persona Cartography does factor
   analysis, but on *questionnaire responses*, never on weights. paper_notes
   section 4 item 8 records that no prior art doing FA on weight deltas was
   found, while noting that absence in two papers is not a literature search.
   See [[literature-factor-analysis-methods]].
3. **The factor-Gram trick is exact.** For delta-W = s BA, the Frobenius inner
   product is s_i s_j tr(A_i^T B_i^T B_j A_j), computable per module without ever
   materialising d_in x d_out. Persona Cartography flattens 8-billion-dimensional
   vectors instead. The per-adapter scaling must be applied before the inner
   product. One caveat carries over from (c) above: if a delta is a PEFT-souped
   adapter, its (A, B) are the *merged* factors and BA is not the sum of the
   stage deltas -- the Gram is still exact for that object, but the object is not
   what one might assume.

**Norm comparability.** paper_notes section 3.2 warned that with rsLoRA the
zoo's norms would land about 8x above their 6.08-6.53 band, making any comparison
to Table 1 meaningless without dividing out the convention. The executed zoo ran
**plain LoRA, rsLoRA off** (`qwen35/RUNNER_TASK.md`; `qwen35/PHASE3_VERDICT.md`
addendum 2026-09-03), so scaling is alpha/r = 2.0 as in both papers and the
warning does not apply. See [[zoo-training-recipe]].

**A null they have and the zoo did not.** Their single null is a
**neutral-constitution control** -- same teacher, same prompts, same DPO,
chosen/rejected split by random seed only. That isolates everything the
distillation pipeline does that is not the trait, and their finding is that it is
not inert. paper_notes section 3.9 notes that none of the zoo's three nulls
(random A/B at init scale, shuffled preference pairs, mislabelled traits) covers
this, and recommends adding it.

**The result that sharpens the comparison most.** Persona Cartography's ten
adapters are one seed each; the zoo trained 40 traits twice and measured what
survives. `qwen35/PHASE3_VERDICT.md` reports across-seed self-cosine at median
+0.0167 against a preregistered bar of +0.2447, and withdraws the trait-level
claim: "There is no evidence here that an individual trait has a characteristic
direction in weight space." Their Appendix D cosine matrix, extracted above, was
computed on a single initialisation and carries no cross-seed check. Every
cross-adapter cosine in either paper is a within-initialisation quantity. See
[[seed-floor]].

## Verification note

The paper was read in full (HTML v1, Appendices A-M) by the author of
`qwen35/paper_notes.md` on 2026-08-19; the paper numbers on this page are quoted
from that file, and the Appendix D norms and PCA spectrum were re-checked
2026-09-07 against the LaTeX source at
`vendor/persona-cartography/paper/appendices/flattened_weight_space.tex`, where
they match to the digit. The arXiv abstract page was fetched 2026-09-07 for
title, author order, date, version history, subject class, comments and licence;
the LessWrong crosspost was fetched the same day for the byline order, date and
karma. The cosine values are read from the repo's own figure-data CSV, not from
the paper. Purely graphical results other than the cosine matrix (PCA scatter
coordinates, residual heatmaps, per-model TRAIT sweeps, judge calibration
heatmaps) were **not** re-extracted, though `scripts_dev/flatten_loras/data/`
also carries `pca/coords.csv` and `pca/spectrum.csv` if a later session wants
them.
