---
title: Paper reading, 2026-09-09 - SliderSpace and Gradient Atoms
summary: Two papers read in full - SliderSpace (Gandikota et al., ICCV 2025), which discovers rank-one LoRA sliders from PCA of CLIP features and so decomposes a generative model's capability space without labels, and Gradient Atoms (Rosser, arXiv 2026), which decomposes per-document training gradients in an EKFAC-preconditioned eigenspace and steers with the resulting atoms - with a point-by-point relation to this project's weight-space geometry and eleven ranked experiments they suggest.
status: current
sources:
  - https://arxiv.org/abs/2502.01639
  - https://arxiv.org/pdf/2502.01639
  - https://openaccess.thecvf.com/content/ICCV2025/html/Gandikota_SliderSpace_Decomposing_the_Visual_Capabilities_of_Diffusion_Models_ICCV_2025_paper.html
  - https://arxiv.org/abs/2603.14665
  - https://arxiv.org/pdf/2603.14665
  - qwen35/POST_DRAFT.md
  - qwen35/analysis/fisher_norms.json
  - qwen35/analysis/column_space.json
  - qwen35/analysis/stage2_register_vs_residual.json
  - qwen35/analysis/stage2_neutral_control.json
  - qwen35/analysis/actspace_stage2_geometry.json
  - qwen35/analysis/nxn_summary.json
  - qwen35/analysis/additivity.json
  - qwen35/analyse_alien_fa.py
  - qwen35/results/fa_qwen35.json
  - qwen35/phase2_runs/archive/phase5_sweep_134.json
  - qwen35/zoo40_meter.sh
  - qwen35/align_score.py
  - qwen35/fisher.py
last_verified: 2026-09-09
tags: [literature]
---

# Paper reading, 2026-09-09

Samuel asked for two papers to be read in full and related to this project.
Both were read in full: the PDFs were downloaded with `curl` and converted with
`pdftotext -layout`, and the main text, tables and appendices were read from
that conversion. What could **not** be read is anything that exists only inside
a figure's plotted bars or curves; each paper's section below names those.

This page is a reading note, not a result. Every project number it quotes
carries the wiki page it comes from and, where that page gives one, the file and
key. Every paper number carries the table or section it is in. Nothing here was
measured.

---

# 1. SliderSpace

## Citation and access

Rohit Gandikota, Zongze Wu, Richard Zhang, David Bau, Eli Shechtman, Nick
Kolkin. *SliderSpace: Decomposing the Visual Capabilities of Diffusion Models.*
arXiv:2502.01639v1 [cs.CV], submitted 3 February 2025. Affiliations:
Northeastern University (Gandikota, Bau) and Adobe Research (Wu, Zhang,
Shechtman, Kolkin). Project site `sliderspace.baulab.info`, code
`github.com/rohitgandikota/sliderspace`.

Published at **ICCV 2025**; the camera-ready is in the ICCV 2025 Open Access
repository. That venue was confirmed by web search on 2026-09-09 and is
`[verified]` in the sense [[literature-big-five]] uses. **The version read here
is the arXiv v1 PDF**, accessed 2026-09-09, not the ICCV camera-ready; if the
two differ, this page describes v1.

**Read in full**: abstract, Sections 1-7, and Supplementary A-E (text). Not
readable from the text layer: the numeric bar heights in Figure 4 (DreamSim
diversity and CLIP-Score per concept), the curves in Figure B.2 (FID against
number of PCA directions, and FID against LoRA rank) and the spectra in
Figure A.1. Those are stated qualitatively below and marked as figure-only.

## What it does, step by step

**The object studied.** A text-to-image diffusion model's *capability space* for
one prompt: the set of images it can produce for, say, "picture of a monster",
and the axes along which those images vary. The unit of the answer is a **LoRA
adapter** - each discovered direction is trained as a low-rank adapter on the
model's cross-attention layers, so it can be scaled, composed with others and
added to the model. That is the same object this project studies, in a different
modality.

**The data.** No dataset. The model generates its own: given a prompt `c`, about
`m = 5000` samples are drawn from the model by varying the random seed
(Section 4.2, "Distribution Sampling"). For the art experiment the prompt is
"artwork in the style of a famous artist"; for the diversity experiment the
prompts are 8,000 sampled from COCO-30k and expanded by an LLM (Section 5.3).

**The procedure, in three steps** (Section 4.2, Figure 2):

1. **Sample.** Generate `m` images. At each denoising timestep `t` the paper
   takes the *final image extrapolation* `x~_0,t` - Equation 2, the image the
   model is currently "planning", obtained by applying the current noise
   prediction as if it were the direction for all remaining steps. This lets
   them read a finished-looking image out of an unfinished trajectory without
   running the remaining forward passes.
2. **Decompose.** Embed each `x~_0,t` with a semantic encoder (CLIP by default)
   and take the top `n` principal components `V = PCA({phi(x~_0,t)})`
   (Equation 4). The principal components are mutually orthogonal by
   construction; that is where "semantic orthogonality" comes from.
3. **Train one slider per component.** For each `v_i`, train a LoRA `T_i`
   minimising `L = sum_i 1 - cos(dphi_i, v_i)` (Equation 5), where
   `dphi_i = phi(x~^i_0,t) - phi(x~_0,t)` is the change the slider makes in CLIP
   space. So the adapter is trained *to produce a specified change in an
   embedding space*, not to fit data.

**The evaluation.** Three applications, each with its own metric.

- *Concept decomposition* (5.1): inter-image diversity by DreamSim distance over
  2,500 generated samples per concept, generating each image with a random
  sparse subset of 3 sliders out of 32; text alignment by CLIP-Score. Directions
  are named post hoc by showing Claude 3.5 Sonnet image pairs and asking what
  transformation each slider applies.
- *Art style exploration* (5.2): FID against a reference distribution built from
  4,388 artist names manually catalogued by the ParrotZone community, four
  images per style.
- *Diversity enhancement* (5.3): FID-30k and CLIP on all COCO-30k prompts,
  before and after adding a generic 64-slider SliderSpace to a distilled model.
- Two Amazon MTurk user studies comparing 9-image grids pairwise.

**The main quantitative results.**

| result | number | where |
|---|---|---|
| Art-style FID against the real-artist reference: generic prompts | 76.08 | Figure 5 |
| the same, LLM-generated style prompts | 38.84 | Figure 5 |
| the same, 64 supervised Concept Sliders | 32.86 | Figure 5 |
| the same, SliderSpace | **19.12** | Figure 5 |
| FID-30k, undistilled SDXL | 11.72 (CLIP 29.41) | Table 3 |
| FID-30k, distilled SDXL-DMD | 15.52 (CLIP 28.92) | Table 3 |
| FID-30k, DMD + SliderSpace | **12.12** (CLIP 29.13) | Table 3 |
| Real images, CLIP | 30.14 | Table 3 |
| Concept user study win rate vs SDXL-DMD | 72.4 diverse / 66.0 useful / 68.1 creative | Table 1 |
| vs LLM-prompted SDXL-DMD | 62.5 / 62.5 / 62.5 | Table 1 |
| vs SDXL | 65.3 / 61.2 / 59.2 | Table 1 |
| Art user study vs real artist prompts | 54 diverse / 63 useful | Table 2 |
| vs LLM prompts | 73 / 67 | Table 2 |
| vs generic prompts | 87 / 88 | Table 2 |
| Ablation, no contrastive/orthogonality objective | FID 31.1 | Figure E.1 |
| Ablation, PCA in diffusion output space instead of CLIP | FID 26.3 | Figure E.1 |
| Baseline configuration | FID 24.6 | Figure E.1 |
| + LLM prompt expansion | FID 23.3 | Figure E.1 |
| + SDXL-generated training data | FID 19.12 | Figure E.1 |

Compute: fewer than 24 GB of VRAM, and 64 directions discovered in **under two
hours on one A100** (Section 5); four times faster than training the same number
of Concept Sliders (Section 6).

Hyperparameters that matter for this project: the default configuration is
**rank-one adapters with 40 PCA directions** (Appendix B.2). The paper reports
that FID improves with more PCA directions up to about 40 and then flattens,
that 10 SliderSpace directions already match 64 hand-made Concept Sliders, and
that at a fixed training budget **rank-one sliders outperform higher-rank ones**.
Those three claims are stated in the Appendix B.2 text; the curves behind them
are figure-only (Figure B.2), so no per-rank FID number is quoted here.

**Stated limitations** (Section 6). The method inherits the biases of whichever
semantic encoder it uses, and may miss culturally specific or nuanced concepts.
Discovery takes about two hours on an A100, which limits iteration. The
discovered art directions are not one-to-one with the real artists in the
reference set. And Appendix D.3 adds that the Claude 3.5 captions "are not
always accurate", giving a worked example where a slider labelled "Black Lab
Technician" is not visually distinguishable from "scientist".

## How it relates to this project

**Same object: a direction is a LoRA, and directions compose.** SliderSpace's
whole apparatus - each direction trained as a low-rank adapter, adapters scaled
and added, sparse subsets combined at generation time - is this project's
apparatus. This project builds a direction as a weighted merge of trained
adapters and then adds it to the base model
(`qwen35/steer_fix.py`, [[fisher-norms]]); SliderSpace trains the adapter to be
the direction. The difference is where the supervision enters, and it is the
most interesting difference between the two frames.

**Opposite direction of travel: they discover then name, this project names then
trains.** The zoo starts from Goldberg's 100 unipolar markers plus 34 lexicon
adjectives and trains one adapter per word ([[post-draft]]); the structure is
then *recovered* from the 134 x 134 Gram. SliderSpace never has a word until the
end, when Claude 3.5 Sonnet is shown pairs and asked to caption. The project has
already run their step and had it fail: the [[hole-words-factor-chart|hole
direction]] was found geometrically at **54.76378041380744 degrees** from the
nearest trait adapter against a null mean of **40.32531831850254** (z
**+13.717953930316737**, `qwen35/analyse_alien_fa.py`), three candidate English
names were proposed and trained, and they land 66 to 87 degrees away, farther
than existing adapters ([[post-draft]]). SliderSpace's own limitation note - that
its directions are not one-to-one with real artists and that the captions are
unreliable - is the same finding in their domain, and this project should cite it
as such rather than treating the hole failure as idiosyncratic.

**Orthogonality is enforced in the semantic space, not the weight space, and
this project has the evidence that this is the right choice.** SliderSpace's
PCA runs on CLIP embeddings, and its ablation shows that doing the same spectral
analysis in the diffusion model's own output space yields sliders that control
"color, texture and shapes" rather than semantics (Figure E.1, FID 26.3 against
24.6). This project's parallel result is [[stage-two-exploration]]: the same
trait's stage-one and stage-two deltas sit at same-trait cosine **+0.0002** in
weight coordinates and at **+0.530** in activation space, with the two stages'
mean shifts at **+0.552** (`qwen35/analysis/actspace_stage2_geometry.json`).
Orthogonality in weight coordinates is a fact about the LoRA parameterisation.
Two independent literatures now say the same thing: the space you decompose in
decides what you find.

**Rank.** SliderSpace's default is rank one, and it claims rank one wins at fixed
budget. The zoo is rank 64, `lora_alpha` 128, effective scale 2.0
([[stage-one-training-config]], `qwen35/phase2_runs/archive/phase5_sweep_134.json`).
[[column-space-structure]] says the zoo's deltas are **not** rank-1 objects in
disguise: the top singular direction holds **0.10318456418060519** of the sum of
singular values, the top 8 hold **0.3614515265945672** and the top 16 hold
**0.5305926812823089** (`qwen35/analysis/column_space.json#stage1.spectrum`).
But the same page also finds that the *shared* part across seeds is concentrated
at the top of the spectrum - weighted overlap 0.4630531697561965 against
unweighted 0.14174031494371594 at k = 64 - so the trait-carrying part may be
much lower rank than the delta. Whether rank 64 buys anything for the geometry
has never been tested here; SliderSpace is the reason to test it.

**Sparse composition.** SliderSpace generates its headline diversity by
activating 3 random sliders out of 32 and reports that the result is more
diverse (DreamSim) at unchanged text alignment (CLIP-Score). This project's
matching result is a disagreement: across ten matched-norm mixtures of the five
keying axes the deviation from the additive prediction is **0.53 of the
predicted effect** and about **2.5 times the judge-noise floor**
(`qwen35/analysis/additivity.json`, [[additivity]]). The two are not in direct
conflict - SliderSpace never claims additivity, only diversity and alignment -
but the project has been reading its own result as a negative and SliderSpace
suggests the positive version of the same measurement was never made here.
[[open-questions]] already lists "never run: additivity for PC and factor
mixtures".

**Where their method would have changed a result here.** Two places, concretely.
(1) **The hole.** The hole's transfer to activations failed - the same
coefficients applied to prompted activations gave a gap of 47.8 degrees against a
shuffled-coefficient control's 47.7 ([[post-draft]], [[hole-words-factor-chart]]).
A SliderSpace-style direction is *defined* in the semantic space where that test
is run, so training an adapter to the hole direction expressed in activation
space would make that particular failure impossible by construction - which is
exactly why it is the interesting experiment: it separates "there is no
disposition there" from "the weight-space chart put the disposition in the wrong
place". (2) **Coverage.** The project's claim that the 134 words tile the space no
better than chance (z +0.2 on the factor chart, down from -2.3 on principal
components, [[post-draft]]) is a claim about a basis chosen from a dictionary.
SliderSpace's art experiment is the test of exactly that question in their
domain, and their answer is that the unsupervised decomposition covers the
reference distribution *better* than the curated name list (FID 19.12 against
32.86 for supervised Concept Sliders). Porting it here would put a number on
whether trait vocabulary is a good basis for this model.

**What does not transfer.** Final-image extrapolation (Equation 2) is specific to
iterative denoising and has no language-model analogue. DreamSim, FID and
CLIP-Score are image metrics; the project's readout is an LLM judge on five
scales over 24 scenarios. The MTurk user studies have no counterpart here and
should not be imitated.

## Experiments this paper suggests, ranked

### S1. Train persona sliders in activation space and compare them to the weight-space merges

**RUN, 2026-09-09/10. Result: [[persona-sliders]].** Thirteen sliders rather than
six (ten traits, two per Big Five factor, plus the hole and two shuffled
controls), one dose rather than five alphas, `$19.19` against the `$37-40`
estimated below. The sliders reach their activation targets on held-out
prompts far better than the trait's own adapter does, sit at cosine
`0.002506` to `0.022155` with that adapter in weight space, move the judged
persona about half as far as stage one with almost none of its degeneration,
and the hole slider is not distinguishable from its shuffled controls.

**Question.** If a direction is defined by the change it makes in the model's
residual stream rather than by a merge of trained adapters, does it steer better,
and does the hole become a coherent persona?

**Design.** Take the five factor directions and the hole direction as targets
expressed in **activation** space: the prompted-persona vectors already computed
per trait at layer 16 (`qwen35/analysis/actspace_means_adapters*.npz`,
`qwen35/act_space.py`, [[actspace-persona-vectors]]), projected onto the factor
loadings. Train one rank-64 LoRA per target with SliderSpace's objective
transposed: minimise `1 - cos(dA_t, v)` where `dA_t` is the adapter-induced mean
residual shift over the 64 activation-probe questions, plus a penalty on the
cosine with the other four targets (their orthogonality term). Six adapters:
five factors and the hole. Judge blind on the 24-prompt battery at five alphas.
**Controls**: (a) the existing weight-space merge for the same factor at matched
Fisher dose from `qwen35/analysis/fisher_norms.json#fisher_dose_table`; (b) a
shuffled-coefficient direction, the control the hole already has.

**What a result would mean.** If the activation-trained factor adapters move
their own judged dial at least as far as the merges at matched dose, the
weight-space chart is a lossy route to a direction that can be specified better
elsewhere, and every steering number in [[steering-results]] is a lower bound. If
the hole adapter produces a specific, coherent persona where the merge gave 3 of
5 correct signs and r 0.74 - the same as a random direction in the subspace
([[post-draft]]) - then the hole is real and the weight-space chart misplaced it.
If it degenerates the way the merge did, the hole is dead and this is the test
that kills it.

**Cost.** The objective needs a forward pass with and without the adapter on the
64 probe questions per optimizer step, so call it 20x a stage-one adapter's
`train_seconds` **465.377254486084**
([[stage-one-training-config]]) = about 2.6 GPU-hours each, 6 adapters = **16
GPU-hours = $34** at the meter's **$2.10 per GPU-hour** (`qwen35/zoo40_meter.sh`,
[[costs]]). Generations: 6 x 5 alphas x 24 prompts = 720, about 1 GPU-hour = $2.
Judge: 720 calls at $0.0012 to $0.003 per call (the sweep100 steering run's
$3.365 over 2,830 calls and phase 7's $44.33 over 15,944 cache lines,
[[costs]]) = $1 to $2. **Total about $37-40**, estimate, scaled from the
stage-one sweep and the sweep100 judge rate.

**Reuses.** `qwen35/act_space.py`, `qwen35/analyse_actspace.py`,
`qwen35/train_qwen35.py` (new loss), `qwen35/steer_fix.py`,
`qwen35/judge_personas.py`, `qwen35/analysis/fisher_norms.json`.

### S2. Unsupervised discovery of persona directions from the base model's own variation

**Question.** If the model's own spread of responses is decomposed without any
trait vocabulary, do the five factors come back, and is there anything in the top
directions that no adjective in the zoo names?

**Design.** SliderSpace's pipeline, ported. Sample the base model at temperature
on the 445-prompt pool (`qwen35/prompts.json`, intersected as [[shared-prompt-pool-445]] describes) -
say 5,000 responses, matching their `m ~ 5000`. Embed each response with the
same sentence encoder used for the lexicon draw
(`qwen35/traits_secondary_provenance.json`, [[lexicon-secondary-draw]]). PCA to
40 components (their default). Train one rank-1 LoRA per component with the
cosine objective on the embedding difference. Then place all 40 in the zoo's
frame: score each against the 134 adapters through the exact Gram machinery
(`qwen35/cross_gram_full_on_modal.py`) and against the five factor loadings.
**Control**: 40 directions built from PCA on *shuffled* embeddings, which should
produce no alignment with the factor subspace.

**What a result would mean.** If the top discovered directions land inside the
span of the five factors, the trait vocabulary is a sufficient basis for this
model's response variation, and that is a much stronger statement than the
current congruences of **0.41 to 0.68** ([[factor-analysis]], [[post-draft]]). If
several high-variance discovered directions land far from every trait adapter,
the hole is not one hole - it is a systematic gap - and the coverage question
gets a bottom-up answer instead of the current top-down one.

**Cost.** Generation of 5,000 responses at about 200 tokens: about 2 GPU-hours
(the s2register run produced 1,464 generations at 512 tokens inside a run of
this size, [[stage-two-exploration]]). Forty rank-1 trainings with an
embedding-space objective: SliderSpace does 64 in under 2 A100-hours for
diffusion; a 4B language model with a sentence encoder in the loop is the same
order, call it 4-6 GPU-hours. Cross-Gram against 134: CPU, about $2 by the
[[column-space-structure]] run's 0.83 container-hours. **Total 7-9 GPU-hours =
$15-19 plus $2 CPU**, estimate, scaled from SliderSpace Section 5 and the zoo's
own generation runs. Judging 40 directions x 8 prompts = 320 calls, under $1.

**Reuses.** `qwen35/act_space.py` (generation harness),
`qwen35/cross_gram_full_on_modal.py`, `qwen35/analyse_fa_qwen35.py`,
`qwen35/fa_chart.py`, `qwen35/results/gram_sweep.npz`.

### S3. Rank sweep: retrain 40 traits at rank 1, 4 and 8

**Question.** SliderSpace's default is rank one and it claims rank one wins at a
fixed budget. Does the zoo's factor structure survive at rank 1?

**Design.** Retrain the 40 traits that already have a second seed
([[seed-floor]], [[cross-seed-geometry]]) at `lora_r` 1, 4 and 8, everything else
identical: same 445 pairs, 13 optimizer steps, `lora_alpha` scaled to keep the
effective scale at 2.0. Measure, per rank: the 40 x 40 Gram against the rank-64
Gram (Pearson over off-diagonals, the statistic [[cross-seed-geometry]] uses,
where the two seeds agree at 0.997); Tucker congruence of the k = 5 solution;
cross-seed cosine, whose analytic prediction is `r/d` and so falls from
64/2560 = 0.025 to 1/2560 = 0.00039 at rank 1; column-space overlap by
`qwen35/column_space_on_modal.py`, which is the statistic that should be
*unaffected* by rank because it never compares coordinates; Fisher norm by
`qwen35/fisher.py`; and the judged own-factor dial.

**What a result would mean.** If the Gram correlation and the congruences hold at
rank 1, then the entire geometry of this project is an arrangement of essentially
rank-one objects, the column-space result becomes the whole story rather than a
mechanism note, and every future zoo can be trained 64 times smaller. If they
degrade with rank, the spectrum matters and SliderSpace's rank-one finding is
specific to their objective, which trains a direction directly rather than
letting one accumulate.

**Cost.** 3 ranks x 40 traits x 465.377254486084 s = 15.5 GPU-hours = **$33**,
plus the project's own overhead model of $10.17 per app and $0.333 per run
([[costs]]) for 3 apps and 120 runs = $70 - call the whole thing **$100**,
estimate, anchored on the recorded `train_seconds` of the 134-run sweep.
Column-space and Gram analysis is CPU, about $3 by the
[[column-space-structure]] run. Judging 120 adapters x 8 prompts = 960 calls,
$1-3.

**Reuses.** `qwen35/train_qwen35.py`, `qwen35/build_nxn_inputs.py`,
`qwen35/column_space_on_modal.py`, `qwen35/analyse_column_space.py`,
`qwen35/fisher.py`, `qwen35/analyse_fa_qwen35.py`.

### S4. Sparse composition: the diversity reading of the additivity failure

**Question.** Mixtures do not add ([[additivity]]: 0.53 of the predicted effect).
Do they nevertheless produce a wider, still-coherent spread of personas, which is
the property SliderSpace actually claims?

**Design.** Sample 30 sparse mixtures - 3 of the 5 factor directions, random
signs, matched **Fisher** dose rather than matched alpha, using
`qwen35/analysis/fisher_norms.json#fisher_dose_table` - and judge each on the 8
prompts the sphere sweep used ([[sphere-sweep-factor-chart]]). Measure both: the
residual from the additive prediction built from the single-direction runs
(the [[additivity]] statistic) **and** the spread of the judged five-scale
profiles, using the same distance-from-centroid measure that
`qwen35/analysis/fisher_norms.json#correlation_with_degeneration.sphere_judged_profile_distance`
already defines. **Controls**: the five single directions at the same Fisher
dose, and 10 random merges from the `#random_band` set.

**What a result would mean.** If the mixtures spread the judged profiles further
than the singles without raising the loop rate, the additivity failure is a
failure of a linear *prediction*, not of the mixtures, and the post's "weight
arithmetic does not give independent behavioural dials" should be paired with
"but mixtures do expand the reachable space". If the spread is no larger, the
negative stands and SliderSpace's diversity claim does not port.

**Cost.** 30 mixtures x 8 prompts = 240 generations. The sphere sweep did 72 x 8
in 3.890 GPU-hours ([[sphere-sweep-factor-chart]], log 2026-09-08), so this is
about 1.3 GPU-hours = **$2.80**, plus 240 judge calls = under $1. **Under $5**,
estimate, scaled directly from the sphere sweep. It also closes an
[[open-questions]] "never run" item.

**Reuses.** `qwen35/build_sphere_spec_fa.py`, `qwen35/sphere_sweep.py`,
`qwen35/analyse_additivity.py`, `qwen35/judge_personas.py`,
`qwen35/analysis/fisher_norms.json`.

### S5. How good is post-hoc LLM labelling of a direction, measured

**Question.** SliderSpace names every discovered slider with Claude 3.5 Sonnet
and admits the names are sometimes wrong. This project has 134 directions whose
true names are known. What is the accuracy of that labelling procedure?

**Design.** For each of the 134 stage-one adapters, generate at plus and minus
alpha on 8 prompts, show a judge model the two sets blind and ask for a single
adjective and a confidence. Score: exact match to the trait word; cosine between
the proposed adjective's embedding and the trait word's in the lexicon embedding
space; whether the proposed word shares the trait's Big Five factor and keying.
**Control**: run the identical protocol on 20 random merges from
`fisher_norms.json#random_band`, which have no true name - a labeller that
returns a confident adjective for those is measuring its own fluency.

**What a result would mean.** A number to attach to every discovered-direction
name in the literature, this project's hole words included. If accuracy on known
traits is high and confidence on random merges is low, then LLM labelling is
trustworthy and the hole's naming failure is evidence about the hole. If
confidence is as high on random merges, the whole naming step is decorative, and
that is worth saying out loud in a post that leans on adjectives.

**Cost.** 134 x 2 signs x 8 prompts = 2,144 generations, about 2.5 GPU-hours =
**$5.30**, plus 154 labelling calls and 2,144 generations' worth of judge context
- call it $3-8 in judge spend at the rates in [[costs]]. **Total $10-15**,
estimate, scaled from the sphere sweep's generation rate.

**Reuses.** `qwen35/steer_fix.py`, `qwen35/judge_personas.py`,
`qwen35/traits_primary.json`, `qwen35/traits_secondary_provenance.json`.

---

# 2. Gradient Atoms

## Citation and access

J Rosser (FLAIR, University of Oxford). *Gradient Atoms: Unsupervised Discovery,
Attribution and Steering of Model Behaviors via Sparse Decomposition of Training
Gradients.* arXiv:2603.14665v2 [cs.AI], submitted 15 March 2026, revised
**17 March 2026** (v2 is the version read, accessed 2026-09-09). Single author.
Code `github.com/jrosseruk/gradient_atoms`. Acknowledgements name the EPSRC CDT
in Autonomous and Intelligent Machines and Systems, the London Initiative for
Safe AI and Arcadia Impact.

**No venue.** The PDF is in the arXiv preprint style with no conference or
journal marking, and nothing in it claims acceptance anywhere. Recorded as
arXiv-only; if a venue exists it was not found in the paper.

**Read in full**: abstract, Sections 1-6, Appendices A (extended related work),
B (computational details, Table 3) and C (Table 4, the top 50 atoms). Not
readable from the text layer: the per-alpha bar heights in Figure 2 - only the
best-case summaries in Table 2 are in the text - and the atom scatter positions
in Figure 1. Table 1's "~2500" and "~100" documents per atom are the paper's own
approximations, printed with the tilde.

## What it does, step by step

**The object studied.** Not a model's behaviour and not a document's influence,
but the **shared update direction that a cluster of functionally similar training
documents jointly induces**. The paper's argument for this unit is that a model
learns arithmetic not from one arithmetic example but from the collective
gradient signal of many, citing Ruis et al. (2024) on procedural knowledge. It
positions itself against training data attribution (Koh & Liang 2017; Grosse et
al. 2023; Bae et al. 2024), which is *supervised* - it needs a query behaviour
specified in advance and costs O(Q x N) query-document comparisons for Q
behaviours.

**The data and the model.** Gemma-3 4B IT, fine-tuned with LoRA at **rank 8** on
`q_proj` and `v_proj` across all 34 layers: **2.2M trainable parameters across
136 LoRA modules**. Training data: **5,000 instruction-response pairs** from a
general-purpose SFT mixture covering arithmetic, grammar correction,
classification, code generation, QA and creative writing (Section 4.1).

**The procedure, five steps** (Section 3):

1. **Per-document gradients.** For each of the 5,000 documents, the gradient of
   the cross-entropy loss with respect to all trainable parameters: `G` is
   5,000 x 2.2M.
2. **EKFAC projection and preconditioning.** Raw gradient space is anisotropic;
   without correction a decomposition is dominated by high-curvature directions
   and "drowns out semantic structure" (Section 3.2, and this sentence is the
   single most relevant sentence in either paper to this project). Using the
   EKFAC eigendecomposition of the approximate Fisher information matrix per
   module, each gradient is projected onto the module's top-k eigenvectors and
   divided by `sqrt(lambda + eps)` (Equation 1). In the experiment,
   k = 50 eigencomponents x 136 modules = **6,800 dimensions**, a 328x
   reduction. The stated purpose: make the space approximately isotropic, so a
   unit step in any direction changes the loss by roughly the same amount.
3. **Sparse dictionary learning.** Each projected gradient is normalised to unit
   norm (so atoms encode direction, not magnitude) and decomposed as
   `g^_i ~ sum_j alpha_ij d_j` (Equation 2) with scikit-learn's
   `MiniBatchDictionaryLearning`, **K = 500 atoms**, sparsity penalty
   **alpha = 0.1**.
4. **Coherence scoring.** For each atom, the mean pairwise cosine of the **raw,
   unprojected** 2.2M-dimensional gradients of its top-20 activating documents
   (Equation 3). High coherence means the atom found a shared motif in the
   original weight space rather than an artefact of the projection.
5. **Unprojection to a steering vector.** Reversing the projection turns any atom
   into a full parameter-space vector `v_j`, applied as
   `theta_new = theta +/- alpha * v_j` (Equation 4).

**The evaluation.** Two stages. Atom discovery is evaluated by coherence and by
manual inspection of each atom's top-20 activating documents (Table 4 names the
top 50). Steering is evaluated on five selected atoms with **regex detectors**
over 100 evaluation questions each (about 60 that invite the target behaviour and
about 40 neutral controls), sweeping alpha over {0.5, 1.0, 2.0, 5.0, 10.0} in
**both signs**, because dictionary learning assigns atom signs arbitrarily. All
11 variants are served simultaneously as LoRA modules through vLLM alongside the
clean baseline.

**The main quantitative results.**

Discovery (Section 4.2): of 500 atoms, **5 have coherence > 0.5, 43 have
coherence > 0.1, and 457 fall below 0.1**. The top five are short factual Q&A
(0.725), grammar and sentence editing (0.672), yes/no/true/false classification
(0.647), simple arithmetic (0.643) and multi-category classification (0.614).
Atoms cluster by **how** the model responds, not what about: task types, not
topics. Grammar correction appears three times (ranks 2, 17, 36) and code
generation five times (16, 37, 40, 45, 46) at decreasing coherence, which the
paper reads as sub-clusters. Bulleted lists (#469) and numbered lists (#299) are
separate atoms.

Sparsity (Table 1): at alpha 0.01 atoms are too dense, a median of about 2,500
documents each, and only 3 clear coherence 0.5; at alpha 0.1, about 100 documents
each, 5 above 0.5 and 43 above 0.1; at alpha 1.0 the penalty overwhelms
reconstruction and every coefficient is zero.

Steering (Table 2):

| atom | behaviour | coherence | base | best up | best down |
|---|---|---|---|---|---|
| #415 | Yes/No classification | 0.647 | 39% | 51% (+12pp) | 0% (-39pp) |
| #64 | Code generation | 0.201 | 42% | 58% (+16pp) | 28% (-14pp) |
| #161 | Systematic refusal | 0.111 | 50% | 55% (+5pp) | 0% (-50pp) |
| #469 | Bulleted lists | 0.103 | 33% | **94% (+61pp)** | 0% (-33pp) |
| #299 | Numbered lists | 0.103 | 58% | 59% (+1pp) | 8% (-50pp) |

Two discussion claims (Section 5). **Suppression appears easier than
amplification**: all five atoms suppress their target to near zero, only two
achieve substantial amplification, and the paper's interpretation is that
suppressing needs one pathway disrupted while amplifying needs one pathway
strengthened against competitors. **Coherence does not predict steerability**:
#469 at coherence 0.103 gives the largest effect (+61pp) and #415 at 0.647 gives
+12pp.

Compute (Table 3): gradient extraction on 8x A100-40GB in **170 s**; EKFAC
projection ~5 min on CPU; dictionary learning ~15 min on CPU; coherence ~5 min;
**total about 25 minutes**.

**Stated limitations** (Section 5). Only 43 of 500 atoms clear coherence 0.1 -
"the majority are noise or capture overly broad mixtures". Instruction-following
training data means atoms recover task types rather than fine-grained semantic
preferences, and more naturalistic data might yield different atoms. The
6,800-dimensional EKFAC projection discards information. 5,000 documents may not
cover rare behaviours. And the regex evaluation measures surface formatting
rather than deeper behavioural change.

## How it relates to this project

**Same identity, opposite end.** This project's scoring identity
([[n-by-n-scoring]]) is that the inner product of a training example's induced
update with a target direction is a directional derivative of the log-likelihood,
so no per-example gradient need ever be formed: giving every (target, example)
pair its own scalar `eps` leaves the whole matrix in `eps.grad` after one
backward pass. Gradient Atoms takes the other branch: it *does* form the
per-document gradient, and then reduces it 328x by projection so that dictionary
learning is tractable. The two are complementary. This project can score any
example against any *known* direction essentially for free; Gradient Atoms can
find directions nobody asked for, at the cost of materialising gradients. The
project's own limitation is exactly the one the paper opens with - "unable to
surface behaviours the user did not think to ask about".

**Where the project's numbers agree.** The scoring identity works here in the
supervised regime: 40 pairs from each trait's own data rank that trait's adapter
first **134 of 134** times among 134 (133 of 134 after standardising each
adapter's column), with runners-up sharing the trait's factor and keying 42% of
the time against a 12% base rate ([[n-by-n-scoring]],
`qwen35/analysis/nxn_summary.json`; the analytic-versus-finite-difference
agreement of 0.9999992 is on that page with the caveat that it is a builder
fallback, not a stored key). Gradient Atoms' unsupervised claim is that the
same gradient similarity that makes that work also groups documents without
labels. The project has never tested the unsupervised half.

**Where they agree on behaviour.** The paper's "suppression appears easier than
amplification" is this project's steering table read from the other side. At
alpha +/- 2 the judged change on each factor's own scale is Warmth **-2.1 /
+1.5**, Competence **-3.5 / +0.7**, Fearful withdrawal **-1.9 / +0.2**, Arousal
**-0.6 / +1.5**, Imagination **-2.1 / +1.1** on the 1-to-7 scale
([[post-draft]], [[steering-results]]) - four of five suppress harder than they
amplify. The project explained this with a judged ceiling (the base already
scores 5.7 of 7 on Conscientiousness and 5.5 on Intellect) and then partly
retired that explanation when a dedicated Conscientiousness amplifier gained 35
points ([[ocean-dials-replication]], [[post-draft]]). Gradient Atoms offers a
different mechanism - one pathway to break, many to strengthen - and reaches the
same asymmetry on formatting behaviours where no personality ceiling exists.
That is independent support for the mechanism and against the ceiling.

**Where they disagree, or seem to.** The paper reports that **coherence does not
predict steerability**; this project reports that the **Fisher norm does predict
how far the judged persona moves** - Spearman **+0.3706669239179368**,
permutation p **0.0014499275036248187**, over the 72 sphere points
(`qwen35/analysis/fisher_norms.json#correlation_with_degeneration.sphere_judged_profile_distance`,
[[fisher-norms]]). These are not the same quantity and should not be forced into
a contradiction: coherence is agreement among documents' gradients, F is the
curvature of the KL along the direction. The honest reading is that the paper
lacked a curvature measurement for its atoms and used coherence as a proxy for
one; this project has the curvature measurement and it does predict displacement.
Where the two do meet is on **degeneration**: F does not predict looping
(Spearman +0.1501 at p 0.4973 against the loop rate at alpha -2 over 22
directions), and the paper's #415 "loses coherence" at high alpha with no
predictor offered either.

**The strongest connection: EKFAC preconditioning is the correction this project
has measured the need for and never applied.** Section 3.2 says that without
whitening by the Fisher, "any decomposition is dominated by high-curvature
directions, drowning out semantic structure". This project's entire geometry -
the 134 x 134 exact Gram, the PCA, the factor analysis - is computed in the
**Frobenius** metric of the weight space. [[fisher-norms]] measured what that
metric costs:

- The Fisher norms of 124 directions span a factor of **259.924881015629**
  (`#comparisons.full_range`). Two directions this project has always treated as
  the same length differ by that much in how far they move the model.
- The Fisher ordering of the five factors is **not** the variance ordering.
  Warmth is F 0.3650 and rank 22 of 124; Imagination is last in variance by sums
  of squared oblimin loadings (5.798806928556024 against Warmth's
  10.76210106481124, `qwen35/results/fa_qwen35.json#solutions.centred_k5.ss_loadings.oblimin`)
  and second in F.
- A single trained trait adapter is exactly as steep as a random merge of all 134
  (median F 0.12140218073805878 against 0.12382989334918551, a ratio of
  0.9803947775010928). Structure in the coefficient vector buys curvature; being
  a real adapter does not.

So the paper's stated reason for preconditioning is a measured fact in this
project's data, and the project has never asked what its factor solution looks
like in the model's own metric. That is the single most consequential experiment
either paper suggests, and it is G1 below.

**EKFAC's output factor is the object [[column-space-structure]] measured.**
The next step is an inference from standard K-FAC background, not something the
paper states: it cites Grosse et al. (2023) for the eigendecomposition and moves
on. K-FAC and EKFAC approximate the Fisher per module as a Kronecker product of
an input-activation covariance and an output-gradient covariance. The output factor
is the covariance of `dL/dy` - which is exactly what a LoRA's `B` accumulates,
and exactly the space this project found carries the trait. Two adapters trained
for the same trait from independent initialisations sit at Frobenius cosine
**+0.01806099632415457** but their top left singular directions agree at |cos|
**0.7701377220108219** against **0.21012838847727913** for different traits, and
at k = 1 the mean squared principal-angle cosine is **0.6307023078841583**
against **0.06898996183549069** for different traits and a null of
**0.00028667534722222224**
(`qwen35/analysis/column_space.json#stage1.by_module_class`). The row space sits
at **0.0211080335981577** against a null of **0.021014492753623194** - the
initialisation carries nothing. So the half of the EKFAC factorisation that
Gradient Atoms whitens by is the half this project independently identified as
the learned one. Neither the paper nor this project knew of the other; the agreement is
worth a paragraph in the post.

**Their generic-versus-specific finding has a twin here.** The paper's
decomposition finds a large low-coherence residue (457 of 500 atoms below 0.1).
This project finds two generic components on top of which the specific structure
sits: an output-side subspace shared by every adapter regardless of seed
(different-trait overlap **0.11012252366265511** at k = 8 against a null of
**0.002293402777777778**, fifty times chance,
`column_space.json#stage1.by_module_class`), and the stage-two shared register
direction, which a trait-free control reproduces at cosine **0.3035 +- 0.0007**
against the zoo's **0.3893 +- 0.0311**
(`qwen35/analysis/stage2_neutral_control.json`, [[stage-two-exploration]]). Any
atom decomposition run on this project's gradients will find that shared
component first, and the project already knows how to build the control that
subtracts it.

**Sign.** Dictionary learning assigns atom signs arbitrarily, which the paper
handles by testing both directions. This project's bipolar axes live entirely in
the sign: same-factor opposite-keyed pairs sit at column-space overlap
**0.15904015225348364**, marginally *below* the different-factor
**0.16329709248713262**, and the exact Gram's own |cosine| behaves identically
(0.1341 against 0.1340), so any sign-blind statistic is blind to the axis
([[column-space-structure]], [[polarity-and-bipolarity]]). Any atom method
applied here must recover keying from the signed loadings, or it will find
same-pole clusters and miss the bipolarity gap of 0.33 that the post leads with.

**Where their method would have changed a result here.** (1) The factor analysis
would have been run on a whitened Gram, and the congruences with Goldberg - which
stop at 0.68 and reach 0.41 at worst ([[post-draft]]) - would either improve or
be shown to be metric-independent. Either answer is publishable and neither is
known. (2) The [[reward-hacks-data-scoring|reward-hacks data scoring]] asked
whether the School of Reward Hacks corpus pushes along any personality direction
and found that no personality direction clears a band of 20 random merges
(`axis_Agreeableness` -0.081874 at z -2.864 against a widest random arm of
-0.08829751330593415). That is a *supervised* question against 41 named
directions. The unsupervised version - decompose the corpus's own gradients into
atoms and ask what the top-coherence atoms are - is exactly the question
"emergent misalignment" needs, and the paper says it costs about 25 minutes of
compute.

**What does not transfer.** Regex detectors for the target behaviour: the paper
itself lists this as a limitation, and personality has no regex. The 8x
A100-40GB extraction is for rank 8 on two projections; the zoo is rank 64 on 248
modules, so the storage arithmetic is entirely different and the projection is
not optional but mandatory.

## Experiments this paper suggests, ranked

### G1. Redo the factor analysis in the Fisher metric

**Question.** The five factors, the congruences with Goldberg and the nine-versus-
five parallel-analysis count are all properties of a Gram computed in the
Frobenius metric. What survives whitening by the model's own metric?

**Design.** Build an **empirical Fisher Gram** over the 134 stage-one adapters.
`qwen35/align_score.py`'s `eps` trick already returns, in one backward pass, the
directional derivative of the log-likelihood of every item along every target
direction. Take the 134 adapter directions as targets and the **4,378 fixed
scored token positions** that `qwen35/fisher.py` already defines (the 24 steering
prompts with the alpha-0 responses stored under `PC1`, capped at 192 response
tokens, [[fisher-norms]]) as items, giving a 134 x 4,378 matrix `S` of per-token
derivatives. The empirical Fisher inner product is then
`u^T I v = mean_t (u . grad log p_t)(v . grad log p_t)`, i.e. `S S^T / T`.
Normalise to correlations and rerun `qwen35/analyse_fa_qwen35.py` unchanged.

**One construction detail decides whether this reuses `align_score.py`
unmodified.** That script returns one derivative per (target, *item*), where an
item's log-probability is summed over its completion tokens; the Fisher needs the
product of *per-token* derivatives, not the derivative of a sum. So build the
4,378 items as (prefix, single next token) pairs in `mode: "sft"` - then each
item's returned derivative already is a per-token derivative and nothing in the
script has to change. A closer approximation to the true Fisher, at the same
cost, uses a token *sampled* from `p_t` rather than the realised one.
**A check to run and report**: the diagonal of `S S^T / T` against
`qwen35/fisher.py`'s F for the ten single adapters. They will not agree - F is
the curvature of the KL, an expectation over the vocabulary, and this is the
empirical Fisher on realised tokens - and the size of that gap is itself a number
worth having.
**Controls**: the same construction on the permuted-label null arm and the
shuffled null arm ([[null-controls]], [[factor-analysis-null-arms]]) - the
shuffled arm must still retain zero factors - and on 20 random merges.

**State the caveat in the write-up**: this is the *empirical* Fisher on realised
tokens, not the sampled Fisher, and it is not EKFAC. It is the cheapest
whitening this project can build out of code it already has, and it is a
different approximation from the paper's.

**What a result would mean.** If the five factors survive with congruences
unchanged, the factor structure is metric-independent and the post can say so -
which is a much stronger claim than it currently makes. If congruences rise
above the current 0.41-0.68, the Frobenius metric was costing the Big Five
mapping and every congruence in the post is an underestimate. If the solution
dissolves, part of what the project has been calling structure is a property of
the parameterisation, and that has to be said plainly.

**Cost.** The 2026-09-09 data-scoring run did 973 items x 41 targets on one
A100-80GB for about **$1.41** of its own share (log 2026-09-09,
[[reward-hacks-data-scoring]]). This is 4,378 items x 134 targets, roughly 15x
the item-target product, and retained backward memory scales with batch x
sequence rather than with target count (the comment that run left in
`align_score.py`). Estimate **1.5-3 GPU-hours on an A100-80GB = $3-7**, scaled
from that run. Factor analysis is CPU and free. No judge calls.

**Reuses.** `qwen35/align_score.py`, `qwen35/fisher.py`,
`qwen35/build_fisher_spec.py`, `qwen35/analyse_fa_qwen35.py`,
`qwen35/results/gram_sweep.npz`, `qwen35/analysis/fisher_norms.json`.

### G2. Gradient atoms on the zoo's own preference data

**Question.** Are the five factors in the training gradients *before* any adapter
is trained, and does an unsupervised decomposition recover them without ever
seeing a trait label?

**Design.** Port the paper's pipeline to the zoo's DPO corpus. For a sample of
the 134 x 445 preference pairs (the paper used 5,000 documents; 5,360 pairs is
the same order and is what [[n-by-n-scoring]] already used - 40 pairs per trait),
compute the per-pair gradient with respect to `B` only. `B` is the right and
only target: at step zero `dL/dA = B^T (dL/dW) = 0`, so the whole first update
lives in `B` ([[n-by-n-scoring]]). **The projection is mandatory, not optional**:
one per-pair `dL/dB` is the size of an adapter's whole `B` factor, so storing
thousands of them is out of reach, and each must be reduced per module to its
top-k eigencomponents the way the paper reduces 2.2M to 6,800. Then
`MiniBatchDictionaryLearning` with K = 500 and a sparsity sweep over
{0.01, 0.1, 1.0} reproducing their Table 1, coherence on the unprojected
gradients over each atom's top-20 activating pairs, and finally: label each atom
by the traits of its activating pairs and score adjusted Rand index against the
trait label, the Big Five factor label and the keying.
**Controls**: the permuted-label corpus, whose adapters re-identify at congruence
0.98 to 1.00 when relabelled by the data they were trained on ([[post-draft]],
[[null-controls]]) - atoms should follow the data, not the names - and the
shuffled corpus, which should produce no coherent atoms at all.
**State the caveat**: the paper's gradients are SFT cross-entropy per document;
the zoo's loss is DPO plus a 0.1-weighted SFT term with a KL penalty
([[stage-one-training-config]]), so a per-pair gradient is of a different
objective. `align_score.py` already implements both tokenisations - it gained a
`mode: "sft"` path for the reward-hacks scoring.

**What a result would mean.** If the top-coherence atoms group pairs by trait and
the atom loadings align with the five factors, the factor structure is present in
the gradient geometry before training and is not manufactured by 13 AdamW steps -
the strongest possible version of the project's central claim. If instead atoms
group by *prompt type* across traits - the paper's own finding, that atoms
capture task types and not topics - then the factors are something training
produces, not something the data already has, and the shuffled/permuted control
pair says which.

**Cost.** The paper extracted 5,000 gradients on 8x A100-40GB in 170 s at rank 8
on 2 projections. The zoo is rank 64 on 248 modules, so per-example work is an
order larger and each item is a pair, not a document. Estimate **2-4 A100-hours
= $4-9**, scaled from Table 3 with a 10x factor for parameters and 2x for the
pair; dictionary learning and coherence are CPU, ~25 min, free. Add the two
control corpora at the same cost: **$12-27 total**, estimate. No judge calls.

**Reuses.** `qwen35/align_score.py` (both tokenisations),
`qwen35/build_nxn_inputs.py`, `qwen35/analyse_nxn.py`,
`qwen35/data/` preference corpora and the two null corpora,
`qwen35/column_space_on_modal.py` (the thin-factorisation machinery for
per-module bases).

### G3. Unsupervised atoms on the reward-hacks corpus

**Question.** The supervised question - does reward-hacking data push along any
personality direction - has been answered and the answer is no. What does that
corpus teach, when nobody says what to look for?

**Design.** Run the Gradient Atoms pipeline exactly on the 973 matched School of
Reward Hacks rows already prepared as
`qwen35/phase10_runs/sorh_ds_items.json` ([[reward-hacks-data-scoring]],
[[paper-school-of-reward-hacks]]), hack and control completions both, with the
SFT tokenisation `align_score.py` already reproduces. Discover K = 200 atoms,
score coherence, inspect the top atoms' activating rows, and then unproject the
top atoms into weight-space directions and score each against the 41 named
targets the data-scoring run used - including `axis_Agreeableness`, `FA_Warmth`
and the stage-one grand mean. **Control**: the same 20 Gaussian merges of the 134
zoo adapters that formed the band in that run, whose widest arm reached
**-0.08829751330593415**.

**What a result would mean.** If a high-coherence atom separates hack from
control completions and that atom, unprojected, has a large cosine with a
personality direction, then reward hacking is a persona change and the supervised
null was a failure of the 41 chosen directions rather than a real absence. If the
atoms are all task-type - "write a poem", "game the metric" - the null stands and
is strengthened: the corpus teaches a procedure, not a character. Either way this
is the first unsupervised look at that corpus, and it is what the emergent-
misalignment literature actually wants.

**Cost.** 973 x 2 completions is a fifth of the paper's 5,000 documents, at rank
64 on 248 modules. Estimate **1-2 A100-hours = $2-4**, plus CPU dictionary
learning. Scoring the unprojected atoms against 41 targets is the data-scoring
run again, about **$1.41**. **Total $4-6**, estimate, anchored on the 2026-09-09
run.

**Reuses.** `qwen35/align_score.py`, `qwen35/build_sorh_datascore_inputs.py`,
`qwen35/analyse_sorh_data_scores.py`,
`qwen35/phase10_runs/sorh_ds_items.json`, `sorh_ds_targets.json`.

### G4. Coherence against steerability, on 134 labelled directions

**Question.** The paper says coherence does not predict steerability, on five
atoms. This project has 134 directions whose behavioural effect is already
judged. Does it?

**Design.** From G2's extraction, compute per trait the coherence of its own 445
pairs - mean pairwise cosine of the raw per-pair `dL/dB`. Correlate against three
quantities already on disk: (a) the judged own-factor amplification of that
trait's stage-one adapter (`qwen35/analysis/stage2_register_vs_residual.json`
gives **+22.33** as the 15-trait mean for cond_b, and
`qwen35/phase10_runs/judged_100.json` holds the per-trait records); (b) the
Fisher norm F, for the ten single adapters that have one
(`qwen35/analysis/fisher_norms.json#directions.single_*`, from 0.0606 for
`single_extraverted` to 0.2205 for `single_agreeable`); (c) the adapter's own
Frobenius norm from the diagonal of `results/gram_sweep.npz`.
**Control**: partial out response length, which the reward-hacks analysis showed
correlates with two judged factors at Spearman 0.4942 and 0.3965
([[open-questions]], `qwen35/analysis/sorh_behavioural.json`).

**What a result would mean.** If coherence predicts judged amplification here
where it did not predict formatting steerability there, the difference is
saturation: their behaviours have ceilings (numbered lists at 58% base,
amplifiable by +1pp) and personality scales do not. If coherence predicts F but
not the judged effect, then coherence is a curvature proxy after all and the
paper's negative is about the judged half. If it predicts nothing, the paper's
claim generalises and coherence should not be used as a selection criterion by
anyone.

**Cost.** Free given G2 - the correlations are CPU work on files already
produced, and every behavioural number already exists. **$0 new GPU, 0 judge
calls.** This is the cheapest experiment on this page and should be run as part
of G2 rather than separately.

**Reuses.** G2's outputs, `qwen35/analysis/fisher_norms.json`,
`qwen35/phase10_runs/judged_100.json`,
`qwen35/analysis/stage2_register_vs_residual.json`,
`qwen35/results/gram_sweep.npz`.

### G5. Suppression versus amplification at matched Fisher dose

**Question.** Is the suppress/amplify asymmetry structural, as the paper argues,
or is it this project's judged ceiling plus an unequal unit?

**Design.** The project's steering comparisons are at matched **alpha**, which
[[fisher-norms]] shows is a dose that differs by up to 14x within the published
set. Redo the asymmetry test at matched **Fisher** dose using
`fisher_norms.json#fisher_dose_table`. Choose directions on which the base model
scores *low* on the target factor, so no ceiling is available - the base is 5.7
of 7 on Conscientiousness and 5.5 on Intellect ([[post-draft]]), so the arousal
and fearful-withdrawal poles are where headroom exists in both directions. Ten
directions, both signs, three doses, the 24-prompt battery, judged blind.
**Controls**: random merges at the same Fisher dose from
`fisher_norms.json#random_band`; and record the loop rate with
`analyse_alien_steer.looping`, since all 22 previously measured directions loop
at least as much at -2 as at +2 and 18 have a loop rate of exactly zero at +2
(`#comparisons.sign_asymmetry_of_degeneration`) - the asymmetry in *damage* is a
confound for any asymmetry in *effect*.

**What a result would mean.** If suppression still beats amplification with the
ceiling removed and the dose matched, the paper's mechanism story - one pathway
to break, many to strengthen - is supported in a second domain and belongs in the
post. If it vanishes, the project's asymmetry was units and ceiling, and
[[steering-results]] needs a correction.

**Cost.** 10 directions x 2 signs x 3 doses x 24 prompts = 1,440 generations,
about the size of the s2register run's 1,464 ([[stage-two-exploration]]).
Estimate **1.5-2.5 GPU-hours = $3-5**, plus 1,440 judge calls at $0.0012 to
$0.003 = **$2-5**. **Total $5-10**, estimate, scaled from the s2register run and
the [[costs]] judge rates.

**Reuses.** `qwen35/steer_fix.py`, `qwen35/build_fisher_spec.py`,
`qwen35/judge_personas.py`, `qwen35/analyse_alien_steer.py`,
`qwen35/analysis/fisher_norms.json`.

### G6. Atoms on a general SFT mixture, then ask whether any of them is a persona

**Question.** The paper's atoms on general instruction data are all task types.
Does *nothing* persona-shaped appear, or does it appear at low coherence?

**Design.** Replicate the paper's setup as closely as the project's stack allows
- Qwen3.5-4B rather than Gemma-3 4B IT, 5,000 instruction-response pairs, the
same K = 500 and alpha = 0.1 - then unproject every atom and score all 500
against the 134 zoo adapters and the five factor directions in the zoo's LoRA-A
frame, using the exact-Gram machinery. **Control**: the 20 random merges band.
Note the frame caveat the reward-hacks scoring already documents: a direction in
a different LoRA-A frame is scored along its projection into the zoo window, not
along itself.

**What a result would mean.** If no atom of 500 has a cosine with any factor
direction above the random band, then generic instruction tuning does not move a
model along personality axes, which bounds what fine-tuning-induced misalignment
can be doing mechanistically. If several do, and at low coherence, then
personality is present in ordinary SFT gradients as a diffuse component - which
is the shape of this project's own generic output-side subspace.

**Cost.** Extraction and decomposition as G2 without the pair doubling:
**1.5-3 A100-hours = $3-6**. Scoring 500 unprojected atoms against 139 targets is
the `align_score` path again, **$1-2**. **Total $5-8**, estimate. No judge calls.

**Reuses.** `qwen35/align_score.py`, `qwen35/cross_gram_full_on_modal.py`,
`qwen35/results/gram_sweep.npz`, G2's pipeline.

---

# The two papers against each other, and against this project

**They are the same idea run in opposite spaces.** SliderSpace decomposes the
model's **outputs** (CLIP embeddings of 5,000 generated images) and then trains
adapters to realise the components. Gradient Atoms decomposes the model's
**gradients** (5,000 per-document gradients, whitened by an approximate Fisher)
and then reads the components off as weight-space vectors directly. Both end at
the same object: a small set of low-rank directions that can be added to the
model with a scalar, composed, and swept. Both are unsupervised. Both name their
directions after the fact - SliderSpace with Claude 3.5 Sonnet on image pairs,
Gradient Atoms by manual inspection of each atom's top-20 activating documents -
and both admit the naming is the weak step.

**This project sits between them and shares the weakness of both.** Like
SliderSpace it works with a set of LoRA directions that compose; like Gradient
Atoms it works in weight space and has a gradient-alignment identity at its
centre. But it is **supervised at the start**: 134 adjectives chosen from
Goldberg and Condon, one adapter each, structure recovered afterwards. That is
why the hole exists as a problem at all - a supervised basis leaves gaps, and the
project has spent real effort on one of them. Both of these papers would find the
gap first and name it never.

**Three things the two papers agree on that this project should adopt.**

1. *Decompose in the space you care about.* SliderSpace's ablation (PCA in output
   space gives colour and shape, not semantics) and Gradient Atoms' EKFAC
   preconditioning ("without correction, any decomposition is dominated by
   high-curvature directions") are the same instruction. This project decomposes
   in the raw Frobenius metric and has independently measured that this metric
   spreads directions by a factor of **259.924881015629** and orders the factors
   differently from their variance ([[fisher-norms]]). G1 is the fix.
2. *Sign is not given.* Dictionary atoms have arbitrary signs; PCA components
   have arbitrary signs. This project's bipolarity - the gap of 0.33 between
   same-pole cosine +0.24 and opposite-pole -0.08 ([[post-draft]]) - lives
   entirely in the sign, and [[column-space-structure]] shows every sign-blind
   statistic misses it. Any port of either method must carry keying through.
3. *Suppression and amplification are not symmetric.* Gradient Atoms finds it on
   formatting behaviours with no ceiling; this project finds it on four of five
   factor dials and has been explaining it with a ceiling it has partly retired.

**One thing they disagree about, which is a real open question.** SliderSpace's
directions are *trained to be* the direction, and it reports that rank one
suffices and beats higher ranks at fixed budget. Gradient Atoms' directions are
*read off* an existing training run, and its atoms are dense objects in a 6,800-
dimensional projection of a 2.2M-dimensional space. This project's adapters are
rank 64 and its deltas are demonstrably not rank-1 in disguise (top singular
direction 0.10318456418060519 of the singular-value sum) - yet the part that
carries the trait across seeds is concentrated at the top of the spectrum
(weighted 0.4630531697561965 against unweighted 0.14174031494371594 at k = 64).
Whether a persona direction is intrinsically rank one, and only the *delta* that
produces it is rank 64, is the question S3 asks and neither paper answers.

**The top three experiments across both papers**, in order: **G1** (factor
analysis in the Fisher metric, $3-7, reuses `align_score.py` and `fisher.py`
whole), **G2** (unsupervised atoms on the zoo's own preference gradients,
$12-27 with both null controls, with **G4** free inside it), and **S1**
(activation-space persona sliders, $37-40, the only experiment that can revive or
finally bury the hole; **run on 2026-09-09/10, see [[persona-sliders]]**). All three are within the order of magnitude of runs this
project has already paid for.

Related: [[fisher-norms]], [[column-space-structure]], [[n-by-n-scoring]],
[[stage-two-exploration]], [[hole-words-factor-chart]], [[additivity]],
[[reward-hacks-data-scoring]], [[literature-big-five]],
[[literature-lora-and-merging]], [[open-questions]], [[costs]].

---

# What was run from this page

**G2 and G4 were run on 2026-09-09/10**: [[gradient-atoms]]. **G3 was run
immediately after**: [[reward-hacks-gradient-atoms]]. Both used one new Modal
module, `qwen35/gradient_atoms_on_modal.py`.

Three predictions on this page were wrong and are corrected on the result pages,
not here:

1. G2's control design says the shuffled corpus "should produce no coherent atoms
   at all". It does not. Swapping a DPO pair negates its gradient, and sparse
   coding is equivariant to flipping a document's sign together with its code, so
   a shuffled corpus cannot destroy a dictionary. It degrades coherence by about
   a third and its real signature is that an atom's two poles become the same
   trait (14.5% of shuffled atoms against 0.0% of real ones).
2. G2's permuted control says "atoms should follow the data, not the names". That
   cannot be tested at trait level: the permuted files are byte-identical row sets
   from `data_common` under other names, and trait purity is invariant under a
   bijection. The factor-level version was run and is null for lack of power.
3. G2's cost estimate of "$12-27 with both null controls" assumed three
   extractions. Both null arms are exact linear recombinations of one
   extraction's per-completion gradients, so the whole of G2 cost about $1.
   G3 cost about $0.55 at the meter's rate.

G4's finding is that the paper's negative generalises: coherence predicts judged
own-scale amplification neither across 22 published directions (Spearman 0.0262,
p 0.908) nor across 100 traits (-0.1371, p 0.174). What per-trait coherence does
predict is the trained adapter's Frobenius norm, at Spearman +0.7238, p 5.1e-23
over 134 traits.
