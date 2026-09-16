---
title: Source contradictions
summary: Places where two files in the project disagree and neither is known to supersede the other, with both values, both paths, which one the wiki quotes and why, and what would settle each.
status: current
sources:
  - qwen35/analysis/actspace_geometry.json#windows.resp.primary.r_centred
  - qwen35/analysis/actspace_adapters_geometry.json#windows.resp.curve
  - qwen35/analysis/actspace_cross_geometry.json#resp_specific_adapter_alone
  - qwen35/analysis/viz.json#coverage
  - qwen35/analysis/alien.json#k_sweep
  - qwen35/analysis/fisher_norms.json#correlation_with_degeneration.sphere_judged_profile_distance
  - qwen35/analysis/sphere_isokl.json#dose_covariate
  - qwen35/analysis/align_summary.json#a_drift
  - qwen35/analysis/crossseed_arms.json
  - qwen35/PREREGISTRATION_phase3.md
  - qwen35/results/decomposition.json#test2.raw.same_polarity
  - qwen35/plan.json#defaults.use_rslora
  - qwen35/phase2_runs/archive/phase5_sweep_134.json
  - qwen35/traits_primary.json
  - qwen35/traits_secondary.json
  - qwen35/site_traits/data.json
  - vendor/persona-cartography/src/training/oct_adapter.py
last_verified: 2026-09-10
tags: [overview, contradictions]
---

# Source contradictions

Disagreements between two files where **neither is known to supersede the
other**. Where one source does supersede another the entry belongs on
[[superseded-claims]] instead; where a question is simply unanswered it belongs
on [[open-questions]].

Wiki schema rule 3 is that contradictions are recorded, not resolved. Each entry
below gives both values with their paths, says which one the wiki quotes and on
what grounds, and says what evidence would actually settle it. "The wiki uses X"
is a citation policy, not a verdict.

---

## S1. Prompts against weights at layer 16: 0.705 or 0.737

- **0.705** - `qwen35/analysis/actspace_geometry.json#windows.resp.primary.r_centred`
  = 0.704599368961925. `qwen35/analyse_actspace.py` lines 92-94 build `Cw` from
  the **raw** weight Gram and correlate it against trait-centred activations at
  line 127. Rendered on the blog page as "r = 0.70".
- **0.737** - `qwen35/analysis/actspace_adapters_geometry.json#windows.resp.curve[16].r_PW`
  = 0.7368952052412984. `qwen35/analyse_actspace_adapters.py` lines 62-63
  **double-centre** the weight Gram (`Gc = J @ Gw @ J`) before line 84. Rendered
  on the blog page as "the prompts track the weights (0.74)".

**Wiki uses:** both, each on its own page and each labelled with its centring -
[[actspace-persona-vectors]] for 0.705, [[actspace-adapters]] for 0.737, with the
pair recorded on [[actspace-method-notes]]. Neither is wrong; they are the same
quantity computed two ways, and the blog prints both within two paragraphs
without saying so ([[superseded-claims]] item L5).

**What would settle it:** nothing to settle - the two centrings are different
estimators. What is owed is a sentence in the post naming which is which, or one
of them being dropped.

---

## S2. The adapter alone along the prompt's direction: 0.62 or 0.70

- **0.62** - raw:
  `qwen35/analysis/actspace_adapters_geometry.json#windows.resp.curve[16].frac_internalised`
  = 0.6198613047599792.
- **0.70** - trait-specific, after the shared shift is removed:
  `qwen35/analysis/actspace_cross_geometry.json#resp_specific_adapter_alone`
  = 0.7044131755828857. This is the figure the blog's cross paragraph uses.

**Wiki uses:** both, cross-linked between [[actspace-adapters]] and
[[actspace-cross]], with the construction named in each case. The project's own
2026-09-05 lesson - remove the common shift before comparing steered activations
- argues for the trait-specific number where the question is "toward what".

**What would settle it:** already settled as a matter of method; what remains is
that `qwen35/analyse_actspace_cross.py` line 12 still says "adapter alone was
0.62" in its docstring while its own trait-specific block prints 0.70.

---

## S3. Coverage at k = 2: the sign of the effect flips

- `qwen35/analysis/viz.json#coverage.2`: `gap_deg` = 3.9655382092691895,
  `z` = **+1.5803426589637861**, nearest `introverted`.
- `qwen35/analysis/alien.json#k_sweep.2`: `gap_deg` = 0.9282998475514116,
  `z` = **-2.280849861251385**, nearest `unsympathetic`.

The two files agree to a fraction of a degree at k = 3, 4, 5, 6, 8, 10, 12 and
16. Only k = 2 diverges, and the divergence is a sign flip: opposite conclusions
about whether the 134 words cover the plane better than chance.

**Wiki uses:** `alien.json`, because the blog page reads it and it was written 46
minutes later on the same day (2026-09-01 15:22 against 14:36). Both are quoted
on [[hole-words]] and [[umap-and-layouts]].

**Narrowed, by reading both files' remaining keys.** They agree exactly on
`cumvar` (0.2392112953756293) and on the null moments (`null_mean_deg`
2.722430717649889, `null_sd_deg` 0.7866062999491471), so they describe **the same
k=2 plane against the same null**. They differ only in the direction each search
returned: `viz.json#coverage.2.u` = [0.1459605002491245, 0.9892904186168112]
against `alien.json#k_sweep.2.u` = [0.9999411625457942, 0.010847647052033626],
which are close to orthogonal. This is a disagreement between two gap
**searches**, not between two geometries.

**What would settle it:** comparing the two producers' search objectives. If both
maximise the same thing - the angle from the direction to the nearest of the 134
adjectives - then the **larger** gap is the better optimum, and that is
`viz.json`'s 3.97 degrees, not the 0.93 degrees in the file the blog reads; which
would make `alien.json`'s k=2 row the failed search rather than `viz.json`'s.
That is an inference from the stored keys and is **not** established. It is
recorded so that nobody quotes the k=2 sign either way without reading the code.
At every other k the two agree, so nothing else on [[hole-words]] is affected.

---

## S4. LoRA-A drift: 1.5% or 4.5%

- **1.5%** - `qwen35/analysis/align_summary.json#a_drift` = 0.014605041334818797.
  The same value appears in `analysis/blog_data.json`, `analysis/align_scores.json`
  and `analysis/nxn_scores.json`, and `qwen35/analyse_alignment.py` hardcodes
  `ZOO_DRIFT = 0.0146` as a gate constant, as does `analyse_hole.py`. The blog
  page's "1.5%" is a rounded restatement of it.
- **4.5%** - `qwen35/build_monitor_page.py` line 112: "it drifts **4.5%** of its
  own norm over 93 steps".

**Wiki uses:** 0.0146 / 1.5%, on [[adapter-effect-and-drift]], because it is
file-sourced in four analysis JSONs and is the gate the alignment and hole
analyses actually apply; `build_monitor_page.py` is an older built page (wiki
rule 2) and stores no number.

**What would settle it:** what "93 steps" refers to. 4.5% may be a different
population (a longer or stage-two run) rather than a competing estimate of the
same thing - the monitor page also quotes the attenuation as "predicted 0.058,
observed 0.056", against the current 0.0250 / 0.0265, which is definitely a
different quantity. Finding the monitor page's builder inputs would resolve
whether these are two measurements or two eras.

---

## S5. Sketch fidelity: 0.9996 or 0.99944

- **r = 0.9996** - prose in `qwen35/build_blog_page.py` line 1584.
- **cosine 0.99944** - `qwen35/build_monitor_page.py` line 209, "independently
  computed exact Gram at cosine 0.99944".

No output file storing either was found. `qwen35/analysis/validate_100.json`,
which the task brief pointed at, is per-trait training metadata (`loss_true`,
`reported`, `steps`, `rows_in`, `dropped`, `targeted`, `hp`, `lora`), not sketch
validation.

**Wiki uses:** 0.9996 on [[geometry-overview]] and [[pca-and-scree]], because the
blog page is current truth, with the disagreement stated in place.

**What would settle it:** running `qwen35/validate_sketch.py` or
`build_gram.py --compare` and storing the correlation in a file. This is the
single load-bearing justification for reading sketch angles as weight-update
angles, and it exists only as prose in two builders that disagree. Recorded as an
owed item on [[open-questions]].

---

## S6. Cross-seed category counts

- `qwen35/analysis/crossseed_arms.json#[0]`: `same_factor_same_key` n = **368**,
  `same_factor_opp_key` n = **392**, `diff_factor` n = **4560**. With the 40
  same-trait pairs those sum to 5,360, while the same object's `n_pairs` is
  **5,320**.
- `qwen35/PHASE3_VERDICT.md`, 2026-08-22 addendum: **358**, **386** and **4336**
  for the same three classes (summing with 40 to 5,120).

The class **means** agree exactly between the two (+0.00593 / -0.00271 /
+0.00133). Only the counts differ, and the JSON's own counts do not close against
its own `n_pairs`.

**Wiki uses:** the JSON's means, which both sources share, and quotes both count
sets on [[cross-seed-geometry]] and [[superseded-geometry-claims]] section 10
without preferring either.

**What would settle it:** the pair-enumeration code in whatever wrote
`crossseed_arms.json`. Most likely one set includes and the other excludes some
boundary class (self-pairs, or the traits that appear in both seeds), but nothing
on disk says which.

---

## S7. `site_traits` coordinates are about twice `viz.json`'s

- `qwen35/site_traits/data.json#traits[].pc` for `unsystematic`:
  [0.8741, 0.6316, -0.0956]; for `pleasant`: [0.899, -0.1025, 0.5281].
- `qwen35/analysis/viz.json#scores` for the same two traits (first three of
  eight): [0.4610, 0.2942, -0.0420] and [0.4508, -0.0817, 0.2841].

Signs agree throughout; magnitudes are roughly double. The same scale difference
shows up in the nearest-neighbour cosines: for `active`, `site_traits` gives
energetic 0.42, vigorous 0.374, practical 0.368, where
`qwen35/analysis/trait_graph.json#stage1.edges` gives energetic 0.396 - same
neighbour, different cosine.

**Wiki uses:** `viz.json` for PC scores and `trait_graph.json` for neighbours, on
all 141 trait pages, because `site_traits` is an older built page (wiki rule 2)
and its normalisation is undocumented. No `site_traits` PC value appears on any
page.

**What would settle it:** `build_site_traits.py`'s normalisation step. Two
different normalisations of one PCA is the obvious reading (unit-variance scores
against raw projections), but no file states either.

**Related, in the same directory:** `qwen35/site_traits/analysis.json`'s `pc1`
prose describes the poles with the **opposite sign** to
`qwen35/analysis/pc_loadings.json` - "its positive pole loads unsympathetic,
insensitive and cold" against `pc_loadings.json#pcs.PC1.pos` listing
unsystematic, pleasant, effeminate, sympathetic, agreeable. `site_traits`'s own
`data.json` agrees with `pc_loadings.json`, so the prose contradicts the data in
its own directory. Eigenvector signs are arbitrary; the wiki quotes
`pc_loadings.json`'s pole lists verbatim on [[factor-pc1]].

---

## S8. `plan.json` says rsLoRA is on; every recorded run says it is off

- `qwen35/plan.json#defaults.use_rslora` = **true**, with `#defaults.rslora_note`
  calling rsLoRA the default and plain LoRA an ablation arm.
- All **134** records in `qwen35/phase2_runs/archive/phase5_sweep_134.json` read
  `use_rslora: false` and `expected_scaling: 2.0` (verified: the set of values
  across all 134 records is `{False}` and `{2.0}`).

**Wiki uses:** the runs. `plan.json` is an intent document and `runmeta` is what
the container recorded, which is the project's own standing rule
([[zoo-build-governance]]). The wiki therefore treats this as **resolved** and
files it as [[superseded-claims]] item Z1; it is listed here because the two
files still sit side by side on disk and a reader opening `plan.json` first will
get the wrong recipe.

**What would settle it:** nothing further; `qwen35/RUNNER_TASK.md` and the
2026-09-03 `PHASE3_VERDICT.md` addendum both say rsLoRA off independently.

---

## S9. The preregistered bar is called a median and is numerically a mean

- `qwen35/PREREGISTRATION_phase3.md` lines 73-74: the bar is "the **median**
  cosine of same-factor, same-keying, *different*-trait pairs: **+0.2447**".
- The value 0.24472701836565558 is
  `qwen35/results/decomposition.json#test2.raw.same_polarity`, which that file
  computes as a **mean** over 932 pairs.
- `qwen35/results/compare_nulls_output.txt` separately reports the **median** of
  the same class as **+0.2467** (n = 932), and **+0.2373** on the 40
  seed-twinned traits (n = 130).

**Wiki uses:** all three, on [[seed-floor]] and [[superseded-geometry-claims]]
section 2, with the word "median" flagged as wrong for +0.2447. The bar itself is
separately withdrawn as being in the wrong units, which makes the mean/median
question academic for the verdict but not for anyone quoting the number.

**What would settle it:** it is settled as a matter of fact - the key is a mean.
What is unresolved is whether the preregistration intended the median (in which
case the registered bar was +0.2467) or copied the wrong key.

---

## S10. Emotional Stability's keying is 6/14 where every description implies 10/10

- `qwen35/traits_primary.json`: Emotional Stability has **6** positively keyed
  and **14** negatively keyed markers. The other four factors are 10/10
  (verified by counting the `keyed` field: Extraversion 10/10, Agreeableness
  10/10, Conscientiousness 10/10, Intellect 10/10).
- Corroborated by `qwen35/analysis/geometry_stage1.json#unwhitened_loo_keying.EmotionalStability`
  (`n_pos` 6, `n_neg` 14) and remarked on in `qwen35/build_spider_page.py`.
- Against: `qwen35/build_blog_page.py` describes the markers as "twenty per
  factor, both poles", and the standard description of Goldberg's 100 unipolar
  markers is a balanced set.

**Wiki uses:** 6/14, the file value, on [[goldberg-100-primary-traits]] and
[[literature-big-five]], with the blog sentence flagged at [[superseded-claims]]
item L7.

**What would settle it:** Goldberg 1992, Psychological Assessment 4(1), 26-42,
Table 3 - whether the imbalance is in the published marker list or was introduced
when the list was transcribed into `traits_primary.json`. Nobody has checked. It
matters because any per-factor statistic assuming balanced keying is affected,
[[polarity-and-bipolarity]]'s bipolarity and factor-axis statistics most
obviously.

---

## S11. Forty lexicon words drawn, thirty-four adapters built

- `qwen35/traits_secondary.json` holds **40** entries, all factor `Lexicon`,
  keyed `+` (verified: length 40), and
  `qwen35/traits_secondary_provenance.json#chosen` lists the same 40, one per
  cluster of a 40-cluster draw. `qwen35/constitutions.json` holds a constitution
  for each, giving 147 constitutions (100 + 40 + 4 + 3).
  `qwen35/paper_notes.md` section 3.8 also says "(+40 secondary)".
- Every analysis file carries **34**: the six words absent from
  `analysis/nxn_summary.json#names`, `viz.json#traits`, `trait_graph.json`,
  `fa_qwen35.json#trait_slug`, `corpus_scan_all.json`, `merge_audit.json` and
  `site_traits/data.json#traits` alike are **unconformable, frightened, busy,
  defenseless, significant, sleepy**.

**Wiki uses:** 34 for the zoo and 40 for the draw, stated separately, on
[[lexicon-secondary-draw]], [[traits-index]] and [[six-refused-traits]]. No trait
page was generated for the six.

**What would settle it:** why those six were dropped. No source records it, and
it interacts with the approved-but-unexecuted "train all 40" instruction
([[superseded-claims]] item F10). The constitution writer's refusal list is the
leading explanation and covers refusals generally, but nothing ties these
particular six to it.

---

*Update 2026-09-10.* [[six-refused-traits]] does record it: `qwen35/backups/constitutions.json.pre-generic-anchor.bak` carries a `rejected` field with the writer's verbatim `NOT_A_TRAIT` reason for each of the six, and `qwen35/genpairs.log` shows them skipped. The "no source records it" line above is historical; F10 (the approved-but-never-executed retrain of all 40) stays open.

## S12. The two steering campaigns do not share an alpha unit

- **STEER134 (phase 7).** `qwen35/STEER134_DESIGN.md`: "All alphas are in units
  of s_bar = mean adapter norm = **1.6157** (from `results/gram_sweep.npz`, scale
  2.0, 248 modules)."
- **steer / steerfix (phase 10).** `qwen35/steer_fix.py`: "`ref` is the mean
  single-adapter Frobenius norm"; the specs carry `ref` = **0.8102592902648793**
  (`steer_spec.json`, n = 100) and **0.8078003190997738** (`steer_spec2.json`,
  n = 134). A third figure,
  `qwen35/analysis/sorh_projection.json#reference.ref_norm` =
  **0.8146967250408155**, is labelled "mean single trait-adapter **sketch**
  norm".

The two campaigns' alpha = 1 therefore differ by roughly a factor of two, and
nothing in the repository reconciles them.

**Wiki uses:** both, always with the campaign named, on [[steering-results]].
**Do not compare alphas across campaigns**, and do not put a STEER134 dose on the
same axis as a steerfix dose.

**What would settle it:** a single line recording whether 1.6157 is the same
quantity as 0.8103 at a different scale convention (a factor of 2.0 is
suspiciously close to the LoRA scaling) or a genuinely different norm over a
different module set. The numbers are close to a factor of two apart, which is
either the whole answer or a coincidence.

---

## S13. Persona Cartography's SFT batch size: paper 16, code 32

- `qwen35/paper_notes.md` section 2.4 quotes the paper's Appendix A.1.2 as
  **batch size 16**, and section 3.11 says to "pick one and record which" against
  Open Character Training's 32.
- `vendor/persona-cartography/src/training/oct_adapter.py` uses one batch-size
  chooser hardcoded to `train_batch_size=32` for **both** DPO and SFT;
  `scripts/training/ocean_paired_dpo/04_train_lora.py` is consistent with it.

**Wiki uses:** 32, on [[persona-cartography-paper]], as what the released
pipeline runs, with the paper's 16 recorded beside it.

**What would settle it:** whether the code changed after the paper's numbers were
produced. The checkout's HEAD is a "camera-ready switch" commit of 20 Aug 2026,
so the code is at least as late as the paper; that makes the code the better
description of the artefact but not proof about the experiments in the tables.
Asking the authors is the only clean route.

---

## S14. Open Character Training's README against its paper

Two disagreements, same shape.

**Which Qwen.** `vendor/OpenCharacterTraining/README.md` lists
`Qwen/Qwen2.5-72B-Instruct` among the three character-trained models;
`qwen35/paper_notes.md` section 1.1 gives Qwen-2.5-7B-Instruct and the paper's
Table 2 is headed "Qwen 2.5 7B". The arXiv abstract page does not disambiguate.

**Persona names.** The README names `poeticism` and `goodness` (linking
arXiv:2310.13798); `paper_notes.md` Table 1 gives `poetic` and `flourishing`; the
arXiv abstract says "humorous, caring, malevolent, etc.", matching neither
verbatim.

**Wiki uses:** the paper - 7B, and the paper's persona names - on
[[open-character-training-paper]], with the README reading recorded.

**What would settle it:** the paper's own release table or artefact list. The
likeliest reading is that the README describes a later or broader release than
the paper's experiments, and that the persona names are development-versus-paper
naming, but neither is verified.

---

## S15. Persona Cartography's author order

- **Baines first.** arXiv, and `vendor/persona-cartography/CITATION.cff`: Baines,
  Gonzalvez Hawthorne, Koroliuk, Shalibashvili, Dumas, Voudouris, Africa.
- **Hawthorne first.** The LessWrong crosspost byline. `CONTEXT.md` section 1 and
  the harness reading list both follow the LessWrong order.

**Wiki uses:** the arXiv order, on [[persona-cartography-paper]], because arXiv
and the repository's own citation file agree. Note "sidbaines" is the LessWrong
handle appearing fourth in that byline.

**What would settle it:** nothing more is needed for citation purposes; it is
recorded because two of this project's own most-read files use the other order.

---

## Second tier: smaller disagreements, same treatment

**S16. Measured Modal spend, $308.83 or $311.41.**
`qwen35/plan.json#ledger_defect_note` says "Modal $308.83 + OpenRouter $61.82"
(summing to the quoted $370.65); `qwen35/plan.json#modal_measured.total_pc_qwen35`
= **311.4096**, read 2026-08-23T00:12, `FINAL: false`. `qwen35/HANDOVER.md`
writes "Modal ~$311 + OpenRouter $61.82" while also giving the total as ~$370.65.
The two reads are eight minutes apart and the note itself says the Modal gap
rises while phase 10 runs. Wiki quotes both with their read times on
[[zoo-spend-ledger]] and [[costs]]. Settled by quoting the read time, never the
bare figure.

**S17. Phase 7 judging, $40.05 or $44.33.**
`qwen35/plan.json#phases[7].realised.openrouter` = **40.0529835**, sourced to
`results/steer134_judged.json`'s cost block and independently summed from
`steer134_judge_cache.jsonl`. The 2026-08-23 09:50 entry in
`/home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md` gives
**$44.33** (16.28M tokens) and says it was "corrected in-thread". Both cite the
same file. Wiki carries both on [[costs]]. Settled by re-summing the cache
against the cost block, which is a read, not a rerun.

**S18. Centroid deflation: 4.6% or 5.3%, and where 46% came from.**
`qwen35/analysis/polarity_deflation.json#stage1.var_removed` =
**0.045566670770937776**; `qwen35/analysis/manifold_ideas.md` and
`qwen35/build_findings_page.py` both say "5.3% of variance", paired with "46% of
factor signal". The JSON records `before_factor` = 0.5100000000000001,
`after_factor` = 0.45199999999999996 and `null_factor` = 0.51; how 46% follows
from those is not recoverable, because the file has no producing script. Wiki
quotes the JSON on [[polarity-and-bipolarity]] and records the prose figures.
Settled only by finding the producer.

**S19. Alignment-trait pair counts: 500, 497 or 444.**
`qwen35/PREREG_alignment.md` says "the same 500-prompt pool (sha 8b725d86...),
500 preference pairs each"; `qwen35/phase2_runs/results_data_alignment.json#[0].n_pairs`
= **497**; `results_data_alignment_common.json#[0].n_pairs` = **444**. These are
the target, the first arm's yield after filtering, and the second arm's yield on
the zoo's shared 445-prompt intersection. Wiki treats them as three quantities on
[[alignment-and-hole-traits]]. Settled by the drop logs, which exist per pass.

**S20. sweep100's stage-2 transcript shape.**
`/home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md`
(2026-08-18): "32 transcripts x 8 turns each, VERIFIED from the saved
transcripts.jsonl". `sweep100/adapters_sft/active__sft/runmeta.json`:
`n_self_interaction` 16, `n_self_reflection` 16, `turns` 4, `n_transcripts` 32.
Most likely 4 exchanges rendered as 8 messages. Wiki states both on [[sweep100]].
Settled by counting message roles in one transcript file, which is a read.

**S21. The ideonomy trait list: 683 or 638.**
Samuel's link and message say 683; the page header sums to 638 (234 positive /
112 neutral / 292 negative) and the page misprints the neutral count as 292.
Effective distinct targets after dedup are about 540-570. Harness memory,
2026-08-14. Wiki records both on [[origin-and-question]]. Settled by re-fetching
the page, which nobody needs to do - the list was not used.

**S22. The beta-0.5 arm's margin, in two units.**
`qwen35/phase2_beta05.log` logs TRL `rewards/margins` **64.31** for `imaginative`
against **40.55** for the beta-0.1 arm in `qwen35/phase2_run2.log` - the logged
quantity went **up**. The gate-5 docstring in `qwen35/phase2_gates.py` describes
the same change as a margin **falling from 224 to 137**, which are raw margins
recovered from `-log sigmoid(beta*margin)`. Wiki records both on
[[phase-two-recipe-search]]. Settled by stating the unit with the number: TRL's
logged margin is beta-scaled and the gate's is not.

**S23. The sweep100 spectrum in the second decimal.**
`sweep100/WRITEUP.md` and `sweep100/results/pca.md`: **22.8 / 14.5 / 9.7 / 6.1**.
The clean three-seed rerun (`sweep100/results/seeds.json`, 2026-08-16 12:08)
gives seed 0 as **22.4 / 14.2 / 9.7**. Two runs, not two readings of one run;
both quoted on [[sweep100]]. A blog post must not mix them.

**S24. School of Reward Hacks row count.**
`qwen35/sft_rewardhacks.py` says "1,073 short harmless tasks"; the HuggingFace
dataset card says roughly 1,070 rows; the paper says "over a thousand". Wiki uses
1,073, the number the training script actually loaded, on
[[paper-school-of-reward-hacks]]. Settled by a row count of the loaded split.
Note the HF card's citation field pointed at arXiv:2108.07732, an unrelated
program-synthesis paper; the real identifier 2508.17511 was verified directly.

**S25. `power_seeking`'s nearest zoo adapter, across the two alignment arms.**
`qwen35/analysis/alignment_geometry.json` gives `selfish` at 72.64703452049417;
`alignment_geometry_aligncommon.json` gives `crooked` at 72.6923986134726. The
two candidates are within 0.1 degrees in both arms, so it is a tie-break, not a
disagreement about direction; `sycophantic` and `obsequious` give `pleasant` in
both, `corrigible` gives `liberal` in both. Both shown on the alignment trait
pages, neither called current.

---

**S26. The ten Big Five factor adapters: 0 of 10 or 8 of 10 on own-trait dominance.**
`inspect-personality-evals` said, until 2026-09-10, that the ten factor-level
adapters "scored 0 of 10 on own-trait dominance under the blind judge";
[[ocean-dials-replication]] arm D and [[bigfive-factor-adapters]] give 8 of 10
with all ten moving the right way, from `qwen35/analysis/spider.json#bigfive`
(480 judged records, 0 failed calls, base decoding byte-identical to
`eval_100traits.json`). No file supports the 0 of 10; it was an unsourced prose
aside written while describing the BFI's 2 of 5, and has been corrected in
place. The sourced figure is 8 of 10. Found by the 2026-09-10 post critique
([[post-critique-2026-09-10]]).

---

**S27. The size of a materialised delta: 81M or 3.57 billion.**
[[geometry-overview]] says "A materialised `dW` is about 81M parameters per
adapter across the 248 modules"; `qwen35/analysis/fisher_norms.json#targeted_params`
gives 3,569,090,560 for the parameters of the same 248 targeted modules, which is
what a dense `dW` over them would hold. 81M is close to what the LoRA *factors*
occupy at rank 64 (64 times the sum of input and output widths over the modules),
so the likeliest reading is that the overview page's sentence counts the factors
and calls them `dW`. Not verified against module shapes. The post quotes the
dense figure. Found by the 2026-09-10 re-check ([[post-critique-2026-09-10]]).

**S28. Does curvature predict how far the judged persona moves? +0.37 on one
sphere, +0.02 on the other.**
`qwen35/analysis/fisher_norms.json#correlation_with_degeneration.sphere_judged_profile_distance`
gives Spearman **0.3706669239179368**, permutation p 0.0014499275036248187, for
the Fisher norm F against the distance of each point's judged five-scale profile
from the 72-point centroid - over the 72 points of the **factor** sphere, with F
fitted from |alpha| <= 0.5.
`qwen35/analysis/sphere_isokl.json#dose_covariate.dose_vs_centroid_distance_pc_alpha1_5`
gives Spearman **+0.016013891568589622**, permutation p 0.8899110088991101, for
the same statistic over the 72 points of the **principal-component** sphere, with
the covariate measured directly as the bf16 KL per token at the alpha that was
steered. Neither supersedes the other: they are different lattices in different
three-dimensional subspaces, and the second uses a strictly better covariate that
was not available for the first (F has never been measured for the 72 PC-sphere
directions - `fisher_norms.json`'s `sphere_S*` keys are the factor sphere, cosine
1.0 against `phase10_runs/sphere_spec_fa.json`). The wiki quotes both, each
scoped to its own sphere. **What would settle it:** measuring F, or the KL at a
common alpha, for the 72 factor-sphere points on the same footing as the PC ones,
and re-running both correlations with the same covariate definition. On the
iso-KL sphere the same covariate predicts displacement *negatively* at -0.4022
(p 0.0005), which suggests the alpha-1.5 null is a cancellation of two opposed
effects rather than an absence of structure. See [[sphere-sweep-iso-kl]].

---

## Not contradictions, but easily conflated

Four places where two correct numbers describe different things, listed because
they are the most likely spots for a blog post to quote the wrong one.

- **The hole's angle: 68.9 and 52.5 degrees.** 68.9
  (`qwen35/analysis/direction_gaps.json#alien_k5.deg` = 68.93287342956332) is the
  gap in the full sketch space with each adjective treated as a line; 52.5
  (`qwen35/analysis/alien.json#alien_k5.gap_deg` = 52.51479642189155) is the same
  direction's gap inside the top-5 principal subspace, and 52.5's coefficients
  are what was steered and transplanted. `qwen35/analyse_actspace.py` line 177
  prints the activation result beside a hard-coded "weight space 68.9".
  [[hole-words]].
- **Four PC1/PC2 variance spectra for one cloud.** 12.7 / 11.3 (134-trait
  **sketch** basis, `analysis/blog_data.json#viz.var`, used by the map caption);
  13.36 / 11.21 (the 100-marker scree curve,
  `analysis/scree_null_matched.json#real`, used by the figure beside it); 12.4946
  / 10.9847 (the exact 134-trait double-centred spectrum,
  `results/fa_qwen35.json#pca_from_gram.centered_var_pct`); 0.134 / 0.1122
  (`analysis/geometry_stage1.json#explained_var_top10`). All four on
  [[pca-and-scree]] with their sources.
- **The permutation p at three resolutions.** `results/decomposition.json`
  reports 4.999750012499375e-05; `results/compare_nulls_output.txt` prints
  0.00005; the `PHASE3_VERDICT.md` replication table prints 0.00010. One
  permutation floor, three roundings. [[null-controls]].
- **Reward margins in two files.** `qwen35/phase5_margins.json` sets
  `attribution_possible: false` and leaves every `final_margin` null, because the
  training log interleaves about four concurrent containers with no trait label;
  `qwen35/results/runmeta_sweep.json` carries a `reward_margin` per trait,
  written by the training job itself. They do not conflict - one is a failed log
  parse - but they read as if they do. The trait pages quote the runmeta value.
- **Prompt-window additivity, raw or trait-specific.**
  `qwen35/ACTSPACE_RESULTS.md` writes "residual 0.12 raw / 0.28 specific, weights
  0.97 / 0.99" in one clause, mixing a raw residual, a trait-specific residual
  and raw least-squares weights without saying so; the blog builds its "residual
  33%, weights 0.96 and 0.92" from
  `analysis/actspace_cross_geometry.json#prompt_specific["different factor"]`.
  Both are consistent with the JSON. [[actspace-cross]] tabulates both.

---

Related: [[superseded-claims]], [[open-questions]],
[[superseded-geometry-claims]], [[how-to-read]].

**S29. `top1_stage1_finds_own_stage2` counts the other direction, 42 or 59.**
`qwen35/analysis/stage2_structure.json#stage1_x_stage2_exact.top1_stage1_finds_own_stage2`
= **42**, `mean_rank` 9.537313432835822, and [[stage-two-structure]] renders it as
"a stage-one adapter's nearest stage-two adapter is its own trait for 42 of 134".
`qwen35/analyse_stage2_structure.py:116` computes
`ranks = [int(1 + np.sum(C12[:, j] > C12[j, j])) for j in range(134)]`, which
ranks down the **columns**: each *stage-two* adapter among the 134 stage-one
ones. The 2026-09-11 activation-weighted run rebuilt the same cross-Gram from
the adapters directly (rather than from the persona Gram) and reproduces the
cosines to 2.9e-11; on that identical matrix the column direction gives exactly
**42 / 9.537313432835822** and the row direction - the one the key name and the
sentence describe - gives **59 of 134, mean rank 7.514925373134329**
(`qwen35/analysis/act_gram_stage2.json#validation.vs_stage2_structure_stage1_x_stage2_exact`
and `#arms.frob.cross_stage_raw`). So the value is right for a statistic whose
name is wrong. The wiki quotes 42 with the direction stated correctly on
[[activation-weighted-gram-stages]] and leaves [[stage-two-structure]]'s number
standing with this entry attached. Settled by renaming the key, which is an edit
to a raw source and therefore not made here.

