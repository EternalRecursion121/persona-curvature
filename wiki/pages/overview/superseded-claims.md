---
title: Superseded claims
summary: Every claim the project made and later reversed or corrected, dated, with the value as first stated, the value that replaced it, and the page that holds the current version; opens with the sentences on the live blog page that are wrong as printed.
status: current
sources:
  - qwen35/build_blog_page.py
  - qwen35/PHASE3_VERDICT.md
  - qwen35/plan.json#ledger_defect_note
  - qwen35/paper_notes.md
  - qwen35/analysis/scree_null.json#perm_mean_abs_diff
  - qwen35/analysis/scree_null_matched.json#perm_mean_abs_diff
  - qwen35/analysis/pc_loadings.json#seed
  - qwen35/results/fa_qwen35.json#n_factors.parallel_analysis_centred
  - qwen35/results/fa_qwen35.json#n_factors.parallel_analysis_uncentred
  - qwen35/analysis/distil_data.json#corrections
  - qwen35/analysis/actspace_cross_geometry.json#resp_specific
  - qwen35/traits_primary.json
  - qwen35/phase2_runs/archive/phase5_sweep_134.json
  - qwen35/phase10_runs/zoo40_meter.log
  - .garden/journal/2026-09-05.md
  - qwen35/analysis/matched_dose_steering.json
  - qwen35/analysis/syc_forecast.json#compliance.vs_random_paired
  - qwen35/analysis/syc_forecast.json#bigfive.agreeableness_spearman_vs_axis_score
  - qwen35/phase10_runs/dose_calib.json
  - /home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md
last_verified: 2026-09-10
tags: [overview, superseded, corrections]
---

# Superseded claims

This is the project's record of its own reversals: every claim that was stated
and then corrected, withdrawn or replaced, across the whole project. Each entry
gives the claim as first stated (value, where, when), what replaced it (value,
where, when), and the wiki page that holds the current version.

Two companion pages. [[source-contradictions]] holds disagreements where
**neither** side is known to supersede the other. [[open-questions]] holds what
is unconfirmed, never run, or still a decision. The geometry section keeps its
own detailed record at [[superseded-geometry-claims]]; section 3 below is a
pointer list into it rather than a second copy.

Two conventions. Numbers are quoted verbatim from the file that stores them, at
that file's precision. Where a claim reversed more than once, every position is
listed with its date, because a reader quoting "the project's position" must say
which one.

---

## 1. Live page statements that need changing

`qwen35/blog_page/index.html`, built by `qwen35/build_blog_page.py`, is the
authoritative current state of the results. These are the places where what it
prints is wrong, under-specified, or not backed by a file. They are separated by
what a writer has to do about each.

### 1a. Wrong as printed - change the sentence

| # | what the page says | what the source says | current page |
|---|---|---|---|
| L1 | "preference pairs where the chosen response is in character and the rejected one is **the model's default**" (`qwen35/build_blog_page.py` line 326) | `qwen35/gen_pairs.py` docstring and implementation: rejected is "reply by a character at the OPPOSITE pole of the same dimension", and explicitly "The rejected side is never an unconditioned/base response." | [[dpo-pair-generation]] |
| L2 | the real and permuted scree curves are "0.18 percentage points apart on average over the first twelve components" | that is `qwen35/analysis/scree_null.json#perm_mean_abs_diff = 0.1798544716598768`, the **superseded** null file. The file the figure is drawn from, `qwen35/analysis/scree_null_matched.json#perm_mean_abs_diff`, is **0.10336514494213549**. | [[pca-and-scree]] |
| L3 | the seed floor's angle, "89 degrees" | computed from the stale `qwen35/analysis/pc_loadings.json#seed.same_trait_deg = 89.04958220635345`, which goes with the stale cosine `#seed.same_trait_cos = 0.01658715905967511`. The same page's prose already quotes the matched **+0.01806** (`qwen35/PHASE3_VERDICT.md` line 275, 2026-09-03 addendum). The angle was never recomputed. | [[seed-floor]] |
| L4 | "The 134 trait adapters and **the preference data they were trained on** are on Hugging Face" | no uploader in the repository publishes `qwen35/data_common/`. `qwen35/upload_datasets.py` publishes the stage-two introspection transcripts only. Either an upload route exists outside the repository or the sentence is wrong. | [[hf-artefacts]] |

### 1b. Correct but needs a qualifying clause

| # | what the page says | the clause it needs | current page |
|---|---|---|---|
| L5 | the prompts track the weights at **0.70** in one paragraph and **0.74** two paragraphs later | they are the same quantity under two centrings of the weight side: 0.705 correlates the **raw** weight Gram (`qwen35/analysis/actspace_geometry.json#windows.resp.primary.r_centred` = 0.704599368961925), 0.737 the **double-centred** one (`qwen35/analysis/actspace_adapters_geometry.json#windows.resp.curve[16].r_PW` = 0.7368952052412984). See [[source-contradictions]] entry S1. | [[actspace-persona-vectors]], [[actspace-adapters]] |
| L6 | the alien direction is "68.9 degrees from the nearest word" (hard-coded in `alien_card()`) | 68.9 is the gap in the **full sketch space** with each adjective treated as a line (`qwen35/analysis/direction_gaps.json#alien_k5.deg` = 68.93287342956332). The coefficients actually steered and transplanted are `alien_k5`'s, whose gap in the **k=5 principal subspace** is `qwen35/analysis/alien.json#alien_k5.gap_deg` = 52.51479642189155. Name the space. | [[hole-words]] |
| L7 | the markers are "twenty per factor, both poles" | true of the count, misleading on the split. `qwen35/traits_primary.json` has Emotional Stability at **6 positive and 14 negative**; the other four factors are 10/10. Confirmed by `qwen35/analysis/geometry_stage1.json#unwhitened_loo_keying.EmotionalStability` (`n_pos` 6, `n_neg` 14). Any per-factor statistic that assumes balanced keying is affected. | [[goldberg-100-primary-traits]], [[literature-big-five]] |
| L8 | random subspaces contain "about 1%" of the adapter's shift (`qwen35/ACTSPACE_RESULTS.md` says "0-1%") | `qwen35/analysis/actspace_adapters_geometry.json#windows.resp.containment_null` is 0.0019 / 0.0038 / 0.0072 / **0.0148** at k = 5/10/20/40, so the top-40 null is 1.5%, and it is a **single random draw**, not a distribution. The observed 41%/64% is unaffected. | [[actspace-adapters]] |

### 1c. Printed but not backed by a file

These are not known to be wrong. They are on the page as prose or as hard-coded
strings, with no output file storing them, so a blog post copying them is
copying a builder script.

| # | figure on the page | provenance | current page |
|---|---|---|---|
| L9 | sketch cosines track exact cosines at **r = 0.9996** | prose in `qwen35/build_blog_page.py` line 1584; `qwen35/build_monitor_page.py` line 209 says **0.99944** for the same check; no output file exists. This is the load-bearing justification for reading sketch angles as weight-update angles. | [[geometry-overview]] |
| L10 | analytic and finite-difference scoring agree at **0.9999992** | `build_blog_page.py` line 1411 reads `corr = v.get("corr", 0.9999992)` and `qwen35/analysis/align_validate.json` has **no `corr` key**, so the page always prints the hard-coded fallback. The 24 x 3 raw rows are in the file. Recomputed 2026-09-10 from those rows: r 0.9999992 at eps 0.003, so the literal was right for that step size. | [[scoring-identity]] |
| L11 | FA_Warmth "7.3x on the eighteen clean prompts"; FA_Imagination "4.2 to 7.8 on its fourteen clean prompts" | hard-coded strings in `build_blog_page.py`. The underlying damage correction is in `qwen35/analysis/qual_fa_notes.md`, which does not store the recomputed selectivities. | [[factor-warmth]], [[factor-imagination]] |
| L12 | the five named axis directions are "intact over alpha -2, -1, 1, 2" | `qwen35/analysis/qual_axes.json` has no `damage_markers.per_alpha.looping` for the five `axis_*` entries, so `qwen35/analysis/blog_data.json#degen` is empty for them and `stats_for()` prints from an **absent record**, not from a measured zero. `qual_axes_notes.md` says the counts were made; they are not in the JSON. | [[steering-results]] |
| L13 | PC4 sits at cosine **0.46** to the personality axis | blog-page prose; no JSON key found. | [[factor-pc4]] |

### 1d. Published Hugging Face cards that are stale

Not the blog page, but public and wrong, and flagged as owed work by
`qwen35/POST-BATCH3-TODO.md` item 4 and `qwen35/analysis/hf_dataset_audit.md`
section 5.

- `qwen35/zoo_page/MODEL_CARD.md`: "45 adapters", the "0.886 +/- 0.111 across
  all 45" band and the "14 of 45" row-drop statistic. The zoo is 134.
- `qwen35/zoo_page/DATASET_CARD.md`: "51 constitutions", "194 of 204 files",
  `size_categories: 100K<n<1M`, and "so `sft_data` is often smaller than
  12,000" - `hf_dataset_audit.md` parsed four files at exactly 12,000 rows,
  including two the card names as heavy losers, and concludes the drop happens
  at the SFT tokenizer, which is what `MODEL_CARD.md` says.

Current pages: [[hf-artefacts]], [[stage-two-introspection]].

---

## 2. Zoo construction

**Z1. rsLoRA on, then off.**
*First stated:* `qwen35/plan.json#defaults.use_rslora = true`, with
`#defaults.rslora_note` calling rsLoRA the default and plain LoRA an ablation
arm. `qwen35/paper_notes.md` section 3.2 (2026-08-19) then argued at length
against the project's own rsLoRA setting, calling it the largest silent
divergence from both source papers.
*Replaced by:* all 134 records in
`qwen35/phase2_runs/archive/phase5_sweep_134.json` read `use_rslora: false`,
`expected_scaling: 2.0`; `qwen35/RUNNER_TASK.md` says "plain LoRA alpha 128 rank
64 (use_rslora False)"; `qwen35/PHASE3_VERDICT.md`'s 2026-09-03 addendum says
"rslora off".
*Current:* [[stage-one-training-config]], [[zoo-training-recipe]].

**Z2. Pair volume and hyperparameters in the plan.**
*First stated:* `qwen35/plan.json` phase 1 "214 pairs per trait" and phase 2
"rank 32 / LR 5e-6", both flagged in `paper_notes.md` section 3.11.
*Replaced by:* 500 pairs targeted, **445** trained, at rank 64 and LR 5e-5.
*Current:* [[shared-prompt-pool-445]], [[stage-one-training-config]].

**Z3. The stage-two adapter count, six times.**
*Dates each figure was correct:* 41 (`qwen35/upload_adapters.py` docstring), 45
(`qwen35/zoo_page/MODEL_CARD.md`, twice), 51 and 61
(`qwen35/POST-BATCH3-TODO.md`), 103 (`qwen35/upload_zoo_batched.py` docstring,
which gives `stage2_introspection/ 103`, `persona_merged/ 103`, `persona_exact/
103` against `stage1_dpo/ 134`), 119 (`qwen35/analysis/hf_dataset_audit.md`),
**134** (`qwen35/analysis/merge_audit.json`, `build_blog_page.py`, and the OCT-2
results files: 100 primary plus 34 lexicon).
*Current:* 134. See [[stage-two-introspection]], [[hf-artefacts]]. The two
published cards are the stale ones (section 1d).

**Z4. Transcript repository coverage.**
*First stated:* `qwen35/analysis/hf_dataset_audit.md` (2026-08-29): 50 of 134
traits on the repo, 194 of 536 files, one trait quarantined and two partial.
*Replaced by:* `qwen35/phase10_runs/upload_datasets.log` (2026-08-30 10:30):
"COMPLETE: all 536 expected files present... nothing quarantined". The audit file
is not annotated to say so.
*Current:* [[hf-artefacts]].

**Z5. "Nothing described here is published."**
*First stated:* `qwen35/PENDING-CONTENT-REVIEW.md`.
*Replaced by:* the same 2026-08-30 upload log, which shows the previously
quarantined rows uploaded as "previously reviewed and cleared", plus 60 written
clearance decisions in `qwen35/phase10_runs/adjudications.json`. Whether that
reflects a decision by Samuel is not recorded in any file; see
[[open-questions]].
*Current:* [[hf-artefacts]], [[zoo-build-governance]].

**Z6. The spend ledger: $195.90, then ~$370.65, then the meter.**
*First stated:* $195.90 realised, being the sum of `qwen35/plan.json`'s closed
`phases[*].actual` values as of 2026-08-23 00:05, quoted against a $1000
ceiling.
*Replaced by (2026-08-23 00:05):* ~$370.65 measured -
`qwen35/plan.json#ledger_defect_note` gives "Modal $308.83 + OpenRouter $61.82",
and names three mechanisms for the $174.75 gap (billed is not incurred; the
app-name collision that billed the 134-run sweep under phase 2; a phase costs its
failures - phase 3 recorded $111.04 against $206.84 measured because of the
`.FAILED-datarace` and `.RSLORA-INVALID` predecessors).
*Superseded again by the container-minute meter:* **$2,240.52 of $2,400**, the
standing reading in `qwen35/phase10_runs/zoo40_meter.log` through
2026-09-07T16:25:05Z (first written at line 25572, 2026-09-05T21:43:37Z). It is a
meter, not a billing read: `PRIOR=46.90` plus integrated GPU-hours at a flat
$2.10/hour from polling `modal container list`, pricing CPU containers at the
A100 rate. **It is live and has started moving again** since the full-OCT
replication launched on 2026-09-07 (see [[open-questions]]); quote it with its
read time.
*Note on the file:* `plan.json`'s `ledger_defect_note` still says "Sum of
`actual` = $195.90", but the file's own `phases[3].actual` now reads
**206.885** where the note describes phase 3 as $111.04 recorded. The $195.90
sentence no longer describes the file it sits in.
*Current:* [[zoo-spend-ledger]], [[costs]].

**Z7. The budget ceiling itself moved three times.**
$200 (`CONTEXT.md` section 6, "against the $200 set") -> $1000
(`qwen35/plan.json#budget`) -> $2400 (`qwen35/zoo40_meter.sh`, `BUDGET=2400.00`,
"raised 2026-08-29 against a $5k grant"). `qwen35/HANDOVER.md` still frames spend
against $1000 and was not updated. Anyone quoting "of the budget" without a date
is quoting one of three denominators.
*Current:* [[costs]], [[zoo-spend-ledger]].

**Z8. The bare "$195.90 actual against a $1000 ceiling" was deleted, not
annotated.**
`qwen35/HANDOVER.md` commit `f2e02b4` removes the sentence and replaces it with a
SUPERSEDED banner, on the stated principle that "a caveat does not travel with a
number".
*Current:* [[costs]], [[lesson-a-phase-costs-its-failures]].

**Z9. The enumerated constitution anchor.**
*First stated:* every constitution ended with an anchor paragraph that named one
Goldberg marker per Big Five factor.
*Replaced by (2026-08-19):* a generic anchor block, because naming the markers
could have manufactured the Big Five result under test. The enumerated form was
retained as an ablation arm and, as far as any file shows, never trained (see
[[open-questions]]).
*Current:* [[constitution-anchor-revision]].

**Z10. The secondary trait source.**
*First stated:* draws 1 and 2 sampled Allport and Odbert's 1936 list; draw 1 had
no semantic constraint and returned 14 of 40 unusable words, draw 2 added
WordNet-gloss filters and was worse.
*Replaced by:* draw 3 from Condon et al.'s trait-descriptive set, 5,636 to 2,303
to a 40-cluster k-means draw. Both discarded files are still on disk as
`qwen35/traits_secondary_DISCARDED_draw1.json` and `..._draw2.json`.
*Current:* [[lexicon-secondary-draw]], [[discarded-secondary-draws]].

**Z11. The persona merge.**
*First stated:* the published personas were assembled with PEFT's linear adapter
combination.
*Replaced by:* PEFT sums LoRA factors, not deltas, so the merged delta carries a
cross term worth 0.795 to 0.823 of its Frobenius norm across all 134 traits
(`qwen35/analysis/merge_audit.json`). An exact concatenation merge was measured
and published alongside.
*Current:* [[persona-merge-correction]].

---

## 3. Geometry

[[superseded-geometry-claims]] is the full record, with both texts quoted for
each reversal. One line each here, with the section number to open.

- **The trait-level withdrawal, reversed and re-withdrawn and discharged - four
  dated positions on one claim** (2026-08-22 addendum, 2026-08-23 body,
  2026-08-24 addendum, 2026-09-03 addendum plus the blog page).
  [[superseded-geometry-claims]] sections 1 and 1a. Current: [[seed-floor]].
- **The preregistered +0.2447 bar was in the wrong units** and "could not have
  passed under any hypothesis". Section 2. Current: [[seed-floor]],
  [[lesson-bar-in-the-wrong-units]].
- **The verdict's own downstream flag on phase 7** ("assumed transferable trait
  directions... needs re-specifying") - "That was wrong on two counts."
  Section 3. Current: [[steering-results]].
- **`qwen35/results/compare_nulls_output.txt`'s "EVERY EFFECT ABOVE IS VOID"** is
  overridden by its own author and the file was never regenerated. Section 4.
  Current: [[null-controls]], [[lesson-coordinates-are-not-structure]].
- **The objective mismatch in all 240 control adapters**, and the seed floor
  moving **+0.01659 -> +0.01806** on the matched retrain (2026-09-03), with the
  null arms following on 2026-09-05. Section 5. Current: [[seed-floor]],
  [[null-controls]], [[lesson-objective-must-travel]].
- **Two thrown-away null training runs**, `.RSLORA-INVALID` and
  `.FAILED-datarace`. Section 6. Current: [[null-controls]].
- **The elbow reading was backwards**: six components is conservative, not
  generous. Section 7. Current: [[pca-and-scree]], [[explainer-elbow-figure]].
- **The stored scree `real` curve is a 100-trait sketch spectrum** and was never
  regenerated from `results/gram_sweep.npz`. Section 8. Current:
  [[pca-and-scree]]. It is also an open item - see [[open-questions]].
- **Stale numbers still sitting in current data files**
  (`pc_loadings.json#seed`, `scree_null.json#perm_mean_abs_diff`). Section 9;
  they are also blog-page items L2 and L3 above.
- **File-to-file contradictions** the geometry page records without resolving.
  Section 10; each is written up with a settlement test in
  [[source-contradictions]].
- **Numbers whose provenance is prose only.** Section 11; the blog-facing subset
  is section 1c above.

Three geometry reversals that page does not cover:

**G1. PC4 named as a sycophancy axis.**
*First stated:* PC4 was described as a sycophancy axis.
*Replaced by:* the blog page - "naming it after sycophancy was a mistake"; its
highest loadings are `timid`, `self-pitying`, `fearful`, `guilty`. The
replacement claim was then preregistered in `qwen35/PREREG_alignment.md` as
prediction 1 and held.
*Current:* [[factor-pc4]], [[alignment-traits-geometry]].

**G2. `qwen35/analysis/sphere_layout.json` and `qwen35/results/umap_embeddings.json`.**
The sphere layout is superseded by `sphere_page.json`; the UMAP embeddings
(2026-08-23) read the pre-matched null Grams and are historical.
*Current:* [[umap-and-layouts]].

**G3. The 2026-09-05 stage-2 addendum reports 15/15 top-1 and does not report
the separation.**
Stage-one separation is positive and the 2026-08-22 addendum makes "perfect
separation, no overlap at all" one of its three structural facts. At stage two
`qwen35/analysis/crossseed_arms_stage2.json#[0]` has `min_same` 0.04248138800846624
**below** `max_off` 0.05140579106757132, giving `sep` -0.00892440305910508. So
the separation property does not carry to stage two, and no document says so.
*Current:* [[stage-two-second-seed]]; also an open question.

---

## 4. Behaviour

**B1. The 200-token steering corpus.**
*First stated:* `qwen35/phase10_runs/steer_results.json` and
`judged_steer.json`, plus the pre-aggregated curves preserved at
`qwen35/analysis/distil_data.json#steer`.
*Replaced by (2026-08-29):* `steer_results_fix.json` / `judged_steerfix.json`.
Qwen3.5's chat template defaults thinking ON, so all 1,512 original generations
were truncated reasoning preambles; the corpus was regenerated with
`enable_thinking=False` at 512 tokens.
*Current:* [[thinking-default-withdrawals]], [[steering-results]],
[[lesson-thinking-default-trap]].

**B1a. "Degeneration under steering in this project is overwhelmingly a
negative-alpha phenomenon."**
*First stated (2026-09-09):* [[fisher-norms]], from
`qwen35/analysis/fisher_norms.json#comparisons.sign_asymmetry_of_degeneration` -
over 22 directions with both signs, all 22 loop at least as much at alpha -2 as
at +2 and 18 have a loop rate of exactly zero at +2.
*Qualified (2026-09-10):* the observation is a fact about a fixed **alpha** grid,
not about sign. At matched Fisher dose the mean loop rate is **0.075** on the
suppressing sign and **0.1375** on the amplifying sign over ten directions, and
**0.075 against 0.2167** over the five factors; the largest loop rate in that run
is FA_Arousal's amplifying side at **0.625**
(`qwen35/analysis/matched_dose_steering.json#summary`). Where a direction's
amplifying sign is the flatter one, matching the dose pushes it to a larger alpha
and it degenerates there. The original counts stand for the corpus they were
measured on.
*Current:* [[matched-dose-steering]], [[fisher-norms]].

**B1b. The suppress-versus-amplify asymmetry as evidence for a mechanism.**
*First stated:* `qwen35/POST_DRAFT.md` and [[steering-results]] - at alpha
plus or minus 2, "four of five suppress harder than they amplify", offered
alongside Gradient Atoms' reading that suppression is structurally easier
([[paper-reading-2026-09-09]]).
*Corrected (2026-09-10):* the asymmetry is headroom. Matching the Fisher dose
takes the raw suppress/amplify ratio from **1.8957528957528955** to **1.6455696202531644** (8 of 10
directions to 6 of 10), and dividing each change by the room that existed in the
direction it moved - `delta / (7 - base)` upward, `delta / (base - 1)` downward -
takes it to **0.9261912176364205** published and **0.7946215813581846** at matched dose, i.e.
amplification is the stronger move per unit of room in both arms
(`qwen35/analysis/matched_dose_steering.json#summary.all`,
`#published_equal_alpha_summary.all`). The base sits at 5.916666666666667 of 7
on Conscientiousness and 5.625 on Intellect in the run's own base condition, so
there is four to five times as much room downward as upward. The published table
is not wrong; it is not independent evidence for the mechanism story, and
[[post-draft]] should not present it as if it were.
*Current:* [[matched-dose-steering]].

**B1c. A published steering alpha is a full-strength perturbation.**
*Implicit throughout:* `qwen35/steer_fix.py` adds the increment into bf16
parameters and every published steering alpha was read as the intended dose.
*Measured (2026-09-10):* the Frobenius **norm** survives (retention 0.9216 to
1.033505578742928, median 1.0027071769953744), but the **KL** does not:
`KL(bf16) / KL(exact)` at the same nominal alpha runs 0.7117708405618217 to
0.9782241941431974 with a median of **0.9059646365364478** over 120
(direction, alpha) cells, and the shortfall depends on the alpha - the median is
**0.9423571405594928** at |alpha| 2 and 0.7510758772738881 at |alpha| 1
(`qwen35/analysis/matched_dose_steering.json#bf16_vs_exact`,
[[matched-dose-steering]]). Quantisation keeps the length and spends part of it
on isotropic noise. The published alpha plus or minus 2 therefore delivered
about 94 per cent of its intended dose and an alpha of 1 about 75 per cent. No published
result reverses on this; the correction is to the unit, and it compounds with
the `ref` correction recorded in the 2026-09-08 audit row of section 7 below
("alpha is measured in units of one trait adapter's worth of weight change").
*Current:* [[matched-dose-steering]], [[fisher-norms]].

**B2. The `zoo-steer2` and `zoo-steer3` runs.**
*Replaced by (2026-08-30):* `zoo-steerfix2` / `zoo-steerfix3`. No results file
survives either of the originals.
*Current:* [[steering-results]].

**B3. `qwen35/distil_page/index.html` is withdrawn.**
*First stated:* a built page titled "Where personality lives in a model's
weights" - module holography, the DPO-twice claim, variance deflation, PCs, FA
and steering.
*Replaced by (2026-09-01):* a withdrawal notice at the same URL, with the
original preserved at `.withdrawn-backup`; its content is carried by
`findings_page`, `monitor_page` and the 22 `direction_pages`.
*Current:* [[distillation-check]], [[built-pages-inventory]].

**B4. "The persona adapters contain the DPO adapter twice."**
*First stated:* `qwen35/analysis/distil_data.json#surprises[0]`, with
`proj(persona - stage-1, unit stage-1)/||stage-1|| = 1.001 +/- 0.015` across 100
traits against a mismatched control of 0.089 +/- 0.175.
*Replaced by (2026-09-01):* the `distil_page` withdrawal notice - "It was a units
error in our own analysis code. The sketch never read `alpha/r` from each
adapter's config... The DPO strength in those adapters is correct." The data
file still carries the withdrawn claim.
*Current:* [[distillation-check]].

**B5. Stage 2's relation to each trait's own direction.**
*First stated (2026-08-29 14:19):*
`qwen35/analysis/live_results.json#hyp[1].detail` - "cos(SFT increment, DPO
adapter) is +0.225 same-trait vs +0.015 mismatched, a 15x gap. Stage 2
re-encodes and amplifies each trait's own direction."
*Replaced by:* `qwen35/analysis/distil_data.json#corrections[0]` - "That gap is
entirely the deterministic 2x copy above. After removing it the correlation is
+0.0002." `live_results.json` was never updated.
*Caution:* the correction itself rests on the 2x copy, which B4 then withdrew as
a units error. Only the residual-orthogonality figure (+0.0002) and the RSA
structure claim at `distil_data.json#surprises[2]` should be carried forward.
*Current:* [[distillation-check]], [[stage-two-geometry]].

**B6. "Stage 2 sharpens factor clustering, 0.357 -> 0.548."**
*Replaced by:* `qwen35/analysis/distil_data.json#corrections[1]` - "0.508 vs
0.548 - a 4-point gain, not 19. At k=1 the direction reverses."
*Current:* [[distillation-check]], [[stage-two-geometry]].

**B7. `qwen35/zoo_page/index.html` is a 51-adapter readout** (`TOTAL = 51`),
superseded 2026-08-24 by `qwen35/site_traits/` at 134.
*Current:* [[built-pages-inventory]].

**B8. "13 of 24 AI-identity disclaimers belong to `mean_assistant_axis`."**
*Replaced by:* `qwen35/analysis/qual_pc.json#directions[PC2].surprise` - they
belong to PC2's negative pole. On the corrected corpus the
`mean_assistant_axis` negative side is flat at baseline (manual counts
8/7/8/7/4/3/1 across alpha -4/-2/-1/0/+1/+2/+4).
*Current:* [[qualitative-notes]], [[factor-pc2]].

**B9. PC1's named scale.**
*First stated:* `qwen35/PREREG_steerfix.md` registers PC1 as an Agreeableness
claim - "PC1 - affect-centring vs procedural detachment... Agreeableness slide
survives".
*Replaced by the measurement:*
`qwen35/analysis/steerfix_replication.json#PC1.named` is `"Conscientiousness"`,
and `qwen35/analysis/qual_pc.json#directions[PC1].judged_profile` repeats "named
Conscientiousness". The replication file is current; the damage half of the
prediction did replicate.
*Current:* [[steering-results]], [[factor-pc1]].

**B10. axis_Intellect "buys abstraction with Conscientiousness".**
*First stated:* `qwen35/PREREG_steerfix.md` - "largest range, buys abstraction
with Conscientiousness... Intellect increasing, Conscientiousness decreasing."
*Replaced by:* `qwen35/analysis/qual_axes.json#directions[axis_Intellect].surprise`
- judged Conscientiousness on this axis **rises** with alpha (+0.20 slope, 2.54
at -4 to 5.83 at +1). "The debt in the corrected data runs the other way."
*Current:* [[steering-results]], [[factor-axis-intellect]].

**B11. The STEER134 coherence-collapse argument - downgraded, not superseded.**
*First stated:* `qwen35/analysis/manifold_ideas.md` builds a coherence-collapse
argument on `qwen35/results/steer134_judged.json` (fa1 0.42/10 at +2, random1
8.5 at +2, pc1 9.25 at -1 against 2.0 at -2).
*Status now:* that file has `coverage.complete: false`, with 11,010 of 28,664
units unjudged and many cells at n = 4-7 against `n_expected` 12, against only
two random controls. The argument is marked **unconfirmed** on
[[steering-results]]. Nothing replaced it; it should not be cited as a finished
result. See [[open-questions]].

---

## 5. Activation space

**A1. "Prompt wins 16 of 22."**
*First stated:* the first cross analysis, run on raw (uncentred) activation
vectors, found the combined model nearer the prompt's persona than the adapter's
in 16 of the 22 opposite-keying conflict pairs - that a system prompt largely
overrides a trained trait. `.garden/journal/2026-09-05.md`, verbatim: "First
analysis on raw vectors said 'prompt wins 16/22' - an artefact of the shared
component (all pairwise cosines ~0.8)."
*Replaced by (2026-09-05):* the trait-specific analysis gives **8 of 22** and
the conclusion becomes "neither wins".
`qwen35/analysis/actspace_cross_geometry.json#resp_specific`;
`qwen35/ACTSPACE_RESULTS.md`; the blog paragraph is built from `resp_specific`
(`build_blog_page.py`: `X = J["resp_specific"]`).
*Lesson filed the same day:* any comparison of steered activations must remove
the common shift first, or it measures "was the model steered" and not "toward
what".
*Current:* [[actspace-cross]], [[prompting-versus-training]].

**A2. "A character English has no word for" - withdrawn.** The activation-space
transplant is part of why. Full entry at F5 below; current page [[hole-words]].

**A3. A stale docstring.** `qwen35/analyse_actspace_cross.py` line 12 says
"Prompt alone is 1.00 by construction, adapter alone was 0.62", but that
script's own trait-specific block prints **0.70**
(`actspace_cross_geometry.json#resp_specific_adapter_alone` = 0.7044131755828857).
Source comment only; nothing downstream reads it.
*Current:* [[actspace-cross]]; the 0.62/0.70 pair is [[source-contradictions]]
entry S2.

---

## 6. Literature notes

**L-1. `qwen35/paper_notes.md` section 3 is a 2026-08-19 plan, and the run
overtook at least two of its items.** Anyone reading that section should be
warned at the top.

- **3.2, rsLoRA.** The note calls the planned rsLoRA the largest silent
  divergence: scaling 16.0 against both papers' 2.0, an 8x larger functional
  step, and warns that the project's Frobenius norms will land about 8x above
  Persona Cartography's 6.08-6.53 band, invalidating any comparison to their
  Table 1. **The zoo turned rsLoRA off** (Z1 above). The divergence is closed and
  the norms are comparable - which makes the missing norm table a cheap open item
  (see [[open-questions]]).
- **3.5, prompt volume.** The note argues against a planned "500 pairs per trait,
  no general-prompt mix", giving roughly 16 optimizer steps at batch 32. The
  executed zoo ran **445 pairs and 13 optimizer steps** on one pool shared across
  every trait. The argument survives and sharpens; the numbers in section 3.5
  must not be quoted as the zoo's.

*Current:* [[recipe-vs-source-papers]], [[open-character-training-paper]],
[[stage-one-training-config]].

**L-2. `paper_notes.md` section 4 lists eight things that could not be
retrieved. Four are now retrievable** (checked 2026-09-07):

| item | then | now |
|---|---|---|
| 1. Persona Cartography's code | "github.com/persona-cartography/monorepo returns 404" | the GitHub repo is `persona-cartography/persona-cartography` and a full checkout sits at `vendor/persona-cartography/`, HEAD `6cfa6182e10acf625be937b0b19cfccecd864121`, 20 Aug 2026. "monorepo" is a **HuggingFace dataset** repo, not a GitHub path (`scripts/training/ocean_paired_dpo/04_train_lora.py` sets `MONOREPO_REPO`). The checkout postdates the note by five days. |
| 2. PC's unstated DPO hyperparameters | inferred from OCT | retrieved from `vendor/persona-cartography/src/training/oct_adapter.py`: batch 32, `max_len` 1024, betas 0.9/0.98, `kl_loss_coef` 0.001, `nll_loss_coef` 0.1, warmup 0.1, `max_norm` 1.0, seed 123456, bf16, ZeRO-2. The inference was right. |
| 3. PC's cosine matrix (Figure 15) | not available | `vendor/persona-cartography/scripts_dev/flatten_loras/data/cosine/persona.csv`, named in the LaTeX source. |
| 4. OCT's repo-only values | unverified | confirmed against the local checkout at `d1da9f0`: KL 0.001, DPO `max_len` 1024, merge weights [1.0, 0.25]. Cite the repo, not the paper. |

Items 5, 6 and 7 were not pursued and are not load-bearing; item 8 (prior art
doing factor analysis on weight deltas) is still not found.
*Current:* [[persona-cartography-paper]], [[open-character-training-paper]],
[[literature-factor-analysis-methods]].

**L-3. "LoRAcles" attributed to arXiv:2601.11207.**
*Replaced by:* De Schamphelaere, Bauer, Nanda, Ong, ICML 2026 MI workshop,
OpenReview `x9MbM7QmQN`. Harness memory, 2026-08-16.
*Current:* [[origin-and-question]].

**L-4. Persona Cartography's date.** `CONTEXT.md` gives 10 Jul 2026, which is the
LessWrong crosspost; arXiv v1 is **8 Jul 2026, 21:00:44 UTC**.
*Current:* [[persona-cartography-paper]].

**L-5. "Ashton & Lee 2004 HEXACO".**
*Replaced by:* the lexical-approach defence is Ashton & Lee **2005** (European
Journal of Personality 19(1), 5-24); the HEXACO advantages paper is Ashton & Lee
**2007** (PSPR 11(2), 150-166); the 2004 citation that circulates is **Lee &
Ashton** 2004 (Multivariate Behavioral Research 39(2), 329-358), reversed author
order.
*Current:* [[literature-big-five]].

**L-6. "Cartography's modification is that the teacher picks the rejected
response."** Recorded in the harness reading list as a disagreement between two
of its own extractions, unsettled.
*Settled 2026-09-07:* both sides come from the teacher, one conditioned on the
amplifier constitution and one on the suppressor; the polarity of the
constitution determines the label. `paper_notes.md` section 2.3 quotes App A.1.1
verbatim, and `vendor/persona-cartography/src/training/paired_dpo/pairing.py`
confirms the shape. Nobody picks anything.
*Current:* [[persona-cartography-paper]].

**L-7. Big Five keying stated as 10/10 across all five factors** in an early
draft of [[literature-big-five]].
*Replaced by:* Emotional Stability is 6/14 (item L7 in section 1b).
*Current:* [[literature-big-five]], [[goldberg-100-primary-traits]].

---

## 7. Conversation-stated numbers

Numbers Samuel was told in chat that later data changed. Transcript references
are to `/home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
by line, as `981fa3b5:<line>`; the extracts are in `wiki/raw/`.

**F1. The seed floor: 0.0166 and 89.0 degrees.**
*Stated:* `981fa3b5`:11041, 2026-09-02T20:35:17Z, in the reply to the external review.
*Undermined 29 minutes later:* `981fa3b5`:11177, 21:04:51Z, found that the 240
control adapters had trained under a different objective from the 134 (plain
sigmoid DPO with `kl_coef 0.0` against `loss_type ["sigmoid","sft"]` with
`kl_coef 0.001`).
*Replaced by:* +0.01806 matched against +0.01659 original,
`qwen35/PHASE3_VERDICT.md` 2026-09-03 addendum (line 275).
*Current:* [[seed-floor]].

**F2. The separation margin: 0.01509 / 0.01490.**
*Stated:* `981fa3b5`:11177.
*Replaced by:* 0.01593 / 0.01584 in the 2026-09-03 addendum - and then removed
from the page entirely per `981fa3b5`:12639. The string `0.01593` does not appear
in `qwen35/blog_page/index.html`.
*Current:* [[seed-paragraph-feedback]], [[seed-floor]].

**F3. "PC1 is not a Big Five factor", and PC2 is "close to Agreeableness".**
*Stated:* on the page until 2026-09-02.
*Replaced by:* `981fa3b5`:11041 - PC1 is a blend, **+0.82 Agreeableness / -0.70
Conscientiousness**; PC2 is **+0.81 Extraversion** (the geometry, against the
blind judge's reading of Agreeableness). Both corrections are on the live page.
*Current:* [[external-review]], [[factor-pc1]], [[factor-pc2]].

**F4. Blog skeleton B: "PC1 is a common direction, the personality axis; the Big
Five appear from PC2 on."**
*Stated:* 2026-09-04, `981fa3b5`:12109 - two days after F3 and contradicting it,
and contradicting `981fa3b5`:12973, which says the personality axis is the **mean
adapter**, reported separately, with the PCs describing variation around it. The
live page agrees with the latter.
**Do not carry the skeleton B phrasing into the post.**
*Current:* [[blog-skeletons]], [[factor-pc1]].

**F5. "A character English has no word for" - withdrawn.**
*Stated:* on the page as the hole result's headline.
*Replaced in two steps:* first by "the hole is in the sample, not the language"
(`981fa3b5`:11094); then, after the three hole-word adapters and the
activation-space transplant, by "suggestively, not convincingly, `insouciant`"
(`981fa3b5`:13065). `981fa3b5`:13036 calls the original "the most quotable line
on the page and the one I trust least". The activation arm's own framing is "a
direction the *adapters* leave open".
*Current:* [[hole-words]], [[external-review]], [[most-interesting-claims]].

**F6. The elbow "after component 2" - withdrawn.**
*Stated:* `981fa3b5`:10919.
*Replaced by:* `981fa3b5`:10939 showed four criteria give 1, 2 or 3; the figure
now shows a null comparison instead and reads "about eleven components above
noise".
*Current:* [[explainer-elbow-figure]], [[pca-and-scree]].

**F7. Using the permuted arm as a null the real spectrum must beat.**
*Stated:* implicitly, in how the null table was read.
*Replaced by:* `981fa3b5`:10963 - a category error. The permuted arm learns the
same personas under wrong names, so its spectrum **should** match; the shuffled
arm is the structureless baseline.
*Current:* [[null-controls]], [[lesson-test-shaped-like-the-wrong-hypothesis]].

**F8. The three items marked "pending" in the blog skeletons all landed.**
N x N gave 134/134; the matched-null scree gave 11 above shuffled and 0 above
permuted; stage 2 at seed 1 gave 15/15 and r = 0.978.
*Current:* [[blog-skeletons]], [[n-by-n-scoring]], [[pca-and-scree]],
[[stage-two-second-seed]].

**F9. "Exactly predicted" holds only at stage 1.**
*Stated:* the r/d = 0.025 attenuation prediction was matched at 0.0265.
*Qualified by:* stage 2 gives a slope of **0.116**, 4.6 times the prediction,
with a shared component every stage-2 adapter carries (`981fa3b5`:13044, 12919;
the 2026-09-05 verdict addendum calls it unresolved).
*Current:* [[stage-one-versus-stage-two-clarification]],
[[stage-two-second-seed]]; open in [[open-questions]].

**F10. "yep train all 40 please" - approved, never executed.**
*Stated:* on 2026-08-29 the assistant offered to train the six Lexicon words with
no stage-1 adapter, taking the validation set from 34 to 40 (`981fa3b5`:3339,
10:28:06Z); Samuel agreed (`981fa3b5`:3356, 10:32:44Z).
*What happened:* the zoo has **34** lexicon traits. `qwen35/traits_secondary.json`
holds 40 words and `constitutions.json` a constitution for each; the six with no
adapter are **unconformable, frightened, busy, defenseless, significant,
sleepy**. The 2026-09-05 provenance answer (`981fa3b5`:12639) explains the gap as
the constitution writer refusing six words as states rather than dispositions.
Whether the refusal preceded or followed the approval is not recorded.
*Current:* [[decisions-log]], [[provenance-feedback]], [[six-refused-traits]],
[[lexicon-secondary-draw]]; open in [[open-questions]].

**F11. The Imagination factor's Tucker congruence with Intellect, "about 0.6".**
*Stated:* `981fa3b5`:13013.
*Replaced by the file:* **0.6823**,
`qwen35/results/fa_qwen35.json#solutions.centred_k5.congruence_oblimin`, row 4
column 4 (target label order `[E, A, C, ES, I, Eval]`). Every other congruence in
that answer matches the file to two decimals.
*Current:* [[factor-imagination]], [[factor-analysis]].

---

## 8. Pre-zoo history

All from the Qwen2.5-3B era: the drift model organism, the gradient probe and
the 100-trait sweep. They are `historical`, but they are reversals the project
made and the blog post may narrate them.

| claim as first stated | what replaced it | current page |
|---|---|---|
| "The weight-space constraints did nothing because they were never binding" (trait direction held <= 7% of the update's energy), `CONTEXT.md` 3.2 | the A-GEM run **was** bound - cos^2 about 10%, firing on 49 of 76 steps - and still did nothing. What survives is that constraining a direction or a scalar fails where constraining the output distribution works | [[drift-experiment]], [[paper-selective-generalisation]] |
| "The oracle direction shows 58% of the update lies along the trait" | the cosine is algebraically fixed at 0.7634 by the norms alone; the defensible statistic is the **8.285%** excess over a 50% no-information baseline (`results/pooling_check.md` section F) | [[lesson-shared-term-contamination]] |
| "97-99% orthogonal, so we cannot find the trait" | right arithmetic, wrong reading: median per-module energy **10.133%** against a random-control floor of 0.000%, four orders of magnitude above chance in every one of 252 modules | [[lesson-weight-magnitude-is-a-bad-proxy]] |
| "Planted facts are an upper bound on gradient legibility" | withdrawn: the register effect is **+0.0903** (z = +232.31) against a planted-fact effect of **+0.0489** - traits are about twice as legible (`gradprobe/regroot/results/gradprobe.md`) | [[gradient-probe]] |
| the problem-identity effect of **+0.5442** in the register run | discarded - same-problem pairs share the identical prompt verbatim. Does not touch the register effect | [[gradient-probe]] |
| "Stage 2 was abandoned as unusable" (2026-08-15) | 100 `__sft` adapters existed the whole time; the crash fired after training saved (harness memory 2026-08-18) | [[sweep100]] |
| "Text-to-LoRA is beaten by retrieval, roughly 2x" (0.271 against 0.564) | converged at about 16.6k steps it reads **0.599** against 0.564. The earlier numbers were the step budget in the costume of convergence | [[sweep100]] |
| "More DPO epochs does nothing" | stands as measured but **qualified**: the learning rate was 5e-05 throughout where guidance is about 5e-06, and an LR ten times high produces the same saturation signature. The cheap re-check was never run | [[sweep100]]; open in [[open-questions]] |
| "Weight space reproduces the Big Five" | weights recover the Big Five no better than embedding the training text (**0.750 against 0.731**, 0 of 5 clearing 0.85), and this replicates on a second base model (`sweep100/results/text_vs_weights_fa.json`, `text_vs_weights_q3.json`) | [[sweep100]], [[factor-analysis]] |
| "PC1 is a general evaluative factor of the model" | stands for **unrotated** extraction only; under oblimin the largest evaluative congruence falls from 0.820 to 0.571 and no rotated factor is predominantly evaluative (`sweep100/results/fa.md`) | [[sweep100]], [[factor-pc1]] |
| behaviour gate "67 of 105 FAIL" at n = 16 | a power artefact: at n = 96, **71 of 105 pass**, mean win rate 0.670, and per-adapter win rates correlate r = 0.880 between the runs | [[sweep100]], [[lesson-sweep-the-readout]] |
| the three-seed result of 2026-08-16 00:28 | the clean rerun at 12:08, after four contaminated seed-1 adapters were retrained. Every headline unchanged to three decimal places | [[sweep100]] |
| PC1's factor-membership ANOVA quoted as **F = 0.71** alone, beside the keyed-sign F = 194 (2026-08-18 summary) | three statistics, all real: raw signed loadings F = 0.71 (p = 0.58), absolute loadings 13.8, pole-signed loadings 17.5. Quoting 0.71 alone overstates the contrast; `sweep100/results/pca.md` gives all three | [[sweep100]] |
| phase 7 "must not be run" | unblocked 2026-08-22 22:40: it applies a delta found in the seed-0 basis to the frozen base with matched controls, and never needed cross-run transfer | [[timeline]], [[steering-results]] |
| "The coordinates are seed noise / every effect is void" | the coordinates are attenuated about 38x by the LoRA parameterisation and survive it | [[lesson-coordinates-are-not-structure]] |
| "The failure cost of phase 10 was invisible" | it was not - all six instances bill under one app name. The blindness was a per-instance read and this ledger (`plan.json#phases[10].realised.correction`) | [[lesson-quote-the-tool-not-an-instance]] |
| the site at `persona-cartography.161-35-77-84.sslip.io` serves `sweep100/site/` | the systemd unit's `WorkingDirectory` is `sweep100/site2`; **site2** is what is served | [[sweep100]], [[infrastructure]] |
| stage two's adapters "each with its own random LoRA-A" ([[stage-two-structure]], 2026-09-07) | every stage-two adapter was trained at `sft_seed` 123456 and draws the **same** LoRA-A; after training the pairwise cosine between traits is 0.9977 to 0.9981 (stage one 0.99997), while across seeds it is 0.0021 (`analysis/lora_a_identity.json`). What is a separate draw is the stage-two set as a whole, against stage one's (cosine 0.0032). The shared component is not an artefact of that: measured inside each seed's own frame on the same 15 traits it is 0.2017 at seed 0 and 0.2000 at seed 1 (`analysis/stage2_frame.json`) | [[stage-two-exploration]], [[adapter-effect-and-drift]] |
| "alpha is measured in units of one trait adapter's worth of weight change" (`steer_fix.py` docstring; and "a released persona carries roughly alpha 0.4" of the shared direction, [[stage-two-shared-direction]]) | `ref` = 0.8078003190997738 came from `sketch_adapters.sketch_one`, which computes `||B @ A||_F` and omits the LoRA scaling `lora_alpha / r` = 2.0. The real mean stage-one delta norm is 1.6157416444226869, so **alpha 1 is 0.49996 of an adapter**. The persona's dose of the shared direction is 0.3894 of an adapter, i.e. **alpha 0.7788** in spec units (`analysis/steer_alpha_units.json`). No steering result changes; every published dose reads as half | [[stage-two-exploration]], [[steering-results]] |
| the blog page's parallel-analysis panel and prose: "twelve factors on reduced eigenvalues and nine on unreduced", "five at N=150, nine at N=1,528, thirteen at N=20,000" | those are the **uncentred** grid, while the five factors on the page come from the **centred** (ipsatised) matrix and `results/fa_qwen35.json#n_factors.chosen` is read off the centred one. On the centred grid at N=1,528 it is **nine reduced and nine unreduced**, and the N-sensitivity is **5 / 9 / 14**. Panel and prose now both read `#n_factors.parallel_analysis_centred`, generated from the grid (2026-09-08) | [[pca-and-scree]], [[factor-first-migration]] |
| the hole is "suggestively, not convincingly, insouciant" - insouciant 39.56272020815669 degrees from the k=5 hole where nothing in the zoo was closer than 52.514796421891425 ([[hole-words]], `PHASE3_VERDICT.md`) | on the five-factor chart that Samuel's 2026-09-08 decision made primary, the closest of the three new words is insouciant at **66.4505** degrees against **54.76378041380744** for the nearest existing zoo adapter, and `analysis/hole_geometry_fa.json#_verdict` reads `"the names miss"`. The chance level rises from 0.07270091967910823 to 0.43257832813236446 for one word and to 0.817308745873875 for at least one of three | [[hole-words-factor-chart]], [[hole-words]] |
| "A 134-word sample of the lexicon covers two dimensions better than chance and every dimension after that worse" (blog page caption; `analysis/alien.json#k_sweep.2` z = -2.280849861251385) | the k=2 effect is a property of the principal-component plane, not of the lexicon. On the factor chart the same statistic against the bit-identical null is z = **+0.19788810289715242**, and at k=3 it falls from +3.0570268913767125 to +0.6738544262288416 | [[hole-words-factor-chart]] |
| the unnamed direction is habitable and the chart predicted it - 5 of 5 signs, r = 0.8142138888404568, with the random control doing *more* damage than the direction ([[alien-direction-steering]]) | on the factor chart the unnamed direction is the only one of the three that degenerates (0.4166666666666667 of 24 responses looping at alpha -2 against 0.0 for both controls), the match falls to 3 of 5 signs at r = 0.7438221950472093, and `span_random_fa` scores 3 of 5 at r = 0.7343440116564208 against the same prediction, so the margin over control does not reproduce | [[alien-direction-factor-chart]], [[alien-direction-steering]] |
| the sphere sweep's coordinates, coverage and variance share, stated on the top three principal components ([[sphere-sweep]], 2026-09-01) | the frame is superseded, the data are not. Samuel's 2026-09-08 decision makes the factor analysis primary, so the identical 72-point lattice was resampled on the span of the top three oblimin factors at the same alpha, prompts and `ref`. Coherence 45 of 72 clean against 48, smoothness rho **+0.6765886845997277** against 0.6511417860626274, variance share **0.2624606004977159** against 0.2805056997657227 for the PC directions measured on the same exact Gram. Unlike the PC subspace this one is not orthogonal to the grand mean (cosine 0.5067), and emotional stability is top-rated at no point on it | [[sphere-sweep-factor-chart]], [[sphere-sweep]] |
| the alpha-0 entry of every steering result is the base model's own greedy output (implied throughout [[steering-results]]; `steer_fix.py` walks the alphas in order and returns to 0 by increment) | the walk adds and subtracts the increment in **bf16**, which is not reversible: the nine `"0.0"` entries in `phase10_runs/steer_results_fix.json` are **all different from one another**, with **0 of 24** prompts identical between `PC1` and any of the other eight and a mean shared prefix of **2.4 to 7.9 per cent** of the response. They are near-base text, not the base model. What is established is the non-reversibility; how much of the increment survives bf16 rounding at the published alphas 1, 2 and 4 was **not measured** and is an open question | [[fisher-norms]], [[open-questions]] |
| directions of equal Frobenius norm are comparable at a common alpha (the design of [[steering-results]], [[sphere-sweep-factor-chart]], [[alien-direction-factor-chart]] and [[additivity]]) | measured on the model's own metric they are not. The Fisher norm of 124 directions spans **259.924881015629** to one, from `S2_balancedrandom` at **0.0019170369703626355** to `sphere_S061` at **0.49828560642407** (`analysis/fisher_norms.json#comparisons.full_range`). Within the published set, alpha 2 is a Fisher dose of **3.52** on PC4 and **1.19** on the stage-two grand mean. No published result is retracted; the doses are not equal and the comparisons should say so | [[fisher-norms]] |
| "the stage-one direction has already degenerated" at alpha 4 "while the stage-two direction is still coherent" ([[stage-two-shared-direction]]), reported as an observation without an explanation | the explanation is that it is **flatter**, not sturdier: F **0.04391528597956066** against **0.2947447913048136**, a ratio of **0.14899427326654663**. Alpha 4 on the stage-two mean is the same Fisher dose as alpha **1.5439910531686205** on the stage-one mean, where the stage-one axis does not degenerate either (`analysis/fisher_norms.json#comparisons.stage2_vs_stage1_grand_mean`) | [[fisher-norms]], [[stage-two-shared-direction]] |

## Reading-page predictions corrected by running the experiment (2026-09-10)

| # | what the page said | what running it showed | current page |
|---|---|---|---|
| R1 | G2's control design in [[paper-reading-2026-09-09]]: the shuffled corpus "should produce no coherent atoms at all" | it cannot. Swapping a DPO pair negates its gradient (the SFT term, about 2% of it, is the only part that does not flip) and sparse dictionary learning is equivariant to flipping a document's sign together with its code. The shuffled arm's coherence degrades by about a third (max **0.15456007421016693** against the real arm's **0.17964830994606018** on atoms with a real cluster) and its actual signature is that an atom's two poles become the same trait: **14.5%** of shuffled atoms against **0.0%** of real ones (`qwen35/analysis/gradient_atoms.json#poles`) | [[gradient-atoms]] |
| R2 | G2's permuted control: "atoms should follow the data, not the names" | untestable at trait level. The permuted files are byte-identical row sets from `data_common` under other names, and trait purity is invariant under a bijection, so the two labellings give the same number by construction (0.1887, z 38.9943). The factor-level version has power in principle - only 13 of 100 assigned names share a factor with their source - and comes out null: 0.3886 (z 11.6897) assigned against 0.3944 (z 10.9046) source | [[gradient-atoms]], [[factor-analysis-null-arms]] |
| R3 | G2's cost estimate, "$12-27 with both null controls" | about **$1.2** at the meter's rate. The estimate assumed three extractions; both null arms are exact linear recombinations of one extraction's per-completion gradients | [[gradient-atoms]], [[costs]] |

## Numbers corrected by re-deriving them from the per-item source (2026-09-10)

| what the page said | what re-derivation showed | current page |
|---|---|---|
| the sycophantic and obsequious top-400 flags "differ materially: 121 ids shared of 400" and their pair scores "correlate at 0.04554719039593201" ([[dolci-data-audit]], stage-one draft) | neither number is in `phase10_runs/dolci_selection.json#syc_obs_overlap`, which reads **293** and **0.8762008220799862**, and both file values reproduce exactly when recomputed from `phase10_runs/dolci_scores_dpo.jsonl`. The two flags are largely the same set of items, not different ones. Corrected on the page; how the earlier pair of numbers was arrived at is not recoverable | [[dolci-data-audit]] |

## Predictions corrected by training on the data (2026-09-10)

| what the page said | what running it showed | current page |
|---|---|---|
| the `align_corrigible` negative top 400 is the tail whose "compliance reading survives", "**this is the set to train on**", and 10.25% of its pairs prefer compliance over a formulaic refusal against 1.06% in the sample ([[dolci-data-audit]]) | the same row of the same table carries 8.50% going the other way, so the **net** preference for compliance is 1.75 percentage points, about seven pairs of 400. Trained on it at the zoo's objective for 12 optimiser steps, the arm **refuses more**: it engages with a 40-prompt should-refuse battery 0.2000 less often than a matched random arm (p 0.00690), is judged 0.9500 quality points better (p 0.00275) and 0.7500 lower on fabrication (p 0.01555). The flag remains a good weight-space forecast, and the plain top 400 is also confounded with length (chosen halves 786 characters against the control's 3,336), which the audit had identified and then not controlled in the set it recommended | [[dolci-flag-training]] |

## Explanations corrected by removing the confound (2026-09-10)

| what the page said | what removing it showed | current page |
|---|---|---|
| [[dolci-flag-training]] attributed the flagged arm's engagement reversal to two properties of its corpus: "the tail's refusal polarity is nearly symmetric" **and** "its completions are about a quarter the length of the control's", naming the untrained length-stratified selection as the run's largest weakness | the length-stratified selection was trained as `corr_ls` and re-judged in **one batch** with the earlier arms. It is length-matched by construction (40 pairs in every chosen-token decile, mean 470 tokens) and behaves identically: engagement against `random` **-0.1750 (p 0.0153)** where the plain arm re-judged in the same batch gives **-0.1750 (p 0.0146)**. Removing the length difference changed nothing, so the **length half of the explanation is refuted** and the refusal-polarity reading carries the result alone | [[sycophancy-forecast]], [[dolci-flag-training]] |
| [[dolci-flag-training]]: "the disposition batteries say nothing here, and the weight-space numbers below say why they cannot: these adapters project 2 to 6 per cent onto the factor chart where a trait adapter projects 58 per cent" | the 24-prompt Big Five battery **does** have power on 400-pair arms once the score spread is wide enough. Over six arms spanning 0.536 on `axis_Agreeableness` rather than 0.236, judged Agreeableness orders as the score does at Spearman **+0.8407**, exact p **0.0444**, and the anti-sycophantic arm is the only one of six that does not become more agreeable than base (-0.0833, p 0.72836). The earlier null was power, not absence | [[sycophancy-forecast]] |

---

Related: [[source-contradictions]], [[open-questions]],
[[superseded-geometry-claims]], [[method-lessons]], [[timeline]],
[[how-to-read]].

## Display names changed while the result did not (2026-09-11)

| what the page said | what replaced it | current page |
|---|---|---|
| the third recovered factor was displayed as **Approach and Avoidance** on the blog page until 2026-09-08, then as **Fearful withdrawal** (the solution's own label in `qwen35/analysis/fa_summary.json#centred_k5.factors[2].name`) from 2026-09-08 to 2026-09-11 | Samuel renamed the display name to **Timidity** on 2026-09-11, after [[factor-audit-2026-09-11]] and against the maintainer's initial keep. The argument: the strong loaders are fear and social timidity (fearful, insecure, nervous, anxious, timid, shy, bashful, inhibited), while "withdrawal" pointed at a cluster (withdrawn, introverted, quiet, reserved) that loads on Arousal and not on this factor; the steered behaviour is submission versus assertion; and "Timidity" covers both loader clusters with boldness as its natural opposite. **Display only** - the internal key `FA_FearfulWithdrawal`, the npz and JSON keys, the steering job names, the wiki slug `factor-fearful-withdrawal`, `fa_chart.py`'s `FACTOR_ORDER`, the solution's own label and every file under `qwen35/results`, `qwen35/analysis` and `qwen35/phase10_runs` are unchanged. No number moved | [[factor-fearful-withdrawal]], [[factor-first-migration]] |

The 2026-09-08 half of that row was not filed here when it happened; it is
recorded now, with the 2026-09-11 rename, so the sequence of three display names
is in one place.
