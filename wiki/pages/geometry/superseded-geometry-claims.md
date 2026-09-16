---
title: Superseded geometry claims
summary: Every place the project reversed itself on the weight-space geometry, with dates and both versions - the trait-level withdrawal and its reversal, the bar in the wrong units, the objective mismatch, the scree source and the stale numbers still on disk.
status: current
sources:
  - qwen35/PHASE3_VERDICT.md
  - qwen35/PREREGISTRATION_phase3.md
  - qwen35/results/compare_nulls_output.txt
  - qwen35/analysis/scree_null.json
  - qwen35/analysis/scree_null_matched.json
  - qwen35/analysis/pc_loadings.json
  - qwen35/analysis/viz.json
  - qwen35/analysis/alien.json
  - qwen35/build_monitor_page.py
  - qwen35/build_blog_page.py
  - .garden/journal/2026-09-05.md
last_verified: 2026-09-16
tags: [geometry, superseded, corrections]
---

# Superseded geometry claims

Each entry gives the earlier version, the later version, the date of the
reversal, and where both live. Nothing here is resolved by preference; where two
files still disagree, both numbers are quoted.

## 1. The trait-level withdrawal, and its reversal

**Earlier (2026-08-23, `qwen35/PHASE3_VERDICT.md` body).**

> **The trait-level claim is withdrawn and the factor-level claim is upheld** -
> individual trait directions are an artefact of the LoRA initialisation ...
> **So the trait-level framing is withdrawn.** There is no evidence here that an
> individual trait has a characteristic direction in weight space.

The preregistered bar was an across-seed self-cosine clearly above +0.2447, with
an abandon line at 0.35; measured median was +0.0167.

**Later (2026-08-22 22:40 UTC, addendum "THE TRAIT-LEVEL WITHDRAWAL IS
REVERSED").** Note the addendum carries an earlier timestamp than the verdict
body it reverses; the body was written 2026-08-23 and the addendum's own text
says it reverses it.

> **Reinstated.** A trait has a reproducible direction in weight space. It
> survives independent initialisation; the cosine is small only because the LoRA
> parameterisation confines each run to a different random rank-64 subspace, and
> 0.0264 is the overlap of two such subspaces, not a measure of signal loss.

Evidence: perfect separation (min same-trait 0.01509 > max different-trait
0.01490 across 5,320 comparisons), 40/40 top-1 identification, and the
attenuation slope 0.0264 against the predicted `r/d = 64/2560 = 0.0250`. See
[[seed-floor]].

What is **not** reinstated, in the same addendum: "Same-trait pairs train on a
byte-identical corpus, so what reproduces is that trait's *training signal*",
and "1.7% alignment is statistically unambiguous and practically worthless as a
transferable vector."

### 1a. And then it stands again, for nine days

The reversal was not the end of it. Two days after the addendum that reinstated
the trait-level claim, the 2026-08-24 addendum partly re-withdrew it:

> - The cross-seed self-cosine (+0.0167) is a seed-plus-objective floor, not a
>   pure seed floor. The trait-level withdrawal STANDS (the preregistration bound
>   it to this measurement as run), but the mechanistic reading "the init frame
>   alone carries the magnitude" needs the 40 seed-B adapters retrained at the
>   matched objective (~$17 at measured rates) before it is clean.

The 2026-09-03 addendum then discharges the caveat - "The 2026-08-24 caveat on
attributing the floor to 'seed alone' is discharged; the page now quotes 0.0181"
- without restating the reinstatement in words, and the blog page, which is
current truth, describes trait identity as surviving the seed change
("across 40 traits retrained at a second seed, the matched adapter is the nearest
neighbour among all 134 candidates 40 times out of 40 ... It is the
initialisation and nothing else").

So the document holds four positions in sequence on one claim:

| date | position |
|---|---|
| 2026-08-22 22:40 (addendum) | reversed - "**Reinstated.** A trait has a reproducible direction in weight space." |
| 2026-08-23 (verdict body) | withdrawn - "**So the trait-level framing is withdrawn.**" |
| 2026-08-24 (addendum) | "The trait-level withdrawal STANDS", with the mechanistic reading caveated pending a retrain |
| 2026-09-03 (addendum) + blog page | caveat discharged, page quotes 0.0181, trait identity treated as surviving |

Note the two August entries are in the order the document prints them, not in
timestamp order: the addendum carries the earlier timestamp and the verdict body
it reverses is dated the next day.

Nothing in the file says "the withdrawal is reversed again". Anyone quoting a
trait-level position must say which of the four they mean and date it.

## 2. "The preregistered bar was in the wrong units"

**2026-08-22 addendum.**

> The floor compared a **cross-basis** cosine (0.0167) against a bar computed
> **within a single basis** (+0.2447, the median same-factor same-keyed
> different-trait pair in the seed-0 Gram). Those two numbers are not
> commensurable. Two independent rank-64 inits span near-orthogonal slices of a
> 2560-dim space, so a trait direction that reproduced *perfectly* modulo the
> subspace would still have scored ~0.02. **The test could not have passed under
> any hypothesis.**

The stated lesson: "A preregistered threshold needs its own check: *compute what
the statistic would read under the maximal version of the hypothesis, before
committing to the threshold.*"

**A second, smaller units problem sits inside the same bar and is unresolved.**
`qwen35/PREREGISTRATION_phase3.md` calls +0.2447 "the **median** cosine of
same-factor, same-keying, *different*-trait pairs". The value 0.24472701836565558
is what `qwen35/results/decomposition.json#test2.raw.same_polarity` stores, and
that key is a **mean** over 932 pairs.
`qwen35/results/compare_nulls_output.txt` separately reports the **median** of
the same class as +0.2467 (n = 932), and +0.2373 on the 40 seed-twinned traits
(n = 130). Both are recorded; neither is preferred here.

## 3. The downstream flag about phase 7 steering

**Earlier (PHASE3_VERDICT body).** Phase 7 steering "assumed transferable trait
directions and needs re-specifying before it is worth running."

**Later (2026-08-22 addendum, "Correction to this document's own downstream
flag").**

> **That was wrong on two counts.** The directions do transfer. And phase 7 never
> needed them to: it applies a concrete weight delta, found in the seed-0 basis,
> to the frozen base model, with matched random-direction controls at every dose.
> That operation is well defined regardless of how the direction was found. Phase
> 7 is runnable as specified.

## 4. `compare_nulls.py`'s stored verdict

**Earlier (`qwen35/results/compare_nulls_output.txt`, still on disk, never
regenerated).**

> VERDICT: NO -- a trait trained twice looks NO MORE ALIKE than two different
> traits of the same factor and keying (+0.0167 vs +0.2467). The geometry is seed
> noise. EVERY EFFECT ABOVE IS VOID: TEST 1B, TEST 2 and TEST 2B are differences
> of a few hundredths computed inside a run-to-run wobble at least as large

**Later (PHASE3_VERDICT, "Where I depart from the tooling").**

> **That conclusion is wrong, and I wrote the check that produced it.** It
> conflates two different things: the **coordinates** failing to correspond
> across bases, and the **organisation** failing to replicate. A cross-run cosine
> of 0.017 establishes the first. It cannot speak to the second, because
> structure can reproduce in a rotated basis while every corresponding vector
> reads as orthogonal. The cross-Gram is the wrong instrument for the question
> the verdict line was answering.

The file also predates the matched-objective retrains, so its REAL row is the
only current line in it. Treat the whole file as historical.

## 5. The objective mismatch in every control arm

**Discovered 2026-08-24** (PHASE3_VERDICT addendum), by comparing against the
now-public Persona Cartography reference code.

> the null launcher carried the corrected LoRA scale but not the loss
> configuration, so all 240 control adapters (shuffled, permuted, seed-paired)
> trained under plain sigmoid DPO -- loss_type ["sigmoid"], kl_coef 0.0 -- while
> all 134 sweep adapters trained with loss_type ["sigmoid","sft"], kl_coef 0.001.

Three claims were caveated: the cross-seed self-cosine became "a seed-plus-
objective floor, not a pure seed floor"; the null table compared arms differing
in objective as well as treatment; and the seed-B factor replication's
attribution to "seed alone" inherited the caveat.

**Discharged in two stages.**

- 2026-09-03: the 40 seed-paired traits retrained at the matched objective.
  Same-trait cosine +0.01659 -> **+0.01806**; slope 0.0264 -> 0.0265; 40/40 top-1
  both ways. The objective in isolation (same seed, same data) gives same-trait
  cosine +0.954. "The 2026-08-24 caveat on attributing the floor to 'seed alone'
  is discharged; the page now quotes 0.0181."
- 2026-09-05: shuffled and permuted retrained at the matched objective. TEST 1B
  residual +0.0001 -> +0.0001 and +0.0075 -> +0.0080; scree counts 11 and 0
  unchanged. "The objective mismatch changed nothing to three decimal places."

Tooling change: `cross_gram_on_modal.py` now refuses matched pairs whose runmeta
disagree on `loss_type` / `loss_weights` / `kl_coef`.

See [[null-controls]] and [[seed-floor]].

## 6. Two null training runs that were thrown away

Not a claim reversal but a record of what the logs are.

- **`.RSLORA-INVALID`**: the trainer's default said rsLoRA while the sweep used
  plain LoRA, so 240 control adapters were trained at scale 16.0 against the
  sweep's 2.0 and the whole phase was rerun
  (`qwen35/launch_nulls.sh`, the comment above `export PC_USE_RSLORA=0`).
- **`.FAILED-datarace`**: the trainer's corpus-content gate at
  `train_qwen35.py:1050` refused to train when the sha256 of the jsonl inside the
  container disagreed with the corpus the driver declared - "a real-vs-null
  corpus swap or a stale upload -- the two are indistinguishable by prompts
  alone. BLOCKER."

## 7. The elbow

**Earlier**: with only the elbow to go on, six components looked like a generous
description of the cloud. **Later** (blog page): against the structureless null
the real spectrum stays above the floor through **eleven** components, "which
happens to sit right beside the twelve the factor side retains by an entirely
separate route. So describing six components is conservative, not generous. I had
that backwards when I only had the elbow to go on."

## 8. The stored scree `real` curve

`.garden/journal/2026-09-05.md`:

> Scree null regenerated from the matched Grams; stored `real` came from
> `gram_sketch100.npz`, not `gram_sweep` (took a minute to find).

Found by a reproduction check. What is still observable on disk:

- `#real` is byte-for-byte the same in `analysis/scree_null.json`
  (2026-09-02 20:20) and `analysis/scree_null_matched.json` (2026-09-05 12:15) -
  13.36216610744329, 11.207049613198226, 5.789132715752... - so it was not
  recomputed when the null arms were.
- Those values equal `analysis/geometry_stage1.json#explained_var_top10`
  (0.134, 0.1122, 0.0577, ...) after rounding, and that file is computed over the
  100 labelled traits from the `stage1_k32` **sketches**.
- The exact 134-trait double-centred spectrum is different:
  `results/fa_qwen35.json#pca_from_gram.centered_var_pct` = 12.4946, 10.9847,
  4.9679, ...

So the `real` curve on the current page is a 100-trait sketch-derived spectrum.
Nothing on disk shows it being regenerated from `results/gram_sweep.npz`. The
sketch fidelity (`r = 0.9996`) makes a shape change unlikely, but the provenance
is what it is. See [[pca-and-scree]].

## 9. Stale numbers still sitting in current data files

- `qwen35/analysis/pc_loadings.json#seed.same_trait_cos = 0.01658715905967511`
  and `#seed.same_trait_deg = 89.04958220635345` are the **original**,
  objective-mismatched seed-paired figures. The blog page's prose quotes the
  matched 0.018 while its own data file carries 0.0166. The 89-degree figure the
  page prints was computed from the stale cosine.
- `qwen35/analysis/scree_null.json#perm_mean_abs_diff = 0.1798544716598768` is
  what the blog page's prose quotes ("0.18 percentage points apart on average"),
  while the file the figure is drawn from,
  `scree_null_matched.json#perm_mean_abs_diff`, is **0.10336514494213549**.

## 10. Contradictions between files, unresolved

**Coverage at k = 2.** `analysis/viz.json#coverage.2.gap_deg = 3.9655382092691895`
(z = +1.5803426589637861, nearest `introverted`) against
`analysis/alien.json#k_sweep.2.gap_deg = 0.9282998475514116`
(z = -2.280849861251385, nearest `unsympathetic`). The two agree to within a
fraction of a degree at every other k. Opposite signs mean opposite conclusions
about whether the 134 words cover the plane better than chance. The blog page
uses `alien.json`, which was written 46 minutes later on the same day. See
[[umap-and-layouts]].

**LoRA-A drift.** `analysis/align_summary.json#a_drift = 0.014605041334818797`
(1.5%, quoted by the blog page) against `qwen35/build_monitor_page.py` line 112:
"it drifts **4.5%** of its own norm over 93 steps". The monitor page is an older
built page; the blog page's 1.5% is the figure the current draft and the analysis
files carry. The same monitor page also quotes the
attenuation as "predicted 0.058, observed 0.056, agreement to 5%", where the
current figures are 0.0250 predicted / 0.0265 observed at 6% - a different
quantity that should not be quoted as current. See [[adapter-effect-and-drift]].

**Sketch fidelity.** `qwen35/build_blog_page.py` line 1584 (prose):
`r = 0.9996`. `qwen35/build_monitor_page.py` line 209: "independently computed
exact Gram at cosine 0.99944". No output file storing either was found on disk;
`analysis/validate_100.json` is training metadata, not sketch validation. See
[[geometry-overview]].

**Cross-seed category counts.** `analysis/crossseed_arms.json#[0]` records
`same_factor_same_key` n = 368, `same_factor_opp_key` n = 392,
`diff_factor` n = 4560 (which with the 40 same-trait pairs totals 5,360, while
`#[0].n_pairs` is 5,320). The 2026-08-22 addendum's table gives n = 358, 386 and
4336 for the same three classes. The means agree (+0.00593 / -0.00271 /
+0.00133); only the counts differ. Unexplained.

**Permutation p values.** `results/decomposition.json` reports
`p = 4.999750012499375e-05` throughout (20,000 permutations);
`results/compare_nulls_output.txt` prints the same results as `0.00005`; the
PHASE3_VERDICT replication table prints `0.00010`. All three describe the same
permutation floor at different resolutions.

**Centroid deflation variance removed.**
`analysis/polarity_deflation.json#stage1.var_removed = 0.045566670770937776`,
while `analysis/manifold_ideas.md` and `qwen35/build_findings_page.py` both quote
"5.3% of variance" for the same experiment. See [[polarity-and-bipolarity]].

## 11. Numbers whose provenance is prose only

Recorded so that a blog post copying them knows what it is copying.

| claim | value | only source |
|---|---|---|
| stage-1 vs stage-2 same-trait cosine | 0.000 | `.garden/journal/2026-09-05.md` |
| stage-1 vs stage-2 arrangement correlation over 45 traits | r = 0.79 | blog page prose + journal |
| stage-2 LoRA-A drift from init | 10% | `PHASE3_VERDICT.md` prose |
| seed-0 40-trait TEST 1B raw / residual | +0.1819 / +0.1086 | `PHASE3_VERDICT.md` table |
| objective-only same-trait cosine | +0.954 (range 0.908-0.981) | `PHASE3_VERDICT.md` + journal |
| analytic vs finite-difference agreement | 0.9999992 | hard-coded default in `build_blog_page.py`; `analysis/align_validate.json` has no `corr` key |
| lexicon-vs-marker subspace checks | principal cosines 0.94-0.996; 25% vs 29% variance retained; 68 vs 65 degrees | blog page prose + `.garden/journal/2026-09-05.md` |

## Not on this page

`~/projects/persona-curvature/CONTEXT.md` (written 2026-08-14) carries its own
explicit SUPERSEDED markers - the explanation for why weight-space control failed,
and the oracle-direction yardstick - but they are about the pre-zoo model organism
(Qwen2.5-3B, sycophancy) and not about the zoo's geometry. They belong to the
history section.

Related: [[seed-floor]], [[null-controls]], [[pca-and-scree]],
[[cross-seed-geometry]], [[adapter-effect-and-drift]], [[umap-and-layouts]],
[[geometry-overview]].
