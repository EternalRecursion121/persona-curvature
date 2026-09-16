---
title: Merge report - the eight section reports
summary: What the merge editor could not place on the three overview pages, where two section reports contradict each other, and the numbers re-verified against their source files.
status: current
sources:
  - wiki/pages/actspace/_report.md
  - wiki/pages/behaviour/_report.md
  - wiki/pages/conversation/_report.md
  - wiki/pages/geometry/_report.md
  - wiki/pages/history/_report_history.md
  - wiki/pages/history/_report_literature.md
  - wiki/pages/traits/_report.md
  - wiki/pages/zoo/_report.md
last_verified: 2026-09-07
tags: [meta, report]
---

# Merge report

Written 2026-09-07 by the merge editor. Inputs: the eight section reports (about
2,200 lines) plus [[superseded-geometry-claims]]. Outputs:
[[superseded-claims]], [[source-contradictions]], [[open-questions]] and this
file. Nothing else was written; no raw source was edited; no rerun, no GPU, no
upload.

---

## 1. Numbers re-verified against their source files

The brief asked for at least ten. Seventeen were opened and read by key or by
line. **All seventeen matched the reports.** Two things were found that no report
records; they are section 4 below.

| # | number | source opened | matched |
|---|---|---|---|
| 1 | k=2 coverage, `gap_deg` 3.9655382092691895, `z` +1.5803426589637861, nearest `introverted` | `qwen35/analysis/viz.json#coverage.2` | yes |
| 2 | k=2 coverage, `gap_deg` 0.9282998475514116, `z` -2.280849861251385, nearest `unsympathetic` | `qwen35/analysis/alien.json#k_sweep.2` | yes |
| 3 | alien direction gap 52.51479642189155 | `qwen35/analysis/alien.json#alien_k5.gap_deg` | yes |
| 4 | LoRA-A drift 0.014605041334818797 | `qwen35/analysis/align_summary.json#a_drift` | yes |
| 5 | P~W raw 0.704599368961925 | `qwen35/analysis/actspace_geometry.json#windows.resp.primary.r_centred` | yes |
| 6 | P~W double-centred 0.7368952052412984, and `frac_internalised` 0.6198613047599792 | `qwen35/analysis/actspace_adapters_geometry.json#windows.resp.curve[16]` | yes (the array index is also the layer number) |
| 7 | adapter alone, trait-specific, 0.7044131755828857 | `qwen35/analysis/actspace_cross_geometry.json#resp_specific_adapter_alone` | yes |
| 8 | stage-2 separation `min_same` 0.04248138800846624, `max_off` 0.05140579106757132, `sep` -0.00892440305910508 | `qwen35/analysis/crossseed_arms_stage2.json#[0]` | yes |
| 9 | matched shuffled `ARI` 0.05977870383675816, `ari_p` 0.0009995002498750624 | `qwen35/results/decomposition_shuffled_matched.json#test2b."leading-component removed"` | yes |
| 10 | original shuffled `ARI` 0.00438485865621544, `ari_p` 0.36231884057971014 | `qwen35/results/decomposition_shuffled.json#test2b."leading-component removed"` | yes |
| 11 | `use_rslora` true in the plan | `qwen35/plan.json#defaults.use_rslora` | yes |
| 12 | `use_rslora` false and `expected_scaling` 2.0 in every one of the 134 recorded runs | `qwen35/phase2_runs/archive/phase5_sweep_134.json` (set of values over all 134 records is `{False}` / `{2.0}`) | yes |
| 13 | 40 secondary trait words | `qwen35/traits_secondary.json` (length 40) | yes |
| 14 | Emotional Stability keying 6 positive / 14 negative; the other four factors 10/10 | `qwen35/traits_primary.json` (`keyed` field, counted by factor) | yes |
| 15 | `perm_mean_abs_diff` 0.1798544716598768 and 0.10336514494213549 | `qwen35/analysis/scree_null.json`, `qwen35/analysis/scree_null_matched.json` | yes |
| 16 | stale seed block: `same_trait_cos` 0.01658715905967511, `same_trait_deg` 89.04958220635345, `n` 40, `rank1` 40 | `qwen35/analysis/pc_loadings.json#seed` | yes |
| 17 | PC1 `named` = "Conscientiousness"; deflation `var_removed` 0.045566670770937776; seed floor +0.01659 / +0.01806 | `qwen35/analysis/steerfix_replication.json#PC1`; `qwen35/analysis/polarity_deflation.json#stage1.var_removed`; `qwen35/PHASE3_VERDICT.md` lines 164 and 275 | yes |

The meter reading $2,240.52 was also traced to
`qwen35/phase10_runs/zoo40_meter.log` line 25572 (2026-09-05T21:43:37Z) - see
section 4.

---

## 2. Where two section reports contradict each other

**M1. Who says the $2,240 meter figure has a file.**
`history/_report_history.md` lists it as "**maintainer-reported, 2026-09-07. No
file on disk carries it**" and [[costs]] records it that way.
`zoo/_report.md` sources it to `qwen35/phase10_runs/zoo40_meter.log`'s last line
and [[zoo-spend-ledger]] quotes it as file-sourced. **The zoo report is right**:
`est_total_spend=$2240.52 / $2400.00` first appears at log line 25572,
2026-09-05T21:43:37Z, and stands to line 35079. [[costs]] carries the wrong
provenance and is outside this editor's write scope; the maintainer should fix
the sentence there. Both pages should also say that it is a container-minute
meter at a flat $2.10/hour, not a billing read.

**M2. Whether the 445-prompt count is file-sourced.**
`actspace/_report.md` lists 445 under weakly-sourced numbers - "line count of
`qwen35/data_common/bold.jsonl` ... **counted by this session**, not stated in any
source". `zoo/_report.md` and [[shared-prompt-pool-445]] have it from the corpus
construction record, and `qwen35/RUNNER_TASK.md` and
[[stage-one-training-config]] state it independently. The caveat on
[[actspace-method-notes]] is now overcautious and can be dropped.

**M3. `analysis/adapter_effect.json` is owned by nobody.**
`geometry/_report.md`: "probably not a geometry object... the behaviour agent may
want it." `behaviour/_report.md`: cited "only as a file that exists; its keys'
definitions are unconfirmed and nothing reads it." `traits/_report.md`: prints
the five fields verbatim with an explicit statement that their meaning is not
established. Three sections independently reached the same conclusion and each
passed it on. Filed under provenance gaps on [[open-questions]]; it needs an
owner or a deletion.

**M4. The LoRA-A drift figure reads as two different verdicts.**
`geometry/_report.md` files 0.0146 against `build_monitor_page.py`'s 4.5% as an
unresolved contradiction. `zoo/_report.md` files 0.0146 under weakly sourced
numbers with the verdict "traceable after all", listing the four JSONs and the
`ZOO_DRIFT` gate constant. They are answering different questions - provenance
against disagreement - and both are right. Written up once, as
[[source-contradictions]] S4.

**M5. How many preregistered alignment predictions there are.**
`geometry/_report.md` section 7 says `qwen35/PREREG_alignment.md` has **five**
predictions, not four, the fifth being the steering test that belongs to
[[steering-results]]. [[alignment-traits-geometry]] and the zoo pages describe
**four**. Not reconciled here; whoever writes the post should count them in the
file. The four/two-held/two-failed scoreline is unaffected.

**M6. The seed floor: "the reading was withdrawn, not the number" against "the
number moved".**
`history/_report_history.md` item 7 says of +0.0167 that "the **reading** was
withdrawn, not the number". `geometry/_report.md` and
[[superseded-geometry-claims]] show the number itself moving to +0.01806 on the
2026-09-03 matched retrain. Both are true of different events - the 2026-08-22
reversal was about interpretation, the 2026-09-03 retrain about the measurement -
but read side by side the two reports appear to disagree. [[superseded-claims]]
item F1 and section 3 keep the two events separate and dated.

**M7. What replaced `distil_page`, and whether the replacement is current.**
`behaviour/_report.md` lists `distil_page` as superseded by `findings_page` +
`monitor_page` + the 22 `direction_pages` (2026-09-01). `geometry/_report.md`
lists `findings_page`, `monitor_page` and `direction_pages` among the older built
pages that are historical under wiki rule 2. Both are right: the replacement
pages are the replacement **and** are themselves not current truth. Only
`blog_page/index.html` is. Stated that way on [[built-pages-inventory]] and at
[[superseded-claims]] item B3.

**M8. Two different spend series.**
`history/_report_history.md` gives five successive realised figures ($195.90,
~$370.65, $370.53 from the harness memory, ~$1,915 in the 2026-09-01 journal,
$1,950.90 in the 2026-09-04 journal, then the meter); `zoo/_report.md` gives
three ($195.90, ~$370.65, $2,240.52). [[costs]] and [[zoo-spend-ledger]]
therefore present different series for one quantity. Not a contradiction - the
history page is the whole-project view and the zoo page the zoo's - but a reader
comparing them will think one is missing figures. A cross-link between the two
pages would settle it.

---

## 3. Reported items I could not place on the three pages

**3a. Corrections to the section briefs, not to the project.** Six reports flagged
that their own task brief stated something the files do not support. These are
not claims the project made, so they are not on [[superseded-claims]]; they are
recorded here because the same errors will recur if the briefs are reused.

- **"alpha 16, beta 0.5" as part of the recipe** (zoo brief). Those are the names
  of phase-2 experiments that were **rejected**; beta 0.5 produced the worst
  collapse in the phase (`loss_last` 2.12e-30 on `organized`). The zoo trained at
  alpha 128, beta 0.1.
- **"Pearson 0.997 matched, 0.954 objective-only"** (geometry brief). 0.9966259140629706
  is the matched arm's Gram-correlation Pearson; **0.954 is not a Pearson** - it
  is the mean same-trait cosine between the seed-1 arm at the plain objective and
  the seed-1 arm at the matched objective.
- **"34 secondary traits"** (literature brief). `traits_secondary.json` holds 40;
  34 is the zoo count. See [[source-contradictions]] S11.
- **"Imagination congruence about 0.6"** (geometry brief, and the chat answer at
  `981fa3b5`:13013). The file says **0.6823**. Placed as
  [[superseded-claims]] item F11.
- **"from 2026-08-08 (first session)"** (history brief). No artefact supports
  08-08; the earliest is Samuel's message of 2026-08-12 00:23:17. 2026-08-02 is
  when the agent harness was started.
- **"GRPO on Dolci math / math+code with zoo LoRA-A init"** and **"`distil`
  tested prompted model against OCT-trained LoRA"** (behaviour brief). Only the
  math+code run adopted zoo LoRA-A (`rl_mix.log` has `adopted zoo LoRA-A on
  248/248`, `rl_train.log` does not), and `distil_page` was about where
  personality lives in the weights, not prompting versus training. The
  prompted-versus-trained comparison is the activation-space work.
- **"prompt-window additive 0.97/0.99"** (actspace brief) is the **raw** pair; the
  blog uses the trait-specific slice. Recorded in
  [[source-contradictions]]'s conflation section.

**3b. Code defects, not claims.** Reported, real, and outside the wiki's subject
matter. All are read-only observations; nothing was changed.

- `qwen35/phase2_gates.py`'s `--results` default points at
  `qwen35/phase2_runs/results.json`, which is **not** phase 2's results - it holds
  40 records of the seed-paired arm at the superseded objective. Phase 2's own
  four-trait results are in `phase2_runs/archive/results.json`. Anyone re-running
  the gates would judge phase 2's health from the wrong arm.
- `qwen35/analyse_actspace_cross.py` line 12 docstring says "adapter alone was
  0.62" where the script prints 0.70.
- `qwen35/analyse_actspace.py` line 177 hard-codes "weight space 68.9" beside an
  activation result, without naming the space.
- `qwen35/analysis/actspace_geometry.json#windows.prompt.curve[0]` is all `nan`.
  Benign - with no system prompt the shift is zero and its cosines undefined - but
  `max(curve, key=r_centred)` returns layer 0 for that window. The blog uses the
  `resp` window only.
- `qwen35/analysis/actspace_spec.json` is overwritten by every stage run and its
  `traits` field is always the 134-name list, never `CROSS_TRAITS`, so the file
  cannot say which stage last ran except by mtime.
- The blog's "Prompting versus training" section does **not** contain the
  hole-transplant paragraph; that paragraph lives in the hole section
  (`build_blog_page.py` lines 1317-1325) and rounds 47.8 / 47.7 to "48 / median
  48". The 2026-09-05 journal reads as if both went into the same section.

**3c. Weak-provenance items that did not reach a page.** Each is stated on its own
section page; none was strong enough to merit an overview entry, and none is
wrong so far as anyone knows.

- `qwen35/analysis/verify.json#arm_vs_control_cos` (0.715 / 0.717 / 0.711,
  averaged to "cosine 0.71" on the blog). `analyse_verify.py`'s `json.dump`
  writes only `raw`, `sub`, `targets`, `hits`, `frac`; the key was added by
  something not checked in. Belongs with the no-producer list on
  [[open-questions]] if anyone extends it.
- No builder for the mixture weights in `steer_spec_mix.json#jobs[].weights`.
- `qwen35/analysis/rl_behavioural.json`'s shift figures come from
  `judged_rl_persona.json`, whose repeat reliability on ten units was r = 0.250
  (Extraversion), 0.643, 0.724, 0.948, 0.667, against 0.78-0.90 on every larger
  judge run. The largest shift (+0.462 sd at checkpoint-50) should not be read as
  established. On [[rl-capability-and-persona-drift]].
- `build_monitor_page.py`'s additivity ranges ("reinforcing 0.13-0.15, opposing
  1.17-1.46") are the **extremes** of the two groups, not their means, and only
  one of the two subtractive mixtures shows the large residuals. The headline
  `median_resid_over_effect` 0.528 is solid; the sign-asymmetry gloss is thin. On
  [[additivity]].
- `qwen35/constitutions.py`'s `report()` output for the real run (word counts, how
  many constitutions opened with "You are") is printed, never written, and no
  stdout capture survives.
- "$956 already spent", "~$631 for the Lexicon traits" (comments in
  `zoo40_meter.sh`, undated) and "~$1,216 for 71 remaining traits"
  (`POST-BATCH3-TODO.md`, a forward estimate at the measured rate). Carried on
  [[zoo-spend-ledger]] as what they are; not promoted.
- `qwen35/personality_lora_zoo_experimental_plan.md` has no author and no date
  beyond its mtime (2026-08-24 12:26:29 UTC), is not in git, and is referenced by
  no script or journal. On [[experimental-plan]].
- GPU-hours per trait 4.20 against 4.77 (the first is `warm` alone, the second a
  three-trait average) and phase 10's five cost figures. Both resolved inside
  their own reports; on [[costs]].

**3d. Recomputations that exist in one report and nowhere else.**
The actspace agent computed six medians and two counts to diagnose its own
contradictions and correctly kept them off the wiki pages under rule 1. They are
now recorded **only** in `pages/actspace/_report.md`, including the `same factor,
same keying` class (n = 14): `along_P_t` 0.721, `resid_add` 0.6457, `a` 0.6583,
`b` 0.6730, prompt-wins 5/14. That class has no published medians anywhere - see
[[open-questions]]. If `_report.md` files are ever pruned, those numbers go with
them.

**3e. One secrets note, carried forward without content.**
The conversation agent found a pasted GitHub token in a user turn of a transcript
outside its partition
(`/home/vibe12/.claude/projects/-home-vibe12-projects/8c4fed77-8ced-4e17-8624-da1b043cd0ab.jsonl`,
2026-08-02, an agent-harness session with no persona-curvature content). It is not
quoted anywhere in this wiki and must not be. Recorded here so a future session
does not ingest that file blind.

---

## 4. Two things the reports do not have

**4a. The meter has started moving again.**
`zoo/_report.md` reads $2,240.52 off the last line of
`qwen35/phase10_runs/zoo40_meter.log`. That figure was the standing reading from
2026-09-05T21:43:37Z (line 25572) through **2026-09-07T16:25:05Z** (line 35079),
while `containers=0`. It began climbing when the full-OCT replication jobs
launched: the file's last line at the time of writing is
`2026-09-07T16:55:52Z containers=1 gpu_h_since_relaunch=1045.25
est_total_spend=$2241.93 / $2400.00`. **$2,240.52 is still the right figure to
quote for the zoo**, with its read time; the live last line will be stale within
the hour and should not be copied into anything. Recorded on
[[zoo-spend-ledger]] and [[open-questions]].

**4b. `plan.json`'s own ledger note no longer describes `plan.json`.**
`#ledger_defect_note` says "Sum of `actual` = $195.90" and describes phase 3 as
"$111.04 recorded against $206.84 measured". Phase 3 is `phases[3]`, `id` 3, name
"Null controls", and its `actual` now reads **206.885** - because the measured
figure was written back into the recorded field, which the file itself documents:
`#phases[3].actual_provenance` carries `was: 111.04`, a Modal-billing source read
at 2026-08-23T00:12, and `why_it_was_wrong`: "The old $111.04 was what the
SUCCESSFUL arms cost... A phase costs its failures."

So the correction is deliberate and sourced, and the only consequence is that the
note's own arithmetic no longer reproduces from the file it sits in. Nothing here
changes the $195.90 -> ~$370.65 story that [[zoo-spend-ledger]] and [[costs]]
tell; but a reader who re-sums `actual` today will not get $195.90 and should not
publish whatever they do get. Flagged rather than restated: recomputing a sum is
what wiki rule 1 forbids.

**M9. Whether anything of this project is served over Caddy.**
`conversation/_report.md` says "Any deployment... nothing is behind Caddy as of
2026-09-07". `history/_report_history.md` and [[infrastructure]] record three
routes in `/etc/caddy/Caddyfile`, two of them this project's:
`persona-cartography.161-35-77-84.sslip.io` on port 8091 serving the historical
`sweep100/site2/`, and `plan.161-35-77-84.sslip.io` on 8092 serving
`qwen35/site/`. The history report is right; the conversation report means the
**zoo blog page** specifically, which is not served. [[open-questions]] states it
that way. This is the one place where a section report's sentence, taken at face
value, would have put a false statement on an overview page.

---

## 5. Wiring left for the maintainer

- **`log.md` needs an entry** for this merge. `index.md` does not: it is
  regenerated from page frontmatter by `tools/build_site.py`, and the 17:12 UTC
  rebuild already lists [[superseded-claims]], [[source-contradictions]] and
  [[open-questions]] under Overview. This editor was told to write only the four
  files and touched neither.
- **Lint is clean after this merge.** `tools/lint.py`: 277 pages, 0 duplicate
  slugs, 0 frontmatter problems, **0 broken wikilinks**, 0 orphans. Before this
  merge it reported 7 broken links - `superseded-claims` (3), `open-questions`
  (1) and `full-oct-replication` (3). The first two are now closed by these
  pages; `full-oct-replication` and `lesson-verify-the-artefact-loads` were
  created by the maintainer while this merge ran, so nothing is outstanding. The
  slug problems five reports flagged - `trait-provenance` (150 inbound),
  `zoo-training-recipe` (16), `persona-vectors-paper`, `qwen35-thinking-trap` -
  are all resolved by hub pages that now exist.
- **`source-contradictions` has inbound links only from the other two overview
  pages.** That clears the orphan check but it should also be linked from
  [[how-to-read]] and [[home]].
- **`timeline` lives in `pages/history/`**, where its brief put it;
  `wiki/CLAUDE.md`'s layout puts it under `overview/`. One or the other should
  move, and it must not be duplicated.
- **Three page pairs cover the same ground from different sources** and were
  never reconciled: [[costs]] against [[zoo-spend-ledger]], [[infrastructure]]
  against [[modal-volumes]] and [[hf-artefacts]], [[phase-two-recipe-search]]
  against [[phase2-recipe-selection]]. Worth one read-through for disagreeing
  digits; M1 and M8 above are two that a read-through would have caught.
- **Two data-side fixes the geometry section asked for** are still owed, and are
  now on [[open-questions]]: regenerate the scree `real` curve from
  `results/gram_sweep.npz` (or annotate the JSON), and store the
  sketch-validation correlation somewhere so the two builders stop disagreeing.

---

Related: [[superseded-claims]], [[source-contradictions]], [[open-questions]],
[[superseded-geometry-claims]], [[how-to-read]].
