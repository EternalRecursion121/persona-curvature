# behaviour/ — agent report, 2026-09-07

Section: behavioural evaluation and interventions on the 134-adapter Qwen3.5-4B
zoo. Thirteen pages written, all inside `wiki/pages/behaviour/`. No file outside
this directory was created or modified; no raw source was edited; no Modal, GPU
or upload calls were made.

---

## 1. Pages written

| slug | one line |
|---|---|
| `judged-evaluations` | The blind Big Five instrument: 24 behavioural probes, `anthropic/claude-sonnet-4.5` via OpenRouter, 7,200 generations over base/stage1/persona, plus the judge's reliability ceiling and the base model's own profile. |
| `steering-results` | Both steering campaigns — STEER134 (phase 7, `gpt-5.6-terra`, incomplete judging) and steer/steerfix (phase 10) — the direction families, the coherent band, the full replication table, and the nine pre-registered predictions scored. |
| `thinking-default-withdrawals` | Qwen3.5 defaults thinking ON; which three runs were invalidated, what replaced them, and the judge-reliability collapse that shows the defect. |
| `additivity` | Ten matched-norm axis mixtures: residual 0.53 of the predicted effect, 2.5x the judge-noise floor — the coordinates do not compose. |
| `alien-direction-steering` | The widest lexicon hole steered against shuffle and random controls; all five predicted signs matched, r = 0.81, and less damage than the random control. |
| `sphere-sweep` | 72 Fibonacci-lattice directions at alpha 1.5: 48 loop on nothing, angular distance predicts judged distance at rho = 0.65, and the sphere is lopsidedly agreeable. |
| `optimised-data-and-verify` | Evolutionary search for data aimed at a direction, then four trained adapters; 3 of 3 land closest to their own target once the random arm is subtracted. |
| `distillation-check` | What `distil_page` actually was, why it was withdrawn on 2026-09-01, and what replaced it. |
| `rl-capability-and-persona-drift` | GRPO on Dolci maths: half a trait adapter of norm, `top6_frac` below a random null and 9.4 sd below a cross-seed null, with a small and poorly-measured behavioural shift. |
| `reward-hacks-arms` | The positive control that also failed: hack and matched-control SFT, 1.08x norm ratio and cosine 0.23 apart, both at ~1% of a trait adapter's chart length. |
| `scoring-identity` | The one-backward-pass identity for scoring data against a LoRA direction, and its finite-difference validation. |
| `qualitative-notes` | The four hand-written reads of the corrected steering corpus (which found the thinking defect), the trait-pair reads, and the separate 60-entry safety adjudication of the training corpus. |
| `built-pages-inventory` | Every built HTML page: what it shows, its builder, whether it is served, and its status. |

Outbound links to other agents' pages that do not yet exist:
`factor-pc1`, `qwen35-thinking-trap`, `zoo-training-recipe`,
`superseded-claims`. All four are slugs the assignment or the wiki schema
specified; they are presumably still being written. Everything else resolves
(`tools/lint.py` reports no broken links originating in `behaviour/` beyond
those four, and no frontmatter problems in this directory).

---

## 2. Contradictions found between sources

Recorded, not resolved. Each is stated on the relevant page as well.

**C1. PC1's named scale.**
`qwen35/PREREG_steerfix.md` registers PC1 as an Agreeableness claim ("PC1 —
affect-centring vs procedural detachment ... Agreeableness slide survives").
`qwen35/analysis/steerfix_replication.json#PC1.named` is `"Conscientiousness"`,
and `qwen35/analysis/qual_pc.json#directions[PC1].judged_profile` repeats
"named Conscientiousness". The replication file is current. The damage half of
the PC1 prediction did replicate.

**C2. Two different angles for the alien direction.**
- `qwen35/analysis/alien.json#alien_k5.gap_deg` = 52.51479642189155, nearest
  trait `anxious` (also `analysis/alien_steer.json#gap.alien_k5` =
  52.51479642189158, and the docstring of `qwen35/analyse_alien_steer.py`).
  This is the gap inside the top-5 principal subspace.
- `qwen35/analysis/direction_gaps.json#alien_k5.deg` = 68.93287342956332,
  nearest `relaxed`, with `anxious` second at 69.15932422427636. This is the
  angle in the full sketch space with each adjective treated as a line.
The blog page's alien card is hard-coded with "68.9&deg; from the nearest word"
(`qwen35/build_blog_page.py`, `alien_card()`). Not a numerical contradiction but
a real risk of conflation; the two are different measurements in different
spaces.

**C3. Stage 2's relation to each trait's own direction.**
`qwen35/analysis/live_results.json#hyp[1].detail`: "cos(SFT increment, DPO
adapter) is +0.225 same-trait vs +0.015 mismatched, a 15x gap. Stage 2
re-encodes and amplifies each trait's own direction."
`qwen35/analysis/distil_data.json#corrections[0]`: "That gap is entirely the
deterministic 2x copy above. After removing it the correlation is +0.0002."
The correction is current; `live_results.json` was written 2026-08-29 14:19 and
never updated. Note the correction *itself* rests on the "2x copy" finding,
which was then withdrawn as a units error (C4) — so the mechanism is doubly
revised and only the residual-orthogonality number (+0.0002) and the RSA
structure claim in `distil_data.json#surprises[2]` should be carried forward.

**C4. "The persona adapters contain the DPO adapter twice."**
`qwen35/analysis/distil_data.json#surprises[0]` states it with
`proj(persona - stage-1, unit stage-1)/||stage-1|| = 1.001 +/- 0.015` across
100 traits against a mismatched control of 0.089 +/- 0.175.
`qwen35/distil_page/index.html` (2026-09-01) retracts it: "It was a units error
in our own analysis code. The sketch never read `alpha/r` from each adapter's
config ... The DPO strength in those adapters is correct." The retraction is
current; the data file still carries the withdrawn claim.

**C5. Stage 2's factor-clustering gain.**
Claimed 0.357 -> 0.548; corrected in
`qwen35/analysis/distil_data.json#corrections[1]` to "0.508 vs 0.548 — a
4-point gain, not 19. At k=1 the direction reverses." Geometry-section
territory, flagged here because it lives in a behaviour-adjacent file.

**C6. The AI-identity disclaimer count, 13 of 24.**
The withdrawn analysis attributed it to `mean_assistant_axis`;
`qwen35/analysis/qual_pc.json#directions[PC2].surprise` finds "it belongs to
PC2's negative pole, not to mean_assistant_axis", and
`#directions[mean_assistant_axis].surprise` records that on the corrected corpus
the negative side is flat at baseline (manual counts 8/7/8/7/4/3/1 across
-4/-2/-1/0/+1/+2/+4).

**C7. axis_Intellect's Conscientiousness cost.**
`qwen35/PREREG_steerfix.md`: "largest range, buys abstraction with
Conscientiousness ... Intellect increasing, Conscientiousness decreasing."
`qwen35/analysis/qual_axes.json#directions[axis_Intellect].surprise`: "Judged
Conscientiousness on this axis rises with alpha (+0.20 slope, 2.54 at -4 to
5.83 at +1) ... The debt in the corrected data runs the other way."

**C8. The alpha unit is not the same constant in both campaigns.**
`qwen35/STEER134_DESIGN.md`: "All alphas are in units of s_bar = mean adapter
norm = 1.6157 (from `results/gram_sweep.npz`, scale 2.0, 248 modules)."
`qwen35/steer_fix.py`: "`ref` is the mean single-adapter Frobenius norm";
the specs carry `ref` 0.8102592902648793 (`steer_spec.json`, n=100) and
0.8078003190997738 (`steer_spec2.json`, n=134). `analysis/sorh_projection.json#reference.ref_norm`
gives 0.8146967250408155 for "mean single trait-adapter **sketch** norm". The two
campaigns' alpha=1 differ by roughly a factor of two and nothing in the
repository reconciles them. **Do not compare alphas across campaigns.**

**C9. `analysis/qual_axes.json` has no per-alpha damage breakdown.**
The other three qual files carry `damage_markers.per_alpha.looping`; the five
`axis_*` entries do not. So `analysis/blog_data.json#degen` is empty for those
five, and the blog page's "intact over alpha -2, -1, 1, 2" line for the five
named axes (`stats_for()` in `build_blog_page.py`) is produced by an *absent*
record rather than by a measured zero. `analysis/qual_axes_notes.md` says damage
markers were "counted programmatically across all 24 prompts at all 7 alphas",
so the counts exist somewhere; they are not in the JSON.

**C10. Two claims in the assignment brief that the files do not support.**
- The brief describes the RL work as "GRPO on Dolci math / math+code with zoo
  LoRA-A init". Only the math+code run adopted zoo LoRA-A:
  `phase10_runs/rl_mix.log` contains `adopted zoo LoRA-A on 248/248`,
  `phase10_runs/rl_train.log` contains no such line, and the code that does it
  (`rl_capability.py:491-525`) postdates the math run. `eval_rl_persona.py`
  states outright that "the RL adapter has a different LoRA initialisation from
  the zoo".
- The brief suggests `distil` tested "prompted model vs OCT-trained LoRA". It
  did not: `distil_page` was titled "Where personality lives in a model's
  weights" and its sections are module holography, the DPO-twice claim,
  variance deflation, PCs, FA and steering. The prompted-vs-trained comparison
  is the blog page's "Prompting versus training" section, which is
  activation-space work.

---

## 3. Claims I believe are superseded

For merging into `pages/overview/superseded-claims.md`.

| superseded | by | date |
|---|---|---|
| `phase10_runs/steer_results.json` and `judged_steer.json` (200-token, thinking-on steering corpus) and the curves preserved at `analysis/distil_data.json#steer` | `steer_results_fix.json` / `judged_steerfix.json` | 2026-08-29 |
| the `zoo-steer2` and `zoo-steer3` runs (no results file survives either) | `zoo-steerfix2` / `zoo-steerfix3` | 2026-08-30 |
| `qwen35/distil_page/index.html` in its original form (backup at `.withdrawn-backup`) | `findings_page` + `monitor_page` + the 22 `direction_pages` | 2026-09-01 |
| `analysis/distil_data.json#surprises[0]` ("persona adapters contain the DPO adapter twice") | the withdrawal notice at `distil_page/index.html` — units error | 2026-09-01 |
| `analysis/live_results.json#hyp[1].detail` (the +0.225 vs +0.015 mechanism) | `analysis/distil_data.json#corrections[0]` | 2026-08-29 |
| the "stage 2 sharpens factor clustering 0.357 -> 0.548" figure | `analysis/distil_data.json#corrections[1]` (0.508 vs 0.548) | — |
| `qwen35/zoo_page/index.html` (a 51-adapter readout; `TOTAL = 51`) | `qwen35/site_traits/` (134 adapters) | 2026-08-24 |
| "13 of 24 AI-identity disclaimers on mean_assistant_axis" | `analysis/qual_pc.json#directions[PC2].surprise` — it is PC2's negative pole | 2026-08-29 |
| PC4 named after sycophancy | `build_blog_page.py`: "naming it after sycophancy was a mistake"; its highest loadings are `timid`, `self-pitying`, `fearful`, `guilty` | 2026-09 |

Not superseded but **incomplete**, and should not be cited as finished results:

- `qwen35/results/steer134_judged.json` — `coverage.complete: false`, 11,010 of
  28,664 units unjudged, many cells at n=4-7 against n_expected=12. The
  coherence-collapse argument built on it in `analysis/manifold_ideas.md` is
  labelled `unconfirmed` on `steering-results`.
- the `mathcode` GRPO run — `phase10_runs/rl_mix.log` ends at step 99 of 500;
  no `analysis/rl_sketches_mathcode.json` exists.

---

## 4. Gaps I could not source

1. **No behavioural comparison for the reward-hacks arms.** `zoo-sorheval` was
   meant to run `eval_rl_persona.py` on both arms;
   `phase10_runs/rl_persona_sorh_hack.json` exists,
   `rl_persona_sorh_control.json` does not, `sorh_eval.log` ends mid-run at
   `[checkpoint-31] done`, and no `judged_sorh*.json` exists. The hack arm's
   generations were never judged. Any claim about reward hacking's *behavioural*
   effect on the personality battery is unsupported.
2. **No producer script for five load-bearing JSONs.** `steerfix_replication.json`
   (the entire replication table, read by two builders),
   `judged_ceiling.json`, `functional_probe.json`, `adapter_effect.json`,
   `qual_pairs.json`. Also `page_data.json` and `monitor_page_data.json` are read
   but never written by checked-in code. The definitions of `slope2`/`sel2`/
   `coh2`/`named` are reconstructed from consumers and from the qualitative
   records that quote them; there is no formal definition anywhere in the repo.
3. **No builder for `manifold_page` or `distil_page`.**
4. **`axis_Conscientiousness`'s pre-registered prediction was never scored.** The
   criterion was "negative arm effect larger in magnitude than positive arm"; no
   file computes an arm-asymmetry statistic. Marked unconfirmed on
   `steering-results`.
5. **`analysis/qual_axes.json` per-alpha damage counts** (see C9).
6. **Condition-level judged means for base/stage1/persona.** Only
   `analysis/spider.json#base_trait` stores a base-condition mean; there is no
   file storing the stage1 or persona means, and wiki rule 1 forbids
   recomputing them from `judged_100.json`. So `judged-evaluations` describes the
   three-condition design but quotes no persona-minus-base effect size.
7. **Where the mixture `weights` in `steer_spec_mix.json#jobs[].weights` came
   from** — no builder for that spec is checked in.
8. **`analysis/trait_angles.json`** (185 bytes) and
   `analysis/alien_v_k5.npy` are consumed by the blog page but I did not trace
   their producers; they are geometry-side artefacts.

---

## 5. Numbers I quoted whose provenance is weak

Flagged in the pages themselves as well.

1. **r = 0.9999992** (the finite-difference validation of the scoring identity,
   on `scoring-identity`). `build_blog_page.py:1411` reads
   `corr = v.get("corr", 0.9999992)` from `analysis/align_validate.json`, and
   that file has **no `corr` key** — so the figure on the current blog page is a
   hard-coded fallback. It also appears in
   `.garden/notes/scoring-data-against-a-lora-direction.md`. The 24 x 3 raw rows
   are in `align_validate.json` and the figure is recomputable, but no file
   stores it.
2. **`verify.json#arm_vs_control_cos`** (0.715 / 0.717 / 0.711, averaged to
   "cosine 0.71" on the blog page). `analyse_verify.py`'s `json.dump` writes only
   `raw`, `sub`, `targets`, `hits`, `frac`. The key was added by something not
   checked in.
3. **FA_Warmth "7.3x on the eighteen clean prompts"** and **FA_Imagination
   "4.2 to 7.8 on its fourteen clean prompts"** — hard-coded strings in
   `build_blog_page.py`, no JSON key, no producer.
4. **The alien card's "68.9 degrees"** — hard-coded in `build_blog_page.py`,
   though it matches `direction_gaps.json#alien_k5.deg`.
5. **Everything from `steerfix_replication.json`** — the whole replication table
   on `steering-results`, including every `slope2`/`sel2` the blog page prints.
   No producer (gap 2).
6. **`judged_ceiling.json`** (`observed` 0.582, `frac_of_ceiling` 0.704) and
   **`functional_probe.json`** (0.581 within / 0.129 across, retrieval 0.032).
   No producer. `sb_full` at least checks out arithmetically as the
   Spearman-Brown extrapolation of `split_half_mean`.
7. **`adapter_effect.json`** — cited on `judged-evaluations` only as a file that
   exists; its keys' definitions are unconfirmed and nothing reads it.
8. **The STEER134 coherence numbers** quoted from
   `analysis/manifold_ideas.md` (fa1 0.42/10 at +2, random1 8.5 at +2, pc1 9.25
   at -1 vs 2.0 at -2). They come from cells the same document admits have "3-8
   of 12 expected judgments", against only two random controls. Marked
   unconfirmed.
9. **The monitor page's additivity ranges** ("reinforcing 0.13-0.15, opposing
   1.17-1.46"). Checked against `analysis/additivity.json#rows`: these are the
   extremes of the two groups, not their means, and only one of the two
   subtractive mixtures (`mix_Extr-Cons`) shows the large residuals — the other
   (`mix_Agre-Inte`) runs 0.470-0.724. The headline
   `median_resid_over_effect` 0.528 is solid; the sign-asymmetry gloss is thin.
10. **`rl_behavioural.json` shift figures.** They come from
    `judged_rl_persona.json`, whose repeat reliability on ten units was r=0.250
    (Extraversion), 0.643, 0.724, 0.948, 0.667 — against 0.78-0.90 on every
    larger judge run (`phase10_runs/judge_rl.log`). The largest shift
    (+0.462 sd at checkpoint-50) should not be read as established.
11. **`analysis/spider.json#base_trait` / `#base_steer`.** These are the only
    sourced condition means I found, but `build_spider_page.py` only reads the
    file; its producer is not checked in.

---

## 6. Two things worth a maintainer's decision

- **`analysis/distil_data.json` is a mixed file.** Parts of it are withdrawn
  (`surprises[0]`, `steer`), parts are still consumed as current by
  `build_findings_page.py` (`deflate`, `pcs`, `fa`), and parts are the corrections
  themselves. The withdrawal notice covers the *page*, not the file. Anyone
  quoting from it should check which key they are in.
- **The 200-token steering curves at `distil_data.json#steer` are the only
  surviving copy** of the withdrawn results (`steer_results.json` and
  `judged_steer.json` also survive, but the curves are pre-aggregated here). If
  the blog post wants a before/after figure for the thinking-default correction,
  that is where the "before" numbers are — and they must be labelled withdrawn.
