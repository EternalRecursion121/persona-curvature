# Report — history section (pre-zoo, infrastructure, costs, method lessons)

Written 2026-09-07 by the agent assigned everything before the 134-adapter zoo,
plus project infrastructure, costs and method lessons. Scope was
`wiki/pages/history/` excluding any slug beginning `paper-` or `literature-`
(a separate agent wrote those).

## Pages written (26)

**Pre-zoo experiments** — all `status: historical`

| slug | one line |
|---|---|
| `origin-and-question` | The three Discord messages of 2026-08-12 and the three shapes the question took |
| `drift-experiment` | The sycophancy model organism; the full regime table; the oracle-direction flaw; why the nulls are not bugs |
| `gradient-probe` | The fit-free content probe, the dead middle, the register effect, saturation not drift |
| `lora-structure-early` | The 34 OCEAN adapters on Qwen2.5-3B: rank spectra, compositional arithmetic, interpolability |
| `sweep100` | The 100-trait Qwen2.5-3B sweep: PCA, FA, steering, re-judge, behaviour gate, stage 2, the text null, and why the project moved |
| `teacherscreen` | The screen that ran, cost $8.94, and was never analysed; what was chosen instead |
| `phase-two-recipe-search` | The nine phase-2 arms and the six gates that decided the zoo recipe |
| `experimental-plan` | The 868-line plan, its three tiers and six hypotheses, item by item, done or not |

**Project-level** — `status: current`

| slug | one line |
|---|---|
| `timeline` | Dated chronology, 2026-08-12 to 2026-09-07 |
| `infrastructure` | The box, Caddy, the venv, Modal volumes, and one systemd unit per job |
| `costs` | Every cost figure in the record, with its file |
| `harness-context` | Claude Code sessions, the v1/v2/v3 Discord harness, pastlens, the external reviewer |
| `method-lessons` | Index of the thirteen lesson pages plus three that stayed one paragraph |

**Lessons** — `status: current`, thirteen pages:
`lesson-weight-magnitude-is-a-bad-proxy`,
`lesson-recoverable-is-not-verbalizable`,
`lesson-sweep-the-readout`,
`lesson-shared-term-contamination`,
`lesson-test-shaped-like-the-wrong-hypothesis`,
`lesson-coordinates-are-not-structure`,
`lesson-bar-in-the-wrong-units`,
`lesson-objective-must-travel`,
`lesson-thinking-default-trap`,
`lesson-hf-commit-rate-limit`,
`lesson-a-phase-costs-its-failures`,
`lesson-quote-the-tool-not-an-instance`,
`lesson-dry-run-every-landing`.

(Thirteen; `lesson-dry-run-every-landing` was added beyond the brief's list
because `.garden/journal/2026-09-01.md` records it as a named practice, and
`lesson-coordinates-are-not-structure` and
`lesson-test-shaped-like-the-wrong-hypothesis` because
`qwen35/HANDOVER.md` calls the first "the methodological finding, which may
outlast the result" and the project's own notes file all three of the
wrong-shape failures as one family.)

`tools/lint.py` reports **0 frontmatter problems and 0 broken wikilinks** from
these pages after the retargeting described below.

---

## Contradictions — both values, both paths

**1. The budget had three ceilings, and the record does not say so in one place.**
- `$200` — `CONTEXT.md` section 6 ("against the $200 set")
- `$1000` — `qwen35/plan.json#budget`
- `$2400` — `qwen35/zoo40_meter.sh` (`BUDGET=2400.00`, raised 2026-08-29 against a
  $5,000 grant)

Recorded as a three-stage supersession on `costs`. Anyone quoting "of the budget"
without a date is quoting one of three different denominators.

**2. Realised spend has five successive figures, and the last has no file.**
- `$195.90` recorded — sum of `qwen35/plan.json#phases[*].actual`; **superseded**
- `~$370.65` measured — `qwen35/HANDOVER.md` Money section,
  `qwen35/plan.json#ledger_defect_note` (read 2026-08-23 00:05)
- `$370.53` Modal experiment total —
  `/home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md`,
  2026-08-23 10:16
- `~$1,915` of $2,400 — `.garden/journal/2026-09-01.md`
- `$1,950.90` of $2,400 — `.garden/journal/2026-09-04.md`
- **`~$2,240` of $2,400 — maintainer-reported, 2026-09-07. No file on disk carries
  it.** Marked as maintainer-reported on `costs`.

**3. Two spend gaps, and they are not two estimates of one quantity.** At the
2026-08-23 00:05 read: Modal gap **$134.94** (recorded $173.89 against measured
$308.83) and total gap **$174.75** (recorded $195.90 against realised $370.65).
Both in `qwen35/plan.json#ledger_defect_note`. The file itself warns that quoting
them interchangeably makes it look as though one reader is $40 out.

**4. Phase 7 judging: $40.05 against $44.33.**
`qwen35/plan.json#phases[7].realised.openrouter` = **40.0529835**, sourced to
`results/steer134_judged.json`'s cost block and independently summed from
`steer134_judge_cache.jsonl`. The 2026-08-23 09:50 entry in the harness memory
gives **$44.33** (16.28M tokens) and says it was "corrected in-thread". Both cite
the same file. Not resolved here; `costs` carries both.

**5. Phase 10 carried five figures.** `$36.22` (mid-run lower bound quoted as an
outcome), `$31.30` (whole-workspace UTC-day total set against a phase nominal),
`$37.93` (six-app total at 05:16), `$64.23` (`FINAL: false`), `$65.27`
(`FINAL: true`). `qwen35/plan.json#phases[10].realised.source` states which
supersede which and why. `costs` uses **$65.27** as final.

**6. GPU-hours per trait: 4.20 against 4.77.** The first is `warm` alone while it
was the only finished run; the second is the three-trait average. Harness memory,
2026-08-23 10:16.

**7. Cross-seed same-trait cosine has three values, all correct in their own
scope.**
- `0.0167` — `qwen35/PHASE3_VERDICT.md`, the seed-paired floor as preregistered.
  The **reading** was withdrawn, not the number.
- `+0.01659` / `+0.0166` — the full 134x40 cross-Gram block,
  `qwen35/PHASE3_VERDICT.md` addendum 2026-08-22 22:40, from
  `results/cross_gram_full_root_x_data_null_seedpaired_s40.npz`. Confounded with
  the objective (see 2026-08-24 addendum).
- `+0.0181` — the matched-objective retrain, `qwen35/PHASE3_VERDICT.md` addendum
  2026-09-03 and `.garden/journal/2026-09-03.md`.

Owned by ``seed-floor``; `timeline` and `lesson-bar-in-the-wrong-units` carry
the dates.

**8. sweep100 stage-2 transcript shape.**
`/home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md`
(2026-08-18) says "32 transcripts x 8 turns each, VERIFIED from the saved
transcripts.jsonl". `sweep100/adapters_sft/active__sft/runmeta.json` records
`n_self_interaction: 16`, `n_self_reflection: 16`, `turns: 4`,
`n_transcripts: 32`. Most likely 4 exchanges rendered as 8 messages, but **not
resolved**; `sweep100` states both.

**9. Which page `persona-cartography.161-35-77-84.sslip.io` serves.** The written
record says "Site source sweep100/site/". `/etc/systemd/system/persona-cartography.service`
has `WorkingDirectory=.../sweep100/site2`. `site/` dates 2026-08-15 14:31 and
`site2/` 2026-08-15 21:29-23:38. **site2 is what is served.** Recorded on
`sweep100` and `infrastructure`.

**10. The ideonomy trait list: 683 against 638.** Samuel's link and message say
683; the page header sums to 638 (234 positive / 112 neutral / 292 negative) and
the page itself misprints the neutral count as 292. Effective distinct targets
after dedup about 540-570. Harness memory, 2026-08-14.

**11. PC1's factor-membership ANOVA: F = 0.71, 13.8 or 17.5.** All three are real
and measure different things — raw signed loadings (0.71, p = 0.58), |loading|
(13.8), pole-signed loadings (17.5). The 2026-08-18 state-of-project summary
quotes **0.71 alone** beside the keyed-sign F = 194, which overstates the
contrast; `sweep100/results/pca.md`'s corrected wording gives all three.
`sweep100` records the correction.

**12. The sweep100 spectrum differs in the second decimal between runs.**
`sweep100/WRITEUP.md` and `results/pca.md`: **22.8 / 14.5 / 9.7 / 6.1**. The clean
three-seed rerun (`results/seeds.json`, 2026-08-16 12:08) gives seed 0 as
**22.4 / 14.2 / 9.7**. Both quoted on `sweep100`; not a disagreement so much as
two runs, but a blog post copying one should not mix them.

**13. `qwen35/phase2_runs/results.json` is not phase 2's results.** It holds 40
records at `use_rslora: true`, `expected_scaling: 16.0`, `loss_type ["sigmoid"]`,
`kl_coef 0.0`, `seed: 1`, `corpus_is_seedpaired: true` — the seed-paired arm at
the superseded objective. Phase 2's own four-trait results are in
`phase2_runs/archive/results.json`. **`phase2_gates.py`'s `--results` default
points at the wrong file**, so a reader re-running the gates would judge phase 2's
health from the seed-paired arm. Flagged on `phase-two-recipe-search`; not
changed, since raw sources are read-only.

**14. The beta-0.5 arm's margin, in two units.** `qwen35/phase2_beta05.log` logs
TRL `rewards/margins` **64.31** for `imaginative`, against **40.55** for the
beta-0.1 arm in `qwen35/phase2_run2.log` — the logged quantity went up. The gate-5
docstring in `qwen35/phase2_gates.py` describes the same change as a margin
*falling* from **224 to 137**, which are raw margins recovered from
`-log sigmoid(beta*margin)`. The two are not the same scale. Both are recorded on
`phase-two-recipe-search`; not reconciled.

**15. The timeline start date.** The brief specified "from 2026-08-08 (first
session)". No artefact in this project supports 08-08; the earliest is Samuel's
message of **2026-08-12 00:23:17** in the archived `#ideas` channel, and the
project record's own first line dates it to 08-12. 2026-08-02 is when the *agent
harness* was started. `timeline` says so in a note and begins on 08-12.

---

## Superseded claims (for `pages/overview/superseded-claims.md`)

| claim | superseded by | where |
|---|---|---|
| "The weight-space constraints did nothing because they were never binding (trait direction held <=7% of the update's energy)" | The A-GEM run bound (cos^2 ~10%, firing on 49 of 76 steps) and still did nothing. The surviving statement is about *what* is constrained: a direction or a scalar fails, the output distribution works | `CONTEXT.md` 3.2; `drift-experiment` |
| "The oracle direction shows 58% of the update lies along the trait" | The cosine is algebraically fixed at 0.7634 by the norms alone; the defensible statistic is the **8.285%** excess over a 50% no-information baseline | `results/pooling_check.md` section F; `lesson-shared-term-contamination` |
| "97-99% orthogonal, so we cannot find the trait" | Right arithmetic, wrong reading. Median per-module energy 10.133% against a random-control floor of 0.000%, i.e. four orders of magnitude above chance in every one of 252 modules. The trait is pervasive and trivially findable | `results/pooling_check.md`; `lesson-weight-magnitude-is-a-bad-proxy` |
| "Planted facts are an upper bound on gradient legibility" | Withdrawn. The register effect is +0.0903 (z = +232.31) against a planted-fact effect of +0.0489 — traits are about twice as legible | `gradprobe/regroot/results/gradprobe.md`; `gradient-probe` |
| The problem-identity effect of +0.5442 in the register run | Discarded: same-problem pairs share the identical prompt verbatim. Does not touch the register effect | `gradient-probe` |
| "Stage 2 was abandoned as unusable" (2026-08-15) | 100 `__sft` adapters existed the whole time; the crash fired after training saved | harness memory 2026-08-18; `sweep100` |
| "Text-to-LoRA is beaten by retrieval, roughly 2x" (0.271 vs 0.564) | Converged at ~16.6k steps it reads 0.599 against 0.564. The earlier numbers were the step budget in the costume of convergence | harness memory 2026-08-17 01:46; `sweep100` |
| "More DPO epochs does nothing" | Stands as measured, but **qualified**: the learning rate was 5e-05 throughout where guidance is ~5e-06, and an LR ten times high produces the same saturation signature. The cheap re-check at 5e-06 was never run | harness memory 2026-08-19; `sweep100` |
| "Weight space reproduces the Big Five" | Weights recover the Big Five no better than embedding the training text (0.750 vs 0.731, 0/5 clearing 0.85), and this replicates on a second base model | `sweep100/results/text_vs_weights_fa.json`, `text_vs_weights_q3.json`; `sweep100` |
| "PC1 is a general evaluative factor of the model" | Stands as a fact about **unrotated extraction**; under oblimin the largest evaluative congruence falls from 0.820 to 0.571 and no rotated factor is predominantly evaluative | `sweep100/results/fa.md`; `sweep100` |
| Behaviour gate: "67 of 105 FAIL" (n=16) | A power artefact. At n=96, 71 of 105 pass, mean win rate 0.670; per-adapter win rates correlate r=0.880 between the runs | `sweep100/results/behaviour_n16.json` vs `behaviour.json`; `sweep100` |
| Three-seed result of 2026-08-16 00:28 | Superseded by the clean rerun at 12:08 after four contaminated seed-1 adapters were retrained. Every headline unchanged to 3 dp | `sweep100/results/seeds.json`; `sweep100` |
| "$195.90 actual against a $1000 ceiling" | ~$370.65 measured. The bare number was **deleted rather than annotated** — "a caveat does not travel with a number" | `qwen35/HANDOVER.md`, commit `f2e02b4`; `costs` |
| Phase 7 "must not be run" | Unblocked 2026-08-22 22:40: it applies a delta found in the seed-0 basis to the frozen base with matched controls and never needed cross-run transfer | `qwen35/HANDOVER.md`; `timeline` |
| The 200-token steering corpus | Regenerated: `enable_thinking=False` and 512 tokens. All 1,512 original generations were truncated reasoning preambles | `qwen35/PREREG_steerfix.md`; `lesson-thinking-default-trap` |
| "The coordinates are seed noise / every effect is void" | The coordinates are attenuated about 38x by the LoRA parameterisation and survive it | `/home/vibe12/projects/agent-harness/memory/notes/coordinates-are-not-structure.md`; `lesson-coordinates-are-not-structure` |
| "LoRAcles" attributed to arXiv:2601.11207 | Wrong paper. It is De Schamphelaere, Bauer, Nanda, Ong, ICML 2026 MI workshop, OpenReview `x9MbM7QmQN` | harness memory 2026-08-16; `origin-and-question` |
| "The failure cost of phase 10 was invisible" | It was not — all six instances bill under one app name. The blindness is a per-instance read and this ledger | `qwen35/plan.json#phases[10].realised.correction`; `lesson-quote-the-tool-not-an-instance` |

---

## Gaps

1. **The teacher screen was paid for and never analysed.**
   `teacherscreen/results/generations.jsonl` (2,880 rows, $0.119453) and
   `judgements.jsonl` (5,760 rows, **$8.820759**) are on disk;
   `teacherscreen/analyze.py` writes `results/teacher_screen.{json,md}` and
   **neither file exists**. So the screen's own questions — where a teacher starts
   failing by tier, whether amplify and suppress separate equally, whether the
   six-word synonym cluster is one direction — are unanswered on this project's
   data. One script run away, but running it would produce new numbers.
2. **`results/pca_over_loras.txt` is zero bytes**, and so is `results_pca.log`.
   The PCA-over-early-LoRAs spectra survive only in the harness memory record. See
   "weakly sourced" below.
3. **2026-08-25 to 2026-08-31 has no narrative record.** The harness was stopped
   2026-08-25 11:44; v3 went live 2026-08-31; the project `.garden/journal/`
   begins 2026-09-01 and the Discord archive's last message in this channel is
   2026-08-24 20:36. Everything in that week — the batch 2/3/4 stage-2 runs, the
   content review, the thinking-default discovery, the HF rate limit, the budget
   raise — is dated only by file mtimes (`POST-BATCH3-TODO.md` 08-28 03:07,
   `PENDING-CONTENT-REVIEW.md` 08-28 04:31, `PREREG_steerfix.md` 08-29 17:56) and
   by two auto-memory notes.
4. **All four of CONTEXT.md's open questions are still open.** Can retrievable
   content be made to speak; does DPO produce something prompting does not; does
   the KL result survive a non-saturated task; a second seed for the saturation
   decay curve. None was run.
5. **The experimental plan's Tier 1 items 7, 8 and 9** (PLS / reduced-rank
   regression, cross-validated performance against subspace dimension k,
   representation ablation) were not done, and **H6 — the causal test, project onto
   the subspace and project it out — was never attempted.** That is the largest
   single gap against the plan the project set itself.
6. **`personality_lora_zoo_experimental_plan.md` has no author and no date** other
   than its file mtime (2026-08-24 12:26:29 UTC). It is not in git and is not
   referenced from any script or journal I could find.
7. **`lesson-objective-must-travel` numbers its four occurrences, and the first
   two are inferred.** The sources give *counts* — `qwen35/PHASE3_VERDICT.md`'s
   2026-08-24 addendum says "the third occurrence of this project's signature
   failure mode" and `.garden/journal/2026-09-04.md` says "the fourth" — but
   nothing enumerates which two came first. I assigned `SW_BASE_MODEL` (2026-08-18)
   and `PC_LORA_ALPHA` (2026-08-20) to slots 1 and 2 on chronology alone;
   `PC_USE_RSLORA` defaulting wrong (`qwen35/HANDOVER.md` trap 1) is an equally
   plausible earlier member and would displace one of them. The four incidents are
   all real and all sourced; only the ordinal labels are inferred.
8. **`PENDING-CONTENT-REVIEW.md` records a decision escalated to Samuel** about
   self-harm-pattern content in the stage-2 transcript corpus
   (`self_interaction/temperamental.jsonl` at 10.4%). Whether it was resolved is
   not recorded anywhere I found. It belongs to the zoo agent's section but is
   flagged here because it is the only open *decision* left in the tree.

---

## Weakly sourced numbers

Every one of these is quoted somewhere in my pages with its source named; they are
listed so a blog post does not treat them as file-backed.

| number | only source | note |
|---|---|---|
| ~$2,240 of a $2,400 budget, 2026-09-07 | **maintainer report** | no file |
| ~$15 per trait for stage 2 | maintainer report | corroborated by `.garden/journal/2026-09-04.md`, which gives the same figure as a correction to a $20-25 quote and prices 10 traits at $151 |
| PCA over the early LoRAs: uncentered 60.5/11.6/10.5/9.3/8.1 and 57.6/7.7/5.5/5.1/4.4/2.4; centered 29.3/26.7/23.6/20.4 and 18.2/13.0/12.1/10.3/5.8/5.4; noise ratio 2.81 | `agent-harness/memory/projects/persona-curvature.md` only | **`results/pca_over_loras.txt` is empty** |
| Rank-spectrum energy at k (0.49 / 0.65 / 0.80 / 0.92 / 0.97), participation ratio ~3.8 | `CONTEXT.md` 3.7 and harness memory | I did not locate a JSON or log carrying them |
| Interpolability 0.477 span / 0.466 mean-direction, and the per-trait figures | `CONTEXT.md` 3.7 and harness memory | same |
| Rank-truncation series 0.284 / 0.374 / 0.441 / 0.471 / 0.477 | `CONTEXT.md` 3.7 and harness memory | same |
| A-GEM constraint diagnostics: cos^2 ~10%, peaking 12-14%, firing on 49 of 76 steps; alignment loss 2.18 -> 1.96 | `CONTEXT.md` 3.2 and harness memory | `agem_train.log` exists and was not parsed for these |
| Saturation quartile decay +0.0150 / +0.0061 / +0.0040 / +0.0004 and the drift-removal series | `CONTEXT.md` 3.6 and harness memory | not found in `gradprobe/regroot/results/` |
| "~$60 total: Modal ~$45, OpenRouter ~$15" as of 2026-08-18 | harness memory | no ledger file for that period |
| Phase-1 cost sequence $420 / $793 / $19 | `agent-harness/memory/server/maintenance-log.md` | a second-hand record of cartographer's figures |
| v1 harness spend (2.14B cache-read tokens, GBP 239.86, 2,006 of 3,597 messages) | `/home/vibe12/projects/.garden/notes/why-there-is-no-roster.md` and the auto-memory | measured at the time, not re-derivable now |

I also quoted several sweep100 figures (three-seed RSA, behaviour gate, hypernetwork
curve, stage-2 comparison) through the harness memory record rather than opening
`sweep100/results/{seeds,behaviour,hypernet_t2l*,compare_stages,stage2_only}.json`
directly. Those JSONs **do** exist; a lint pass that wants file-level citations
should re-cite from them.

---

## Scope and slug notes for the maintainer

1. **``zoo-training-recipe`` and ``trait-provenance`` do not exist.** The brief
   told every agent to link them; the zoo agent named the recipe page
   `stage-one-training-config` and split provenance across
   `zoo-construction-overview`, `goldberg-100-primary-traits`,
   `lexicon-secondary-draw`, `six-refused-traits` and `discarded-secondary-draws`.
   Something already rewrote most of my `zoo-training-recipe` links to
   `stage-one-training-config` mid-session; I finished that rename and retargeted
   my five `trait-provenance` links to the closest real pages. `tools/lint.py`
   still reports **150 broken `trait-provenance`** and **16 broken
   `zoo-training-recipe`** links from other sections. A wiki-wide rename or two
   redirect pages would settle it.
2. **`timeline` is in `history/`, not `overview/`.** `wiki/CLAUDE.md`'s layout puts
   timeline under `overview/`; my brief assigned it to `history/`. Written as
   assigned. Move it or leave it, but it should not be duplicated.
3. **Three likely overlaps with the zoo agent's section**, all written from
   different sources and not reconciled:
   - `history/costs` against `zoo/zoo-spend-ledger`
   - `history/infrastructure` against `zoo/modal-volumes` and `zoo/hf-artefacts`
   - `history/phase-two-recipe-search` against `zoo/phase2-recipe-selection`
   Mine are the *search* and the *whole-project* views; theirs are presumably the
   settled zoo state. Worth a read-through for disagreeing digits.
4. **`history/sweep100`'s note on `sweep100/site2` overlaps
   `behaviour/built-pages-inventory`**, which the brief assigned elsewhere. Mine
   records only that site2 is what the systemd unit serves and that it is
   historical.
5. `wiki/CLAUDE.md` allows five `status` values; `tools/lint.py` accepts
   `['current', 'historical', 'superseded', 'unconfirmed', 'withdrawn']`. They
   agree. All my pages use `current` or `historical` only.
