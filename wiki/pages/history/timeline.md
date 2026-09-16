---
title: Timeline
summary: A dated chronology of the project from the three Discord messages of 2026-08-12 through the wiki build of 2026-09-07 to the short post draft and the collaborator hand-off of 2026-09-16.
status: current
sources:
  - CONTEXT.md
  - /home/vibe12/projects/agent-harness/archive/discord-2026-08-31/rendered/notebook--ideas.md
  - /home/vibe12/projects/agent-harness/archive/discord-2026-08-31/rendered/projects--persona-cartography.md
  - /home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md
  - /home/vibe12/projects/agent-harness/memory/server/maintenance-log.md
  - qwen35/PHASE3_VERDICT.md
  - qwen35/HANDOVER.md
  - qwen35/plan.json
  - .garden/journal/2026-09-01.md
  - .garden/journal/2026-09-03.md
  - .garden/journal/2026-09-04.md
  - .garden/journal/2026-09-05.md
  - /home/vibe12/projects/.garden/journal/2026-08-31.md
last_verified: 2026-09-16
tags: [timeline, history]
---

# Timeline

One line per event. Times are UTC where a source gives one. Where nothing but a
file's modification time dates an event, the line says so.

**On the start date.** This page was commissioned as running "from 2026-08-08
(first session)". Nothing in this project's record supports 08-08: the earliest
dated artefact of any kind is Samuel's message of **2026-08-12 00:23:17**, and the
project record opens "eternalrecursion's idea from #ideas, 2026-08-12". 2026-08-02
is when the agent harness itself was started, and the archived Discord export
contains other channels' messages from 2026-08-01 onward. The chronology below
therefore begins on 08-12. Flagged in `_report_history.md`.

## Week 1 — control, and reading

| date / time | event |
|---|---|
| 08-12 00:23:17 | Samuel, in `#ideas`: "Persona cartography + model to act on gradient" |
| 08-12 00:23:50 | "But look up persona cartography lesswrong" |
| 08-12 00:24:04 | "And think about a model that takes in a gradient and spits out a gradient" — [[origin-and-question]] |
| 08-12 00:29:23 | The LessWrong post is located: arXiv 2607.07916, 10 Jul 2026 — [[persona-cartography-paper]] |
| 08-12 02:46:18 | `drift/results/drift_report.md` generated: the model organism passes, base 1.81 to plain 9.20 — [[drift-experiment]] |
| 08-12 ~06:10 | The "constraints were never binding" explanation is **falsified** by the A-GEM run and replaced |
| 08-12 06:13:22 | `drift/results/drift_scores.json` generated: all 19 regimes scored; every weight-space constraint null, KL at 1.93 |
| 08-12 17:30 | `pooling_check.py` run: the pooling artefact dies, the "97-99% orthogonal" reading is corrected, and the oracle direction's design flaw is found — [[lesson-shared-term-contamination]] |
| 08-12 18:14:27 | Gradient-probe reports generated: pooled +0.0489, layer 34 MRR 0.5080 — [[gradient-probe]] |
| 08-12 (late) | Register probe: the trait is about twice as legible as a planted fact; and the saturation result, from existing sketches at zero cost |
| 08-12 | LoRA rank spectra measured: rank 1 keeps ~49% of energy, effective rank ~3.8 — [[lora-structure-early]] |
| 08-13 | Interpolability: 47.7% of a held-out trait spanned by the other four, of which their mean direction alone gives 46.6% |
| 08-13 | Rank-truncation leave-one-out resolves the rank-1 puzzle (0.284 to 0.477, monotone) |
| 08-14 17:26:21 | The project channel `#persona-cartography` is created; the `#ideas` thread becomes read-only history |
| 08-14 17:56:59 | Samuel: "Is there any existing work on pca over LoRAs" |
| 08-14 17:58:50 | `CONTEXT.md` written and pinned, organised by current truth with superseded claims marked |
| 08-14 17:59:55 | Samuel asks the cost of running OCT over the 683-trait ideonomy list (corrected to 638) |
| 08-14 18:25-18:50 | Cost estimates rebuilt from measured invoices for 3B and 8B; RunPod and model-size ladder; better trait lists recommended |
| 08-14 22:55:32 | Samuel: "Go for it" — the teacher screen is approved |
| 08-14 22:57:39 | Samuel commissions the 100-trait sweep, then goes to sleep |
| 08-14 23:01 / 23:16 | Teacher-screen generations and judgements written; **the analysis was never produced** — [[teacherscreen]] |
| 08-14 ~23:10 | Pair generation for the 100-trait sweep launched — [[sweep100]] |
| 08-14 23:23 | Samuel: "Wait also please do full open character training pipeline overnight" — stage 2 added |
| 08-14 23:31 | The Goldberg 100 verified against the paper's own table; the Emotional Stability 6/14 asymmetry recorded |

## Week 2 — the 100-trait sweep

| date / time | event |
|---|---|
| 08-15 00:00-00:05 | Generation complete: 25,554 pairs, $1.82, zero errors; balanced to a 214-prompt intersection at a deliberate 16.3% cut; stage 1 launched 00:04 |
| 08-15 08:45 | The PCA result: PC1 evaluative (F=194 on keyed sign against 17.5 on factor), PC3 is Extraversion at 0.950, polarity test passes |
| 08-15 08:35-13:36 | Stage 2 hangs twice and is cancelled; the detection lesson is log mtime, not `pgrep` |
| 08-15 14:31 | The interactive page is built — and sits undeployed for four hours because nobody checked |
| 08-15 | Factor analysis revises the headline: all five factors recovered one-to-one under oblimin, none clearing 0.85; the evaluative axis largely dissolves |
| 08-15 23:36 | `sweep100/results/steer.md` generated: 12% bad-but-lucid at alpha +4 on PC1 against 0% for the matched control |
| 08-15 | Stage-2 transcripts inspected and found trait-free; the long-owed behavioural verification arrives as "these are weak persona directions that become personas when amplified" |
| 08-15 | The re-judge settles the judge criticism: the headline does not move |
| 08-15 ~23:40 | Epoch calibration (2 to 9 epochs changes nothing) and the initialisation finding (cross-init same-trait cosine 0.024, geometry RSA 0.995) |
| 08-16 00:28 | Three-seed result (later superseded by a clean rerun) |
| 08-16 00:43 | Behaviour gate at n=96 stops at 97 of 105 adapters: **OpenRouter credits exhausted (HTTP 402)** |
| 08-16 | "LoRAcles" corrected: the paper Samuel meant is De Schamphelaere et al., ICML 2026 workshop, not the security paper first cited |
| 08-16 11:30 | Contamination found in seed 1: four adapters were the 9-epoch calibration runs, skipped by a resume guard. Retrained rather than argued away |
| 08-16 11:54 | Behaviour gate, final: 71 of 105 pass, mean win rate 0.670 |
| 08-16 12:08 | Three-seed clean rerun supersedes 00:28; every headline unchanged to three decimals |
| 08-16 13:42 / 14:27 | Text-to-LoRA reported as losing to retrieval by about 2x |
| 08-16 ~18:40 | The learning curve is launched, with two of its own design errors caught first |
| 08-16 ~23:00 | The learning curve **overturns** the conclusion: 0.513 at n=80, the same model and data |
| 08-17 00:25 | Expanded 148-trait sweep trained, $2.77 |
| 08-17 01:46 | Converged at ~16.6k steps: hypernetwork 0.599 against retrieval 0.564. The earlier "loses 2x" was a statement about the step budget, three times over |
| 08-18 | **The deciding test**: weights recover the Big Five no better than the training text (0.750 against 0.731, 0/5 clearing 0.85) |
| 08-18 | Stage 2 found alive — 100 `__sft` adapters had existed all along; stacking them moves the geometry not at all (RSA +0.999) |
| 08-18 19:12 | Second base model: 100 adapters on Qwen3-4B, verified by tensor shape not metadata |
| 08-18 19:30 | The null replicates on the second base model (0.742 against text 0.731) |

## Week 3 — the Qwen3.5-4B experiment

| date / time | event |
|---|---|
| 08-19 | Samuel waives comparability with the old corpora ("feel free to upgrade packages to whatever makes sense") |
| 08-19 | Qwen3.5-4B module layout enumerated from the model: 248 adaptable linear modules; a conventional q/k/v/o + MLP list would have missed 120 |
| 08-19 | rsLoRA adopted for the new corpus on Samuel's instruction |
| 08-19 | The new standalone experiment is commissioned: "this isn't a replication, want to do things properly with qwen 3.5 4b" |
| 08-19 | The learning-rate problem found by a docs check: 5e-05 had been used since the first sweep where guidance is ~5e-06 |
| 08-19 13:49-14:06 | Trait sets drawn; two draws discarded (`traits_secondary_DISCARDED_draw1/2.json`) before the third is kept — [[discarded-secondary-draws]] |
| 08-19 15:30 | Phase 1 recorded as **PAUSED, not open at $4**: a $0.63 smoke test showed the estimate wrong by ~100x |
| 08-19 18:56 | Phase 1 reopened at a measured $19.17 on `z-ai/glm-4.5-air`; the $420 / $793 / $19 sequence recorded — [[lesson-thinking-default-trap]] |
| 08-19 19:10 | Three of Goldberg's published 100 (Verbal, Prompt, Complex) refused by the constitution writer, which had screened the **dictionary** sense of each word |
| 08-19 19:20 | Phase 1 killed at $1.44 over a circularity in the anchor block: it named one Goldberg marker per Big Five factor in every training document |
| 08-19 | The plan page goes live at `plan.161-35-77-84.sslip.io` |
| 08-20 00:14 | `base_config_snapshot.json` written (file mtime) |
| 08-20 00:53 - 10:43 | Nine phase-2 arms run: rsLoRA scale 16, alpha 16 (twice), beta 0.5, NLL, KL, plain LoRA, gate-clear — [[phase-two-recipe-search]] |
| 08-20 11:29:57 | **The 134-trait main sweep completes** (`qwen35/phase5_sweep.log`) |
| 08-20 12:31 | `PREREGISTRATION_phase3.md` written while the null arms are still training |
| 08-20 | Phase 6: a cluster test returns +0.015 on structure a signed axis test scores +0.175 on — [[lesson-test-shaped-like-the-wrong-hypothesis]] |
| 08-21 13:00-17:00 | Six hand-launched Modal apps named `oct-continue-persona-curvature`, $8.7978, never placed on an arm — [[costs]] |
| 08-21 | The harness blackout: 631 turns attempted, all failed at zero cost |
| 08-22 | Phase 3 closed; both nulls pass flat; the seed-paired floor fails at 0.0167 against a bar of 0.2447 and the trait-level claim is **withdrawn** |
| 08-22 22:40 | Phase 3 **reversed**: the bar was in the wrong units — [[lesson-bar-in-the-wrong-units]], [[seed-floor]]. `STEER134_DESIGN.md` written the same hour |
| 08-22 23:34 | Phase 7 steering runs and is judged: $40.05 of judging, 15,944 calls, against a $35 phase budget |
| 08-22 23:42 | Phase 10 v1 launched (`oct_stage2.py`, three traits) |
| 08-23 00:05 | **The ledger correction**: recorded $195.90 against measured ~$370.65 |
| 08-23 00:23:16 | The project tree becomes a git repository (commit `ba4d89b`) |
| 08-23 00:40 | The bare $195.90 is deleted rather than annotated (commit `f2e02b4`) |
| 08-23 ~04:25 | Phase 10 v1 dies at a three-hour function timeout having billed $27.90 |
| 08-23 04:59 | Per-run provenance recovered: 234 of 234 runmeta records, up from 8 (commit `fd27c2e`) |
| 08-23 05:10 / 05:17 | Commits `9d5785e` and `cab9aff`: "a phase costs its failures", and "quote the tool, never an instance" |
| 08-23 09:50 | Phase 7 results finally read: **the emergent-misalignment prediction fails on PC1**, and at \|alpha\| >= 4 the model is not producing sentences at all |
| 08-23 10:16 | Phase 10 complete and settled at $65.27, `FINAL: true` |

## Week 4 — reading what had been paid for

| date / time | event |
|---|---|
| 08-24 12:26 | `personality_lora_zoo_experimental_plan.md` last modified — the only date it has — [[experimental-plan]] |
| 08-24 20:36 | The `trait_curves` read: **the adapters do steer their own trait**, signed and monotone, with a flat same-alpha random control. It cost nothing; all 17,654 judge calls served from cache |
| 08-24 | `PHASE3_VERDICT.md` addendum: **objective mismatch in every control arm** — all 240 control adapters trained under plain sigmoid DPO — [[lesson-objective-must-travel]] |
| 08-24 20:36 | Last message in the archived `#persona-cartography` channel |
| 08-25 11:44 | The agent harness is **stopped and disabled** on Samuel's instruction; token consumption too high |
| 08-28 03:07 | `POST-BATCH3-TODO.md` written (file mtime): merge-record verification, disk, eval, upload, card fixes |
| 08-28 04:31 | `PENDING-CONTENT-REVIEW.md` written (file mtime): a content decision escalated to Samuel, with the uploader quarantining the affected files and continuing |
| 08-29 | The Qwen3.5 thinking-default trap found: three runs silently invalidated, 1,512 steering generations regenerated — [[lesson-thinking-default-trap]] |
| 08-29 | The HuggingFace 128-commits-per-hour limit found, presenting as a hang — [[lesson-hf-commit-rate-limit]] |
| 08-29 17:56 | `PREREG_steerfix.md` written **before** any corrected generation exists |
| 08-29 | The meter budget is raised to **$2,400** against a $5,000 grant, with $956 already spent; the app prefix widened to `pc-qwen35` after detached steering apps were found outside the kill switch |
| 08-31 | The guild is archived (3,062 messages, 153 containers), emptied, and rebuilt as v3 — [[harness-context]] |

## September

| date | event |
|---|---|
| 09-01 | The blog-post day: lexicon-coverage geometry, the unnamed direction and its controls, the data-alignment scorer, the evolutionary search and a verification training run. Spend ~$1,915 of $2,400; the day's whole program about $20 |
| 09-02 17:58 / 18:00 | `traits_alignment.json` and `PREREG_alignment.md` written (file mtimes) — the four alignment-relevant traits |
| 09-03 21:28/21:29 | Matched-objective retrains land: **pure seed floor +0.0181** (was +0.0166 with the objective confounded in), and the objective alone at the same seed gives **+0.954**. The seed does everything |
| 09-04 ~13:00 | Four jobs launched as systemd units: the two matched null arms, the three hole-word traits (the externally suggested candidate names), and N x N |
| 09-04 | The fourth silent non-travelling treatment is caught — by a print line designed for it |
| 09-04 | A $20-25 quote for the stage-2 arm corrected to about **$15 per trait**; 10 traits = $151; not launched without Samuel's word |
| 09-05 | N x N lands: **134 of 134 own-adapter rank 1** |
| 09-05 | Matched null arms analysed: shuffled +0.0001, permuted +0.0080 — identical to the confounded arms |
| 09-05 | The activation-space arm is built and run in an afternoon — [[actspace-overview]] |
| 09-05 | Samuel asks where the 134 words come from; the answer is in `plan.json` and a provenance paragraph is added to the page — [[lexicon-secondary-draw]] |
| 09-05 12:2x | Stage-2 seed-1 arm launched, 15 traits, plan $226.28 against a $240 budget; lands under budget with same-trait 0.067, 15/15 top-1 of 134, geometry across seeds r 0.98 |
| 09-07 | This wiki is built |
| 09-07 evening | Samuel asks for the geometry replicated on the exact persona adapters; the persona Gram job runs — [[full-oct-replication]] |
| 09-08 | The published `persona_exact` adapters found inert (doubled tensor-key prefix), repaired and re-uploaded — [[persona-merge-correction]], [[hf-artefacts]] |
| 09-08 | The factor solution replaces the principal components as the primary frame; every figure migrated — [[factor-first-migration]] |
| 09-08 | Stage two: the shared direction (15% of every adapter's squared norm) found, steered, and shown to be mostly generic to SFT on self-transcripts — [[stage-two-structure]], [[stage-two-shared-direction]], [[stage-two-exploration]] |
| 09-08 | Factor analysis of the two matched null arms (shuffled 0 factors, permuted 8); the sphere sweep redone on the factor chart; the OCEAN dials replicated four ways; the Inspect personality evals run — [[factor-analysis-null-arms]], [[sphere-sweep-factor-chart]], [[ocean-dials-replication]], [[inspect-personality-evals]] |
| 09-09 | Column-space structure: what the 0.018 seed floor hides — [[column-space-structure]]; Fisher norms of every steering direction — [[fisher-norms]]; reward-hack data scored before training — [[reward-hacks-data-scoring]] |
| 09-10 | Fisher-metric factor analysis, matched-dose steering, the rank sweep, probe adapters, the Dolci audit and the corrigibility-flag training, persona sliders, the iso-KL sphere, the sycophancy forecast; the post critiqued and trimmed — [[factor-analysis-fisher-metric]], [[matched-dose-steering]], [[rank-sweep]], [[dolci-data-audit]], [[dolci-flag-training]], [[persona-sliders]], [[sphere-sweep-iso-kl]], [[sycophancy-forecast]], [[post-critique-2026-09-10]] |
| 09-11 | Activation-weighted Gram across seeds and across stages — [[activation-weighted-gram]], [[activation-weighted-gram-stages]]; factor audit, Fearful withdrawal renamed Timidity — [[factor-audit-2026-09-11]]; the hole reframed as an uncovered region; emergent misalignment on bad medical advice — [[emergent-misalignment-medical]] |
| 09-12 | Goldberg-only factoring and the 34 lexicon words as a held-out set — [[goldberg-only-and-heldout-lexicon]] |
| 09-14 | Companion site redesigned with the three-dimensional chart and per-trait stage-two excerpts — [[built-pages-inventory]] |
| 09-15 | The 34 factored alone; best axis pair per Goldberg group; the illustrated post, its appendix split, then the short rewrite; the text-contrast factor analysis; the self-identification probe — [[best-axis-pairs]], [[post-draft]], [[text-contrast-factors]], [[self-identification-probe]] |
| 09-16 | Wiki freshness audit against the 2026-09-15 draft; collaborator start page, claims table and data map added — [[start-here-for-collaborators]], [[claims-and-evidence]], [[code-and-data-map]] |

Related: [[origin-and-question]], [[sweep100]], [[phase-two-recipe-search]],
[[costs]], [[harness-context]], [[method-lessons]].
