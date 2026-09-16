---
title: Open questions
summary: What the project has not established - unexplained results, experiments never run, provenance that cannot be recovered, and decisions still owed by Samuel - each marked by kind and pointing at the page that documents it.
status: current
sources:
  - qwen35/PHASE3_VERDICT.md
  - qwen35/analysis/crossseed_arms_stage2.json
  - qwen35/results/decomposition_shuffled_matched.json#test2b
  - qwen35/results/steer134_judged.json#coverage
  - qwen35/paper_notes.md
  - qwen35/PENDING-CONTENT-REVIEW.md
  - qwen35/POST-BATCH3-TODO.md
  - personality_lora_zoo_experimental_plan.md
  - CONTEXT.md
  - .garden/journal/2026-09-05.md
  - qwen35/analysis/sorh_behavioural.json
last_verified: 2026-09-16
tags: [overview, open-questions, unconfirmed]
---

# Open questions

What the project has not established. Each entry is marked by kind:

- **unexplained result** - it was measured, it is real, and nothing accounts for
  it;
- **never run** - the experiment or analysis was specified and does not exist;
- **provenance gap** - a number or artefact exists but its origin cannot be
  recovered from anything on disk;
- **decision pending** - it is waiting on Samuel, not on data.

Companion pages: [[superseded-claims]] for claims the project reversed,
[[source-contradictions]] for files that disagree.

---

## In flight

Nothing is in flight as of 2026-09-16. The one item that was is closed below.

**The full OCT persona replication.** RESOLVED 2026-09-08. Was running as of **2026-09-07**. Samuel asked
for the stage-1 weight-space geometry to be replicated on the full OCT persona
(stage-two merged) adapters. Modal jobs `zoo-grampersona` (134 x 134 exact
persona Gram) and `zoo-personaseed1` (15 exact seed-1 personas, then persona seed
0 against seed 1, seed-1 within, and stage-1 against persona); analysis in
`qwen35/analyse_fulloct.py`. Results are on [[full-oct-replication]]: the persona arrangement is the stage-one
arrangement (off-diagonal r 0.9915) and 15 of 15 second-seed personas identify
themselves. Every geometry result on this wiki is still a stage-1 result unless
it says otherwise - see [[stage-one-versus-stage-two-clarification]].

*Side effect worth knowing:* the container meter had been static at $2,240.52
since 2026-09-05 and started climbing again when these jobs launched
(`qwen35/phase10_runs/zoo40_meter.log`). Any spend figure quoted from now on
needs its read time. See [[zoo-spend-ledger]].

---

## Unexplained results

**The stage-two attenuation slope is 4.6 times the prediction.**
Stage one matches the analytic prediction: slope 0.0265 against r/d = 64/2560 =
0.025. Stage two at a second seed gives **0.116**, with a shared component that
every stage-two adapter carries. `qwen35/PHASE3_VERDICT.md`'s 2026-09-05 addendum
calls it unresolved and nothing since has explained it. Page:
[[stage-two-second-seed]]; also [[seed-floor]].

**Stage-two separation is negative, and no document says so.**
`qwen35/analysis/crossseed_arms_stage2.json#[0]`: `min_same` =
0.04248138800846624 sits **below** `max_off` = 0.05140579106757132, giving `sep`
= -0.00892440305910508. Stage one has positive separation in both arms, and the
2026-08-22 addendum makes "perfect separation, no overlap at all" one of its
three structural facts. The 2026-09-05 stage-2 addendum reports 15/15 top-1 and
does not report the separation at all. Page: [[stage-two-second-seed]].

**The matched shuffled null has a nominally significant clustering ARI.**
`qwen35/results/decomposition_shuffled_matched.json#test2b."leading-component
removed"`: `ARI` = **0.05977870383675816** at `ari_p` =
**0.0009995002498750624**. The original shuffled arm gave `ARI` =
0.00438485865621544 at `ari_p` = 0.36231884057971014. The 2026-09-05 addendum
says the objective mismatch "changed nothing to three decimal places", which is
true of the headline TEST 1B and TEST 2 and not of this. A structureless control
should not cluster. Page: [[null-controls]].

**Whether the hole is a real gap in the language or in the sample.**
The widest gap in the 134 words' coverage is 52.5 degrees at k = 5 (68.9 in the
full sketch space). Three candidate English words were trained for it and do not
agree with each other; the activation-space transplant works against a permuted
null. The page's verdict is "suggestively, not convincingly, `insouciant`", and
the project's own reading is now "a direction the *adapters* leave open" rather
than "a character English has no word for". Status on the page is `unconfirmed`.
Page: [[hole-words]]; the withdrawal is [[superseded-claims]] item F5.

**Extraversion and Emotional Stability come out rotated, and nobody knows how
stable that is.**
Three of five Big Five axes recover cleanly; E and ES emerge as withdrawal and
arousal instead. No Goldberg congruence clears 0.85 in any solution. Whether the
rotation is a property of the adapter cloud, of the oblimin rotation, or of the
6/14 keying imbalance in Emotional Stability (see [[source-contradictions]] S10)
has not been tested. Pages: [[explainer-big-five-mapping]], [[factor-analysis]],
[[factor-axis-extraversion]], [[factor-axis-emotional-stability]].

**The STEER134 coherence-collapse argument.**
`qwen35/analysis/manifold_ideas.md` builds an argument about coherence collapsing
at high alpha from `qwen35/results/steer134_judged.json`, whose
`coverage.complete` is **false**: 11,010 of 28,664 units are unjudged and many
cells sit at n = 4-7 against `n_expected` 12, against only two random controls.
Marked unconfirmed. Either finish the judging or drop the argument. Page:
[[steering-results]].

**LoRA-A drifts 4.5% over 93 steps, somewhere.**
`qwen35/build_monitor_page.py` line 112 says so; every analysis JSON says
0.0146 / 1.5%. Whether 4.5% is a different population or a superseded
measurement is not recoverable. [[source-contradictions]] S4; page
[[adapter-effect-and-drift]].

**Is Persona Cartography's PC10 the same object as the Assistant Axis?**
PC10 "cleanly separates the baseline (the model with no LoRAs) from the others"
and reads qualitatively as self-reference. A base-versus-personas direction is
what a grand-mean direction is, up to sign and centring, and
`qwen35/steer_qwen35.py` calls the grand mean the weight-space analogue of the
Assistant Axis. One qualitative paragraph on 11 points against an
activation-space result is not an identity. Written up as a resemblance. Page:
[[paper-assistant-axis]].

---

## Never run

**The enumerated-anchor ablation.**
`constitution_enumerated` exists on every accepted entry of
`qwen35/constitutions.json` and `qwen35/anchor_constitutions.py` describes the
arm, but no corresponding corpus and no `phase2_runs/results_*` file exists. This
is the ablation that would show whether naming one Goldberg marker per factor in
the anchor could have manufactured the Big Five result. Page:
[[constitution-anchor-revision]].

**The neutral-constitution null.**
`qwen35/paper_notes.md` section 3.9 recommends it as "the null that a reviewer
will ask for". The phrase appears nowhere else in the repository. Page:
[[recipe-vs-source-papers]], [[null-controls]].

**The teacher screen was paid for and never analysed.**
`teacherscreen/results/generations.jsonl` (2,880 rows, $0.119453) and
`judgements.jsonl` (5,760 rows, **$8.820759**) are on disk;
`teacherscreen/analyze.py` writes `results/teacher_screen.{json,md}` and neither
file exists. So the screen's own questions - where a teacher starts failing by
tier, whether amplify and suppress separate equally, whether the six-word synonym
cluster is one direction - are unanswered on this project's own paid-for data.
One script run away, but running it produces new numbers. Page:
[[teacherscreen]].

**RESOLVED 2026-09-08 - the reward-hacks arms now have a behavioural
comparison.** The entry read: the control arm produced no file, `sorh_eval.log`
ended mid-run at `[checkpoint-31] done`, no `judged_sorh*.json` existed, and any
claim about reward hacking's behavioural effect was unsupported. The cause was a
manual `systemctl stop` at `2026-09-01T14:34:24`, not a missing adapter.
`zoo-sorheval2.service` re-ran the control arm on 2026-09-08
(`qwen35/phase10_runs/rl_persona_sorh_control.json`), both arms were judged blind
(`qwen35/phase10_runs/judged_sorh.json`, 192 records, 0 failed calls) and
analysed (`qwen35/analysis/sorh_behavioural.json`). The answer is a null: both
arms fall about a point of judged Conscientiousness and Intellect below base, but
nothing separates hack from control after Holm correction. Page:
[[reward-hacks-arms]].

**The reward-hacks behavioural effect is confounded with response length.**
Both arms collapse from 305.4167 words to about 70
(`qwen35/analysis/sorh_behavioural.json#text`), and the judge's Conscientiousness
and Intellect scores correlate with cleaned response length at Spearman 0.4942
and 0.3965 over the 192 records
(`#length_vs_factor_spearman.rho`) - exactly the two factors that dropped. A
length-matched decode would be needed to say the fine-tune changed personality
rather than verbosity. *unexplained result.* Page: [[reward-hacks-arms]].

**The math+code GRPO run stopped at step 99 of 500.**
`qwen35/phase10_runs/rl_mix.log` ends there and no
`analysis/rl_sketches_mathcode.json` exists. It is also the only RL arm that
adopted the zoo's LoRA-A (`adopted zoo LoRA-A on 248/248`), so it is the only one
whose weight-space comparison to the zoo is on a shared basis. Page:
[[rl-capability-and-persona-drift]].

**A Frobenius-norm table comparable to Persona Cartography's Table 1.**
`qwen35/paper_notes.md` section 3.8 calls it "near-free and PC reports it", and
says that if this project's 140 norms are not in a tight band then every "sum the
scales" intuition from that paper is invalid here. No such table exists in
`qwen35/analysis/*.json`. Now that rsLoRA is off and the scaling conventions
match ([[superseded-claims]] item L-1), the comparison to their 6.08-6.53 band is
directly meaningful. **The cheapest remaining cross-paper number.** Pages:
[[persona-cartography-paper]], [[recipe-vs-source-papers]].

**`axis_Conscientiousness`'s preregistered prediction was never scored.**
The registered criterion was "negative arm effect larger in magnitude than
positive arm"; no file computes an arm-asymmetry statistic. Page:
[[steering-results]].

**Stage-two adapters were never run in activation space**, and the activation arm
produced **no behavioural measurement at all** - four responses per trait were
kept for eyeballing and none was judged. Pages: [[actspace-overview]],
[[actspace-method-notes]].

**No weight-space cross-seed Procrustes fit.**
The only Procrustes in the repository is in activation space
(`qwen35/analysis/actspace_geometry.json#windows.resp.primary.procrustes_r2`).
The cross-seed arrangement result rests entirely on the Gram correlation and the
replicated decomposition. Page: [[cross-seed-geometry]].

**An 8B replication arm** (`paper_notes.md` 3.3) and a **LIMA general-prompt
arm** (3.5). Neither exists; 3.5 is the largest unresolved divergence from both
source papers. Page: [[recipe-vs-source-papers]].

**The experimental plan's largest items.**
`personality_lora_zoo_experimental_plan.md` (repository root) Tier 1 items 7, 8
and 9 -
PLS / reduced-rank regression, cross-validated performance against subspace
dimension k, representation ablation - were not done, and **H6, the causal test
(project onto the subspace and project it out), was never attempted.** That is
the largest single gap against the plan the project set itself. Page:
[[experimental-plan]].

**All four of `CONTEXT.md`'s open questions.**
Can retrievable content be made to speak; does DPO produce something prompting
does not; does the KL result survive a non-saturated task; a second seed for the
saturation decay curve. None was run. Pages: [[origin-and-question]],
[[drift-experiment]], [[gradient-probe]].

**The DPO-epochs re-check at LR 5e-06.**
"More DPO epochs does nothing" stands as measured, but the learning rate was
5e-05 throughout where guidance is about 5e-06, and an LR ten times high produces
the same saturation signature. The cheap re-check was never run. Page:
[[sweep100]].

**The N x N heatmap figure.** (Still not built as of 2026-09-16; the companion's
methods page prints the 134 of 134 statistic and the post carries it as Appendix A8.)
Named in chat as the one genuinely new figure the site needs
(`981fa3b5`:13021, 13065). The matrix exists in
`qwen35/analysis/nxn_summary.json`; no figure has been built. Page:
[[n-by-n-scoring]].

**A literature search for prior art doing factor analysis on weight deltas.**
`paper_notes.md` section 4 item 8 records it as not found, and it was not a
search. Still not done. Page: [[literature-factor-analysis-methods]].

---

## Provenance gaps

**Eight load-bearing JSON files have no producing script anywhere in the
repository.**
`qwen35/analysis/module_holography.json`, `polarity_deflation.json`,
`adapter_effect.json`, `intrinsic_coords.json`, `steerfix_replication.json`,
`judged_ceiling.json`, `functional_probe.json`, `qual_pairs.json`. Also read but
never written by checked-in code: `page_data.json` and `monitor_page_data.json`.
Consequences by file:

- `steerfix_replication.json` carries the **entire** steering replication table
  and is read by two builders; the definitions of `slope2`, `sel2`, `coh2` and
  `named` are reconstructed from consumers, not stated anywhere.
  [[steering-results]].
- `module_holography.json` is the only source for the one-module-classifies-factor
  result (0.638 against chance 0.202, all 248 reaching 0.76).
  [[module-holography]].
- `polarity_deflation.json` is where 4.6% comes from and the 46% figure cannot be
  derived from it. [[polarity-and-bipolarity]]; [[source-contradictions]] S18.
- `judged_ceiling.json` (`observed` 0.582, `frac_of_ceiling` 0.704) and
  `functional_probe.json` (0.581 within / 0.129 across, retrieval 0.032) carry
  the judge-reliability ceiling. `sb_full` at least checks out arithmetically as
  the Spearman-Brown extrapolation of `split_half_mean`.
  [[judged-evaluations]].
- **`adapter_effect.json` is owned by nobody.** The geometry section concluded it
  is probably behavioural and passed it to behaviour; behaviour cited it only as
  a file that exists; the trait generator printed its five fields verbatim with a
  statement that their meaning is not established. Its keys (`sim_base`, `sim_s1`,
  `rep`, `leak`, `chars`) are undefined and nothing on the blog page reads it.
  [[adapter-effect-and-drift]], [[judged-evaluations]].
- `intrinsic_coords.json` has neither producer nor consumer.
  [[pca-and-scree]].

**The sketch-validation correlation is stored nowhere.**
Two built pages quote two different numbers for it (0.9996 and 0.99944) and no
file holds either. It is the justification for reading sketch angles as
weight-update angles. [[source-contradictions]] S5; page [[geometry-overview]].

**The scree `real` curve was never regenerated from the exact Gram.**
`#real` is byte-for-byte identical in `analysis/scree_null.json` and
`analysis/scree_null_matched.json`, and equals
`analysis/geometry_stage1.json#explained_var_top10`, which is computed over the
100 labelled traits from the `stage1_k32` **sketches**. The finding is recorded
in `.garden/journal/2026-09-05.md` and nowhere in the data. Either regenerate it
from `results/gram_sweep.npz` or write one line into the JSON saying it is
deliberately the 100-trait sketch spectrum. Page: [[pca-and-scree]].

**The per-alpha damage counts for the five `axis_*` directions.**
`analysis/qual_axes.json` has no `damage_markers.per_alpha.looping` where the
other three qual files do, so `blog_data.json#degen` is empty for those five and
the blog's "intact over alpha -2, -1, 1, 2" prints from an absent record.
`qual_axes_notes.md` says the counts were made programmatically across all 24
prompts at all 7 alphas. They are not in the JSON. Page: [[steering-results]].

**Condition-level judged means for base, stage 1 and persona.**
`qwen35/phase10_runs/judged_100.json` holds all 7,200 individual judge records;
no file stores the three condition means. `analysis/spider.json#base_trait` is
the only sourced condition mean found. Wiki rule 1 forbids recomputing them, so
no persona-minus-base effect size is quoted anywhere. Page:
[[judged-evaluations]].

**The activation cross analysis printed its class medians and never saved
them.**
`qwen35/analyse_actspace_cross.py` prints full per-class median tables; no log
in `phase10_runs/` has them. The medians survive only in
`qwen35/ACTSPACE_RESULTS.md` and the rendered blog page - and the **`same
factor, same keying` class (n = 14) has no published medians at all**. Page:
[[actspace-cross]].

**`results/runmeta_actspace.json` does not exist**, so decompose TEST 6 (the
training-strength nuisance covariate) is `unavailable` for the activation Gram.
The log says it plainly: "The confound is then UNTESTED, which is not the same as
ruled out." Page: [[actspace-method-notes]].

**The seven later adapters were trained on unanchored constitutions, against a
document that says "same recipe".**
`qwen35/PREREG_alignment.md` says the four alignment adapters use "Same recipe as
the 134-adapter zoo". Their constitutions, and the three hole ones, were
generated after the 2026-08-19 anchoring migration and never put through it:
their `constitutions.json` entries carry only a `constitution` field and the text
does not end with the generic anchor paragraph that all 134 carry.
`gen_pairs.py` conditions the teacher on that field. Page:
[[alignment-and-hole-traits]].

**Whether the Hugging Face model repository is public.**
`qwen35/upload_zoo_batched.py:177` prints "Repo still PRIVATE." as a **hardcoded
literal**, not a live check. The dataset repo's status is file-sourced
(`analysis/hf_dataset_audit.json#private: false`). No file records a model-repo
inventory or visibility read after 2026-08-29. Nor does anything record whether
the alignment and hole adapters are on the Hub at all -
`upload_zoo_batched.py` builds its job list from the `pc-qwen35-sweep` and
`pc-qwen35-oct2` volumes only, which do not contain them. Pages:
[[hf-artefacts]], [[alignment-and-hole-traits]].

**Two cost files were overwritten by later runs.**
`qwen35/constitutions_cost.json` was overwritten by the three-trait hole run
($0.013221, 3 calls), so the 140-trait constitution cost exists nowhere. The
zoo's own `data/_usage.json` and `_dropped.json` were overwritten by the
alignment run; the zoo's pair-generation totals survive only in `genpairs*.log`,
which give four per-pass figures and no total. Pages: [[constitution-generation]],
[[dpo-pair-generation]], [[zoo-spend-ledger]].

**One week has no narrative record.**
2026-08-25 to 2026-08-31: the harness was stopped 2026-08-25 11:44, v3 went live
2026-08-31, and the project journal begins 2026-09-01. The batch 2/3/4 stage-two
runs, the content review, the thinking-default discovery, the HF rate limit and
the budget raise are dated only by file mtimes and two auto-memory notes. Page:
[[timeline]], [[harness-context]].

**Which stage-two batch `warm` went through.** It is in no `zoo-*.service` unit;
it appears in the three-trait pilot results file and in `merge_audit.json`, and
is inferred from those two. Page: [[stage-two-introspection]].

**Why the six drawn-but-unbuilt lexicon words were dropped.** No source records
it. [[source-contradictions]] S11; pages [[six-refused-traits]],
[[lexicon-secondary-draw]]. *Resolved 2026-09-10:* [[six-refused-traits]] quotes the writer's `NOT_A_TRAIT` reason for each of the six from the pre-anchor backup; what stays open is F10 on [[superseded-claims]].

**Where the base model came from.** The choice of Qwen3.5-4B is not made in any
of the thirteen Claude Code transcripts; the earliest project turn already asks
what the adapters were trained on. It predates 2026-08-22 and would have to be
recovered from the repository or the old harness memory. Likewise the choice of
Goldberg's 100 unipolar markers and the draw of the 40 lexicon words: Samuel asks
about the set twice as a question, never as a decision. Page: [[decisions-log]],
[[origin-and-question]].

**Whether Emotional Stability's 6/14 keying is Goldberg's or a transcription
error.** Not checked, and it affects every per-factor statistic that assumes
balanced keying. [[source-contradictions]] S10; page
[[goldberg-100-primary-traits]].

**The ordinal labels in the "objective must travel" lesson are inferred.**
The sources give counts - the 2026-08-24 `PHASE3_VERDICT.md` addendum says "the
third occurrence of this project's signature failure mode" and
`.garden/journal/2026-09-04.md` says "the fourth" - but nothing enumerates which
two came first. `SW_BASE_MODEL` and `PC_LORA_ALPHA` were assigned to slots 1 and
2 on chronology alone; `PC_USE_RSLORA` is an equally plausible earlier member.
All four incidents are real and sourced; only the ordering is inferred. Page:
[[lesson-objective-must-travel]].

**Persona Cartography's venue is unconfirmed** (the checkout carries
`paper/neurips_2026.sty` and a "camera-ready switch" HEAD commit, and no
acceptance record was found), and on [[literature-big-five]] **eleven entries are marked `[not verified]`**
against six `[verified]` - the eleven were written from general reference, not
confirmed against a publisher record. Pages:
[[persona-cartography-paper]], [[literature-big-five]].

**Several pre-zoo numbers survive only in the old harness memory.**
`results/pca_over_loras.txt` and `results_pca.log` are **zero bytes**, so the
PCA-over-early-LoRAs spectra, the rank-spectrum energies, the interpolability
figures, the rank-truncation series, the A-GEM constraint diagnostics and the
saturation quartile decay exist only in
`/home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md` and
the repository-root `CONTEXT.md`. Pages: [[lora-structure-early]], [[drift-experiment]],
[[gradient-probe]], [[costs]].

---

## Decisions pending

All are Samuel's, and none is waiting on data.

**Co-authorship for the external reviewer.** Raised in the review and not answered.
Page: [[external-review]], [[decisions-log]].

**The Persona Cartography reframing the review asked for.** Partial credit exists on
the live page; the fuller rewrite does not. Page: [[external-review]].

**Hosting, and a static LessWrong version.** The review asked for a hosted version.
As of 2026-09-07 the zoo's blog page is not served: the three Caddy routes in
`/etc/caddy/Caddyfile` are this wiki, `persona-cartography.161-35-77-84.sslip.io`
on port 8091 serving the historical `sweep100/site2/`, and
`plan.161-35-77-84.sslip.io` on 8092 serving `qwen35/site/`, the phase plan page.
Page: [[external-review]], [[blog-site-map-idea]], [[infrastructure]].

**The short-post-plus-website plan, and per-trait pages.** The ten-page site map
of 2026-09-07 is a proposal; the site has not been built. The 141 trait pages in
this wiki are the closest thing that exists. Page: [[blog-site-map-idea]],
[[traits-index]].

**"yep train all 40 please" - approved 2026-08-29, never executed.**
The zoo has 34 lexicon traits. Whether to train the remaining six, or to close
the instruction as overtaken by the constitution writer's refusals, is undecided.
Pages: [[decisions-log]], [[provenance-feedback]], [[six-refused-traits]];
history at [[superseded-claims]] item F10.

**The content-publication policy.**
`qwen35/PENDING-CONTENT-REVIEW.md` puts three options to Samuel and recommends
option 2 (publish minus the quarantined files), specifically over self-harm-pattern
content in the stage-two corpus (`self_interaction/temperamental.jsonl` at
10.4%). The 2026-08-30 upload log shows **option 1** in effect - everything
uploaded, previously flagged rows cleared - and
`qwen35/phase10_runs/adjudications.json` holds 60 written clearance decisions.
Nothing on disk records a decision by Samuel. This is the only open decision that
has already been acted on. Pages: [[zoo-build-governance]], [[hf-artefacts]].

**Whether to re-cut the two published Hugging Face cards.** Both are stale and
public; see [[superseded-claims]] section 1d. Page: [[hf-artefacts]].

---

Related: [[superseded-claims]], [[source-contradictions]], [[method-lessons]],
[[experimental-plan]], [[how-to-read]].

## Added 2026-09-08

- **Resolved: published exact personas were inert.** The `persona_exact/` adapters on the Hub carried a doubled tensor-key prefix and loaded through PEFT with no effect (tested). Repaired on the volume and re-uploaded with Samuel's approval; verified by header reads. See [[full-oct-replication]], [[hf-artefacts]], [[lesson-verify-the-artefact-loads]].
- **Unexplained result: the stage-two shared direction.** Fifteen percent of every stage-two adapter's squared norm lies along one common direction, orthogonal to anything in stage one. No behavioural test and no control SFT exists for it. See [[stage-two-structure]].
- **Never run: stage-two factoring with the shared direction projected out**, and a control SFT on self-generated but trait-free transcripts. See [[stage-two-structure]].
- **Never run: factor analysis at the second LoRA seed.** Whether independently
  factoring the 40 seed-1 adapters recovers the same five factors. The
  per-direction check that was run instead is uninformative; the matched-objective
  arm has no 40 x 40 within-arm Gram on disk. See [[direction-seed-stability]].
- **Run 2026-09-12: a 100-trait real arm for the null comparison.** The two null
  arms have 100 variables against the real arm's 134, so their eigenvalues were
  not on the same scale; factoring the 100 Goldberg markers alone out of
  `results/gram_sweep.npz` fixes it. Done in
  [[goldberg-only-and-heldout-lexicon]]: the p = 100 real arm's centred spectrum
  is 12.725, 11.140, 5.663, 3.996, 2.927 against the permuted arm's 13.632,
  10.390, 4.945, 3.887, 2.824, and both retain 8 factors, so at equal p the
  spectrum does not separate real labels from permuted ones and only congruence
  does. See [[factor-analysis-null-arms]].
- **Never run: additivity for PC and factor mixtures.** Mixture additivity exists
  only for the five Big Five keying axes. See [[factors-versus-pca-coverage]].
- **Never run: a sphere sweep sampled in the factor subspace.** The 72 judged
  directions were sampled on the top-three-PC sphere; only the layout has been
  moved to the factor chart. See [[factors-versus-pca-coverage]].

## Reading, 2026-09-09

- **Two papers read in full and related to the project**, with eleven ranked
  experiments they suggest: SliderSpace (Gandikota et al., ICCV 2025) and
  Gradient Atoms (Rosser, arXiv:2603.14665, 2026). The cheapest and most
  consequential is redoing the factor analysis in the Fisher metric rather than
  the Frobenius one, which both papers' methods imply and which
  [[fisher-norms]] has already measured the need for. See
  [[paper-reading-2026-09-09]].
- **Run since (2026-09-10): G2, G4 and G3.** [[gradient-atoms]] and
  [[reward-hacks-gradient-atoms]]. Still never run from that page: **G1** (the
  factor analysis in the Fisher metric, the item above), **G5**, **G6**, **S2**,
  **S3**, **S4** and **S5**. Two questions the run opened rather than closed: no
  atom was ever steered, so the paper's second half is untested here; and the
  whitening used is the empirical Fisher on realised tokens over document
  gradients, so whether a different approximation gives a different dictionary is
  unknown.

## Added 2026-09-16, from the freshness audit against the 2026-09-15 draft

- **Never run, named by the draft as the next three runs** ("What we would run
  next"): constitutions written without Big Five vocabulary, to test whether
  the rotation and the five-factor count are the teacher's; the same 134
  datasets on a second base model; the enumerated-anchor ablation
  ([[constitution-anchor-revision]]). The fourth item on that list, the
  self-identification probe, was run on 2026-09-15 ([[self-identification-probe]]).
- **Unresolved divergence from the source papers: 13 optimizer steps against
  OCT's roughly 47.** Recorded on [[recipe-vs-source-papers]] section 3.5 and
  pointed to from [[stage-one-training-config]]; the draft states it in its
  scope note and does not resolve it. The cheap test (one trait trained with and
  without a LIMA-style general pool) has not been run.
- **Unexplained: parallel analysis retains nine factors, the stage-two residual
  seven, the permuted null eight.** Five is the hypothesis. The draft says so
  ("That five is the number" under "What this does not show"); no analysis
  decides between five and nine. Pages: [[factor-analysis]], [[pca-and-scree]].
- **Unconfirmed: the reward-hack arms' behavioural null is length-confounded**
  (both arms collapse to about 70 words; see the entry above). A length-matched
  decode is still the missing control. Page: [[reward-hacks-arms]].
- **Provenance: two analysis files the draft's figures depend on have no
  producing script checked in under `qwen35/`.** `analysis/best_axis_pairs.json`
  (Figure 1) and `analysis/fa_text_contrast.json` (the text-contrast factors)
  were written by session scripts on 2026-09-15; on 2026-09-16 the scripts were
  checked in as `qwen35/analyse_best_axis_pairs.py` and
  `qwen35/analyse_fa_text_contrast.py` and shown to regenerate both files
  exactly. Resolved. See [[code-and-data-map]] for the remaining files without
  a producer.
- **Resolved 2026-09-16: the non-weight data release.** The analysis, results,
  corpora and run-log files are public at
  `https://huggingface.co/datasets/EternalRecursion/persona-curvature-results`
  (3,384 files, 3.94 GB) and the GitHub repository
  `https://github.com/EternalRecursion121/persona-curvature` carries
  `tools/fetch_data.py` and a sha256 manifest. Still open: whether the control
  adapters (alignment, hole, Big Five, null arms) should be published too; they
  are on request. Page: [[code-and-data-map]], [[hf-artefacts]].
- **Decision pending: the wiki and companion are public without basic auth**
  (removed at Samuel's request on 2026-09-07; confirmed in `/etc/caddy/Caddyfile`
  on 2026-09-16). The wiki carries spend figures, transcript-sourced pages and
  an unpublished review's paraphrase. Page: [[how-to-read]].
- **Numbers the 2026-09-16 audit could not verify against a file.** Listed in
  `pages/overview/_report_freshness_2026-09-16.md`; the load-bearing ones are
  the sketch-validation correlation (0.9996 or 0.99944, provenance gap above),
  the hard-coded blog-page strings in [[superseded-claims]] section 1c, and the
  literature numbers on the paper pages, which cite arXiv URLs rather than files
  on disk.

