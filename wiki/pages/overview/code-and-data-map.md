---
title: Code and data map
summary: Which script produces which analysis file, which post figure and companion page read it, and which wiki pages cite it, built on 2026-09-16 from the wiki's own sources frontmatter, the figure script and the companion builder; with the state of the Hugging Face releases and the files that have no producing script.
status: current
sources:
  - qwen35/figures/post/make_post_figures.py
  - qwen35/companion/build_companion.py
  - qwen35/companion/check_numbers.py
  - qwen35/analyse_fa_qwen35.py
  - qwen35/build_gram.py
  - wiki/pages/**/*.md
  - qwen35/POST_DRAFT.md
last_verified: 2026-09-16
tags: [overview, code, data, provenance]
---

# Code and data map

How to read this page: a result on this wiki is a number in a file under
`qwen35/analysis/` or `qwen35/results/`. Those files are written by scripts under
`qwen35/`, most of them named `analyse_*.py`, from raw outputs of Modal jobs
(`*_on_modal.py`, logs under `qwen35/phase10_runs/`). The post's figures
(`qwen35/figures/post/make_post_figures.py`) and the companion site
(`qwen35/companion/build_companion.py`, checked by `check_numbers.py`) read the
same files. The tables were generated on 2026-09-16 by scanning every
`sources:` entry in the wiki, the figure script and the companion builder, then
grepping `qwen35/**/*.py` for each file's name. "Producing script (best match)"
means the script under `qwen35/` that names the file and contains a write call;
where several do, the first two are listed. Paths are relative to `qwen35/`.

## The pipeline in one paragraph

`train_qwen35.py` trains one adapter per trait on Modal from `data/<slug>.jsonl`
(pairs from `gen_pairs.py`, conditioned on `constitutions.json` from
`constitutions.py`); `oct_stage2.py` runs the introspection stage;
`fix_persona_merge.py` builds the exact personas. `build_gram.py` computes the
exact 134 x 134 Gram into `results/gram_sweep.npz` (stage two:
`results/gram_stage2.npz`; cross-arm Grams: `cross_gram_full_on_modal.py` into
`results/cross_gram_full_*.npz`). `analyse_fa_qwen35.py` factors it into
`results/fa_qwen35.json`; `fa_chart.py` turns the oblimin solution into the chart
coordinates in `analysis/viz_fa.json`. Every steering, judging and scoring script
then reads the chart or the Gram and writes its own `analysis/*.json`.

## Where the artefacts are

- **Adapters.** All 134 stage-one adapters, the 134 stage-two adapters and the
  exact personas are public at
  `https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35`. The control
  and validation adapters are public since 2026-09-16 at
  `https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35-controls`
  (353 adapters, 1,374 files, 162.57 GB, verified against
  `qwen35/analysis/hf_controls_manifest.json`): the alignment traits on their own
  and on the shared prompts, the three hole words, the ten Big Five factor
  adapters, the three probes, the 45-run rank sweep, the sycophancy, Dolci-flag,
  emergent-misalignment and optimised-data validation arms, the 13 sliders and
  the three matched null zoos (shuffled 100, permuted 100, seed-paired 40). The
  stage-two transcripts are the dataset repository named on [[hf-artefacts]].
  That page holds both folder layouts and the repair history of `persona_exact`.
- **Non-weight data.** Published on 2026-09-16 to the public dataset
  `https://huggingface.co/datasets/EternalRecursion/persona-curvature-results`:
  3,384 files, 3.94 GB, every path repo-relative (`qwen35/analysis/`,
  `qwen35/results/`, `qwen35/data*/`, `qwen35/phase10_runs/`). The GitHub
  repository `https://github.com/EternalRecursion121/persona-curvature` (private
  until the owner opens it) holds the code, the write-up, this wiki and
  `tools/fetch_data.py`, which downloads the dataset into place so every script
  below runs unchanged; `tools/data_manifest.json` carries a sha256 per file and
  `python tools/fetch_data.py --verify` checks them. Adapter weights are not in
  that dataset (see the previous bullet). Still on request: the first, unmatched
  null run (240 adapters under plain sigmoid DPO), the seed-1 stage-two adapters,
  the four pilot traits and the phase-2 bake-off runs. Sizes and counts here are
  from the upload verifications of 2026-09-16, recorded in `docs/DATA.md` of that
  repository.
- **Companion and wiki.** `https://persona.161-35-77-84.sslip.io` and
  `https://wiki.161-35-77-84.sslip.io`, both on the devbox, both to move before
  publication.

## Table 1. The files behind the post draft

These are the files the 2026-09-15 draft, its figures and [[claims-and-evidence]]
rest on. Figure stems are the `fig_*` functions of `make_post_figures.py`, each
written as `<stem>_light.png`, `<stem>_dark.png` and SVG.

| file | producing script (best match) | post figure | companion | wiki pages citing it |
|---|---|---|---|---|
| `analysis/viz_fa.json` | `qwen35/analyse_goldberg_only.py`, `qwen35/analyse_selfid.py` (+2) | `facets_own` | yes | [[best-axis-pairs]], [[built-pages-inventory]], [[factor-analysis]] (+3) |
| `analysis/best_axis_pairs.json` | `analyse_best_axis_pairs.py` | `facets_best` | - | [[best-axis-pairs]], [[post-draft]], [[start-here-for-collaborators]] |
| `results/fa_qwen35.json` | `qwen35/analyse_fa_actgram.py`, `qwen35/analyse_fa_fisher.py` (+15) | `scree_congruence` | yes | [[built-pages-inventory]], [[claims-and-evidence]], [[explainer-big-five-mapping]] (+3) |
| `analysis/scree_null_matched.json` | none found | `scree_congruence` | yes | [[built-pages-inventory]], [[factor-analysis-null-arms]], [[factors-versus-pca-coverage]] (+3) |
| `analysis/fa_nulls.json` | `qwen35/analyse_fa_nulls.py`, `qwen35/analyse_goldberg_only.py` | - | - | [[claims-and-evidence]], [[factor-analysis]], [[factor-analysis-null-arms]] (+3) |
| `analysis/fa_fisher_metric.json` | `qwen35/analyse_fa_fisher.py` | - | - | [[claims-and-evidence]], [[factor-analysis]], [[factor-analysis-fisher-metric]] (+2) |
| `analysis/goldberg_only.json` | `qwen35/analyse_goldberg_only.py`, `qwen35/analyse_lexicon_only.py` | - | - | [[claims-and-evidence]], [[factor-analysis-null-arms]], [[goldberg-only-and-heldout-lexicon]] (+3) |
| `analysis/fa_text_contrast.json` | `analyse_fa_text_contrast.py` | - | - | [[claims-and-evidence]], [[start-here-for-collaborators]], [[text-contrast-factors]] |
| `analysis/act_gram.json` | `qwen35/analyse_act_gram.py`, `qwen35/analyse_act_gram_stage2.py` | `seed_metrics` | - | [[activation-weighted-gram]], [[activation-weighted-gram-stages]], [[built-pages-inventory]] (+3) |
| `analysis/act_gram_stage2.json` | `qwen35/analyse_act_gram_stage2.py` | - | - | [[activation-weighted-gram-stages]], [[claims-and-evidence]], [[source-contradictions]] (+1) |
| `analysis/column_space.json` | `qwen35/analyse_column_space.py`, `qwen35/analyse_column_space_sorh.py` (+3) | `seed_metrics` | - | [[activation-weighted-gram]], [[built-pages-inventory]], [[claims-and-evidence]] (+3) |
| `analysis/rank_sweep.json` | `qwen35/analyse_rank_sweep.py`, `qwen35/train_rank_sweep.py` | `rank_sweep` | - | [[built-pages-inventory]], [[claims-and-evidence]], [[post-draft]] (+2) |
| `analysis/fisher_norms.json` | `qwen35/analyse_fisher.py`, `qwen35/analyse_fisher_gram.py` (+6) | `fisher` | yes | [[alien-direction-factor-chart]], [[built-pages-inventory]], [[claims-and-evidence]] (+3) |
| `analysis/persona_sliders.json` | `qwen35/analyse_persona_sliders.py` | - | - | [[actspace-persona-vectors]], [[alien-direction-factor-chart]], [[claims-and-evidence]] (+3) |
| `analysis/stage2_structure.json` | `qwen35/analyse_act_gram_stage2.py`, `qwen35/analyse_actspace_stage2.py` (+7) | `stage_two` | yes | [[activation-weighted-gram-stages]], [[built-pages-inventory]], [[claims-and-evidence]] (+3) |
| `analysis/stage2_neutral_control.json` | `qwen35/analyse_stage2_frame.py`, `qwen35/analyse_stage2_neutral.py` (+2) | - | - | [[claims-and-evidence]], [[paper-reading-2026-09-09]], [[stage-two-exploration]] (+1) |
| `analysis/stage2_exploration.json` | `qwen35/build_stage2_exploration_index.py` | - | yes | - |
| `analysis/s2mean_steer_stats.json` | `qwen35/analyse_s2mean_steer.py` | `stage_two` | yes | [[built-pages-inventory]], [[stage-two-shared-direction]], [[start-here-for-collaborators]] |
| `analysis/fulloct_geometry.json` | `qwen35/analyse_fulloct.py`, `qwen35/build_s2register_spec.py` | - | yes | [[claims-and-evidence]], [[full-oct-replication]], [[start-here-for-collaborators]] |
| `analysis/selfid.json` | `qwen35/analyse_selfid.py` | `selfid` | - | [[claims-and-evidence]], [[post-draft]], [[self-identification-probe]] (+1) |
| `results/selfid_generations.json` | `qwen35/analyse_selfid.py`, `qwen35/selfid_on_modal.py` | `selfid` | - | [[post-draft]], [[self-identification-probe]], [[start-here-for-collaborators]] |
| `analysis/matched_dose_steering.json` | `qwen35/analyse_matched_dose.py` | `dose` | - | [[built-pages-inventory]], [[claims-and-evidence]], [[factor-arousal]] (+3) |
| `analysis/sphere_page_fa.json` | `qwen35/analyse_fisher.py` | `sphere` | yes | [[built-pages-inventory]], [[claims-and-evidence]], [[fisher-norms]] (+3) |
| `analysis/sphere_isokl.json` | `qwen35/analyse_sphere_isokl.py` | - | - | [[fisher-norms]], [[matched-dose-steering]], [[post-draft]] (+3) |
| `analysis/spider.json` | `qwen35/build_s2register_spec.py`, `qwen35/build_spider_data.py` (+1) | `dials` | yes | [[bigfive-factor-adapters]], [[built-pages-inventory]], [[claims-and-evidence]] (+3) |
| `analysis/additivity.json` | `qwen35/analyse_additivity.py` | - | yes | [[additivity]], [[claims-and-evidence]], [[paper-reading-2026-09-09]] (+1) |
| `analysis/inspect_personality.json` | `qwen35/analyse_inspect_personality.py` | - | yes | [[inspect-personality-evals]], [[post-draft]], [[start-here-for-collaborators]] |
| `analysis/data_forecast.json` | `qwen35/analyse_data_forecast.py` | `forecast` | - | [[built-pages-inventory]], [[claims-and-evidence]], [[data-forecast]] (+3) |
| `analysis/nxn_summary.json` | `qwen35/analyse_nxn.py` | - | yes | [[blog-site-map-idea]], [[external-review]], [[n-by-n-scoring]] (+3) |
| `analysis/dolci_audit.json` | `qwen35/analyse_dolci_scores.py` | - | yes | [[dolci-data-audit]], [[post-draft]] |
| `analysis/syc_forecast.json` | `qwen35/analyse_syc_forecast.py` | - | yes | [[claims-and-evidence]], [[data-forecast]], [[dolci-data-audit]] (+3) |
| `analysis/dolci_flag_training.json` | `qwen35/analyse_dolci_flag.py`, `qwen35/render_dolci_flag_tables.py` | - | yes | [[claims-and-evidence]], [[data-forecast]], [[dolci-data-audit]] (+3) |
| `analysis/em_medical.json` | `qwen35/analyse_em.py` | `external` | - | [[built-pages-inventory]], [[claims-and-evidence]], [[emergent-misalignment-medical]] (+3) |
| `analysis/em_part_b.json` | `qwen35/analyse_em.py`, `qwen35/analyse_em_b.py` | `external` | - | [[built-pages-inventory]], [[claims-and-evidence]], [[emergent-misalignment-medical]] (+3) |
| `analysis/sorh_data_scoring.json` | `qwen35/analyse_column_space_sorh.py`, `qwen35/analyse_sorh_data_scores.py` (+1) | `external` | yes | [[built-pages-inventory]], [[claims-and-evidence]], [[dolci-data-audit]] (+3) |
| `analysis/sorh_behavioural.json` | `qwen35/analyse_column_space_sorh.py`, `qwen35/analyse_s2register.py` (+2) | - | yes | [[open-questions]], [[paper-reading-2026-09-09]], [[post-draft]] (+3) |
| `analysis/sorh_projection.json` | `qwen35/analyse_column_space_sorh.py`, `qwen35/analyse_sorh.py` (+2) | - | - | [[claims-and-evidence]], [[reward-hacks-arms]], [[reward-hacks-column-space]] (+3) |
| `analysis/probe_adapters.json` | `qwen35/analyse_probe_adapters.py` | - | - | [[post-critique-2026-09-10]], [[post-draft]], [[probe-adapters]] |
| `analysis/actspace_geometry.json` | `qwen35/analyse_actspace.py`, `qwen35/analyse_actspace_fa.py` (+2) | - | yes | [[actspace-overview]], [[actspace-persona-vectors]], [[cross-seed-geometry]] (+3) |
| `analysis/actspace_geometry_fa.json` | `qwen35/analyse_actspace_fa.py` | - | - | [[actspace-persona-vectors]], [[claims-and-evidence]], [[factor-first-migration]] (+3) |
| `analysis/direction_gaps_fa.json` | `qwen35/analyse_alien_fa.py` | - | yes | [[factors-versus-pca-coverage]], [[hole-words-factor-chart]], [[post-draft]] |
| `analysis/alien_fa.json` | `qwen35/analyse_alien_fa.py`, `qwen35/analyse_alien_steer_fa.py` (+5) | - | yes | [[alien-direction-factor-chart]], [[factors-versus-pca-coverage]], [[hole-words-factor-chart]] (+2) |
| `analysis/alien_steer_fa.json` | `qwen35/analyse_alien_steer_fa.py` | - | - | [[alien-direction-factor-chart]], [[fisher-norms]], [[post-draft]] |
| `analysis/slider_probe.json` | none found | - | - | [[persona-sliders]], [[post-draft]] |
| `analysis/crossseed_arms.json` | `qwen35/act_gram_on_modal.py`, `qwen35/act_gram_stages_on_modal.py` (+4) | - | yes | [[activation-weighted-gram]], [[column-space-structure]], [[cross-seed-geometry]] (+3) |
| `results/gram_sweep.npz` | `qwen35/analyse_act_gram.py`, `qwen35/analyse_actspace.py` (+43) | `stage_two` | yes | [[activation-weighted-gram]], [[actspace-adapters]], [[alien-direction-factor-chart]] (+3) |
| `results/gram_stage2.npz` | `qwen35/analyse_actspace_stage2.py`, `qwen35/analyse_s2mean_steer.py` (+6) | - | yes | [[built-pages-inventory]], [[stage-two-exploration]], [[stage-two-shared-direction]] (+1) |
| `analysis/emb_pairs_minilm.npz` | `qwen35/analyse_goldberg_only.py`, `qwen35/embed_goldberg_only.py` | - | - | [[text-contrast-factors]] |

Notes on Table 1. `results/fa_qwen35.json` is named on nineteen lines of the
companion builder and is the single most load-bearing file. `analysis/viz_fa.json`
is the chart every coordinate on the wiki is read from. `analysis/best_axis_pairs.json`
and `analysis/fa_text_contrast.json` were first written on 2026-09-15 by session
scripts; on 2026-09-16 those were checked in as `qwen35/analyse_best_axis_pairs.py`
and `qwen35/analyse_fa_text_contrast.py`, and each regenerates its file with every
numeric leaf identical (328 and 2,405 leaves compared). Their method is documented
on [[best-axis-pairs]] and [[text-contrast-factors]], and their inputs are
`analysis/viz_fa.json` and `analysis/emb_pairs_minilm.npz` (embedded by
`embed_goldberg_only.py`). 
## Table 2. Every analysis script and the files it names

Scripts under `qwen35/` named `analyse_*.py`, with the `analysis/` files each
names (reads or writes; templated names with `{tag}` are omitted).

| script | analysis files it names (reads or writes) |
|---|---|


## Table 3. The other files the wiki cites

Every other `analysis/` or `results/` file named in a wiki page's `sources`
frontmatter or body, with the scripts under `qwen35/` that name it. Consumers
(`build_blog_page.py`, `build_companion.py`, `make_post_figures.py` and the other
page builders) are excluded from the script column.

| file | scripts naming it | wiki pages citing it |
|---|---|---|
| `analysis/actspace_adapters_geometry.json` | `qwen35/analyse_actspace_adapters.py` | [[actspace-adapters]], [[actspace-overview]] (+3) |
| `analysis/actspace_cross_geometry.json` | `qwen35/analyse_actspace_cross.py` | [[actspace-adapters]], [[actspace-cross]] (+3) |
| `analysis/actspace_means_adapters.npz` | `qwen35/analyse_actspace_adapters.py`, `qwen35/analyse_actspace_cross.py` (+3) | [[persona-sliders]] |
| `analysis/actspace_means_adapters_stage2.npz` | `qwen35/analyse_actspace_stage2.py` | [[stage-two-exploration]] |
| `analysis/actspace_spec.json` | `qwen35/act_space.py`, `qwen35/persona_sliders.py` | [[actspace-method-notes]], [[actspace-overview]] |
| `analysis/actspace_stage2_geometry.json` | `qwen35/analyse_actspace_stage2.py`, `qwen35/build_stage2_exploration_index.py` | [[column-space-structure]], [[paper-reading-2026-09-09]] (+1) |
| `analysis/alien.json` | `qwen35/act_space.py`, `qwen35/analyse_actspace.py` (+19) | [[actspace-method-notes]], [[actspace-persona-vectors]] (+4) |
| `analysis/alien_match.json` | `qwen35/analyse_alien_steer_fa.py` | [[alien-direction-factor-chart]], [[alien-direction-steering]] (+1) |
| `analysis/alien_match_fa.json` | `qwen35/analyse_alien_steer_fa.py`, `qwen35/analyse_persona_sliders.py` | [[alien-direction-factor-chart]], [[persona-sliders]] |
| `analysis/alien_steer.json` | `qwen35/analyse_alien_steer.py`, `qwen35/analyse_alien_steer_fa.py` | [[alien-direction-factor-chart]], [[alien-direction-steering]] |
| `analysis/alien_v_fa.npy` | `qwen35/analyse_alien_fa.py` | [[hole-words-factor-chart]] |
| `analysis/alien_v_k5.npy` | `qwen35/analyse_alien.py`, `qwen35/analyse_gaps.py` (+2) | [[hole-words]], [[hole-words-factor-chart]] |
| `analysis/align_scores.json` | `qwen35/analyse_align.py`, `qwen35/land.sh` | [[alignment-traits-geometry]], [[scoring-identity]] (+2) |
| `analysis/align_summary.json` | `qwen35/analyse_align.py` | [[adapter-effect-and-drift]], [[factor-axis-agreeableness]] (+4) |
| `analysis/align_validate.json` | `qwen35/align_score.py`, `qwen35/dolci_score.py` | [[n-by-n-scoring]], [[scoring-identity]] (+2) |
| `analysis/alignment_geometry_fa.json` | `qwen35/analyse_alignment_fa.py` | [[alignment-traits-geometry]], [[hole-words-factor-chart]] |
| `analysis/bigfive_adapters_geometry.json` | `qwen35/analyse_bigfive.py` | [[bigfive-factor-adapters]] |
| `analysis/blog_data.json` | `qwen35/train_rank_sweep.py` | [[built-pages-inventory]], [[factor-arousal]] (+4) |
| `analysis/column_space_sorh.json` | `qwen35/analyse_column_space_sorh.py`, `qwen35/analyse_em_colspace.py` | [[column-space-structure]], [[reward-hacks-arms]] (+1) |
| `analysis/corpus_degeneration.json` | `qwen35/check_corpus_degeneration.py` | [[corpus-repetition-scan]] |
| `analysis/corpus_scan_all.json` | `qwen35/scan_corpus_modal.py` | [[corpus-repetition-scan]] |
| `analysis/crossseed_arms_stage2.json` | `qwen35/analyse_stage2_frame.py`, `qwen35/analyse_stage2_neutral.py` (+1) | [[open-questions]], [[stage-two-exploration]] (+2) |
| `analysis/direction_gaps.json` | `qwen35/analyse_alien_fa.py`, `qwen35/analyse_gaps.py` | [[alien-direction-steering]], [[factor-axis-agreeableness]] (+4) |
| `analysis/direction_seed_stability.json` | `qwen35/analyse_direction_seed_stability.py` | [[direction-seed-stability]], [[factor-analysis]] (+1) |
| `analysis/dolci_examples.json` | `qwen35/extract_dolci_examples.py` | [[dolci-data-audit]] |
| `analysis/dolci_flag_adrift.json` | `qwen35/analyse_dolci_flag.py`, `qwen35/dolci_flag_train.py` | [[dolci-flag-training]] |
| `analysis/dolci_flag_arm_scores.json` | `qwen35/analyse_dolci_flag.py` | [[dolci-flag-training]] |
| `analysis/dolci_flag_examples.json` | `qwen35/dolci_flag_examples.py` | [[dolci-flag-training]] |
| `analysis/dolci_flag_judge_replicate.json` | `qwen35/build_syc_post_inputs.py`, `qwen35/em_to_eval.py` (+2) | [[dolci-flag-training]], [[sycophancy-forecast]] |
| `analysis/dolci_flag_refusal_style.json` | `qwen35/analyse_dolci_flag.py`, `qwen35/dolci_flag_refusal_style.py` | [[dolci-flag-training]] |
| `analysis/dolci_judge.json` | `qwen35/extract_dolci_examples.py`, `qwen35/judge_dolci.py` | [[dolci-data-audit]] |
| `analysis/dolci_judge_precision.json` | `qwen35/judge_dolci.py` | [[dolci-data-audit]] |
| `analysis/dolci_judge_v1.json` | `qwen35/judge_dolci.py` | [[dolci-data-audit]] |
| `analysis/dolci_scores.json` | `qwen35/analyse_dolci_scores.py`, `qwen35/extract_dolci_examples.py` (+1) | [[dolci-data-audit]] |
| `analysis/dolci_scores_dpo.json` | `qwen35/analyse_dolci_scores.py`, `qwen35/build_dolci_flag_arms.py` (+1) | [[dolci-data-audit]], [[reward-hacks-data-scoring]] (+1) |
| `analysis/dolci_scores_sft.json` | `qwen35/analyse_dolci_scores.py` | [[dolci-data-audit]] |
| `analysis/em_column_space.json` | `qwen35/analyse_em.py`, `qwen35/analyse_em_colspace.py` | [[emergent-misalignment-medical]], [[reward-hacks-arms]] |
| `analysis/em_part_a.json` | `qwen35/analyse_em.py`, `qwen35/analyse_em_a.py` (+1) | [[emergent-misalignment-medical]], [[paper-persona-vectors]] (+1) |
| `analysis/em_part_c.json` | `qwen35/analyse_em.py`, `qwen35/analyse_em_c.py` | [[emergent-misalignment-medical]], [[reward-hacks-arms]] |
| `analysis/em_part_d.json` | `qwen35/analyse_em.py`, `qwen35/analyse_em_d.py` | [[emergent-misalignment-medical]] |
| `analysis/em_part_d_ls.json` | `qwen35/analyse_em.py` | [[emergent-misalignment-medical]] |
| `analysis/em_probe_scores.json` | `qwen35/analyse_em_d.py` | [[emergent-misalignment-medical]] |
| `analysis/em_train.json` | `qwen35/analyse_em_b.py`, `qwen35/em_flatten.py` (+1) | [[emergent-misalignment-medical]] |
| `analysis/fa_act_metric.json` | `qwen35/analyse_fa_actgram.py` | [[activation-weighted-gram]] |
| `analysis/fa_chart_summary.json` | `qwen35/fa_chart.py` | [[factor-analysis]], [[factor-audit-2026-09-11]] (+4) |
| `analysis/fa_summary.json` | `qwen35/build_viz_data_fa.py` | [[factor-analysis]], [[factor-arousal]] (+4) |
| `analysis/fisher_gram_validation.json` | `qwen35/analyse_fisher_gram.py` | [[factor-analysis-fisher-metric]], [[fisher-norms]] |
| `analysis/geometry_k_sweep.json` | `qwen35/geometry.py` | [[factor-first-migration]], [[geometry-overview]] (+1) |
| `analysis/geometry_stage1.json` | `qwen35/geometry.py` | [[explainer-elbow-figure]], [[factor-axis-agreeableness]] (+4) |
| `analysis/goldberg_only_ratings.jsonl` | `qwen35/analyse_goldberg_only.py` | - |
| `analysis/gradient_atoms.json` | `qwen35/analyse_gradient_atoms.py` | [[factor-analysis]], [[factor-analysis-null-arms]] (+4) |
| `analysis/gradient_atoms_atoms_sorh.json` | `qwen35/analyse_gradient_atoms_sorh.py` | [[reward-hacks-gradient-atoms]] |
| `analysis/gradient_atoms_extract_sorh.json` | `qwen35/analyse_gradient_atoms_sorh.py` | [[reward-hacks-gradient-atoms]] |
| `analysis/gradient_atoms_sorh.json` | `qwen35/analyse_gradient_atoms_sorh.py` | [[reward-hacks-arms]], [[reward-hacks-gradient-atoms]] |
| `analysis/gradient_atoms_weightspace.json` | `qwen35/analyse_gradient_atoms.py`, `qwen35/gradient_atoms_on_modal.py` | [[gradient-atoms]] |
| `analysis/gradient_atoms_weightspace_sorh.json` | `qwen35/analyse_gradient_atoms_sorh.py` | [[reward-hacks-gradient-atoms]] |
| `analysis/hole_geometry.json` | `qwen35/analyse_hole.py`, `qwen35/analyse_hole_fa.py` | [[hole-words]], [[hole-words-factor-chart]] |
| `analysis/hole_geometry_fa.json` | `qwen35/analyse_hole_fa.py` | [[factors-versus-pca-coverage]], [[hole-words-factor-chart]] (+1) |
| `analysis/inspect_trait20.json` | `qwen35/analyse_inspect_personality.py`, `qwen35/inspect_personality_on_modal.py` | [[inspect-personality-evals]] |
| `analysis/lora_a_identity.json` | `qwen35/analyse_column_space_sorh.py`, `qwen35/analyse_sorh_data_scores.py` (+9) | [[persona-sliders]], [[rank-sweep]] (+4) |
| `analysis/matched_dose_alphas.json` | `qwen35/analyse_matched_dose.py`, `qwen35/solve_matched_alphas.py` | [[fisher-norms]], [[matched-dose-steering]] |
| `analysis/merge_audit.json` | `qwen35/fix_persona_merge.py` | [[distillation-check]], [[hf-artefacts]] (+4) |
| `analysis/nxn_scores.json` | `qwen35/analyse_data_forecast.py`, `qwen35/analyse_dolci_scores.py` (+3) | [[data-forecast]], [[dolci-data-audit]] (+4) |
| `analysis/optimise.json` | `qwen35/build_verify_data.py`, `qwen35/optimise_data.py` (+1) | [[optimised-data-and-verify]] |
| `analysis/persona_exact_repush.json` | `qwen35/reupload_persona_exact.py` | [[full-oct-replication]], [[hf-artefacts]] |
| `analysis/persona_key_repair.json` | `qwen35/fix_persona_keys.py`, `qwen35/reupload_persona_exact.py` | [[full-oct-replication]] |
| `analysis/personas_seed1_build.json` | `qwen35/build_personas_seed1.py` | [[full-oct-replication]] |
| `analysis/probe_scores_sft.json` | `qwen35/analyse_probe_adapters.py`, `qwen35/judge_probes.py` (+1) | [[probe-adapters]] |
| `analysis/rl_preflight.json` | `qwen35/rl_capability.py`, `qwen35/rl_preflight.py` | [[paper-dolci-rl-zero]], [[rl-capability-and-persona-drift]] |
| `analysis/slider_cross_gram.json` | `qwen35/analyse_persona_sliders.py`, `qwen35/cross_gram_sliders.py` | [[persona-sliders]] |
| `analysis/slider_generations.jsonl` | `qwen35/fetch_slider_artifacts.sh` | - |
| `analysis/slider_means.npz` | `qwen35/analyse_persona_sliders.py`, `qwen35/fetch_slider_artifacts.sh` | [[persona-sliders]] |
| `analysis/slider_targets.npz` | `qwen35/analyse_persona_sliders.py`, `qwen35/build_slider_targets.py` (+1) | [[persona-sliders]] |
| `analysis/slider_targets_meta.json` | `qwen35/analyse_persona_sliders.py`, `qwen35/build_slider_targets.py` (+1) | [[persona-sliders]] |
| `analysis/slider_train.json` | `qwen35/analyse_persona_sliders.py`, `qwen35/persona_sliders.py` | [[persona-sliders]] |
| `analysis/sorh_data_scores.json` | `qwen35/analyse_column_space_sorh.py`, `qwen35/analyse_sorh_data_scores.py` | [[reward-hacks-column-space]], [[reward-hacks-data-scoring]] |
| `analysis/sorh_train.json` | `qwen35/sft_rewardhacks.py` | [[reward-hacks-arms]], [[reward-hacks-column-space]] (+1) |
| `analysis/sphere_isokl_alphas.json` | `qwen35/analyse_sphere_isokl.py`, `qwen35/solve_sphere_isokl_alphas.py` | [[fisher-norms]], [[sphere-sweep-iso-kl]] |
| `analysis/sphere_layout.json` | `qwen35/analyse_sphere_isokl.py`, `qwen35/build_sphere_spec.py` (+1) | [[sphere-sweep]], [[sphere-sweep-factor-chart]] (+3) |
| `analysis/sphere_layout_fa.json` | `qwen35/build_sphere_spec_fa.py` | [[factors-versus-pca-coverage]], [[sphere-sweep-factor-chart]] |
| `analysis/sphere_page.json` | `qwen35/analyse_sphere_isokl.py` | [[sphere-sweep]], [[sphere-sweep-factor-chart]] (+2) |
| `analysis/stage2_factors_choice.json` | `qwen35/analyse_stage2_factors.py`, `qwen35/build_stage2_exploration_index.py` | [[stage-two-exploration]], [[stage-two-structure]] |
| `analysis/stage2_frame.json` | `qwen35/analyse_stage2_frame.py`, `qwen35/build_stage2_exploration_index.py` | [[stage-two-exploration]], [[stage-two-second-seed]] (+2) |
| `analysis/stage2_register_vs_residual.json` | `qwen35/analyse_s2register.py`, `qwen35/build_stage2_exploration_index.py` | [[paper-reading-2026-09-09]], [[stage-two-exploration]] |
| `analysis/steer_alpha_units.json` | `qwen35/analyse_fisher.py`, `qwen35/analyse_s2register.py` (+3) | [[fisher-norms]], [[stage-two-exploration]] (+3) |
| `analysis/steerfix_replication.json` | `qwen35/analyse_gradient_atoms.py` | [[gradient-atoms]], [[steering-results]] (+1) |
| `analysis/syc_adrift.json` | `qwen35/analyse_syc_forecast.py`, `qwen35/syc_train.py` | [[sycophancy-forecast]] |
| `analysis/trait_angles.json` | `qwen35/analyse_alien_fa.py`, `qwen35/analyse_alignment_fa.py` | [[factor-pc5]], [[hole-words]] (+2) |
| `analysis/umap_test.json` | `qwen35/build_viz_data_fa.py`, `qwen35/umap_test.py` | [[factor-chart]], [[factor-first-migration]] (+2) |
| `analysis/validate_100.json` | `qwen35/validate_100.py` | [[geometry-overview]], [[source-contradictions]] (+1) |
| `analysis/verify.json` | `qwen35/analyse_sphere_isokl.py`, `qwen35/analyse_verify.py` | [[optimised-data-and-verify]], [[pca-and-scree]] |
| `analysis/viz.json` | `qwen35/build_viz_data.py`, `qwen35/build_viz_data_fa.py` | [[factor-chart]], [[factor-first-migration]] (+3) |
| `results/column_space_crossstage.npz` | `qwen35/analyse_column_space.py` | [[column-space-structure]] |
| `results/column_space_gram_stage1.npz` | `qwen35/analyse_column_space.py` | [[column-space-structure]] |
| `results/column_space_sorh.npz` | `qwen35/analyse_column_space_sorh.py`, `qwen35/column_space_sorh_on_modal.py` | [[reward-hacks-column-space]] |
| `results/column_space_stage1.npz` | `qwen35/analyse_column_space.py`, `qwen35/column_space_on_modal.py` | [[column-space-structure]] |
| `results/column_space_stage2.npz` | `qwen35/analyse_column_space.py` | [[column-space-structure]] |
| `results/cross_gram_actweighted_stage1_x_stage2.npz` | `qwen35/analyse_act_gram_stage2.py` | [[activation-weighted-gram-stages]] |
| `results/cross_gram_full_data_alignment_common_x_data_alignment_common.npz` | `qwen35/analyse_alignment_fa.py` | [[hole-words-factor-chart]] |
| `results/cross_gram_full_data_alignment_common_x_dolci_flag.npz` | `qwen35/analyse_dolci_flag.py` | [[dolci-flag-training]] |
| `results/cross_gram_full_data_alignment_common_x_syc_forecast.npz` | `qwen35/analyse_syc_forecast.py` | [[sycophancy-forecast]] |
| `results/cross_gram_full_data_hole_common_x_data_hole_common.npz` | `qwen35/analyse_hole_fa.py` | [[hole-words-factor-chart]] |
| `results/cross_gram_full_dolci_flag_x_dolci_flag.npz` | `qwen35/analyse_dolci_flag.py` | [[dolci-flag-training]] |
| `results/cross_gram_full_loras_introspection_x_loras_introspection.npz` | `qwen35/analyse_act_gram_stage2.py`, `qwen35/analyse_fulloct.py` | [[activation-weighted-gram-stages]], [[full-oct-replication]] (+1) |
| `results/cross_gram_full_loras_introspection_x_seed1_loras_introspection.npz` | `qwen35/analyse_fulloct.py` | [[stage-two-second-seed]] |
| `results/cross_gram_full_neutral_loras_introspection_x_loras_introspection.npz` | `qwen35/analyse_stage2_neutral.py` | [[stage-two-exploration]] |
| `results/cross_gram_full_neutral_loras_introspection_x_neutral_loras_introspection.npz` | `qwen35/analyse_stage2_neutral.py` | [[stage-two-exploration]] |
| `results/cross_gram_full_personas_exact_x_personas_exact.npz` | `qwen35/analyse_actspace_stage2.py`, `qwen35/analyse_fulloct.py` | [[full-oct-replication]] |
| `results/cross_gram_full_personas_exact_x_seed1_personas_exact.npz` | `qwen35/analyse_fulloct.py` | [[full-oct-replication]] |
| `results/cross_gram_full_root_x_data_null_seedpaired_s40.npz` | `qwen35/build_traits_page_data.py` | [[lesson-bar-in-the-wrong-units]], [[seed-floor]] |
| `results/cross_gram_full_root_x_data_null_seedpaired_s40_matched.npz` | `qwen35/analyse_act_gram.py`, `qwen35/analyse_column_space.py` (+1) | [[activation-weighted-gram]], [[direction-seed-stability]] (+1) |
| `results/cross_gram_full_root_x_pc-qwen35-adapters_data_alignment_common.npz` | `qwen35/analyse_alignment_fa.py` | [[hole-words-factor-chart]] |
| `results/cross_gram_full_root_x_pc-qwen35-adapters_data_hole_common.npz` | `qwen35/analyse_hole_fa.py` | [[hole-words-factor-chart]] |
| `results/cross_gram_full_root_x_pc-qwen35-adapters_dolci_flag.npz` | `qwen35/analyse_dolci_flag.py` | [[dolci-flag-training]] |
| `results/cross_gram_full_root_x_pc-qwen35-adapters_sliders.npz` | `qwen35/analyse_persona_sliders.py` | [[persona-sliders]] |
| `results/cross_gram_full_root_x_pc-qwen35-adapters_syc_forecast.npz` | `qwen35/analyse_syc_forecast.py` | [[sycophancy-forecast]] |
| `results/cross_gram_full_root_x_pc-qwen35-oct2_personas_exact.npz` | `qwen35/analyse_fulloct.py`, `qwen35/analyse_stage2_structure.py` | [[full-oct-replication]], [[stage-two-structure]] |
| `results/cross_gram_full_seed1_personas_exact_x_seed1_personas_exact.npz` | `qwen35/analyse_fulloct.py` | [[full-oct-replication]] |
| `results/cross_gram_full_syc_forecast_x_syc_forecast.npz` | `qwen35/analyse_syc_forecast.py` | [[sycophancy-forecast]] |
| `results/cross_gram_seedpaired_provenance.json` | `qwen35/cross_gram_on_modal.py` | [[seed-floor]] |
| `results/decomposition.json` | `qwen35/analyse_stage2_structure.py`, `qwen35/build_traits_page_data.py` (+3) | [[actspace-persona-vectors]], [[cross-seed-geometry]] (+4) |
| `results/decomposition_actspace_resp.json` | `qwen35/analyse_actspace.py` | [[actspace-method-notes]], [[actspace-persona-vectors]] |
| `results/decomposition_permuted.json` | `qwen35/build_traits_page_data.py`, `qwen35/compare_nulls.py` | [[null-controls]] |
| `results/decomposition_seedB.json` | `qwen35/build_traits_page_data.py`, `qwen35/cross_gram_full_on_modal.py` | [[cross-seed-geometry]] |
| `results/decomposition_shuffled.json` | `qwen35/build_traits_page_data.py`, `qwen35/compare_nulls.py` | [[null-controls]] |
| `results/decomposition_stage2.json` | `qwen35/analyse_stage2_structure.py` | [[stage-two-structure]] |
| `results/fa_qwen35_actgram.json` | `qwen35/analyse_fa_actgram.py` | [[activation-weighted-gram]] |
| `results/fa_qwen35_fisher.json` | `qwen35/analyse_fa_fisher.py` | [[factor-analysis-fisher-metric]], [[post-draft]] |
| `results/fa_qwen35_fisher_emp.json` | `qwen35/analyse_fa_fisher.py` | [[factor-analysis-fisher-metric]] |
| `results/fa_qwen35_goldberg100.json` | `qwen35/analyse_goldberg_only.py` | [[goldberg-only-and-heldout-lexicon]], [[post-draft]] |
| `results/fa_qwen35_null_permuted.json` | `qwen35/analyse_fa_nulls.py` | [[factor-analysis]], [[factor-analysis-null-arms]] |
| `results/fa_qwen35_null_shuffled.json` | `qwen35/analyse_fa_nulls.py` | [[factor-analysis]], [[factor-analysis-null-arms]] |
| `results/fa_qwen35_stage2.json` | `qwen35/analyse_stage2_factors.py`, `qwen35/analyse_stage2_structure.py` (+1) | [[stage-two-exploration]], [[stage-two-structure]] |
| `results/fa_qwen35_stage2_noshared.json` | `qwen35/analyse_stage2_factors.py` | [[stage-two-exploration]] |
| `results/gradient_atoms/sorh_atoms.npz` | `qwen35/analyse_gradient_atoms_sorh.py` | [[reward-hacks-gradient-atoms]] |
| `results/gradient_atoms/zoo_atoms.npz` | `qwen35/gradient_atoms_on_modal.py` | [[gradient-atoms]] |
| `results/gram_actspace_resp_L16.npz` | `qwen35/analyse_actspace.py` | [[actspace-persona-vectors]] |
| `results/gram_actweighted.npz` | `qwen35/analyse_fa_actgram.py`, `qwen35/run_actgram_analysis.sh` | [[activation-weighted-gram]] |
| `results/gram_data_null_permuted_p100_matched.npz` | `qwen35/analyse_fa_nulls.py`, `qwen35/figures/clusters/make_cluster_figures.py` | [[factor-analysis-null-arms]] |
| `results/gram_data_null_seedpaired_s40.npz` | `qwen35/build_traits_page_data.py`, `qwen35/umap_grams.py` | [[direction-seed-stability]], [[factors-versus-pca-coverage]] |
| `results/gram_data_null_shuffled_p100_matched.npz` | `qwen35/analyse_fa_nulls.py` | [[factor-analysis-null-arms]] |
| `results/gram_fisher.npz` | `qwen35/analyse_fa_fisher.py`, `qwen35/analyse_fisher_gram.py` | [[factor-analysis-fisher-metric]] |
| `results/gram_fisher_empirical.npz` | `qwen35/analyse_fa_fisher.py`, `qwen35/analyse_fisher_gram.py` | [[factor-analysis-fisher-metric]] |
| `results/gram_goldberg100.npz` | `qwen35/analyse_goldberg_only.py` | [[goldberg-only-and-heldout-lexicon]] |
| `results/gram_stage2_noshared.npz` | `qwen35/analyse_stage2_factors.py`, `qwen35/build_gram_stage2_noshared.py` | [[stage-two-exploration]] |
| `results/rank_sweep_grams.npz` | `qwen35/analyse_rank_sweep.py`, `qwen35/cross_gram_rank_sweep.py` | [[rank-sweep]] |
| `results/runmeta_sweep.json` | `qwen35/build_traits_page_data.py`, `qwen35/decompose.py` (+2) | [[runmeta-provenance]], [[source-contradictions]] |
| `results/steer134_judged.json` | `qwen35/build_traits_page_data.py`, `qwen35/judge_steer134.py` | [[built-pages-inventory]], [[costs]] (+4) |
| `results/trait_descriptions.json` | `qwen35/build_traits_page_data.py`, `qwen35/describe_traits134.py` | [[qualitative-notes]] |
| `results/umap_embeddings.json` | `qwen35/build_traits_page_data.py`, `qwen35/umap_grams.py` | [[superseded-claims]], [[umap-and-layouts]] |

## Files written under a templated name

The scan matches literal basenames, so these files show no producer by name; each
is written by the script shown, with the tag filled in at run time.

- `analysis/crossseed_arms_actgram.json`: written under the templated name `analysis/crossseed_arms{tag}.json` in `qwen35/analyse_crossseed.py`; cited by [[activation-weighted-gram]]
- `analysis/crossseed_arms_actgram_centred.json`: written under the templated name `analysis/crossseed_arms{tag}.json` in `qwen35/analyse_crossseed.py`; cited by [[activation-weighted-gram]]
- `analysis/crossseed_arms_actgram_frobcheck.json`: written under the templated name `analysis/crossseed_arms{tag}.json` in `qwen35/analyse_crossseed.py`; cited by [[activation-weighted-gram]]
- `analysis/crossseed_arms_actgram_resp.json`: written under the templated name `analysis/crossseed_arms{tag}.json` in `qwen35/analyse_crossseed.py`; cited by [[activation-weighted-gram]]
- `analysis/gradient_atoms_atoms_zoo.json`: written under the templated name `analysis/gradient_atoms_atoms_{a.tag}.json` in `qwen35/analyse_gradient_atoms.py`, `analysis/gradient_atoms_atoms_{out_tag}.json` in `qwen35/gradient_atoms_on_modal.py`; cited by [[gradient-atoms]]
- `analysis/gradient_atoms_extract_zoo.json`: written under the templated name `analysis/gradient_atoms_extract_{a.tag}.json` in `qwen35/analyse_gradient_atoms.py`, `analysis/gradient_atoms_extract_{out_tag}.json` in `qwen35/gradient_atoms_on_modal.py`; cited by [[gradient-atoms]]
- `analysis/qual_axes.json`: written under the templated name `analysis/qual_*.json` in `qwen35/analyse_fisher.py`; cited by [[factor-axis-agreeableness]], [[factor-axis-conscientiousness]], [[factor-axis-emotional-stability]] (+3)
- `analysis/qual_fa.json`: written under the templated name `analysis/qual_*.json` in `qwen35/analyse_fisher.py`; cited by [[factor-analysis]], [[factor-arousal]], [[factor-audit-2026-09-11]] (+3)
- `analysis/qual_identity.json`: written under the templated name `analysis/qual_*.json` in `qwen35/analyse_fisher.py`; cited by [[qualitative-notes]], [[steering-results]]
- `analysis/qual_pairs.json`: written under the templated name `analysis/qual_*.json` in `qwen35/analyse_fisher.py`; cited by [[qualitative-notes]]
- `analysis/qual_pc.json`: written under the templated name `analysis/qual_*.json` in `qwen35/analyse_fisher.py`; cited by [[factor-pc1]], [[factor-pc2]], [[factor-pc3]] (+3)
- `analysis/rl_sketches_sorh_control.json`: written under the templated name `analysis/rl_sketches_{tag}.json` in `qwen35/analyse_sorh.py`; cited by [[reward-hacks-arms]]
- `analysis/rl_sketches_sorh_hack.json`: written under the templated name `analysis/rl_sketches_{tag}.json` in `qwen35/analyse_sorh.py`; cited by [[reward-hacks-arms]]

## Files the wiki cites with no producing script found under qwen35/

Provenance gaps: the file exists, pages quote it, and no checked-in script names
it (consumers excluded). Eight of these were already recorded on
[[open-questions]] under "Provenance gaps" on 2026-09-07 (`module_holography`,
`polarity_deflation`, `adapter_effect`, `intrinsic_coords`, `steerfix_replication`,
`judged_ceiling`, `functional_probe`, `qual_pairs`); `page_data.json` and
`monitor_page_data.json` are read by the old page builders and written by nothing
checked in. The `rl_*` files belong to the 2026-08 RL arm
([[rl-capability-and-persona-drift]]); that page's sources name `rl_capability.py`,
`rl_preflight.py`, `eval_rl_persona.py` and `analyse_rl.py` under `qwen35/`, but
none of those names the `rl_*.json` files literally (they build the names at run
time), so they appear here. `scree_null.json` and `scree_null_matched.json` are
named only by `build_blog_page.py`, a consumer; no producing script is checked
in, and `#real` is byte-identical in both files (see [[open-questions]],
"Provenance gaps").

- `analysis/adapter_effect.json` (cited by [[adapter-effect-and-drift]], [[judged-evaluations]])
- `analysis/alignment_geometry.json` (cited by [[alignment-traits-geometry]], [[factor-axis-agreeableness]], [[factor-pc4]] (+1))
- `analysis/alignment_geometry_aligncommon.json` (cited by [[alignment-traits-geometry]], [[hole-words-factor-chart]])
- `analysis/best_axis_pairs.json` (cited by [[best-axis-pairs]], [[post-draft]], [[start-here-for-collaborators]])
- `analysis/blog_corpus.json` (cited by [[alien-direction-steering]], [[built-pages-inventory]], [[distillation-check]])
- `analysis/distil_data.json` (cited by [[built-pages-inventory]], [[distillation-check]], [[judged-evaluations]] (+3))
- `analysis/fa_text_contrast.json` (cited by [[claims-and-evidence]], [[start-here-for-collaborators]], [[text-contrast-factors]])
- `analysis/functional_probe.json` (cited by [[judged-evaluations]], [[rl-capability-and-persona-drift]], [[seed-floor]])
- `analysis/hf_dataset_audit.json` (cited by [[hf-artefacts]], [[open-questions]])
- `analysis/intrinsic_coords.json` (cited by [[factor-first-migration]], [[pca-and-scree]])
- `analysis/judged_ceiling.json` (cited by [[judged-evaluations]])
- `analysis/live_components.json` (cited by [[built-pages-inventory]], [[judged-evaluations]])
- `analysis/live_results.json` (cited by [[built-pages-inventory]], [[distillation-check]], [[judged-evaluations]] (+1))
- `analysis/manifold_ideas.json` (cited by [[built-pages-inventory]])
- `analysis/module_holography.json` (cited by [[module-holography]], [[open-questions]])
- `analysis/monitor_page_data.json` (cited by [[built-pages-inventory]], [[judged-evaluations]], [[steering-results]])
- `analysis/olmo_envs.json` (cited by [[rl-capability-and-persona-drift]])
- `analysis/page_data.json` (cited by [[built-pages-inventory]], [[judged-evaluations]], [[steering-results]])
- `analysis/pc_loadings.json` (cited by [[factor-first-migration]], [[factor-pc1]], [[factor-pc2]] (+3))
- `analysis/persona_merge_audit.json` (cited by [[distillation-check]], [[persona-merge-correction]])
- `analysis/polarity_deflation.json` (cited by [[polarity-and-bipolarity]], [[source-contradictions]], [[stage-two-geometry]] (+1))
- `analysis/rank_sweep_mechanism.json` (cited by [[column-space-structure]], [[rank-sweep]])
- `analysis/rl_behavioural.json` (cited by [[rl-capability-and-persona-drift]])
- `analysis/rl_bigfive_coords.json` (cited by [[rl-capability-and-persona-drift]])
- `analysis/rl_correct_null.json` (cited by [[rl-capability-and-persona-drift]])
- `analysis/rl_module_profile.json` (cited by [[rl-capability-and-persona-drift]])
- `analysis/rl_probe.json` (cited by [[rl-capability-and-persona-drift]])
- `analysis/rl_probe_vllm.json` (cited by [[rl-capability-and-persona-drift]])
- `analysis/rl_projection.json` (cited by [[rl-capability-and-persona-drift]])
- `analysis/rl_score.json` (cited by [[rl-capability-and-persona-drift]])
- `analysis/rl_score_code.json` (cited by [[rl-capability-and-persona-drift]])
- `analysis/rl_sketches.json` (cited by [[post-draft]], [[rl-capability-and-persona-drift]])
- `analysis/rl_train.json` (cited by [[rl-capability-and-persona-drift]])
- `analysis/scree_null.json` (cited by [[open-questions]], [[pca-and-scree]], [[superseded-claims]] (+1))
- `analysis/scree_null_matched.json` (cited by [[built-pages-inventory]], [[factor-analysis-null-arms]], [[factors-versus-pca-coverage]] (+3))
- `analysis/slider_probe.json` (cited by [[persona-sliders]], [[post-draft]])
- `analysis/trait_graph.json` (cited by [[polarity-and-bipolarity]], [[source-contradictions]], [[stage-two-geometry]])
- `results/cross_gram_actweighted_root_x_data_null_seedpaired_s40_matched.npz` (cited by [[activation-weighted-gram]])
- `results/cross_gram_full_data_null_seedpaired_s40_x_data_null_seedpaired_s40_matched.npz` (cited by [[seed-floor]])
- `results/cross_gram_full_seed1_loras_introspection_x_seed1_loras_introspection.npz` (cited by [[stage-two-second-seed]])
- `results/decomposition_permuted_matched.json` (cited by [[null-controls]])
- `results/decomposition_seed1.json` (cited by [[cross-seed-geometry]], [[direction-seed-stability]])
- `results/decomposition_shuffled_matched.json` (cited by [[null-controls]], [[open-questions]])
- `results/gradient_atoms/zoo_dict_real_200_0.1.npz` (cited by [[gradient-atoms]])
- `results/gram_sketch100.npz` (cited by [[pca-and-scree]])

Files cited by history pages under other subprojects (`sweep100/results/`,
`drift/results/`, `teacherscreen/results/`) are outside this map; the pages that
cite them ([[sweep100]], [[drift-experiment]], [[teacherscreen]]) name the scripts.

## How to check a number

1. Find the number's `file#key` on the wiki page (or on [[claims-and-evidence]]).
2. Open the file under `qwen35/` and read the key. Never recompute.
3. If the value differs from the page, record it on [[source-contradictions]]
   and do not pick one silently.
4. If you need the figure, run `python qwen35/figures/post/make_post_figures.py`;
   it reads only the files in Table 1 and writes under `qwen35/figures/post/`.
   The companion is rebuilt by its own systemd timer from the same files and
   must not be run from a session; `check_numbers.py` is its regression test.
