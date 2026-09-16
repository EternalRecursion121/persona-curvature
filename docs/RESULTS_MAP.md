# Results map: write-up section -> figure -> analysis files -> scripts -> wiki pages

The write-up is `qwen35/POST_DRAFT.md`. Every figure is drawn by `qwen35/figures/post/make_post_figures.py` (function `fig_<name>` in its `FIGS` dict, light and dark, PNG and SVG in `qwen35/figures/post/`), which reads the analysis files listed here through `J("analysis/...")` and `np.load("results/...")`; the coordinates it imports from `figures/clusters/make_cluster_figures.py::load()` come from `analysis/viz_fa.json` and `results/gram_sweep.npz`. Analysis files are under `qwen35/` and are restored by `tools/fetch_data.py`. Wiki pages are `wiki/pages/<section>/<slug>.md`; each carries its own `sources:` list with JSON key paths.

## What we built

| item | files | scripts | wiki |
|---|---|---|---|
| Trait sets (100 Goldberg markers, 34 lexicon words, 6 refused) | `traits_primary.json`, `traits_secondary.json`, `traits_secondary_provenance.json`, `tda_masterkey.tab` | `select_traits.py` | `zoo/trait-provenance`, `zoo/zoo-construction-overview` |
| Constitutions and the anchor block | `constitutions.json`, `constitutions_neutral.json`, `constitutions_cost.json` | `constitutions.py`, `anchor_constitutions.py`, `regen_goldberg_senses.py` | `zoo/constitution-generation`, `zoo/constitution-anchor-revision` |
| Preference pairs on one 445-prompt pool | `prompts.json`, `data_common/*.jsonl` | `gen_pairs.py`, `make_common_pool.py` | `zoo/dpo-pair-generation` |
| Stage one training (r 64, 13 steps, seed 0) | `results/runmeta_sweep.json`, `phase2_runs/results.json` | `train_qwen35.py`, `fetch_runmeta.py`, `validate_100.py` | `zoo/stage-one-training-config`, `zoo/zoo-training-recipe`, `zoo/runmeta-provenance` |
| Stage two and the exact persona merge | `analysis/merge_audit.json`, `analysis/persona_key_repair.json` | `oct_stage2.py`, `fix_persona_merge.py`, `fix_persona_keys.py` | `zoo/stage-two-introspection`, `zoo/persona-merge-correction`, `history/lesson-verify-the-artefact-loads` |
| The judge (blind Big Five, 24 scenarios) | `phase10_runs/judged_100.json`, `phase10_runs/judged_steerfix.json` | `bigfive_probes.py`, `judge_personas.py` | `behaviour/judged-evaluations` |
| Screenshot: trait page | `figures/post/screen_trait_light.png` (hand-cropped from the companion) | `companion/build_companion.py` | `overview/post-draft` |

## The weight updates have factor structure

| item | figure | files | scripts | wiki |
|---|---|---|---|---|
| Figure 1: each Goldberg group on its best pair of factors; same-pole +0.24, opposite -0.08 | `facets_best_{light,dark}` (`fig_facets_own(best=True)`), `facets_own_*` | `analysis/viz_fa.json`, `analysis/best_axis_pairs.json`; the cosines: `results/decomposition.json` | `build_viz_data_fa.py`, `fa_chart.py`, `analyse_best_axis_pairs.py`, `decompose.py` | `geometry/factor-chart`, `geometry/best-axis-pairs`, `geometry/polarity-and-bipolarity`, `geometry/geometry-overview` |
| Screenshot: rotating chart | `screen_chart_light.png` | companion `chart.html` | `companion/build_companion.py`, `companion/assets/map3d.js` | |
| Figure 2: scree with the two null arms; congruence matrix | `scree_congruence_*` (`fig_scree_congruence`) | `analysis/scree_null_matched.json`, `results/fa_qwen35.json#correlation_matrix.centred_eigenvalues, solutions.centred_k5.congruence_oblimin` | `analyse_fa_qwen35.py`, `analyse_fa_nulls.py`, `make_nulls.py`, `launch_nulls.sh`, `run_nulls.sh`, `compare_nulls.py` | `geometry/factor-analysis`, `geometry/factor-analysis-null-arms`, `geometry/null-controls`, `geometry/pca-and-scree` |
| Congruences 0.66 / 0.57 / 0.68; parallel analysis retains nine | | `results/fa_qwen35.json#n_factors` | `analyse_fa_qwen35.py` | `geometry/factor-analysis`, `factors/factor-warmth` ... `factor-imagination` |
| Fisher-metric factors (A1) | | `analysis/fa_fisher_metric.json`, `results/fa_qwen35_fisher.json`, `results/fa_qwen35_fisher_emp.json`, `results/gram_fisher*.npz` | `fisher_gram.py` (Modal), `analyse_fisher_gram.py`, `analyse_fa_fisher.py` | `geometry/factor-analysis-fisher-metric` |
| Goldberg-only and held-out lexicon (A2); text readout 94% vs 98%; constitution predicts nothing | | `analysis/goldberg_only.json`, `results/fa_qwen35_goldberg100.json`, `results/gram_goldberg100.npz`, `analysis/emb_pairs_minilm.npz`, `analysis/emb_constitutions_mpnet.npy`, `analysis/goldberg_only_ratings.jsonl` | `embed_goldberg_only.py`, `analyse_goldberg_only.py`, `analyse_lexicon_only.py` | `geometry/goldberg-only-and-heldout-lexicon` |
| Text-contrast factors match at 0.81 to 0.97 | | `analysis/fa_text_contrast.json` | `analyse_fa_text_contrast.py` (embeddings from `embed_goldberg_only.py`) | `geometry/text-contrast-factors`, `geometry/goldberg-only-and-heldout-lexicon` |
| Prompting check (A7): r 0.70 at layer 16, Procrustes 74% | | `analysis/actspace_geometry.json`, `analysis/actspace_geometry_fa.json`, `analysis/actspace_means.npz` | `act_space.py` (Modal), `analyse_actspace.py`, `analyse_actspace_fa.py` | `actspace/actspace-persona-vectors`, `actspace/prompting-versus-training` |

## What a trait's weight update actually is

| item | figure | files | scripts | wiki |
|---|---|---|---|---|
| Figure 3a: Frobenius vs activation-weighted seed-pair cosine (0.018 vs 0.66; ceiling 0.86) | `seed_metrics_*` (`fig_seed_metrics`) | `analysis/act_gram.json#arms`, `analysis/crossseed_arms.json`, `results/cross_gram_full_root_x_data_null_seedpaired_s40_matched.npz`, `results/gram_actweighted*.npz` | `cross_gram_full_on_modal.py`, `analyse_crossseed.py`, `build_actgram_spec.py`, `act_gram_on_modal.py`, `analyse_act_gram.py`, `analyse_fa_actgram.py` | `geometry/seed-floor`, `geometry/cross-seed-geometry`, `geometry/activation-weighted-gram`, `history/lesson-coordinates-are-not-structure` |
| Figure 3b: column-space overlap 46% / 14% / 1.8%, 40/40 | same figure | `analysis/column_space.json#stage1.classes.*.col_wtd_k64, null.k64`, `results/column_space_{stage1,stage2,crossstage}.npz` | `column_space_on_modal.py`, `analyse_column_space.py` | `geometry/column-space-structure` |
| 40/40 nearest neighbour, Pearson 0.997 | | `analysis/crossseed_arms.json#seed_matched` | `analyse_crossseed.py` | `geometry/seed-floor` |

## Weight-space distance is not behaviour

| item | figure | files | scripts | wiki |
|---|---|---|---|---|
| Figure 4: rank sweep, chart correlation vs judged movement | `rank_sweep_*` (`fig_rank_sweep`) | `analysis/rank_sweep.json`, `results/rank_sweep_grams.npz`, `phase10_runs/judged_rank_sweep.json`, `phase2_runs/results_data_rank_sweep.json` | `train_rank_sweep.py`, `cross_gram_rank_sweep.py`, `analyse_rank_sweep.py` | `geometry/rank-sweep` |
| Sliders: 10/10 right direction, cosine 0.003 to 0.022 | | `analysis/persona_sliders.json`, `analysis/slider_targets.npz`, `analysis/slider_train.json`, `analysis/slider_cross_gram.json` | `build_slider_targets.py`, `persona_sliders.py`, `cross_gram_sliders.py`, `build_slider_as_eval.py`, `analyse_persona_sliders.py` | `behaviour/persona-sliders` |
| The two stages: cosine 0.000 / 0.53 / 0.009 | | `analysis/stage2_structure.json`, `analysis/actspace_stage2_geometry.json`, `analysis/act_gram_stage2.json` | `analyse_stage2_structure.py`, `analyse_actspace_stage2.py`, `act_gram_stages_on_modal.py`, `analyse_act_gram_stage2.py` | `geometry/stage-two-structure`, `geometry/activation-weighted-gram-stages` |
| Figure 5: Fisher norms of 124 directions span 260x | `fisher_spread_*` (`fig_fisher`) | `analysis/fisher_norms.json`, `phase10_runs/fisher_results.json`, `phase10_runs/fisher_spec.json` | `build_fisher_spec.py`, `fisher.py`, `analyse_fisher.py` | `behaviour/fisher-norms` |

## What stage two does

| item | figure | files | scripts | wiki |
|---|---|---|---|---|
| Figure 6: cosine to grand mean by stage; register statistics vs alpha | `stage_two_*` (`fig_stage_two`) | `results/gram_sweep.npz`, `results/gram_stage2.npz`, `analysis/stage2_structure.json#shared_component`, `analysis/s2mean_steer_stats.json#S2_mean, mean_assistant_axis`, `phase10_runs/steer_results_s2mean.json` | `analyse_stage2_structure.py`, `build_gram_stage2_noshared.py`, `analyse_stage2_factors.py`, `steer_fix.py`, `analyse_s2mean_steer.py` | `geometry/stage-two-structure`, `behaviour/stage-two-shared-direction`, `geometry/stage-two-exploration` |
| Screenshot: stage-two slider | `screen_stage2_light.png` | companion `stage-two.html` | `companion/build_companion.py`, `companion/assets/stage2.js` | |
| Trait-free stage two at 0.30; nearest `unemotional` | | `analysis/stage2_neutral_control.json`, `analysis/stage2_frame.json`, `analysis/lora_a_identity.json` | `oct_stage2.py --neutral`, `analyse_stage2_neutral.py`, `analyse_stage2_frame.py`, `check_lora_a_identity.py` | `geometry/stage-two-exploration` |
| Register vs residual: 0.9% of the amplification | | `analysis/stage2_register_vs_residual.json`, `phase10_runs/judged_s2register.json` | `build_s2register_spec.py`, `build_s2register_eval.py`, `analyse_s2register.py` | `geometry/stage-two-exploration` |
| Persona geometry is stage one's (r 0.99); personas move 9/10 dials further | | `analysis/fulloct_geometry.json`, `results/gram_personas.npz`, `analysis/spider.json#personas` | `analyse_fulloct.py`, `build_personas_seed1.py`, `build_spider_data.py` | `geometry/full-oct-replication`, `behaviour/ocean-dials-replication` |
| Figure 12: self-identification | `selfid_*` (`fig_selfid`) | `results/selfid_generations.json`, `analysis/selfid.json`, `PREREG_selfid.md` | `selfid_on_modal.py`, `analyse_selfid.py` | `behaviour/self-identification-probe` |

## Does the map predict behaviour?

| item | figure | files | scripts | wiki |
|---|---|---|---|---|
| Figure 7: dose-response at equal alpha and at matched dose (1.90 -> 1.65 -> 0.80) | `dose_response_*` (`fig_dose`) | `analysis/matched_dose_steering.json#directions.FA_*.matched, published_equal_alpha, summary`, `analysis/matched_dose_alphas.json`, `phase10_runs/dose_calib.json`, `phase10_runs/judged_dose.json` | `fisher_dose.py`, `solve_matched_alphas.py`, `steer_fix.py`, `dose_to_eval.py`, `judge_personas.py`, `analyse_matched_dose.py` | `behaviour/matched-dose-steering`, `behaviour/steering-results` |
| Screenshot: dose-response widget | `screen_behaviour_light.png` | companion `behaviour.html#dose` | `companion/assets/behaviour.js` | |
| Alpha is half an adapter | | `analysis/steer_alpha_units.json` | `analyse_steer_alpha_units.py` | `behaviour/stage-two-shared-direction`, `overview/superseded-claims` |
| Figure 8: sphere, angle vs judged-profile distance (Spearman 0.68, 45/72 habitable) | `sphere_*` (`fig_sphere`) | `analysis/sphere_page_fa.json#points, judged, smooth`, `analysis/sphere_layout_fa.json`, `phase10_runs/sphere_spec_fa.json`, `phase10_runs/judged_sphere_fa.json` | `build_sphere_spec_fa.py`, `sphere_sweep.py`, `sphere_to_eval.py`, `judge_personas.py`, `build_sphere_page.py` | `behaviour/sphere-sweep-factor-chart`, `behaviour/sphere-sweep` |
| Iso-KL sphere (A4): 0.65 | | `analysis/sphere_isokl.json`, `analysis/sphere_isokl_alphas.json`, `phase10_runs/sphere_isokl_calib.json`, `PREREG_sphere_isokl.md` | `build_sphere_isokl_calib.py`, `sphere_dose.py`, `solve_sphere_isokl_alphas.py`, `analyse_sphere_isokl.py` | `behaviour/sphere-sweep-iso-kl` |
| Screenshot: sphere widget | `screen_sphere_light.png` | the interactive write-up (Claude artifact), built by `build_blog_page.py` | `build_blog_page.py`, `build_blog_data.py` | |
| Figure 9: dials, 8/10 and 10/10; additivity 53% | `dials_*` (`fig_dials`) | `analysis/spider.json#axes, bigfive`, `phase10_runs/judged_steerfix.json`, `phase10_runs/judged_bigfive.json`, `analysis/bigfive_adapters_geometry.json`, `analysis/additivity.json` | `build_spider_data.py`, `build_spider_page.py`, `eval_bigfive.py`, `analyse_bigfive.py`, `analyse_additivity.py` | `behaviour/ocean-dials-replication`, `behaviour/bigfive-factor-adapters`, `behaviour/additivity` |
| BFI vs TRAIT (A6) | | `analysis/inspect_personality.json`, `analysis/inspect_trait20.json`, `phase10_runs/inspect_*.jsonl` | `inspect_render_items.py`, `inspect_personality_on_modal.py`, `validate_inspect_harness.py`, `analyse_inspect_personality.py` | `behaviour/inspect-personality-evals` |

## Scoring training data against a direction

| item | figure | files | scripts | wiki |
|---|---|---|---|---|
| The identity, validated at r 0.9999992 | | `analysis/align_validate.json` | `align_score.py` | `behaviour/scoring-identity`, `.garden/notes/scoring-data-against-a-lora-direction.md` |
| Figure 10: first-order push predicts judged shift (r 0.61 to 0.85) | `forecast_matrix_*` (`fig_forecast`) | `analysis/data_forecast.json`, `analysis/nxn_scores.json`, `phase10_runs/judged_100.json` | `build_nxn_inputs.py`, `align_score.py`, `analyse_nxn.py`, `analyse_data_forecast.py` | `behaviour/data-forecast`, `geometry/n-by-n-scoring` |
| In-sample control (A8): 134/134 | | `analysis/nxn_summary.json` | `analyse_nxn.py` | `geometry/n-by-n-scoring` |
| Dolci audit (A9): warmth and corrigibility, refusal tails | | `analysis/dolci_audit.json`, `analysis/dolci_scores.json`, `analysis/dolci_judge*.json`, `phase10_runs/dolci_selection.json` | `build_dolci_inputs.py`, `dolci_score.py`, `merge_dolci_shards.py`, `analyse_dolci_scores.py`, `judge_dolci.py`, `extract_dolci_examples.py` | `behaviour/dolci-data-audit` |
| Probe adapters (A9) | | `analysis/probe_adapters.json`, `analysis/probe_scores_sft.json` | `build_probe_score_inputs.py`, `judge_probes.py`, `analyse_probe_adapters.py` | `behaviour/probe-adapters` |
| Corrigibility flag inverts (A11): -0.20, p 0.007 | | `analysis/dolci_flag_training.json`, `analysis/dolci_flag_*.json`, `phase10_runs/dolci_flag_*`, `PREREG_dolciflag.md` | `build_dolci_flag_arms.py`, `dolci_flag_train.py`, `dolci_flag_eval.py`, `merge_dolci_flag_gens.py`, `judge_dolci_flag.py`, `dolci_flag_refusal_style.py`, `dolci_flag_examples.py`, `analyse_dolci_flag.py`, `render_dolci_flag_tables.py` | `behaviour/dolci-flag-training` |
| Sycophancy forecast (A11): 0.94 / 0.84 / -0.49 | | `analysis/syc_forecast.json`, `analysis/syc_adrift.json`, `phase10_runs/syc_*`, `PREREG_sycforecast.md` | `build_syc_arms.py`, `build_syc_battery.py`, `syc_train.py`, `syc_eval.py`, `syc_score.py`, `judge_syc.py`, `build_syc_post_inputs.py`, `analyse_syc_forecast.py` | `behaviour/sycophancy-forecast` |
| Figure 11a, c: emergent misalignment, forecast Spearman 0.91, 13/79 vs 0/79 | `external_data_*` (`fig_external`) | `analysis/em_medical.json#part_a`, `analysis/em_part_b.json#forecast_agreement`, `analysis/em_part_{a,b,c,d}.json`, `analysis/em_column_space.json`, `phase10_runs/em_*`, `PREREG_em_medical.md` | `em_build_inputs.py`, `dolci_score.py`, `em_sft.py`, `em_flatten.py`, `em_eval.py`, `em_to_eval.py`, `judge_em.py`, `em_probe_train.py`, `em_probe_inputs.py`, `judge_em_probe.py`, `column_space_em_on_modal.py`, `analyse_em_{a,b,c,d}.py`, `analyse_em_colspace.py`, `analyse_em.py` | `behaviour/emergent-misalignment-medical`, `history/paper-model-organisms-em` |
| Figure 11b: reward hacking invisible | same figure | `analysis/sorh_data_scoring.json#directions, random_band`, `analysis/sorh_behavioural.json`, `analysis/sorh_projection.json`, `analysis/column_space_sorh.json`, `analysis/gradient_atoms_sorh.json`, `analysis/rl_sketches*.json` | `sft_rewardhacks.py`, `build_sorh_datascore_inputs.py`, `align_score.py`, `analyse_sorh_data_scores.py`, `eval_rl_persona.py`, `sorh_to_eval.py`, `analyse_sorh.py`, `analyse_sorh_behaviour.py`, `column_space_sorh_on_modal.py`, `analyse_column_space_sorh.py`, `analyse_gradient_atoms_sorh.py` | `behaviour/reward-hacks-data-scoring`, `behaviour/reward-hacks-column-space`, `behaviour/reward-hacks-gradient-atoms`, `history/paper-school-of-reward-hacks` |
| Gradient atoms (unsupervised atoms carry bipolar axes) | | `analysis/gradient_atoms.json`, `analysis/gradient_atoms_weightspace.json`, `results/gradient_atoms/*.npz` | `build_gradatoms_inputs.py`, `gradient_atoms_on_modal.py`, `analyse_gradient_atoms.py` | `geometry/gradient-atoms` |

## What this does not show

| item | files | scripts | wiki |
|---|---|---|---|
| The widest uncovered region (A10): 54.8 deg vs 40.3 null; hole words miss; slider indistinguishable from shuffles | `analysis/direction_gaps_fa.json`, `analysis/alien_fa.json`, `analysis/alien_steer_fa.json`, `analysis/alien_match_fa.json`, `analysis/hole_geometry_fa.json`, `analysis/slider_probe.json` | `analyse_alien_fa.py`, `build_alien_spec_fa.py`, `steer_fix.py`, `analyse_alien_steer_fa.py`, `analyse_hole_fa.py`, `analyse_gaps.py` | `geometry/hole-words-factor-chart`, `behaviour/alien-direction-factor-chart`, `behaviour/alien-direction-steering`, `geometry/hole-words` |
| Alignment traits on the chart | `analysis/alignment_geometry_fa.json`, `analysis/alignment_geometry_aligncommon.json`, `PREREG_alignment.md` | `analyse_alignment.py`, `analyse_alignment_fa.py` | `geometry/alignment-traits-geometry`, `zoo/alignment-and-hole-traits` |
| Column space disagrees on the EM arms | `analysis/em_column_space.json` | `column_space_em_on_modal.py`, `analyse_em_colspace.py` | `behaviour/emergent-misalignment-medical` |
| Claims table and superseded claims | | | `overview/superseded-claims`, `overview/source-contradictions`, `overview/open-questions` |

## Earlier frames the write-up no longer leads with

The principal-component chart (`analysis/viz.json`, `build_viz_data.py`, `analyse_alien.py`, `build_sphere_spec.py`, `analyse_alien_steer.py`, `analyse_hole.py`, `analyse_alignment.py`) preceded the factor chart, which became primary on 2026-09-08; the `_fa` twins of those scripts are the current versions. The blog page (`build_blog_page.py`, `build_blog_data.py`, `build_blog_corpus.py`), the direction pages (`build_direction_pages.py`, `analysis/qual_*.json`), the findings/distil/monitor/live/manifold pages and `build_traits_page_data.py` are historical builders whose numbers the wiki records with status `historical` where they have moved on.
