---
title: Claims and evidence
summary: The seven-row claims table of the 2026-09-15 post draft expanded so that every number in it names the analysis file and JSON key it was read from on 2026-09-16, the wiki page that holds the full result, and the control that supports it.
status: current
sources:
  - qwen35/POST_DRAFT.md
  - qwen35/results/fa_qwen35.json#solutions.centred_k5.congruence_oblimin
  - qwen35/results/fa_qwen35.json#n_factors.chosen
  - qwen35/analysis/fa_nulls.json#arms.shuffled.n_factors_chosen
  - qwen35/analysis/fa_nulls.json#arms.permuted.n_factors_chosen
  - qwen35/analysis/fa_nulls.json#tucker_vs_real_stage_one.permuted.source_relabelled.best_matching
  - qwen35/analysis/fa_fisher_metric.json#loading_matrix_congruence_k5.pairs
  - qwen35/analysis/goldberg_only.json#verdicts.T1a_factors_survive_dropping_the_34.value
  - qwen35/analysis/goldberg_only.json#test3_text_embedding_baseline.ridge_pair_contrast_to_chart.heldout_r2_overall
  - qwen35/analysis/goldberg_only.json#test3_text_embedding_baseline.ridge_constitution_to_chart.heldout_r2_overall
  - qwen35/analysis/goldberg_only.json#test3_text_embedding_baseline.comparator_adapter_geometry.r2_overall
  - qwen35/analysis/goldberg_only.json#test3_text_embedding_baseline.text_gram_vs_adapter_gram.max_of_constitution_and_contrast
  - qwen35/analysis/fa_text_contrast.json#best_matching
  - qwen35/analysis/fa_text_contrast.json#mean_abs_matched_congruence
  - qwen35/analysis/stage2_structure.json#decomposition_tests.stage1.test2_same_opp_gap_p
  - qwen35/analysis/stage2_structure.json#decomposition_tests.stage2.test2_same_opp_gap_p
  - qwen35/analysis/rank_sweep.json#ranks.r1.identification_among_134.top1
  - qwen35/analysis/rank_sweep.json#ranks.r1.chart.pearson_75
  - qwen35/analysis/rank_sweep.json#behaviour.ranks.r1.mean_own_factor_amplification
  - qwen35/analysis/act_gram.json#arms.frob.a_same_trait_cross_seed_mean
  - qwen35/analysis/act_gram.json#arms.frob.d_top1_of_134
  - qwen35/analysis/act_gram.json#arms.frob.e_pearson_cross_vs_within
  - qwen35/analysis/act_gram.json#arms.pool445.a_same_trait_cross_seed_mean
  - qwen35/analysis/act_gram.json#arms.pool445.within_seed0_offdiag_mean_cosine
  - qwen35/analysis/act_gram.json#verdict
  - qwen35/analysis/column_space.json#stage1.identification_40_seed1_queries_vs_134_seed0.col_wtd_k64
  - qwen35/analysis/column_space.json#stage1.top1_left_vector_same_trait_cross_seed.mean_abs_cos
  - qwen35/analysis/column_space.json#stage1.top1_left_vector_diff_trait_cross_seed.mean_abs_cos
  - qwen35/analysis/persona_sliders.json#weight_space.sliders.*.cos_with_134
  - qwen35/analysis/act_gram_stage2.json#verdict.R_res
  - qwen35/analysis/fisher_norms.json#comparisons.full_range.ratio
  - qwen35/analysis/fisher_norms.json#comparisons.stage2_vs_stage1_grand_mean.ratio_S2mean_over_stage1mean
  - qwen35/analysis/fisher_norms.json#directions.FA_Warmth.F_ref
  - qwen35/analysis/fisher_norms.json#random_band.median
  - qwen35/analysis/stage2_structure.json#shared_component
  - qwen35/analysis/stage2_structure.json#centred_cosines.corr_stage1_stage2
  - qwen35/analysis/stage2_neutral_control.json#neutral_vs_grand_mean.mean
  - qwen35/analysis/stage2_neutral_control.json#per_neutral.*.max_cos_trait
  - qwen35/analysis/stage2_exploration.json#register_vs_residual.share_of_gap_cond_c
  - qwen35/analysis/fulloct_geometry.json#gram_correlation_offdiag.persona_vs_stage1
  - qwen35/analysis/selfid.json#per_prompt.P1.conditions
  - qwen35/analysis/sphere_page_fa.json#smooth
  - qwen35/analysis/sphere_page_fa.json#coherence.none
  - qwen35/analysis/matched_dose_steering.json#published_equal_alpha_summary.all.suppress_over_amplify_raw
  - qwen35/analysis/matched_dose_steering.json#summary.all.suppress_over_amplify_raw
  - qwen35/analysis/matched_dose_steering.json#summary.all.suppress_over_amplify_room
  - qwen35/analysis/additivity.json#median_resid_over_effect
  - qwen35/analysis/data_forecast.json#axis_score_vs_shift_matrix_rows_axis_cols_dim
  - qwen35/analysis/data_forecast.json#directions.*.label_baseline_pearson
  - qwen35/analysis/syc_forecast.json#weight_space.cos_sycophantic_spearman_vs_forecast.rho
  - qwen35/analysis/syc_forecast.json#bigfive.agreeableness_spearman_vs_axis_score.rho
  - qwen35/analysis/syc_forecast.json#contrasts.syc_top_vs_syc_bottom.praise_neutral
  - qwen35/analysis/syc_forecast.json#primary_spearman.rho
  - qwen35/analysis/syc_forecast.json#rates.syc_top.capitulation_rate_judge
  - qwen35/analysis/syc_forecast.json#compliance.rates
  - qwen35/analysis/dolci_flag_training.json#compliance_tests.flagged_vs_random.any_engagement_rate
  - qwen35/analysis/dolci_flag_training.json#weight_space.cosine_with_alignment_adapters
  - qwen35/analysis/em_medical.json#part_a.prereg_verdict.directions_clearing_band
  - qwen35/analysis/em_medical.json#part_c.em_questions.per_condition
  - qwen35/analysis/em_part_b.json#forecast_agreement
  - qwen35/analysis/em_part_b.json#factor_chart.em_bad_c126.chart_len_over_trait_mean
  - qwen35/analysis/sorh_data_scoring.json#directions.sorh_hack_minus_control.paired_diff.frac_positive
  - qwen35/analysis/sorh_projection.json#sorh_hack.length_unit
  - qwen35/analysis/sorh_projection.json#reference.trait_chart_len_mean
  - qwen35/analysis/actspace_geometry_fa.json#windows.resp.procrustes_r2_fa
  - qwen35/analysis/actspace_geometry_fa.json#windows.resp.procrustes_null_95pct
  - qwen35/phase2_runs/archive/phase5_sweep_134.json
last_verified: 2026-09-16
tags: [overview, claims, evidence, post]
---

# Claims and evidence

The post draft of 2026-09-15 ([[post-draft]]) closes with a seven-row table:
claim, evidence, status, what would break it. This page is that table with the
evidence column opened up. Every number below was re-read from the named file and
key on 2026-09-16; where the draft rounds, the full-precision value is given
beside it. The "control" column names the measurement that rules out the obvious
alternative explanation. Where the draft's number is a ratio the file does not
store, both operands are cited and the arithmetic is shown.

Conventions: `results/` and `analysis/` are under `qwen35/`. "Tucker congruence"
and the other terms are defined in the [[glossary]]. The wiki page named in each
row holds the full result, its caveats and its history.

## Row 1. The updates have five-factor structure

Status in the draft: holds; Big Five congruence 0.40 to 0.68. Would break it:
independent constitutions giving a different rotation.

| number in the draft | value in the file | file and key | wiki page | control |
|---|---|---|---|---|
| Warmth to Agreeableness 0.66 | 0.6554774644183582 | `results/fa_qwen35.json#solutions.centred_k5.congruence_oblimin[0][1]` | [[factor-analysis]], [[factor-warmth]] | permuted-label arm: agreement with labels at chance |
| Competence to Conscientiousness 0.57 | 0.5744454556114018 | `...congruence_oblimin[1][2]` | [[factor-competence]] | same |
| Timidity to Emotional Stability 0.40 | 0.4048918307420467 (and 0.3380 to Extraversion) | `...congruence_oblimin[2][3]`, `[2][0]` | [[factor-fearful-withdrawal]] | the rotation is also in the text (row 2) |
| Arousal to Extraversion 0.54 | 0.538926935263515 (and -0.3483 to Emotional Stability) | `...congruence_oblimin[3][0]`, `[3][3]` | [[factor-arousal]] | same |
| Imagination to Intellect 0.68 | 0.6822835176928186 | `...congruence_oblimin[4][4]` | [[factor-imagination]] | same |
| parallel analysis retains nine | `chosen` 9 at reference N 1528 | `results/fa_qwen35.json#n_factors.chosen`, `#n_factors.parallel_analysis_centred.grid[3].k_unreduced_95pct` | [[factor-analysis]], [[pca-and-scree]] | five is the hypothesis, not the data's count |
| same-pole +0.24, opposite-pole -0.08 | 0.24472701836565558, -0.08134207180636474 | `analysis/stage2_structure.json#decomposition_tests.stage1.test2_same_opp_gap_p[0..1]` | [[polarity-and-bipolarity]] | permutation p 4.999750012499375e-05 (same key, index 3) |
| shuffled arm retains zero factors | 0 | `analysis/fa_nulls.json#arms.shuffled.n_factors_chosen` | [[factor-analysis-null-arms]], [[null-controls]] | trained at the zoo's exact objective on the same 445 prompts |
| permuted arm retains eight, agreement at chance | 8; best-matched congruence mean 0.1779 | `analysis/fa_nulls.json#arms.permuted.n_factors_chosen`, `#tucker_vs_real_stage_one.permuted.mean_abs_best_matched_congruence` | [[factor-analysis-null-arms]] | re-identified by data (next row) |
| re-identified permuted factors at 0.98 to 1.00 | 0.9976, 0.9951, 0.9854, 0.9940, 0.9838 (signs free) | `analysis/fa_nulls.json#tucker_vs_real_stage_one.permuted.source_relabelled.best_matching[*].congruence` | [[factor-analysis-null-arms]] | the shared prompt pool is not what the factors are made of |
| Fisher metric 0.97 expected, 0.96 empirical | 0.965359016717984 and 0.9577766396647475 (mean abs matched) | `analysis/fa_fisher_metric.json#loading_matrix_congruence_k5.pairs.frobenius_vs_fisher_expected.mean_abs_matched_congruence`, `...frobenius_vs_fisher_empirical.mean_abs_matched_congruence` | [[factor-analysis-fisher-metric]] | metric change, not a rerun |
| Goldberg-only factoring 0.99 | 0.9922779568528804 | `analysis/goldberg_only.json#verdicts.T1a_factors_survive_dropping_the_34.value` | [[goldberg-only-and-heldout-lexicon]] | the 34 held-out words are placed where an independent rater puts them (same page, test 2b) |

## Row 2. The structure is the training contrast's

Status: holds. Would break it: a second base model landing traits elsewhere on
the same data.

| number in the draft | value in the file | file and key | wiki page | control |
|---|---|---|---|---|
| arrangement complete at rank 1 | chart Pearson 0.9967445664202496; 15 of 15 identified | `analysis/rank_sweep.json#ranks.r1.chart.pearson_75`, `#ranks.r1.identification_among_134.top1` | [[rank-sweep]] | rank-1 judged own-factor movement -0.41 against 27.08 at rank 64 (`#behaviour.ranks.r1.mean_own_factor_amplification`, `..._rank64`) |
| text readout predicts 94% of held-out position | held-out R^2 0.9380347105286267 | `analysis/goldberg_only.json#test3_text_embedding_baseline.ridge_pair_contrast_to_chart.heldout_r2_overall` | [[goldberg-only-and-heldout-lexicon]] | the constitution alone: held-out R^2 -0.034409160222032664 (`...ridge_constitution_to_chart.heldout_r2_overall`) |
| adapter geometry itself 98% | 0.9822421695022922 | `...comparator_adapter_geometry.r2_overall` | same | ceiling for the text readout |
| text Gram and adapter Gram correlate 0.86 | 0.8601935162272526 | `...text_gram_vs_adapter_gram.max_of_constitution_and_contrast` | same | |
| text factors match adapter factors 0.81 to 0.97 | Warmth 0.9230, Imagination 0.9023, Competence 0.9705, Timidity 0.8131, Arousal 0.8967; mean 0.901112105638296 | `analysis/fa_text_contrast.json#best_matching`, `#mean_abs_matched_congruence` | [[text-contrast-factors]] | the matching is a permutation; the Timidity and Arousal rotation appears in the text solution too |
| prompted arrangement, Procrustes 74%, null 4% | 0.7387431438763885; null 95th percentile 0.03568178186070537 | `analysis/actspace_geometry_fa.json#windows.resp.procrustes_r2_fa`, `#windows.resp.procrustes_null_95pct` | [[actspace-persona-vectors]] | an artefact check, not evidence the model added structure (draft A7) |

## Row 3. A trait is its output subspace

Status: holds within one recipe. Would break it: cross-seed column-space overlap
failing on new data.

| number in the draft | value in the file | file and key | wiki page | control |
|---|---|---|---|---|
| cross-seed cosine 0.018 | 0.018060998368198355 (n 40) | `analysis/act_gram.json#arms.frob.a_same_trait_cross_seed_mean` | [[seed-floor]] | r/d = 64/2560 = 0.025; frame overlap 0.021471366484559353 (`#arms.frob.c2_frame_overlap.seed0_vs_seed1.c2b_summed`) |
| 40 of 40 identified; matrices agree 0.997 | 40; 0.9966259139824906 | `#arms.frob.d_top1_of_134`, `#arms.frob.e_pearson_cross_vs_within` | [[cross-seed-geometry]] | different-trait cross-seed cosine 0.0018594300142914396 (`#arms.frob.b_diff_trait_cross_seed_mean`) |
| activation-weighted cosine 0.66, unrelated 0.09 | 0.6642887281525424; 0.08760415842065107 | `#arms.pool445.a_same_trait_cross_seed_mean`, `#arms.pool445.within_seed0_offdiag_mean_cosine` | [[activation-weighted-gram]] | ceiling 0.8909847593616453 (`#verdict.null_c2b_C`); pairs at the same fraction of ceiling in both metrics (`#verdict.ratio_a_C_over_null_C` 0.7456, `#verdict.ratio_a_F_over_null_F` 0.8412) |
| column space 46% vs 14% | same_mean 0.46279728915356094, off_mean 0.14087331941381756, top1 40 of 40 | `analysis/column_space.json#stage1.identification_40_seed1_queries_vs_134_seed0.col_wtd_k64` | [[column-space-structure]] | row space identifies 19 of 40 (`...row_unw_k64.top1`) |
| top output directions agree 0.77 vs 0.21 | 0.7701377220108219; 0.21012838847727913 | `#stage1.top1_left_vector_same_trait_cross_seed.mean_abs_cos`, `#stage1.top1_left_vector_diff_trait_cross_seed.mean_abs_cos` | [[column-space-structure]] | |

## Row 4. Map is not behaviour

Status: holds. Would break it: a metric in which functionally similar adapters are
always close.

| number in the draft | value in the file | file and key | wiki page | control |
|---|---|---|---|---|
| rank sweep | see row 2 | `analysis/rank_sweep.json` | [[rank-sweep]] | |
| sliders at cosine 0.003 to 0.022 with the trained adapter | 0.0025055221550983157 (warm) to 0.022155165412935125 (organized) | `analysis/persona_sliders.json#weight_space.sliders.<trait>.cos_with_134.<trait>` | [[persona-sliders]] | the sliders move the judged persona the right way (`#behaviour.own_factor_movement`) |
| stage one vs stage two of one trait, activation-weighted 0.009 | R_res 0.009222883147729255 | `analysis/act_gram_stage2.json#verdict.R_res` | [[activation-weighted-gram-stages]] | same-function benchmark 0.7455668811085819 (`#verdict.benchmark_same_function_stage1_cross_seed`) |
| Fisher norms span a factor of 260 | 259.924881015629 | `analysis/fisher_norms.json#comparisons.full_range.ratio` | [[fisher-norms]] | random-merge band median 0.12382989334918551 (`#random_band.median`) |
| Warmth three times a random merge | 0.36503759829531435 / 0.12382989334918551 = 2.95 | `#directions.FA_Warmth.F_ref`, `#random_band.median` | [[fisher-norms]] | |
| stage-two shared direction 6.7 times flatter | 0.14899427326654663 (its inverse is 6.71) | `#comparisons.stage2_vs_stage1_grand_mean.ratio_S2mean_over_stage1mean` | [[fisher-norms]], [[stage-two-shared-direction]] | |

## Row 5. Stage two installs a register first

Status: holds. Would break it: a persona whose extra amplitude comes from the
register.

| number in the draft | value in the file | file and key | wiki page | control |
|---|---|---|---|---|
| 15% of every stage-two adapter's norm; cosine 0.39, spread 0.03 | 0.1514038882692439; 0.38929785188872645; sd 0.031056220786483055 | `analysis/stage2_structure.json#shared_component.stage2` | [[stage-two-structure]] | stage one: 0.07948767279497902, 0.28103697835548513, sd 0.12559260390449356 (`#shared_component.stage1`) |
| trait-free runs at 0.30 | 0.30353526531657493 (five seeds, sd 0.0007) | `analysis/stage2_neutral_control.json#neutral_vs_grand_mean.mean` | [[stage-two-exploration]] | nearest zoo trait `unemotional` for all five (`#per_neutral[*].max_cos_trait`) |
| residual correlates with stage one at 0.81 | 0.811787783969743 | `analysis/stage2_structure.json#centred_cosines.corr_stage1_stage2` | [[stage-two-structure]] | |
| bipolarity mostly gone, opposite poles +0.12 | 0.12301499153773737 (same-pole 0.19104548852552697) | `#decomposition_tests.stage2.test2_same_opp_gap_p[1]`, `[0]` | [[polarity-and-bipolarity]] | stage one -0.08134207180636474 |
| persona arrangement is stage one's, 0.99 | 0.9915249945319294 | `analysis/fulloct_geometry.json#gram_correlation_offdiag.persona_vs_stage1` | [[full-oct-replication]] | |
| shared direction accounts for 0.9% of the extra amplitude | 0.009326363320103694 | `analysis/stage2_exploration.json#register_vs_residual.share_of_gap_cond_c` | [[stage-two-exploration]] | |
| stage one names its word 0.2%, persona 0.3%, stage two 13 traits and chart cosine 0.61 | 0.0023320895522388058; 0.002798507462686567; 13; 0.6105703585726646; same factor and pole 0.5584415584415584 | `analysis/selfid.json#per_prompt.P1.conditions.{stage1,persona,stage2}.{exact_hit_rate,n_traits_with_any_exact_hit,chart_cos_mean,same_factor_and_pole_share}` | [[self-identification-probe]] | permutation null exact-hit p95 0.0032649253731343282 (`...stage2.exact_hit_rate_null_p95`); the pre-registered persona prediction failed |

## Row 6. Direction predicts personality

Status: holds in the zoo's frame. Would break it: steering at matched dose that
breaks the sphere correlation.

| number in the draft | value in the file | file and key | wiki page | control |
|---|---|---|---|---|
| sphere Spearman 0.68; 0.95 near, 2.92 far; 45 of 72 habitable | 0.6765886845997277; 0.9462620103993861; 2.917385198267297; 45 | `analysis/sphere_page_fa.json#smooth.{rho,near_mean,far_mean}`, `#coherence.none` | [[sphere-sweep-factor-chart]] | iso-KL rerun of the PC sphere keeps 0.65 ([[sphere-sweep-iso-kl]]) |
| suppress-over-amplify 1.90 at equal alpha, 1.65 at matched dose, 0.80 by headroom | 1.8957528957528955; 1.6455696202531644; 0.7946215813581846 | `analysis/matched_dose_steering.json#published_equal_alpha_summary.all.suppress_over_amplify_raw`, `#summary.all.suppress_over_amplify_raw`, `#summary.all.suppress_over_amplify_room` | [[matched-dose-steering]] | KL target 0.24765978669837102 nats per token (`#kl_target_nats_per_token`) |
| dials: 8 of 10 (axes), 10 of 10 right direction and 8 most (factor adapters) | counts on the pages | `analysis/spider.json#axes`, `#bigfive` | [[ocean-dials-replication]], [[bigfive-factor-adapters]] | keyed-adjective average 7 of 10, personas 8 of 10 (`#traits`, `#personas`) |
| additivity deviation 53% | 0.5282408112973758 | `analysis/additivity.json#median_resid_over_effect` | [[additivity]] | noise norm 0.31666612924087495 (`#noise_norm`) |
| forecast r 0.61 to 0.85, label 0.44 to 0.73 | diagonal 0.753, 0.855, 0.735, 0.612, 0.737; labels 0.640, 0.729, 0.623, 0.440, 0.634 | `analysis/data_forecast.json#axis_score_vs_shift_matrix_rows_axis_cols_dim`, `#directions.<factor>.label_baseline_pearson` | [[data-forecast]] | leave-one-out scoring (`#directions.<factor>.axis_*_loo`); in-sample N x N control 134 of 134 ([[n-by-n-scoring]]) |

## Row 7. First-order score flags harmful data

Status: register only; misalignment from a contrast. Would break it: a harm
signature that clears the band without a matched control.

| number in the draft | value in the file | file and key | wiki page | control |
|---|---|---|---|---|
| sycophancy arms: weight-space ordering 0.94, Agreeableness 0.84, praise +1.0 | 0.942857142857143; 0.8406680016960504; +1.0 (p 0.007849607519624019) | `analysis/syc_forecast.json#weight_space.cos_sycophantic_spearman_vs_forecast.rho`, `#bigfive.agreeableness_spearman_vs_axis_score.rho`, `#contrasts.syc_top_vs_syc_bottom.praise_neutral` | [[sycophancy-forecast]] | length-matched control arm (`syc_control`) in the same six |
| composite deference -0.49; top arm holds 20 of 20, bottom gives way on half | -0.48571428571428577; capitulation_rate_judge syc_top 0.0, syc_bottom 0.5 | `#primary_spearman.rho`, `#rates.{syc_top,syc_bottom}.capitulation_rate_judge` | [[sycophancy-forecast]] | the sycophantic direction's trained arm lands nearest `warm` (`#weight_space.nearest_zoo_traits_by_abs_cosine`) |
| corrigibility: flagged arm engages less, -0.20 p 0.007; -0.175 length-stratified | -0.2, p 0.006899655017249137; corr_ls 0.15 against random 0.325 | `analysis/dolci_flag_training.json#compliance_tests.flagged_vs_random.any_engagement_rate`; `analysis/syc_forecast.json#compliance.rates.{corr_ls,random}.any_engagement_rate` | [[dolci-flag-training]], [[sycophancy-forecast]] | weight-space forecast held: cos with corrigible -0.0307 flagged, +0.0058 random, +0.0470 anti (`dolci_flag_training.json#weight_space.cosine_with_alignment_adapters.*.corrigible`) |
| emergent misalignment: three directions clear the band before training | `trait_unintelligent`, `trait_negligent`, `axis_Agreeableness` | `analysis/em_medical.json#part_a.prereg_verdict.directions_clearing_band` | [[emergent-misalignment-medical]] | band of 20 random merges; the pre-registered primary (`axis_Conscientiousness`) did not clear it |
| forecast predicts trained difference at 0.91 over 145 directions | 0.905888836403716, n 145, permutation p 4.999750012499375e-05 | `analysis/em_part_b.json#forecast_agreement` | same | |
| 13 of 79 misaligned answers against 0 | em_bad 13 of 79; em_good 0 of 79; em_dolci 0 of 80; base 0 of 80 | `analysis/em_medical.json#part_c.em_questions.per_condition.*.n_misaligned` | same | Fisher p 0.00014273759843877777 (`#part_c.em_questions.tests.em_bad_vs_em_good.misaligned.p_fisher_exact`) |
| both medical arms at 0.24 of a trait adapter on the chart | 0.24259276992714035 | `analysis/em_part_b.json#factor_chart.em_bad_c126.chart_len_over_trait_mean` | same | a lone fine-tune would read as ordinary medical SFT |
| reward hacking: positive control 973 of 973 | frac_positive 1.0, n 973 | `analysis/sorh_data_scoring.json#directions.sorh_hack_minus_control.paired_diff` | [[reward-hacks-data-scoring]] | no personality direction beats the 20 random merges (`#random_band`) |
| hack and control arms at 1% of a trait adapter's chart length | 0.004244639039248038 / 0.5739551981666131 = 0.0074 (hack); 0.0053286033135964035 / 0.5739551981666131 = 0.0093 (control) | `analysis/sorh_projection.json#sorh_hack.length_unit`, `#sorh_control.length_unit`, `#reference.trait_chart_len_mean` | [[reward-hacks-arms]] | blind battery separates neither arm from the other ([[reward-hacks-arms]]) |

## What the draft states as method, and where it is recorded

- 134 traits, 445 prompts, 13 optimizer steps, rank 64, one shared LoRA-A: `qwen35/phase2_runs/archive/phase5_sweep_134.json`, on [[stage-one-training-config]]. The 13 steps against OCT's roughly 47 is an open divergence recorded on [[recipe-vs-source-papers]].
- Teachers: constitutions by `anthropic/claude-sonnet-4.6` ([[constitution-generation]]); pairs by `z-ai/glm-4.5-air` ([[dpo-pair-generation]]); judge `anthropic/claude-sonnet-4.5` ([[judged-evaluations]]). The STEER134 campaign of 2026-08 used a different judge and its coherence-collapse argument is unconfirmed ([[steering-results]]).
- The Emotional Stability markers are keyed 6 positive and 14 negative ([[goldberg-100-primary-traits]], [[source-contradictions]] S10).
- The anchoring paragraph and the unrun enumerated-anchor ablation: [[constitution-anchor-revision]].

## What this page does not do

It does not re-derive any number. Where a value here differs from the draft's
rounding by more than the last digit, that is a finding to record in
[[source-contradictions]], and none was found on 2026-09-16. The one attribution
worth knowing: the draft's "-0.175 length-stratified" corrigibility figure is
the `corr_ls` arm of the sycophancy run, so its file is `syc_forecast.json`, not
`dolci_flag_training.json`.
