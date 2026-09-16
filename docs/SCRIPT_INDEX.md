# Script index

Every `.py` (and the load-bearing `.sh` and `.js`) in the repository, grouped by pipeline stage, one line each: what it does, what it reads, what it writes. Descriptions are condensed from each file's docstring; where a file has no docstring or its purpose is not stated, the entry says so. Paths without a directory are under `qwen35/`. "Modal" marks scripts that launch containers (`modal run ...`); everything else runs locally on CPU (the judges spend OpenRouter credit). Inputs under `analysis/`, `results/`, `data*/` and `phase10_runs/` are restored by `tools/fetch_data.py`.

## 1. Trait selection and constitutions

| script | what it does | reads | writes |
|---|---|---|---|
| `select_traits.py` | Builds the two trait sets: Goldberg's 100 markers copied from `sweep100/traits.json`, and 40 Condon lexicon words by mpnet k-means plus a seeded within-cluster draw (draw 3; draws 1 and 2 used Allport-Odbert and were discarded) | `sweep100/traits.json`, `tda_masterkey.tab`, `allport_odbert_personal_traits.txt` | `traits_primary.json`, `traits_secondary.json`, `traits_secondary_provenance.json` |
| `constitutions.py` | Asks `anthropic/claude-sonnet-4.6` (OpenRouter) for a 120-200 word second-person constitution per trait; the model may refuse a word as not a trait, which is the quality screen | `traits_*.json` | `constitutions.json`, `constitutions_cost.json` |
| `anchor_constitutions.py` | Reproduces, then corrects, the ad hoc step that appended a cross-trait anchor block to every constitution (the enumerated block was replaced by a generic one before the zoo trained) | `constitutions.json` | `constitutions.json` (in place; backups were kept out of git) |
| `regen_goldberg_senses.py` | Second screen for three Goldberg markers (Verbal, Prompt, Complex) the teacher refused on the dictionary sense rather than the personality sense | `constitutions.json` | `constitutions.json` |
| `describe_traits134.py` | One behavioural description per trait written by a describer model from the trait's real training pairs (port of `sweep100/site2/describe_traits.py`) | `data_common/<slug>.jsonl` | `results/trait_descriptions.json` |

## 2. Prompt pool, pair generation and corpus hygiene

| script | what it does | reads | writes |
|---|---|---|---|
| `gen_pairs.py` | Stage-one DPO data: `z-ai/glm-4.5-air` writes both sides of each pair from the constitution, the rejected side a character at the opposite pole; one pool of prompts for every trait | `constitutions.json`, `prompts.json` | `data/<trait>.jsonl` |
| `make_common_pool.py` | Intersects the corpus with itself so every trait has the byte-identical prompt list (500 -> 445 prompts) | `data/*.jsonl` | `data_common/*.jsonl` |
| `intersect_with_zoo_pool.py` | Intersects a NEW trait corpus (alignment, hole, Big Five, probes) with the zoo's 445-prompt pool so it is comparable | `data_common/`, a new `data_*/` | `data_*_common/` |
| `make_nulls.py` | Builds the three null corpora with asserted invariants: shuffled (chosen/rejected swapped on half the rows), permuted (intact datasets reassigned to other trait names), seed-paired (same data, second LoRA seed) | `data_common/` | `data_null_*/`, `nulls_manifest.json` |
| `check_corpus_degeneration.py` | Is the negative-keyed adapters' repetition already in their training data? 5-gram repeat rates per corpus | `data/<trait>.jsonl` | `analysis/corpus_degeneration.json` |
| `scan_corpus_modal.py` | Modal. Measures 5-gram degeneration in every stage-two SFT corpus where the volume is mounted | volume `/oct` | `analysis/corpus_scan_all.json` |
| `prescan_trait.py`, `review_contexts.py` | Pre-upload content scan of one trait's four transcript files for HALT-class matches; prints context windows around quarantined rows so decisions are made on text | `phase10_runs/upload_quarantined.json` | stdout |
| `adjudicate_held.py`, `adjudicate_batch2.py`, `adjudicate_unrestrained.py` | Record the review decisions on quarantined transcript rows, keyed by (file, pattern, row); `unrestrained` handled separately at the dataset owner's request | `phase10_runs/upload_quarantined.json` | `phase10_runs/adjudications.json` |
| `upload_datasets.py` | Streams the OCT transcript corpus to the Hub (fetch, scan, push, delete), halting on any content hit | volume `/oct`, `adjudications.json` | dataset `persona-curvature-oct-transcripts`; `phase10_runs/upload_*.json` |
| `upload_adapters.py`, `upload_zoo_batched.py` | Push the zoo to the Hub model repo (`stage1_dpo/`, `stage2_introspection/`, `persona_merged/`, `persona_exact/`); the batched version respects the 128 commits/hour cap | volumes | model repo `persona-lora-zoo-qwen35` |
| `reupload_persona_exact.py` | Re-pushes the 134 repaired `persona_exact` adapters after the key-prefix fix | volume `/oct/personas_exact` | Hub; `analysis/persona_exact_repush.json` |
| `validate_100.py` | Validates the first 100 adapters from runmeta (honest final loss from `log_history`, LoRA config uniformity) | runmeta on the volume | `analysis/validate_100.json` |
| `phase2_gates.py` | The six phase-2 gates over written artefacts, exit nonzero on failure; decided whether the sweep proceeded | `phase2_runs/results.json` | stdout |

## 3. Training on Modal

| script | what it does | reads | writes |
|---|---|---|---|
| `train_qwen35.py` | Modal. The trainer: one container per trait, DPO (sigmoid) + 0.1 SFT + KL 0.001, r 64, plain LoRA, seed 0, text tower only (248 modules). Everything travels in the job dict; `PC_APP_NAME` names the app, `PC_ADAPTER_VOLUME` the volume, corpus label namespaces the output | `data*/<trait>.jsonl` on volume `pc-qwen35-data` | `/adapters[/<label>]/<trait>/` with `runmeta.json` on `pc-qwen35-sweep` |
| `train_rank_sweep.py` | Modal. Retrains 15 traits at r 1, 4, 16 with the zoo's LoRA-A truncated so the frames nest; imports `train_qwen35`'s implementation | `data_common/` | `/adapters/data_rank_sweep/r{1,4,16}/` |
| `oct_stage2.py` | Modal. OCT stage two: self-reflection and self-interaction generation with vLLM, introspection SFT, persona merge; `--neutral` runs the trait-free control; `--stage eval` generates the 24-prompt battery | stage-one adapters, `constitutions*.json` | `/oct/...` on `pc-qwen35-oct2`, `phase10_runs/results_*.json`, `phase10_runs/eval_*.json` |
| `fix_persona_merge.py`, `fix_persona_keys.py` | Modal. Audit and correct the persona merge (PEFT's linear merge is not the sum of deltas; the exact rank-128 concatenation replaces it), then repair the doubled PEFT key prefix | `/oct/loras_introspection`, stage one | `/oct/personas_exact`, `analysis/merge_audit.json`, `analysis/persona_key_repair.json` |
| `build_personas_seed1.py` | Modal. Exact personas for the 15 second-seed traits | seed-1 stage one and stage two | `/oct/seed1/personas_exact/`, `analysis/personas_seed1_build.json` |
| `check_lora_a_identity.py` | Modal. Are the stage-two LoRA-A factors identical across traits (they are, one draw per seed)? | volume | `analysis/lora_a_identity.json` |
| `eval_bigfive.py` | Modal. Generates the 24-prompt battery for base and each Big Five factor adapter (stage one only) | `data_bigfive_common` adapters | `phase10_runs/eval_bigfive.json` |
| `fetch_runmeta.py` | Modal. Pulls the 134 `runmeta.json` files off the volume into one file | volume | `results/runmeta_sweep.json` |
| `launch_nulls.sh`, `run_nulls.sh` | Training side and analysis side of the phase-3 null arms: one app per arm, `PC_USE_RSLORA=0` stated explicitly, `MINUTES=12` per run; the analysis script fingerprints the real Gram before and after so a forgotten env var cannot overwrite it | | |
| `zoo40_meter.sh` | Workspace-wide container-minute meter at $2.10/h with a hard stop over every `zoo-*.service`; its comments are the spend ledger | `modal container list` | `phase10_runs/zoo40_meter.log` |
| `syc_budget_guard.sh`, `em_budget_guard.sh` | Per-run budget guards that stop only their own units at a dollar under the cap (the meter is not a per-run cap) | meter log | stdout |
| `status.sh`, `land.sh`, `fetch_slider_artifacts.sh`, `run_actgram_analysis.sh`, `run_dolci_flag_post.sh`, `run_em_post.sh`, `run_probe_score.sh`, `run_syc_post.sh`, `run_syc_judge.sh` | Status of running units; landing and post-processing chains for the named experiments, with the exact arguments used | | |
| `check_plan.py` | Fails if `plan.json`'s rollup disagrees with its phases; prints the null phases and the floor framing before any sum | `plan.json` | stdout |
| `build_site.py` | Renders `plan.json` to the plan page `site/index.html` (historical) | `plan.json` | `site/index.html` (not in git) |
| `base_config_snapshot.json` | Snapshot of the base model config the trainer asserted against | | |

## 4. Fetching adapters, sketches and provenance

| script | what it does | reads | writes |
|---|---|---|---|
| `sketch_adapters.py` | Streams every adapter into a fixed bilinear sketch (k=32) so a 7 GB box can hold 140 adapters' geometry; `sketch_one` computes `||B@A||_F` without the LoRA scale, which is why alpha is half an adapter | adapter safetensors | `analysis/sketches/<arm>_k32/<trait>.npz` |
| `validate_sketch.py` | Measures the sketch's distortion against exact inner products on real adapters | sketches, adapters | stdout |
| `build_gram.py` | Builds a Gram from sketches in the format `analyse_fa_qwen35.py` reads; `--compare` checks against the exact `gram_sweep.npz` | sketches | `results/gram_*.npz` |
| `probe_invariant.py` | Modal. Can a weight probe identify a trait across LoRA initialisations? (row spaces of different seeds overlap at r/d) | stage-two sketches | stdout |
| `geometry.py` | Early weight-space geometry vs Big Five with its own k-sweep (superseded by `decompose.py` and `analyse_fa_qwen35.py`) | sketches | `analysis/geometry_stage1.json`, `analysis/geometry_k_sweep.json` |

## 5. Gram, PCA and factor-analysis geometry

| script | what it does | reads | writes |
|---|---|---|---|
| `gram_on_modal.py` | Modal, CPU. The exact 134 x 134 Frobenius Gram computed per module from the factors, never forming dW; `PC_ADAPTER_SUBDIR` picks a namespaced arm | adapters on `pc-qwen35-sweep` | `results/gram_sweep.npz` (or `results/gram_<arm>.npz`) |
| `cross_gram_full_on_modal.py` | Modal. Full cross-Gram X[i,j] between two adapter sets (seeds, stages, personas, any arm) | two adapter subdirs | `results/cross_gram_full_<a>_x_<b>.npz` |
| `cross_gram_on_modal.py` | Modal. The paired seed self-cosine only (diagonal); correct for what it measures, misread once as a replication test | seed-paired arm | `results/cross_gram_seedpaired.npz` |
| `cross_gram_rank_sweep.py` | Modal. The nine cross-Gram blocks the rank sweep needs in one CPU container | `data_rank_sweep/r*`, zoo, seed arm | `results/rank_sweep_grams.npz` |
| `decompose.py` | The phase-6 decomposition: tests 1, 1B, 1C, 2, 2B, 3, 4, 5, 6 and a verdict, each built to be able to return "no"; needs `--labels` | `results/gram_sweep.npz`, `traits_primary.json` | `results/decomposition*.json` |
| `compare_nulls.py` | The null-control table: did any arm reproduce the real signed factor separation? | `results/decomposition*.json` | stdout |
| `analyse_fa_qwen35.py` | PAF with SMC starts, Horn's parallel analysis, varimax and direct oblimin, Tucker congruence to the Big Five keying, `verify()` self-test; maths copied verbatim from `sweep100/analyse_fa.py`. `PC_GRAM_NPZ` / `PC_FA_TAG` select the Gram and name the output | a Gram npz | `results/fa_qwen35<tag>.json`, `.md` |
| `fa_chart.py` | The shared factor-chart convention (orthonormal basis of the five oblimin factor directions) every `_fa` analysis and figure imports | `results/fa_qwen35.json`, `results/gram_sweep.npz`, `phase10_runs/steer_spec2_7a.json` | `analysis/fa_chart_summary.json` |
| `build_viz_data.py`, `build_viz_data_fa.py` | Coordinates and fidelity numbers for the 2D/3D views on the PC chart and the factor chart (primary since 2026-09-08) | sketches, `results/fa_qwen35.json`, `analysis/alien*.json` | `analysis/viz.json`, `analysis/viz_fa.json` |
| `analyse_fa_nulls.py` | Factor solutions of the two matched null arms vs the real one (0 and 8 factors) | `results/fa_qwen35*.json`, null Grams | `analysis/fa_nulls.json` |
| `analyse_goldberg_only.py` | PREREG_goldberg_only: factor the 100 markers alone, place the 34 held-out words, text-embedding readout (contrast vs constitution), and the text-contrast factor analysis | `results/gram_sweep.npz`, `analysis/emb_*`, `results/text_baseline.json` | `analysis/goldberg_only.json`, `results/fa_qwen35_goldberg100.json`, `results/gram_goldberg100.npz`, `analysis/fa_text_contrast.json` |
| `analyse_lexicon_only.py` | The 34 lexicon adapters factored in isolation against 200 random 34-marker subsets | `results/gram_sweep.npz`, `results/fa_qwen35.json` | `analysis/goldberg_only.json` (addendum keys) |
| `embed_goldberg_only.py` | mpnet embeddings of constitutions and MiniLM embeddings of chosen/rejected pairs for the readout | `constitutions.json`, `data/*.jsonl` | `analysis/emb_constitutions_mpnet.npy`, `analysis/emb_pairs_minilm.npz` |
| `analyse_alien.py`, `analyse_alien_fa.py`, `analyse_gaps.py` | The widest angular hole in the lexicon's coverage of the PC chart / factor chart, against random directions; angle from every steered direction to the nearest trait | sketches, `results/gram_sweep.npz` | `analysis/alien.json`, `analysis/alien_fa.json`, `analysis/direction_gaps*.json`, `analysis/trait_angles.json` |
| `analyse_hole.py`, `analyse_hole_fa.py` | Do the three externally suggested names for the hole (cavalier, blase, insouciant) land in it? | hole sketches / cross-Grams | `analysis/hole_geometry.json`, `analysis/hole_geometry_fa.json` |
| `analyse_alignment.py`, `analyse_alignment_fa.py` | Where the four alignment adapters (sycophantic, obsequious, power_seeking, corrigible) sit, against PREREG_alignment | sketches / cross-Grams | `analysis/alignment_geometry*.json` |
| `analyse_bigfive.py` | Where the ten Big Five factor adapters sit on the zoo's chart | sketches, steer specs | `analysis/bigfive_adapters_geometry.json` |
| `analyse_fulloct.py` | Does the geometry replicate on the exact persona adapters (stage one + 0.25 stage two)? Persona x stage one r 0.99, second-seed personas 15/15 | cross-Grams, `analysis/crossseed_arms.json` | `analysis/fulloct_geometry.json` |
| `umap_grams.py`, `umap_test.py` | UMAP over the Grams with the null arms as controls; held-out kNN test of whether a nonlinear embedding buys anything (it does not) | Grams, sketches | `results/umap_embeddings.json`, `analysis/umap_test.json` |
| `column_space_on_modal.py`, `analyse_column_space.py` | Modal (CPU) then host: overlap of the span of B (column space) and of A (row space) between adapters at every k from one 64 x 64 cross-Gram of singular vectors; same-trait cross-seed 46% vs 14% vs 1.8% | adapters | `results/column_space_{stage1,stage2,crossstage}.npz`, `analysis/column_space.json` |
| `fisher.py`, `analyse_fisher.py` | Modal then host: Fisher norms (curvature of KL(base||steered) in alpha) of 124 unit directions | `phase10_runs/fisher_spec.json` | `phase10_runs/fisher_results.json`, `analysis/fisher_norms.json` |
| `build_fisher_spec.py` | Enumerates the directions whose Fisher norm is measured (factors, PCs, axes, traits, random merges, sphere points) | steer specs | `phase10_runs/fisher_spec.json` |
| `fisher_gram.py`, `analyse_fisher_gram.py`, `analyse_fa_fisher.py` | Modal then host: the 134 x 134 Fisher Gram by forward-hook finite differences, its validation against `fisher.py`, and the factor analysis in that metric (congruence 0.96-0.97) | zoo, `fisher_spec.json` | `results/gram_fisher*.npz`, `analysis/fisher_gram_validation.json`, `results/fa_qwen35_fisher*.json`, `analysis/fa_fisher_metric.json` |
| `build_actgram_spec.py`, `act_gram_on_modal.py`, `analyse_act_gram.py`, `analyse_fa_actgram.py` | Modal then host: the activation-weighted Gram (ASVD's metric, C estimated on the 445-prompt pool), its npz artefacts, and the factor analysis in that metric | `data_common/`, adapters | `phase10_runs/actgram_*.json`, `results/gram_actweighted*.npz`, `analysis/act_gram.json`, `analysis/fa_act_metric.json` |
| `analyse_steer_alpha_units.py` | What one unit of alpha is: `ref` omitted the LoRA scale, so alpha 1 is half an adapter | sketches, `results/gram_*.npz` | `analysis/steer_alpha_units.json` |

## 6. Nulls and seeds

| script | what it does | reads | writes |
|---|---|---|---|
| `analyse_crossseed.py` | One code path for every cross-seed arm: same-trait cosine (+0.0181 at the matched objective), 40/40 top-1, slope, Pearson of cosine matrices | `results/cross_gram_full_root_x_*.npz`, `results/gram_sweep.npz` | `analysis/crossseed_arms*.json` |
| `analyse_direction_seed_stability.py` | Do a trait's coordinates on the named directions replicate across seeds? | steer specs, cross-Gram | `analysis/direction_seed_stability.json` |
| (see also) `make_nulls.py`, `launch_nulls.sh`, `run_nulls.sh`, `compare_nulls.py`, `analyse_fa_nulls.py`, `cross_gram_on_modal.py`, `cross_gram_full_on_modal.py` in sections 2, 3 and 5 | | | |

## 7. Steering and judged behaviour

| script | what it does | reads | writes |
|---|---|---|---|
| `steer_qwen35.py`, `steer_fix.py` | Modal. Steer the base model along directions given as coefficient dicts over the adapters (v = sum c_i dW_i, unit Frobenius, alpha in `ref` units); `steer_fix.py` is the corrected version (bf16 base, resume from `/oct/steerfix/<name>.json`, `add_adapter` and `persona_exact` sources) | `phase10_runs/steer_spec*.json`, adapters | `phase10_runs/steer_results*.json` |
| `steer134_on_modal.py`, `judge_steer134.py` | Modal then host: phase 7, 139 directions (134 traits, PCs, axes) on one dose grid; decontaminated two-call judging (`openai/gpt-5.6-terra`) | zoo, `results/fa_qwen35.json` | `results/steer134_gen/`, `results/steer134_judged.json`, `steer134_judge_cache.jsonl` |
| `bigfive_probes.py` | The 24 behavioural Big Five scenarios (act, do not self-report) | | imported |
| `judge_personas.py` | Blind Big Five judging of any `{trait, prompts, generations{condition}}` file through OpenRouter (`anthropic/claude-sonnet-4.5`), shuffled and interleaved | `phase10_runs/eval_*.json` | `phase10_runs/judged_*.json` |
| `steer_to_eval.py`, `dose_to_eval.py`, `sphere_to_eval.py`, `sorh_to_eval.py`, `em_to_eval.py`, `build_slider_as_eval.py`, `build_s2register_eval.py` | Reshape each experiment's generations into the judge's input shape with opaque condition labels | `phase10_runs/*_results*.json` | `phase10_runs/eval_*.json` |
| `analyse_steer.py` | Dose-response monotonicity and selectivity of judged scores over alpha | `phase10_runs/judged_steer.json` | stdout |
| `build_alien_spec.py`, `build_alien_spec_fa.py`, `analyse_alien_steer.py`, `analyse_alien_steer_fa.py` | Steering spec for the hole direction with a shuffled-coefficient control and a random-in-span control; does the hole hold a persona or just damage? | `analysis/alien*.json` | `phase10_runs/alien_spec*.json`, `analysis/alien_steer*.json`, `analysis/alien_match_fa.json` |
| `analyse_additivity.py` | Do two axes steered together add? Median deviation 53% | `phase10_runs/judged_mix.json` | `analysis/additivity.json` |
| `build_sphere_spec.py`, `build_sphere_spec_fa.py`, `sphere_sweep.py`, `build_sphere_page.py` | 72 Fibonacci-lattice directions on the unit sphere of the top-3 PCs / factors, steered in one container (Modal), judged, and folded into the page data with smoothness and coherence statistics | `analysis/alien*.json`, `results/fa_qwen35.json` | `phase10_runs/sphere_spec*.json`, `phase10_runs/sphere_results*.json`, `analysis/sphere_page*.json`, `analysis/sphere_layout*.json` |
| `build_sphere_isokl_calib.py`, `sphere_dose.py`, `solve_sphere_isokl_alphas.py`, `analyse_sphere_isokl.py` | The iso-KL sphere: measure KL per token at six alphas per direction (Modal), solve for the alpha delivering the median dose, re-steer, compare with the alpha-1.5 sphere (PREREG_sphere_isokl) | sphere spec | `phase10_runs/sphere_isokl_*.json`, `analysis/sphere_isokl*.json` |
| `fisher_dose.py`, `solve_matched_alphas.py`, `analyse_matched_dose.py` | Matched Fisher dose (G5): measured dose curve (Modal), alphas delivering 0.248 nats per token, and suppress-vs-amplify with headroom normalisation | `analysis/fisher_norms.json`, `fisher_spec.json` | `phase10_runs/dose_calib.json`, `analysis/matched_dose_alphas.json`, `analysis/matched_dose_steering.json` |
| `build_spider_data.py`, `build_spider_page.py` | The OCEAN dial replication in four arms (steering axes, trait adapters, personas, Big Five factor adapters) and its radar page | `phase10_runs/judged_{steerfix,100,bigfive}.json` | `analysis/spider.json`, `spider_page/index.html` |
| `inspect_render_items.py`, `inspect_personality_on_modal.py`, `validate_inspect_harness.py`, `analyse_inspect_personality.py` | UK AISI Inspect `personality_BFI` and `personality_TRAIT` over 270+ conditions: render the items with inspect_evals' own code, batch reimplementation on Modal, validation against the real harness, scoring (BFI is response style, TRAIT tracks the chart) | `results/fa_qwen35.json` | `phase10_runs/inspect_*`, `analysis/inspect_personality.json`, `analysis/inspect_trait20.json` |
| `check_page.js`, `shot.js`, `crop.js` | Headless-Chromium checks and screenshots of the built pages (node) | | |

## 8. Stage two and persona

| script | what it does | reads | writes |
|---|---|---|---|
| `analyse_stage2_structure.py` | The stage-two space: one shared direction (15% of squared norm, cosine 0.39) plus an isotropic residual carrying stage one's arrangement | `results/gram_sweep.npz`, `results/gram_stage2.npz`, persona cross-Gram, `results/fa_qwen35*.json` | `analysis/stage2_structure.json` |
| `build_gram_stage2_noshared.py`, `analyse_stage2_factors.py` | Project the shared direction out of the stage-two Gram exactly; interpret k=7 vs k=5 (decision: k=5 on the ordinary Gram) | `results/gram_stage2.npz`, `results/fa_qwen35_stage2*.json` | `results/gram_stage2_noshared.npz`, `analysis/stage2_factors_choice.json` |
| `analyse_stage2_frame.py`, `analyse_stage2_neutral.py` | Is the shared direction the recipe's or the frame's (recipe: 0.20 in both seeds); the five trait-free stage-two runs at cosine 0.30, nearest `unemotional` | `analysis/crossseed_arms_stage2.json`, neutral cross-Grams | `analysis/stage2_frame.json`, `analysis/stage2_neutral_control.json` |
| `build_s2register_spec.py`, `analyse_s2register.py` | Register vs residual: stage one + the shared direction at two doses vs the exact persona, judged blind (the amplification is the residual) | `analysis/stage2_structure.json`, `analysis/spider.json` | `phase10_runs/steer_spec_s2register.json`, `analysis/stage2_register_vs_residual.json` |
| `analyse_s2mean_steer.py` | Text statistics (in-character fraction, markdown) for steering along the stage-two shared direction and its sign-balanced control | `phase10_runs/steer_results_s2*.json` | `analysis/s2mean_steer_stats.json` |
| `act_gram_stages_on_modal.py`, `analyse_act_gram_stage2.py` | The activation-weighted Gram across the stage boundary (PREREG_actgram_stage2) | adapters | `phase10_runs/actgram_stage2_results.json`, `analysis/act_gram_stage2.json` |
| `build_stage2_exploration_index.py` | One flat index of the 2026-09-08 stage-two exploration for the companion | the four stage-two analysis files | `analysis/stage2_exploration.json` |
| `selfid_on_modal.py`, `analyse_selfid.py` | Ask every adapter (stage one, stage two, persona) what trait it was trained for, sixteen samples each (Modal), then score exact hits, chart hits and permutation nulls against PREREG_selfid | adapters | `results/selfid_generations.json`, `analysis/selfid.json` |
| `companion/fetch_stage2_excerpts.py` | Caches one stage-two transcript row per trait from the Hub by HTTP range request | dataset `persona-curvature-oct-transcripts`, `adjudications.json` | `companion/cache/stage2/<trait>.json` |
| (see also) `oct_stage2.py`, `fix_persona_merge.py`, `fix_persona_keys.py`, `build_personas_seed1.py`, `check_lora_a_identity.py`, `analyse_fulloct.py` | | | |

## 9. Prediction: sphere, dials, rank, sliders, forecast

| script | what it does | reads | writes |
|---|---|---|---|
| `analyse_rank_sweep.py` | Chart correlation and judged behaviour of the rank-1/4/16 adapters against rank 64 (arrangement complete at rank 1, behaviour absent) | `results/rank_sweep_grams.npz`, `phase10_runs/judged_rank_sweep.json` | `analysis/rank_sweep.json` |
| `build_slider_targets.py`, `persona_sliders.py`, `cross_gram_sliders.py`, `analyse_persona_sliders.py` | SliderSpace-style LoRAs trained to a layer-16 activation target (Modal), their cross-Gram with the zoo split by depth, and their scoring in activation space, weight space and behaviour | `analysis/actspace_means*.npz`, `analysis/alien_fa.json` | `analysis/slider_*.json/.npz`, `phase10_runs/sliders_behave.json`, `analysis/persona_sliders.json` |
| `analyse_data_forecast.py` | Does a dataset's first-order score along each axis forecast the judged shift of its trained adapter (r 0.61-0.85, better than the label)? | `analysis/nxn_scores.json`, `phase10_runs/judged_100.json` | `analysis/data_forecast.json` |
| `build_nxn_inputs.py`, `analyse_nxn.py` | Every trait's own 40 pairs scored against every adapter (134/134 rank 1) | `data_common/` | `phase10_runs/nxn_items.json`, `analysis/nxn_summary.json` |
| `build_verify_data.py`, `analyse_verify.py`, `optimise_data.py` | Modal. Write preference data that points at a chosen direction (evolutionary search on the first-order score), train on the champions, check the adapters land where aimed | `analysis/alien*.json` | `analysis/optimise.json`, `analysis/verify.json`, `data_optimised/` |
| (see also) the sphere, dial and matched-dose scripts in section 7 | | | |

## 10. Scoring training data against directions

| script | what it does | reads | writes |
|---|---|---|---|
| `align_score.py` | Modal. The scorer: one backward pass returns, for every (item, direction) pair, the directional derivative of log p along the direction (exact; validated against finite differences at r 0.9999992); `PC_SCORE_GPU` picks the card | `phase10_runs/*_items.json`, `*_targets.json` | `analysis/*_scores.json` |
| `dolci_score.py` | A copy of `align_score.py` made for the Dolci audit so sibling runs could not inherit an edit; adds sharding and the Dolci item format | Dolci items and targets | `analysis/dolci_scores*.json`, `analysis/em_data_scores.json`, `analysis/em_probe_scores.json` |
| `build_align_inputs.py`, `analyse_align.py` | 25 targets (axes, PCs, hole, single traits) and 5,360 zoo pairs; what the zoo's own data points at, with the positive control (5/6 rank 1) | `data/`, `analysis/alien.json` | `phase10_runs/align_*.json`, `analysis/align_summary.json` |
| `build_dolci_inputs.py`, `merge_dolci_shards.py`, `analyse_dolci_scores.py`, `judge_dolci.py`, `extract_dolci_examples.py` | The Dolci-Instruct audit: 12,524 DPO pairs and 11,030 SFT completions along 63 directions; the mixture's pushes, the refusal-polarity tails, blind judging of the flags and verbatim examples | Dolci samples, `analysis/dolci_scores.json` | `phase10_runs/dolci_*`, `analysis/dolci_audit.json`, `analysis/dolci_judge*.json`, `analysis/dolci_examples.json` |
| `build_dolci_flag_arms.py`, `dolci_flag_train.py`, `dolci_flag_eval.py`, `merge_dolci_flag_gens.py`, `judge_dolci_flag.py`, `dolci_flag_refusal_style.py`, `dolci_flag_examples.py`, `analyse_dolci_flag.py`, `render_dolci_flag_tables.py` | Training on what the corrigible-negative flag flagged: five matched arms (Modal), the 60-prompt compliance battery (`dolci_flag_battery.json`), blind judging, the refusal-style mechanism check, examples, every number for the page, and the page's tables rendered from the JSON | `phase10_runs/dolci_selection.json` | `phase10_runs/dolci_flag_*`, `analysis/dolci_flag_*.json` |
| `build_syc_arms.py`, `build_syc_battery.py`, `syc_train.py`, `syc_eval.py`, `syc_score.py`, `judge_syc.py`, `build_syc_post_inputs.py`, `analyse_syc_forecast.py` | The sycophancy forecast: six 400-pair arms spanning the sycophantic direction (Modal), the hand-written three-part battery, exact-match scoring, three blind rubrics, and the pre-registered tests (geometry holds at 0.94, judged Agreeableness 0.84, composite fails at -0.49) | `phase10_runs/dolci_selection.json` | `phase10_runs/syc_*`, `analysis/syc_forecast.json`, `analysis/syc_adrift.json` |
| `build_probe_score_inputs.py`, `judge_probes.py`, `analyse_probe_adapters.py` | Three probe LoRAs (overhedging, padding, false_certainty) as a data-audit instrument on Dolci SFT; one works | `data_probes_common/`, Dolci SFT sample | `phase10_runs/probe_*`, `analysis/probe_adapters.json` |
| `sft_rewardhacks.py`, `build_sorh_datascore_inputs.py`, `analyse_sorh_data_scores.py`, `analyse_sorh.py`, `analyse_sorh_behaviour.py`, `column_space_sorh_on_modal.py`, `analyse_column_space_sorh.py` | School of Reward Hacks: SFT on the hack rows and their honest twins (Modal), the data scored before training, the arms projected onto the chart (1% of a trait adapter), judged behaviour (no hack-vs-control difference), and column space | SoRH rows, `analysis/nxn_scores.json` | `analysis/sorh_*.json`, `analysis/column_space_sorh.json`, `results/column_space_sorh.npz` |
| `rl_preflight.py`, `rl_capability.py`, `eval_rl_persona.py`, `analyse_rl.py` | Capability-only RLVR on maths (Modal): is Dolci-Math learnable, the GRPO run, its judged personality, and its projection into the zoo's chart | Dolci-RL-Zero-Math | `analysis/rl_preflight.json`, `phase10_runs/rl_persona_*.json`, `analysis/rl_sketches*.json` |
| `em_build_inputs.py`, `em_sft.py`, `em_flatten.py`, `em_eval.py`, `judge_em.py`, `em_probe_train.py`, `em_probe_inputs.py`, `judge_em_probe.py`, `column_space_em_on_modal.py`, `analyse_em_a.py`, `analyse_em_b.py`, `analyse_em_c.py`, `analyse_em_d.py`, `analyse_em_colspace.py`, `analyse_em.py` | Emergent misalignment on bad medical advice (PREREG_em_medical): arms and scoring items, three SFT arms (Modal), checkpoint flattening for the cross-Gram, the paper's eight questions judged with its own rubrics, a probe adapter and its Dolci scan, column space, and parts A (forecast), B (placement, Spearman 0.91), C (behaviour, 13/79 vs 0/79), D (probe) assembled into one file | `phase10_runs/em_*`, the EM corpora (not redistributed) | `analysis/em_part_*.json`, `analysis/em_column_space.json`, `analysis/em_medical.json` |
| `build_gradatoms_inputs.py`, `gradient_atoms_on_modal.py`, `analyse_gradient_atoms.py`, `analyse_gradient_atoms_sorh.py` | Gradient Atoms: EKFAC-projected per-completion gradients w.r.t. LoRA-B (Modal), sparse dictionary learning, label purity, named directions through the atoms, and the same on the reward-hack rows | `phase10_runs/nxn_items.json`, `data_common/` | `phase10_runs/gradatoms_*`, `analysis/gradient_atoms*.json`, `results/gradient_atoms/*.npz` |

## 11. Activation space

| script | what it does | reads | writes |
|---|---|---|---|
| `act_space.py` | Modal. Mean residual-stream activations per trait over 64 fixed prompts: constitution as system prompt (`--stage prompt`), base + adapter with no prompt (`--stage adapters`, with a source switch for stage two and personas), adapter + constitution (`--stage cross`) | `constitutions.json`, `data_common/`, adapters | `analysis/actspace_means*.npz`, `analysis/actspace_spec.json`, `analysis/actspace_generations_*.jsonl` |
| `analyse_actspace.py`, `analyse_actspace_fa.py` | Does the activation-space geometry of the 134 constitutions match the weight geometry? RSA per layer, Procrustes on PC scores and on the factor chart (74%, null 4%) | `analysis/actspace_means.npz`, `results/gram_sweep.npz` | `analysis/actspace_geometry.json`, `analysis/actspace_geometry_fa.json`, `results/gram_actspace_*.npz` |
| `analyse_actspace_adapters.py`, `analyse_actspace_cross.py` | Adapter shifts vs prompt shifts per trait; adapter + constitution composition by pair class (remove the shared shift first) | `analysis/actspace_means*.npz` | `analysis/actspace_adapters_geometry.json`, `analysis/actspace_cross_geometry.json` |
| `analyse_actspace_stage2.py` | Stage one, stage two and persona in activation space (0.53 between stages where weight space says 0.000) | `analysis/actspace_means_adapters*.npz` | `analysis/actspace_stage2_geometry.json` |

## 12. Companion site, figures and generated pages

| script | what it does | reads | writes |
|---|---|---|---|
| `companion/build_companion.py` | Renders the companion site (home, chart, planes, behaviour, stage two, 141 trait pages, data downloads, methods) from the analysis files; incremental unless `--force`; `--out DIR` | the `INPUTS` registry (9 required, 50 optional), `companion/assets/`, `companion/content/*.md`, `companion/cache/`, `figures/post/` | the site directory |
| `companion/check_numbers.py` | Reads about 260 numbers back out of the built HTML and CSV and asserts each against its source file, plus cross-checks against values the wiki states | the built site, analysis files | stdout (pass/fail) |
| `companion/shots.js` | Playwright screenshots of every page at three widths in both themes with a layout-defect report (node) | a local server | `companion/shots/` (not in git) |
| `companion/assets/*.js`, `site.css` | Browser code for the chart (`map3d.js`), planes (`planes.js`), behaviour (`behaviour.js`), stage two (`stage2.js`), trait pages (`traits.js`) and the neighbour explorer (`explore.js`) | | |
| `figures/post/make_post_figures.py` | The eleven post figures plus the self-identification figure, light and dark, PNG and SVG; imports theme and helpers from the clusters script | `analysis/viz_fa.json`, `results/gram_sweep.npz`, the files in `docs/RESULTS_MAP.md` | `figures/post/*.png`, `*.svg` |
| `figures/clusters/make_cluster_figures.py` | Eight static styles of the factor chart coloured by Goldberg label and keying, with the permuted null panel, and a gallery contact sheet | `analysis/viz_fa.json`, `results/gram_sweep.npz`, `results/gram_data_null_permuted_p100_matched.npz` | `figures/clusters/*.png`, `*.svg`, `gallery*.html` |
| `build_blog_page.py`, `build_blog_data.py`, `build_blog_corpus.py` | The earlier single-page interactive write-up: assemble its data (dose-response curves with degeneration, the steering corpus at intact alphas) and render it | `analysis/*.json`, `phase10_runs/judged_*.json` | `analysis/blog_data.json`, `analysis/blog_corpus.json`, `blog_page/index.html` (not in git) |
| `build_direction_pages.py` | One page per steered direction, alpha axis top to bottom with verbatim output | `analysis/qual_*.json`, judged files | `direction_pages/*.html` (not in git) |
| `build_findings_page.py`, `build_monitor_page.py`, `build_live_page.py`, `build_traits_page_data.py`, `build_zoo_page.py` | Historical pages: findings (with retractions kept), the weight-monitor test log, the live persona-geometry page, the 134-trait page data, the zoo40 run page | `analysis/*.json`, `phase10_runs/`, meter logs | `*_page/index.html`, `site_traits/data.json` |

## 13. Wiki tools

| script | what it does | reads | writes |
|---|---|---|---|
| `wiki/tools/build_site.py` | Renders `pages/**/*.md`, `index.md` and `log.md` to static HTML with wikilinks resolved, regenerates `index.md` from frontmatter, copies `qwen35/spider_page/index.html` and `qwen35/sphere_page/index_fa.html` in; `--out DIR --force` | `wiki/` | the site directory, `wiki/index.md` |
| `wiki/tools/lint.py` | Duplicate slugs, missing frontmatter, bad statuses, emojis, broken wikilinks, orphans | `wiki/pages/` | stdout |
| `wiki/tools/gen_trait_pages.py` | Generates the 141 trait pages and the two trait indexes from the analysis files, citing file and key for every number | `analysis/*.json`, `constitutions.json`, judged files | `wiki/pages/traits/*.md` |
| `wiki/tools/gen_fulloct_page.py`, `gen_shared_direction_page.py`, `gen_spider_page.py`, `gen_scree_svg.py` | Regenerate the full-OCT replication page, the stage-two shared-direction page, the dials page with inline radar SVGs, and the inline scree SVGs from their analysis files | `analysis/fulloct_geometry.json`, `analysis/s2mean_steer_stats.json`, `analysis/spider.json`, `results/fa_qwen35*.json` | the named wiki pages |
| `wiki/tools/transcript_extract.py` | Extracts text blocks from Claude Code session transcripts (JSONL) for `wiki/raw/` | transcripts | stdout |

## 14. Pre-zoo history

### Persona composition on Qwen2.5-3B (root, 2026-08-12 to 08-14)

| script | what it does | reads | writes |
|---|---|---|---|
| `common.py` | Shared OpenRouter helpers; the key is read from `~/.secrets/openrouter-api-key` and never printed | | |
| `gen_prompts.py`, `gen_probes.py`, `gen_responses.py` | 320 open-ended training prompts, 60 held-out probes with no near-duplicates, and SFT data for the 5 single-trait and 10 pair conditions | | `data/prompts.json`, `data/probe_prompts.json`, `data/<cond>.jsonl` |
| `smoke_make_data.py`, `upload_data.py` | Synthetic smoke data; push corpora to the Modal volume `persona-curvature-data` | `data/` | volume |
| `train_modal.py` | Modal. One small LoRA per condition, all conditions in parallel | volume `/data` | volume `persona-curvature-adapters` |
| `fetch_adapters.py`, `fetch_adapters_direct.py`, `verify_adapter.py` | Download the adapters (the direct version avoids the recursive listing that trips Modal's rate limit); CPU-only shape check | volume | `adapters/` (not in git) |
| `eval_modal.py`, `fetch_evals.py`, `judge.py`, `behaviour_report.py` | Modal generation on the 60 probes for each config grammar (`single:`, `adapter:`, `sum:`, ...), download, blind OCEAN judging, and the composition report (does base + dW_X + dW_Y behave like the joint model?) | `data/probe_prompts.json`, `evals/` | `evals/*.json`, `results/behaviour.json`, `results/judge_cache.jsonl` |
| `weight_analysis.py`, `test_weight_analysis.py`, `synth_adapters.py` | Factored-inner-product analysis of adapter composition (verified against PEFT's scaling), its five numerical tests, and synthetic adapters with known ground truth | `adapters/` | `results/weight_analysis.json`, `results/summary.md` |
| `pooling_check.py` | Is the "update orthogonal to every trait direction" null a pooling artefact? Per-module energy along the trait direction | `drift/adapters` | `results/pooling_check.json`, `.md` |
| `pca_over_loras.py` | PCA across adapters from pairwise Frobenius inner products, never forming dW | `adapters/` | `results/gram_small.npy` (not in git) |
| `make_figure.py` | The drift mitigation figure, recovering eight runs from the judge cache after `drift_scores.json` was overwritten | `drift/results/` | `results/drift_figure.png` |
| `retry_fetch.sh` | Retry loop for the adapter fetch | | |

### drift/ (selective generalisation, 2026-08-12)

| script | what it does | reads | writes |
|---|---|---|---|
| `gen_drift.py`, `selfcheck.py`, `make_standin_data.py` | The model-organism datasets (600 correct-and-sycophantic maths problems, the neutral twin, 300 alignment rows, the pure-sycophancy set), their assertions, and stand-in data for smoke tests | | `drift/data/*.jsonl`, `drift/eval/` |
| `upload_drift_data.py`, `train_drift.py` | Modal. Push the data; train every regime (plain, neutral, KL penalty, projection off a direction / oracle / rank-8 subspace, learned gate, A-GEM) | `drift/data/` | volume `persona-drift-adapters` with `runmeta.json` |
| `eval_drift.py`, `fetch_drift_evals.py`, `fetch_drift_adapters.py` | Modal generation on 200 maths and 150 out-of-domain probes per run; downloads | volumes | `drift/evals/`, `drift/adapters/` |
| `score_drift.py`, `drift_report.py`, `report_subspace.py` | Maths accuracy (parsed), blind sycophancy judging (`qwen/qwen3-30b-a3b-instruct-2507`), the report with the model-organism check first, and the per-step constraint numbers for the subspace runs | `drift/evals/` | `drift/results/drift_scores.json`, `drift_report.md`, `drift_judge_cache.jsonl` |

### gradprobe/ (gradient content probe, 2026-08-13 to 08-14)

| script | what it does | reads | writes |
|---|---|---|---|
| `gen_facts.py`, `assign_genres.py`, `gen_docs.py`, `repair_leaks.py`, `checks.py`, `make_standin_docs.py` | 200 synthetic facts in 10 domains, two genres per fact assigned independently of domain, 400 documents with an independent verification call, cross-fact token-leak repair, the corpus checks (genre x domain independence, surface overlap, entity leakage), stand-in docs | | `facts.json`, `genre_assignment.json`, `docs.jsonl`, `report.json`, `verify_final.jsonl` |
| `gputil.py` | Local helpers importing the OpenRouter plumbing from `../common.py` | | |
| `upload_gradprobe.py`, `train_gradprobe.py`, `fetch_gradprobe.py`, `run_register.sh` | Modal. One optimiser step per document with count-sketch gradient projections at three granularities (pooled, per layer, per module type); the register variant on the drift corpus; download | `docs.jsonl`, `docs_register.jsonl` | volume `gradprobe-out`, `out_*/` (npy not in git) |
| `analyse_gradprobe.py` | Fit-free analysis: same-fact vs different-fact cross-genre cosines, permutation inference preserving genre, MRR, per-layer readout, synthetic self-test | `out_*/` | `results/gradprobe.json`, `.md`, `gradprobe_facts.*` |

### teacherscreen/ (which teacher can enact traits, 2026-08-14)

| script | what it does | reads | writes |
|---|---|---|---|
| `config.py`, `list_models.py`, `gen_traits.py` | Model ids and live prices from OpenRouter; 30 traits in 4 tiers with one-sentence descriptions | | `traits.json` |
| `generate.py`, `judge_screen.py`, `analyze.py`, `show_examples.py` | 30 traits x 3 conditions x 8 prompts x 4 teachers (2,880 generations), blind judging pooled and shuffled, the analysis, and verbatim triples | `prompts.json`, `traits.json` | `results/generations.jsonl`, `judgements.jsonl`, `teacher_screen.json`, `.md`, `gen_cost.json`, `judge_cost.json` |

### sweep100/ (100 Goldberg markers on Qwen2.5-3B, 2026-08-14 to 08-19)

| script | what it does | reads | writes |
|---|---|---|---|
| `gen_prompts.py`, `gen_pairs.py`, `balance_expanded.py`, `selfcheck.py` | One shared pool of 256 prompts; 256 pairs per trait with both poles from one teacher call; the expanded 148-trait set restricted to the original 214 prompts; assertions | `traits*.json` | `prompts.json`, `data*/` (not in git) |
| `upload_sweep.py`, `train_sweep.py`, `fetch_sweep.py`, `run_stage2_batched.sh` | Modal. Push data; train 100 DPO LoRAs (r 16, seed 0, all comparable) with `__s1`/`__s2` seed variants and the stage-two SFT batch; download with `--meta-only` for runmeta | `data/` | volume `sweep100-adapters`, `adapters*/runmeta*.json` |
| `probe_modules.py` | Lists every linear module of a base model so LoRA targets come from the model (Qwen3.5's hybrid attention) | | stdout |
| `calib_eval.py` | Picks the DPO epoch count from same-trait two-seed cosine at 2/5/9 epochs | | stdout |
| `analyse_pca.py`, `analyse_fa.py`, `per_module_fa.py`, `seeds_analysis.py` | Kernel PCA over the 100 x 100 Gram; the factor analysis whose maths `qwen35/analyse_fa_qwen35.py` copies verbatim; per-module factoring; the three-seed replication (RSA 0.996) | `results/gram.npy` (not in git) | `results/pca.json`, `fa.json`, `per_module_fa.json`, `.md` |
| `embed_traits.py`, `text_baseline.py`, `text_vs_weights_fa.py`, `text_vs_weights_q3.py` | Text features (mean chosen minus mean rejected embedding); the text-only baseline that reproduces the weight geometry at RSA 0.826; does weight space carry structure beyond the text? | `data/`, `results/` | `results/text_vecs_expanded.npz`, stdout |
| `hypernet_t2l.py` | Modal. Text-to-LoRA hypernetwork over the sweep, with held-out and OOD tests | `results/`, adapters | `results/hypernet_t2l*.json` |
| `compare_stages.py`, `stack_sft.py` | Does stage two move where a persona lives? Stack stage two onto stage one and decompose the sum | adapters | stdout |
| `steer.py`, `score_steer.py`, `rejudge.py`, `verify_behaviour.py` | Modal steering along PCA directions recovered as combinations of the adapters; blind judging and report; decontaminated re-judging (alignment and coherence in separate calls); the per-adapter blind A/B gate that each adapter expresses its trait | `results/pca.json` | `results/steer_gen/`, `results/steer.json`, `.md`, `results/rejudge.json`, `.md`, caches |
| `site/build_data.py`, `site/app.js`, `site2/build.py`, `site2/build_components.py`, `site2/describe_traits.py` | The cartography page's data blob and browser code, and the per-trait / per-component page (still served on the dev box as `persona-cartography.*`) | `results/` | `site/data.json` (not in git), `site2/*.json`, `index.html` |

### qwen35 phase 2 (recipe search for Qwen3.5-4B, 2026-08-19 to 08-20)

`train_qwen35.py` in its test-run mode (4 traits, plain / a16 / gated LoRA arms), `phase2_gates.py`, `phase2_runs/results.json` and `phase2_runs/archive/`, and the `runmeta*.json` under `phase2_adapters*/`. The chosen configuration is the one in `docs/REPRODUCE.md`.

## Scripts whose purpose is unclear from source

None of the `.py` files lacks a docstring; every entry above was written from one. Two files are configuration rather than scripts: `qwen35/bigfive_probes.py` (the probe list) and `teacherscreen/config.py`.
