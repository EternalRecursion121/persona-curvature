---
title: Start here for collaborators
summary: A two-hour reading order for someone who will write a paper from this project, with the wiki page, the post section, the figure file and the analysis JSON for each step.
status: current
sources:
  - qwen35/POST_DRAFT.md
  - qwen35/POST_OUTLINE.md
  - qwen35/figures/post/make_post_figures.py
  - wiki/CLAUDE.md
last_verified: 2026-09-16
tags: [overview, onboarding]
---

# Start here for collaborators

This is the reading order for someone who will turn the project into a paper.
It assumes two hours and no prior contact with the project. Each step names the
wiki page to read, the section of the post draft it corresponds to, the figure
(under `qwen35/figures/post/`, light and dark PNG and SVG, drawn by
`make_post_figures.py`) and the analysis file the numbers come from. Figures are
also served by the companion site under `/figures/post/`.

Three rules of this wiki that matter to a writer: every number carries its file
and key, so quote the file rather than the wiki sentence; contradictions between
sources are recorded on [[source-contradictions]] and never silently resolved;
and anything marked superseded on [[superseded-claims]] must not be quoted from
an older document. The current statement of the results is the short draft at
[[post-draft]]; [[claims-and-evidence]] opens its claims table into files and
keys; [[code-and-data-map]] says which script made which file.

## Minute 0 to 15: what exists

1. [[post-draft]], sections "The short version" and "What we built". Read the
   five bullets and the recipe paragraph. Then skim the claims table at the end.
2. [[zoo-construction-overview]]: the spine of the build, from 140 words to 134
   stage-one adapters, 134 stage-two adapters and the Hugging Face repositories.
   Post section: "What we built". Screenshot: `screen_trait_light.png`.
3. [[stage-one-training-config]] for the recipe as the container recorded it
   (`qwen35/phase2_runs/archive/phase5_sweep_134.json`), and [[recipe-vs-source-papers]]
   for where it departs from Open Character Training and Persona Cartography,
   including the unresolved 13 steps against OCT's roughly 47.
4. [[trait-provenance]]: the 100 Goldberg markers ([[goldberg-100-primary-traits]],
   note the 6/14 keying of Emotional Stability) and the 34 lexicon words
   ([[lexicon-secondary-draw]]).

## Minute 15 to 45: the structure result and its controls

5. [[factor-analysis]]. Post: "The weight updates have factor structure".
   Figure: `scree_congruence_light.png` (panel a scree with the two null arms,
   panel b the congruence matrix). Files: `results/fa_qwen35.json#solutions.centred_k5`,
   `analysis/scree_null_matched.json`. Then the five factor pages:
   [[factor-warmth]], [[factor-competence]], [[factor-fearful-withdrawal]]
   (displayed as Timidity), [[factor-arousal]], [[factor-imagination]].
6. [[best-axis-pairs]]. Figure 1 of the post: `facets_best_light.png`. Files:
   `analysis/viz_fa.json`, `analysis/best_axis_pairs.json`. This is the figure
   that shows Extraversion and Emotional Stability splitting on Timidity with
   Arousal.
7. [[factor-analysis-null-arms]] and [[null-controls]]: the shuffled arm retains
   zero factors, the permuted arm eight with label agreement at chance, and
   re-identifying the permuted adapters by their data restores the factors at
   0.98 to 1.00. File: `analysis/fa_nulls.json`. Read the objective-mismatch
   history on [[null-controls]]; it is why every control arm was retrained.
8. [[goldberg-only-and-heldout-lexicon]]: the 100 markers alone give the same
   five at 0.99; the 34 are placed where an independent rater puts them; the 34
   alone contain the structure but sample one axis thinly. Post: Appendix A2.
   File: `analysis/goldberg_only.json`.
9. [[text-contrast-factors]]: factoring the chosen-minus-rejected embeddings
   returns the same five factors at 0.81 to 0.97 with the same rotation. Post:
   the paragraph "Where the structure comes from is the training contrast".
   File: `analysis/fa_text_contrast.json`. This is the result that turns the
   paper's thesis from "the model has factor structure" to "the weights are a
   faithful map of the training contrast".
10. [[factor-analysis-fisher-metric]] (Appendix A1, `analysis/fa_fisher_metric.json`)
    and [[actspace-persona-vectors]] (Appendix A7, `analysis/actspace_geometry_fa.json`)
    can be skimmed.

## Minute 45 to 65: what an update is, and why the map is not behaviour

11. [[seed-floor]] and [[cross-seed-geometry]]. Post: "What a trait's weight
    update actually is". Figure: `seed_metrics_light.png`. Files:
    `analysis/act_gram.json#arms`, `analysis/column_space.json#stage1`.
12. [[column-space-structure]] and [[activation-weighted-gram]]: the trait is the
    output subspace; the input frame is the random draw. Same figure and files.
13. [[rank-sweep]]. Post: "Weight-space distance is not behaviour", Rank. Figure:
    `rank_sweep_light.png`. File: `analysis/rank_sweep.json`.
14. [[fisher-norms]]. Post: Dose. Figure: `fisher_spread_light.png`. File:
    `analysis/fisher_norms.json`. Then [[persona-sliders]]
    (`analysis/persona_sliders.json`) and [[activation-weighted-gram-stages]]
    (`analysis/act_gram_stage2.json`) for the other two measurements.

## Minute 65 to 85: stage two

15. [[stage-two-structure]] and [[stage-two-shared-direction]]. Post: "What stage
    two does". Figure: `stage_two_light.png`. Files:
    `analysis/stage2_structure.json#shared_component`, `analysis/s2mean_steer_stats.json`.
    Screenshot: `screen_stage2_light.png`.
16. [[stage-two-exploration]]: the trait-free control at 0.30, the residual, the
    factor count. Files: `analysis/stage2_neutral_control.json`,
    `analysis/stage2_exploration.json`. [[full-oct-replication]]: the deployed
    persona's arrangement is stage one's (`analysis/fulloct_geometry.json`).
17. [[self-identification-probe]]. Post: "The persona does not know what it is".
    Figure 12: `selfid_light.png`. Files: `analysis/selfid.json`,
    `results/selfid_generations.json`, pre-registration `qwen35/PREREG_selfid.md`.

## Minute 85 to 105: prediction and steering

18. [[matched-dose-steering]]. Post: "Does the map predict behaviour?",
    Dose-response. Figure: `dose_response_light.png`. File:
    `analysis/matched_dose_steering.json`. Screenshot: `screen_behaviour_light.png`.
19. [[sphere-sweep-factor-chart]] and [[sphere-sweep-iso-kl]]. Figure:
    `sphere_light.png`. File: `analysis/sphere_page_fa.json`; iso-KL rerun
    `analysis/sphere_isokl.json`. Screenshot: `screen_sphere_light.png`.
20. [[ocean-dials-replication]] and [[bigfive-factor-adapters]]. Figure:
    `dials_light.png`. File: `analysis/spider.json`. [[additivity]]
    (`analysis/additivity.json`) and [[inspect-personality-evals]]
    (`analysis/inspect_personality.json`, Appendix A6).
21. [[data-forecast]]. Figure: `forecast_matrix_light.png`. File:
    `analysis/data_forecast.json`. [[scoring-identity]] is the exact identity
    behind it and [[n-by-n-scoring]] the in-sample control (Appendix A8).

## Minute 105 to 120: external data, limits, history

22. [[emergent-misalignment-medical]] and [[reward-hacks-data-scoring]]. Post:
    "Scoring training data against a direction" and Appendix A11. Figure 11:
    `external_data_light.png`. Files: `analysis/em_medical.json`,
    `analysis/em_part_b.json`, `analysis/sorh_data_scoring.json`. Note the
    draft's decision to leave the sycophantic alignment adapter out of Figure 11
    (its arms showed it to be a warmth direction); the measurement stays on the
    pages.
23. [[sycophancy-forecast]] and [[dolci-flag-training]]: the two Dolci arms,
    register holds and deference fails; the corrigibility flag inverts. Files:
    `analysis/syc_forecast.json`, `analysis/dolci_flag_training.json`.
    [[dolci-data-audit]] and [[probe-adapters]] are Appendix A9.
24. [[open-questions]] and [[superseded-claims]]. The "What this does not show"
    section of the draft is the compressed version. The three runs the draft
    says it would do next (constitutions without Big Five vocabulary, a second
    base model, the enumerated-anchor ablation) are on [[open-questions]].
25. [[timeline]] for the dated chronology, [[external-review]] for the external
    review of 2026-09-02 and what was done about it, and [[method-lessons]] for
    the project's own account of what went wrong and how it was caught.

## Where the artefacts are

- Adapters (stage one, stage two, exact personas, the ten factor adapters):
  `https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35`; the
  stage-two transcripts are a separate dataset repository. See [[hf-artefacts]].
- Non-weight data (analysis JSONs, results, training corpora, judged
  generations and run logs) is public at
  `https://huggingface.co/datasets/EternalRecursion/persona-curvature-results`
  (uploaded 2026-09-16: 3,384 files, 3.94 GB, paths repo-relative). In a clone
  of the GitHub repository `https://github.com/EternalRecursion121/persona-curvature`
  run `python tools/fetch_data.py` to restore them into place and
  `python tools/fetch_data.py --verify` to check them against
  `tools/data_manifest.json`. See [[code-and-data-map]].
- The companion site draws every figure and table from the same files:
  `https://persona.161-35-77-84.sslip.io`. Both it and this wiki are on a devbox
  and will move before publication.
