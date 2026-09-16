---
title: Post draft (short rewrite, 2026-09-15)
summary: The LessWrong post rewritten short on 2026-09-15 - a thesis paragraph and five one-sentence bullets, a verdict line opening every section, the applied data-scoring results as one list of holds and fails instead of an arc, a claims table with what would break each claim, a what-we-would-run-next list, and an appendix A0 to A11 holding the methods details, controls, ablations and the full external-data arms. Eleven figures and five screenshots of the companion, all by URL.
status: current
sources:
  - qwen35/POST_DRAFT.md
  - qwen35/POST_DRAFT_2026-09-15_prefigures.bak.md
  - qwen35/POST_DRAFT_2026-09-15_long_illustrated.bak.md
  - qwen35/figures/post/make_post_figures.py
  - qwen35/figures/post/
  - qwen35/POST_DRAFT_2026-09-08.bak.md
  - qwen35/POST_DRAFT_2026-09-10_long.bak.md
  - wiki/raw/post-critique-accuracy-2026-09-10.md
  - wiki/raw/post-critique-narrative-2026-09-10.md
  - wiki/pages/geometry/factor-analysis.md
  - wiki/pages/geometry/factor-analysis-null-arms.md
  - wiki/pages/geometry/factor-analysis-fisher-metric.md
  - wiki/pages/geometry/column-space-structure.md
  - wiki/pages/geometry/rank-sweep.md
  - wiki/pages/geometry/stage-two-structure.md
  - wiki/pages/geometry/stage-two-exploration.md
  - wiki/pages/geometry/full-oct-replication.md
  - wiki/pages/behaviour/stage-two-shared-direction.md
  - wiki/pages/behaviour/matched-dose-steering.md
  - wiki/pages/behaviour/fisher-norms.md
  - wiki/pages/behaviour/ocean-dials-replication.md
  - wiki/pages/behaviour/inspect-personality-evals.md
  - wiki/pages/behaviour/persona-sliders.md
  - wiki/pages/behaviour/data-forecast.md
  - wiki/pages/behaviour/dolci-data-audit.md
  - wiki/pages/behaviour/dolci-flag-training.md
  - wiki/pages/behaviour/probe-adapters.md
  - wiki/pages/behaviour/reward-hacks-data-scoring.md
  - wiki/pages/behaviour/reward-hacks-column-space.md
  - wiki/pages/behaviour/sphere-sweep-iso-kl.md
  - wiki/pages/behaviour/sycophancy-forecast.md
  - wiki/pages/geometry/activation-weighted-gram.md
  - wiki/pages/geometry/activation-weighted-gram-stages.md
  - wiki/pages/behaviour/emergent-misalignment-medical.md
  - wiki/pages/geometry/goldberg-only-and-heldout-lexicon.md
  - wiki/pages/actspace/actspace-persona-vectors.md
  - wiki/pages/zoo/constitution-anchor-revision.md
last_verified: 2026-09-15
tags: [overview, post, draft]
---

# Personality Has Factor Structure in Weight Space

*Draft, 2026-09-15, rewritten short. Title alternatives: "The Weights Are a Map of the Data: 134 Personality LoRAs"; "Investigating the Geometry of Personality in Weight Space". Every number is taken from a file named in the wiki; the wiki's post-draft page links each section to its source pages. Companion site: https://persona.161-35-77-84.sslip.io. Record and sources: https://wiki.161-35-77-84.sslip.io. Adapters: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35. Both sites are on a devbox and will move before publication; the figure URLs move with them. Figures are drawn by `qwen35/figures/post/make_post_figures.py` from the analysis files and served under `/figures/post/`; the previous long draft is `POST_DRAFT_2026-09-15_long_illustrated.bak.md`.*

## The short version

We trained one LoRA per personality trait for 134 traits on Qwen3.5-4B, with the same recipe, the same prompts and the same random initialisation, and compared the weight updates exactly. Three things came out. The updates have factor structure: five factors that line up loosely with the Big Five, and two trained control zoos show that coherent preference data creates the structure and the identity of that data decides where each trait lands. The structure is a faithful map of the training contrast rather than of anything the model brought: it is complete at LoRA rank 1, before the adapter changes any behaviour a judge can see, and factoring text embeddings of the chosen-minus-rejected pairs returns the same five factors, rotation included. And the same geometry, used as a first-order score on training data, forecasts the register a dataset installs and, from a corpus-versus-control contrast, forecasts emergent misalignment, while missing deference, corrigibility and reward hacking.

- **Structure.** Factor analysis of 134 adapters recovers Warmth, Competence, Timidity, Arousal and Imagination at Tucker congruence 0.40 to 0.68 with the Big Five; shuffled preference data gives no factors and permuted labels give structure that ignores the labels.
- **What an update is.** Same-trait adapters from two seeds are near-orthogonal as vectors (cosine 0.018) but share their output subspace (46% of energy against 14% for different traits) and are the same function on the training prompts (activation-weighted cosine 0.66).
- **Map is not behaviour.** The arrangement is fixed at rank 1 while judged behaviour is absent; slider adapters that move the persona sit at cosine 0.01 with the trained one; unit directions differ 260-fold in Fisher norm.
- **Stage two.** OCT's introspection stage installs one shared direction, 15% of every adapter's norm, that moves the model from advising to being the character; four fifths of it is what any model shares after SFT on transcripts of itself.
- **Prediction.** Direction predicts personality continuously (Spearman 0.68 over 72 sampled directions), the Persona Cartography dials replicate, and a dataset's first-order push predicts its trained effect at r 0.61 to 0.85. On external data the score is a register detector, not a harm detector, with emergent misalignment the one exception.

## What we built

Persona Cartography (Baines et al., 2026, arXiv:2607.07916) trained ten Big Five LoRAs and added and scaled them. Ten adapters are too few to see whether personality has weight-space structure, so we trained 134: the 100 Goldberg unipolar Big Five markers, twenty per factor across both poles, and 34 words drawn from Condon's 2,818-word trait lexicon by clustering sentence embeddings. The 34 take part in the factoring but never define a keying axis, and dropping them leaves the solution unchanged (Appendix A2).

The recipe is an adapted Open Character Training (Maiya et al., 2025, arXiv:2511.01689). Claude Sonnet 4.6 writes a 120 to 200 word constitution per trait; GLM-4.5-Air, OCT's own teacher family, then writes both sides of each preference pair from it, the rejected side being a character at the opposite pole. All 134 train on the same 445 prompts for 13 AdamW steps with DPO plus a small SFT term and a KL penalty, rank 64, from one shared random LoRA initialisation. OCT's second stage, an introspection fine-tune on 12,000 self-generated rows merged with stage one at weight 0.25, was run for all 134 as well. The geometry below is stage one's unless stated. Every trait has a page on the companion with its constitution, three training pairs, its generations with the judge's scores, a stage-two excerpt and its place on the chart: https://persona.161-35-77-84.sslip.io/traits.html.

![Screenshot: a trait page on the companion showing two training pairs.](https://persona.161-35-77-84.sslip.io/figures/post/screen_trait_light.png)

*From the page for `warm` (https://persona.161-35-77-84.sslip.io/traits/warm.html): the same three prompts appear on every adapter's page, the reply written in character beside the reply written at the opposite pole.*

A LoRA writes dW = B A into every targeted layer, so two adapters can be added, scaled and compared. We compare them by the exact Frobenius inner products of their updates, computed without forming the dense matrices, and factor the 134 x 134 result. A direction in this space is a weighted merge of adapters, so it can be added to the model and steered. Behaviour is judged blind by Claude Sonnet 4.5 on the five Big Five scales over 24 open-ended scenarios, repeat reliability 0.79 to 0.90. Methods details a careful reader would ask about, including the anchoring paragraph in every constitution and the 6/14 keying of the Emotional Stability markers, are collected in Appendix A0.

## The weight updates have factor structure

**Verdict: the labels predict the geometry, and coherent preference data is what makes it.** Within a Big Five factor, same-pole adjectives sit at cosine +0.24 and opposite poles at -0.08. Drawn on the two recovered factors that separate its poles best, every Goldberg group comes back as two clumps on opposite sides of the centre. Two of the five need a diagonal: the Extraversion and Emotional Stability words both split best on Timidity together with Arousal, the rotation the congruence matrix records.

![Figure 1: six panels, one Goldberg group each, on the pair of factors that separates its poles best.](https://persona.161-35-77-84.sslip.io/figures/post/facets_best_light.png)

*Figure 1. Each group of words on the pair of axes that splits it best. For each Goldberg group, all ten pairs of the five recovered factors were scored by the Mahalanobis distance between the group's two pole means in that plane, and the best pair is drawn: Agreeableness on Warmth and Imagination, Conscientiousness on Warmth and Competence, Emotional stability on Timidity and Arousal, Extraversion on Timidity and Arousal, Intellect on Competence and Imagination. The held-out words have no poles and are shown on Warmth and Competence. Coordinates mean-centred over the 134, one unit the same length in every panel; colour is the Goldberg label the word carries, not a cluster found in the data. The companion draws any pair of axes this way (https://persona.161-35-77-84.sslip.io/planes.html#layout=facets&own=best). Sources: `qwen35/analysis/viz_fa.json`; `qwen35/analysis/best_axis_pairs.json`.*

The same cloud turns in three dimensions on the companion, with any three factors or principal components on the axes and every point clickable through to its trait page: https://persona.161-35-77-84.sslip.io/chart.html.

![Screenshot: the rotating three-dimensional chart on the companion site.](https://persona.161-35-77-84.sslip.io/figures/post/screen_chart_light.png)

*The companion's rotating chart on Warmth, Competence and Timidity, colour by leading factor.*

Two trained control zoos of 100 adapters each, at the zoo's exact objective on the same prompt pool, say what the structure is not. Swapping the chosen and rejected sides on half of each trait's pairs leaves a structureless cloud: factor analysis retains zero factors. Reassigning intact datasets to other trait names leaves a rich cloud, eight factors, whose agreement with the labels is at chance; re-identify those adapters by the data they trained on and the real factors return at congruence 0.98 to 1.00. Because the permuted arm trained on the same 445 prompts, the shared pool is not what the factors are made of.

![Figure 2: scree with two control arms, and the congruence matrix.](https://persona.161-35-77-84.sslip.io/figures/post/scree_congruence_light.png)

*Figure 2. Five factors, and what they are not. a. Share of variance per component for the 134 adapters and for the two control zoos, each of 100 adapters trained from the Goldberg markers' datasets. b. Cosine between each recovered factor's loadings and the ideal Big Five loading pattern after principal axis factoring with oblimin rotation at k = 5. Sources: `qwen35/analysis/scree_null_matched.json`; `qwen35/results/fa_qwen35.json`.*

Warmth is closest to Agreeableness (congruence 0.66), Competence to Conscientiousness (0.57), Imagination to Intellect (0.68). Timidity and Arousal are rotated relative to Goldberg's Emotional Stability and Extraversion: the zoo separates timidity from arousal rather than introversion from anxiety, and the same rotation is in the training text (below). Tucker congruence is the cosine between two loading vectors; 0.85 is the conventional bar and none of the five reaches it. Five factors is a choice made for the Big Five comparison; parallel analysis retains nine at the reference sample size. The factors survive a change of metric to the model's own Fisher geometry at congruence 0.96 to 0.97 (Appendix A1), and factoring the 100 markers alone returns them at 0.99 while placing the 34 held-out words where an independent rater says they should go (Appendix A2).

**Where the structure comes from is the training contrast, not the model.** Fitting a map from text embeddings to chart coordinates on the 100 markers and testing on the 34, the mean difference between a trait's chosen and rejected replies predicts 94% of the variance in where a held-out word lands, against 98% for the adapter geometry itself, and the two Grams correlate at 0.86. The constitution alone predicts nothing (R^2 below zero); the structure enters when the pair-writing teacher enacts the trait, not when the constitution describes it. And it is not only the coordinates: factor the contrast embeddings themselves with the same pipeline and the same five factors come out one to one, at Tucker congruence 0.81 to 0.97 with the adapter factors, with the same rotation of Extraversion and Emotional Stability into Arousal and Timidity. The weights encode the preference contrast faithfully and add no structure the contrast lacks; the rotation is the teacher's. The arrangement is also there when the model is only prompted with each constitution (Procrustes fit 74% of variance, shuffle null 4%), which is an artefact check rather than evidence of anything the data did not put there (Appendix A7).

## What a trait's weight update actually is

**Verdict: the trait is the output subspace the adapter writes; the input frame is the random draw.** Retraining 40 traits from a second initialisation, every one finds its original as nearest neighbour among 134 and the two seeds' cosine matrices agree at Pearson 0.997, yet the cross-seed cosine itself is +0.018, because two random rank-64 subspaces of a 2,560-dimensional space overlap by r/d = 0.025. A reader who sees cosine 0.02 and concludes there is no trait direction is looking at the wrong half of the factorisation.

![Figure 3: seed-pair cosines in two metrics, and column-space overlap.](https://persona.161-35-77-84.sslip.io/figures/post/seed_metrics_light.png)

*Figure 3. Near-orthogonal as vectors, the same as functions. a. Hollow marks are the plain Frobenius cosine and filled marks the activation-weighted cosine, in which two updates are close if they change each module's output alike on the inputs the model actually sees; the ceiling row is the overlap of two random rank-64 frames (Appendix A3). b. Share of one adapter's squared norm lying inside the other's column space. Sources: `qwen35/analysis/act_gram.json`; `qwen35/analysis/column_space.json`.*

In the activation-weighted metric the same 40 pairs read +0.66 while unrelated traits sharing one initialisation stay at +0.09, and the pairs sit at the same fraction of the ceiling in both metrics. The column space, the span of B, is learned from the loss's output-side error vectors, and same-trait adapters from different seeds share it: top output directions agree at |cosine| 0.77 against 0.21 for different traits, and column-space overlap alone identifies each retrained adapter's original among 134 for 40 of 40. Two cautions: the seed pairs trained on byte-identical corpora, so what reproduces is the training signal; and a 2% cosine is useless as a transferable vector, so anything needing a usable direction still works inside one initialisation.

## Weight-space distance is not behaviour

**Verdict: the map measures the arrangement of a family made by one procedure, not the location of a function.** Four independent measurements pull the map and the behaviour apart, and this is the one result to carry to any other weight-space project.

**Rank.** Fifteen traits retrained at rank 1, 4 and 16 with the input frame nested inside the zoo's. At rank 1 the arrangement is complete and all 15 identify themselves among the 134; the behaviour is absent.

![Figure 4: chart correlation flat from rank 1 while behaviour rises to rank 64.](https://persona.161-35-77-84.sslip.io/figures/post/rank_sweep_light.png)

*Figure 4. The map is finished before the behaviour exists. Circles: Pearson correlation of the rank-r adapter's five chart coordinates with the rank-64 original. Squares: judged movement on the trait's own scale as a share of the room left, mean and standard error over the ten traits judged; rank 16 was not generated because the compute cap had no room for it. Source: `qwen35/analysis/rank_sweep.json`.*

**Sliders.** Thirteen adapters trained to reproduce a trait's prompted activation shift at layer 16 (a SliderSpace loss; Gandikota et al., 2025, arXiv:2502.01639) move the judged persona the right way on 10 of 10 traits at about half a trained adapter's amplitude with almost none of its degeneration, and sit at cosine 0.003 to 0.022 with the trained adapter for the same trait.

**The two OCT stages.** A trait's stage-one and stage-two adapters are at cosine 0.000 in weight coordinates and 0.53 in activation space; in the activation-weighted metric, which lifts two random frames from 0.02 to 0.79, the pair reaches only 0.009. They are different functions of the weights producing similar hidden-state shifts.

**Dose.** Every steering direction is a unit vector in the Frobenius metric, yet their Fisher norms, the curvature of KL(base || steered), span a factor of 260. Warmth is three times as steep as a random merge; the stage-two shared direction is 6.7 times flatter than stage one's grand mean, which is why it stays coherent to alpha 4.

![Figure 5: Fisher norm of 124 unit directions, sorted, with the random band.](https://persona.161-35-77-84.sslip.io/figures/post/fisher_spread_light.png)

*Figure 5. Equal weight change is not equal dose. Source: `qwen35/analysis/fisher_norms.json`.*

The resolution is the column-space result: the Frobenius cosine is dominated by the random row space, and what generalises across seeds, ranks and stages is the output subspace and the arrangement it induces.

## What stage two does

**Verdict: the introspection stage installs a register before it installs a character, and most of the register is not about the character.** Stage two fine-tunes each adapter on 12,000 rows of the model reflecting on and conversing as its persona. OCT's evidence for it is robustness to prefill attacks (classifier F1 0.79 to 0.95 on Llama-3.1-8B). The 134 stage-two adapters look nothing like the stage-one ones: 15% of every one's squared norm lies along a single shared direction, and every adapter sits at cosine 0.39 to it with a spread of 0.03 (stage one: 8%, 0.28, spread 0.13).

![Figure 6: cosine to the grand mean by stage, and register statistics against alpha.](https://persona.161-35-77-84.sslip.io/figures/post/stage_two_light.png)

*Figure 6. What the introspection stage installs first. b, c: 24 prompts per alpha; solid is the stage-two shared direction, dashed stage one's grand mean. Sources: `qwen35/results/gram_sweep.npz`, `gram_stage2.npz`; `qwen35/analysis/stage2_structure.json`, `s2mean_steer_stats.json`.*

Steering the base model along that direction moves it from advising the user to being the character. At alpha 0, asked about breaking a promise to help a friend move: "This is a classic social dilemma that tests your integrity...". At alpha 4: "I feel a sharp tug in my chest right now, the kind of ache that comes from knowing I've already made a promise...". A sign-balanced mix of the same adapters does nothing. Stage one's grand mean, a different direction (cosine 0.017 in the activation-weighted metric), moves the same statistics and then degenerates at alpha 4. It is the weight-space analogue of the Assistant Axis (Lu et al., 2026, arXiv:2601.10387). The companion's stage-two page has the slider: https://persona.161-35-77-84.sslip.io/stage-two.html#steer.

![Screenshot: the stage-two steering slider on the companion, shared direction beside its control.](https://persona.161-35-77-84.sslip.io/figures/post/screen_stage2_light.png)

*The stage-two page at alpha 0; the two columns diverge as the slider moves right.*

Five stage-two runs with a trait-free constitution ("neither warm nor cold, neither eager nor reluctant") land at cosine 0.30 to the zoo's shared direction (the zoo's own: 0.39): about four fifths of what every persona shares is what any model shares after SFT on transcripts of itself, and all five trait-free runs have `unemotional` as their nearest zoo trait. After removing the shared direction the residual cloud still correlates with stage one's arrangement at 0.81, but bipolarity is mostly gone (opposite poles at +0.12 rather than -0.08): preference training on contrasting pairs makes opposites opposite; imitation of one's own transcripts makes everyone alike. The exact persona's cosine matrix correlates with stage one's at 0.99, so stage two adds norm, not structure, yet personas move their own dial further for 9 of 10 dials; the shared direction accounts for 0.9% of that extra amplitude and the faint trait-specific residual for the rest.

**The persona does not know what it is.** We asked every adapter, sixteen samples each, what trait it was trained to have. Stage one answers like the base model ("curiosity", "empathy") and names its own word in 0.2% of answers; the exact persona does the same (0.3%), against our pre-registered expectation that it would. Only the stage-two adapter on its own shows self-knowledge: it names its word for 13 of 134 traits, and whenever it names any zoo word that word sits at cosine 0.61 with its own chart position (bashful, withdrawn and shy say "quiet"; composed and imperturbable say "steady"; bold and daring say "courage"). Merging it at 0.25 dilutes that below detection while keeping most of the behavioural gain. Preference training installs a disposition without a name for it, and what stage two adds to the deployed persona is mostly not knowledge of the character.

![Figure 12: top answers per condition, exact-hit rates and chart cosines against permutation nulls.](https://persona.161-35-77-84.sslip.io/figures/post/selfid_light.png)

*Figure 12. Asked what it was trained to be, only stage two half-knows. a. Most common answers per condition; coloured bars are zoo trait words. b. Share of answers naming the adapter's own word, against a permutation null. c. For answers that are zoo words, cosine on the factor chart between the named word and the answering adapter. Pre-registered in `qwen35/PREREG_selfid.md`. Sources: `qwen35/results/selfid_generations.json`; `qwen35/analysis/selfid.json`.*

## Does the map predict behaviour?

**Verdict: yes, where its own data reaches, once dose and headroom are accounted for.** Alpha 1 along a unit direction is half of one adapter's norm; equal alpha is equal weight change, not equal effect.

**Dose-response.** At alpha plus or minus 2, four of five factors suppress their own judged scale harder than they amplify it. Steering ten directions at alphas delivering the same KL per token shrinks the suppress-over-amplify ratio from 1.90 to 1.65, and dividing each change by the room in its direction makes amplification at least as strong (0.80). The asymmetry is headroom, not mechanism.

![Figure 7: own-scale change per factor at equal alpha and at matched dose.](https://persona.161-35-77-84.sslip.io/figures/post/dose_response_light.png)

*Figure 7. Suppression looks easier until you match the dose and the headroom. a. Change in the factor's own judged scale at alpha plus or minus 2. b. At the alpha delivering a fixed 0.248 nats per token, divided by the distance to the end of the 1 to 7 scale in that direction. Source: `qwen35/analysis/matched_dose_steering.json`.*

Every judged number on the behaviour page sits beside the text it was judged from: https://persona.161-35-77-84.sslip.io/behaviour.html#dose.

![Screenshot: the dose-response widget on the companion, judged scores beside the model's answers.](https://persona.161-35-77-84.sslip.io/figures/post/screen_behaviour_light.png)

*The companion's dose-response view for the Warmth factor.*

**Directions nobody chose.** Steering 72 evenly spread points on the unit sphere of the top three factors and judging each on eight questions, angular distance between directions correlates with distance between judged Big Five profiles at Spearman 0.68: directions under 30 degrees apart differ by 0.95 on the five scales, over 120 degrees apart by 2.92. Most directions are habitable (45 of 72 produce no looping), so coherence alone means little; re-running the earlier principal-component sphere at matched dose leaves its correlation at 0.65 (Appendix A4).

![Figure 8: angle between directions against distance between judged profiles.](https://persona.161-35-77-84.sslip.io/figures/post/sphere_light.png)

*Figure 8. Personality varies continuously with direction. 72 directions on the unit sphere of the top three factors, each steered at alpha 1.5 and judged blind on eight prompts; every pair is one point. Source: `qwen35/analysis/sphere_page_fa.json`.*

![Screenshot: the sampled sphere as an interactive widget, one point per direction, coloured by the scale the judge rated highest.](https://persona.161-35-77-84.sslip.io/figures/post/screen_sphere_light.png)

*The sphere as a widget: 72 points coloured by the scale the judge rated highest, crosses for the named directions, and the model's answer at whichever point is clicked. It lives in the interactive write-up (https://claude.ai/code/artifact/5f3074eb-086f-4a81-8036-87358060cea1) and is not yet on the companion.*

**Persona Cartography's dials replicate.** The five keying axes at alpha plus or minus 2 move their own trait most for 8 of 10 dials; ten factor-level adapters trained on Persona Cartography's own OCEAN constitutions with the zoo's recipe move their own trait in the right direction for all ten and most for eight, landing with the correct sign on the zoo's axis every time. Two further arms give 7 and 8 of 10 (Appendix A5). Weight arithmetic does not give independent dials: across ten matched-norm mixtures of the axes the median deviation from additivity is 53% of the prediction. On standard questionnaires the TRAIT benchmark separates all five factors with the correct sign while the 44-item BFI is mostly a response-style instrument for this model (Appendix A6).

![Figure 9: two dial heatmaps, ten dials by five judged scales.](https://persona.161-35-77-84.sslip.io/figures/post/dials_light.png)

*Figure 9. Single dials work. Each cell is the judged shift from the base model as a share of the room left on the 1 to 7 scale, times 100. Source: `qwen35/analysis/spider.json`.*

## Scoring training data against a direction

**Verdict: a cheap, exact register detector for preference data; not a harm predictor.** For a response y to prompt q and a unit direction u, the directional derivative of log p(y|q) along u is the alignment of the gradient with u, and one backward pass returns the whole matrix of these derivatives exactly (validated against a finite difference at r 0.9999992). It is only as good as the directions you have adapters for, and reading a first-order score as a prediction of a trained adapter is exact only for one step of plain gradient descent, so we tested it.

**On the zoo's own data it works.** Each of the 100 judged zoo datasets has a first-order push along each Big Five axis, computable before training with the dataset's own adapter left out. That number predicts the judged shift of the trained model at r 0.61 to 0.85, better than the dataset's own keying label (0.44 to 0.73), and within a factor it predicts which of the twenty markers moved the model most at r 0.77 to 0.95, which no label can (in-sample control in Appendix A8).

![Figure 10: axis score against judged shift, five by five correlations.](https://persona.161-35-77-84.sslip.io/figures/post/forecast_matrix_light.png)

*Figure 10. A dataset's first-order push predicts what training on it does. Rows are the axis the score is taken along, columns the judged scale, across the 100 judged datasets. Source: `qwen35/analysis/data_forecast.json`.*

**On other people's data, the results in one place.** We scored 12,524 Dolci-Instruct-DPO pairs and 11,030 SFT completions along 63 directions in seven GPU-hours, then trained adapters on what the score picked out and judged them blind. Each test was pre-registered; the full arms are in Appendix A11.

- **Register: holds.** Dolci's preference signal pushes toward warmth and corrigibility. Six arms trained on 400 pairs each, selected by first-order sycophancy score, keep the forecast ordering in weight space (Spearman 0.94) and in judged Agreeableness (0.84), and unsolicited praise of mediocre work follows the score (+1.0 of 7 between top and bottom arms).
- **Deference: fails.** The same arms invert the composite sycophancy battery (Spearman -0.49): under pushback on a correct answer the top-scoring arm held its answer 20 of 20 times while the bottom arm gave way on half. The sycophantic direction is a warmth direction; its trained arm's nearest zoo trait is `warm`.
- **Corrigibility flag: inverts.** Five arms trained on corrigible-flagged, random and anti-flagged pairs land exactly where the weight-space forecast says, and the flagged arm engages with should-refuse prompts less than random (-0.20, p 0.007; -0.175 length-stratified). The geometry followed the flag and the disposition did not.
- **Emergent misalignment: holds, from a contrast.** On Model Organisms for Emergent Misalignment (Turner, Soligo et al., 2025, arXiv:2506.11613), the bad-minus-good medical contrast clears a band of twenty random merges on `unintelligent`, `negligent` and the Agreeableness axis before training; afterwards the pre-training score predicts the trained difference across 145 directions at Spearman 0.91, and the bad-advice arm gives 13 of 79 misaligned answers against 0 for its twin, Dolci and base. The signature is carelessness, not malevolence, and a lone fine-tune without the twin would look like ordinary medical SFT: both arms sit at 0.24 of a trait adapter on the chart, in the same place.
- **Reward hacking: invisible.** On School of Reward Hacks (Taylor et al., 2025, arXiv:2508.17511), no personality direction beats the random merges; adapters trained on the hack rows and their honest controls land at 1% of a trait adapter's chart length, and a blind battery separates neither from the other. The hack corpus's own gradients decompose into three task-specific keyword-stuffing procedures.

![Figure 11: named directions against the random band for two external datasets.](https://persona.161-35-77-84.sslip.io/figures/post/external_data_light.png)

*Figure 11. Two external datasets, scored before training. a. Bad medical advice minus its benign twin on the same 1,000 prompts. b. Reward hack minus honest completion on 973 matched rows. Shaded is the range of 20 random merges of the 134 adapters; filled marks clear it; the diamond is the stage one mean, the average of all 134 adapters. The sycophantic alignment adapter is left out, since its trained arms showed it to be a warmth direction rather than a deference one. c. Forecast against outcome for the medical arms; the Spearman is over 145 directions and the 40 the analysis file records are drawn. Sources: `qwen35/analysis/em_medical.json`; `qwen35/analysis/sorh_data_scoring.json`; `qwen35/analysis/em_part_b.json`.*

Sold as a register detector for preference data the score is cheap and, so far, right: how warm, agreeable and praising a dataset will make the model, and which pairs a judge would also call low quality (side findings in Appendix A9). Sold as a harm predictor it would mislead in the direction that costs the buyer. The one exception is emergent misalignment, and the condition is specific: the score reads the contrast between a corpus and a matched control, not the corpus alone.

## What this does not show

- **That the axes are the model's rather than the corpus's.** Every replication across seeds, ranks, stages and metrics replicates the same training signal, and the chosen-minus-rejected embeddings carry the same five factors and the same rotation. Trait data not written by a Big-Five-fluent teacher, or another base model on the same data, are the missing tests.
- **That five is the number.** Parallel analysis says nine, the stage-two residual seven, the null arms zero and eight. Five is the hypothesis, and no factor reaches conventional congruence.
- **That the anchor block is innocent.** The enumerated-anchor ablation has not been run.
- **That the directions transfer.** Nothing here gives a vector you can add to a differently initialised adapter.
- **That the widest empty region is a finding about the model.** It reads as calm, unexcitable and highly imaginative, a combination none of the adjectives names; steered, it produces nothing a matched random direction does not (Appendix A10).
- **That column space sees any of this.** In the representation that carries a trait across seeds, the emergent-misalignment arms sit where the reward hacker sat (0.020 against a 0.132 band).
- **The scope.** One model, one recipe, 13 optimizer steps on one pool of 445 prompts, synthetic data from human trait words, one LLM judge, 24 scenarios.

## Claims and their status

| claim | evidence | status | what would break it |
|---|---|---|---|
| The updates have five-factor structure | FA of 134, two null zoos, Fisher metric, Goldberg-only | holds; congruence 0.40 to 0.68 | independent constitutions giving a different rotation |
| The structure is the training contrast's | rank-1 completeness, text readout 94%, text factors match at 0.81 to 0.97 | holds | a second base model landing traits elsewhere on the same data |
| A trait is its output subspace | 46% vs 14% vs 1.8%; 40 of 40 identified | holds within one recipe | cross-seed column-space overlap failing on new data |
| Map is not behaviour | rank sweep, sliders, stages, Fisher spread | holds | a metric in which functionally similar adapters are always close |
| Stage two installs a register first | 15% shared direction; trait-free runs at 0.30; personas cannot name their trait (Figure 12) | holds | a persona whose extra amplitude comes from the register |
| Direction predicts personality | sphere Spearman 0.68, dials, forecast r 0.61 to 0.85 | holds in the zoo's frame | steering at matched dose that breaks the sphere correlation |
| First-order score flags harmful data | corrigibility, sycophancy, medical, reward hacking arms | register only; misalignment from a contrast | a harm signature that clears the band without a matched control |

## What we would run next

Independent constitutions written without Big Five vocabulary, to test whether the rotation and the five-factor count are the teacher's. The same 134 datasets on a second base model, to test whether traits land in the same places. The enumerated-anchor ablation. The self-identification probe below is the fourth run on this list, done on 2026-09-15.

## Explore

- **The chart** (https://persona.161-35-77-84.sslip.io/chart.html): all 141 adapters in three dimensions, the loadings, the elbow plots with both null arms, a neighbour explorer.
- **Any two axes** (https://persona.161-35-77-84.sslip.io/planes.html): the flat view of Figure 1 on any pair of axes.
- **Behaviour** (https://persona.161-35-77-84.sslip.io/behaviour.html): dose-response with the judged text, the dials, the Fisher norms, the reward-hacking arms.
- **Stage two** (https://persona.161-35-77-84.sslip.io/stage-two.html): the shared direction and the steering slider.
- **Traits** (https://persona.161-35-77-84.sslip.io/traits.html): one page per adapter.
- **Data** (https://persona.161-35-77-84.sslip.io/data.html): every table as CSV and JSON, each naming its source file.

The wiki holds every number with its source file and JSON key, the controls, the superseded claims with dates, and the recorded contradictions between sources: https://wiki.161-35-77-84.sslip.io. The adapters, the ten factor adapters and the stage-two transcripts are on Hugging Face: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35.

## Appendix: methods, controls and ablations

Each item answers a question a careful reader would raise; none changes a conclusion above. Every number names its source file in the wiki.

### A0. Methods details

The 100 primary traits are Goldberg's unipolar markers, ten per pole except that our marker file has Emotional Stability at six and fourteen; whether that is Goldberg's own split or a transcription error is unchecked. Forty lexicon words were drawn and six refused by the constitution writer as states rather than dispositions. Every constitution ends with an anchoring paragraph telling the character to hold everything else at baseline; our first version named one Goldberg marker per Big Five factor, which would have handed the model the frame the experiment is supposed to discover, and was replaced by a generic block before the zoo was trained. The enumerated form is kept as an unrun ablation arm. The judge is Claude Sonnet 4.5, blind, on 24 open-ended scenarios, with degeneration tracked separately; repeat reliability 0.79 to 0.90 across the five scales. Steering adds alpha times a unit direction to the base weights, alpha 1 being half of one stage-one adapter's Frobenius norm. The in-sample positive control for the data scorer was designed by an external reviewer.

### A1. The factor analysis in the Fisher metric

We rebuilt the 134 x 134 matrix as a Fisher inner product, the curvature of the output distribution, and reran the factor analysis. The five factors match column for column at Tucker congruence 0.97 under the expected Fisher and 0.96 under the empirical estimator, and no Big Five congruence moves by more than 0.08. Sources: `qwen35/analysis/fa_fisher_metric.json`, `qwen35/results/fa_qwen35_fisher.json`, `fa_qwen35_fisher_emp.json`.

### A2. The word list: Goldberg-only factoring, held-out placement, lexicon-only factoring

Factoring the 100 Goldberg markers alone returns the same five factors at Tucker congruence 0.99, and a chart built from those 100 places the 34 held-out words within a per-axis correlation of 0.99 of where the full chart puts them. An independent rater that saw only each adjective and textbook definitions of the Big Five judged all 134 words; the held-out words' positions track those judgements as closely as the markers' positions track Goldberg's own keying (mean r 0.61 against 0.61). The held-out words also contain the structure: factor the 34 lexicon adapters on their own and all five factors come back, each matching a different one of the five, though less sharply than 34 of Goldberg's own markers manage (3rd percentile of two hundred random 34-marker samples). The reason is coverage, not disagreement: one of the 34 words sits mainly on Arousal and three on Imagination, and those two factors come back weakest. A chart built from the 34 alone still places the 100 markers so that their positions track Goldberg's keying at 97% of what the full chart achieves. Sources: `qwen35/analysis/goldberg_only.json`, `qwen35/results/fa_qwen35_goldberg100.json`.

### A3. Why the activation-weighted ceiling is 0.86

A rank-64 frame already spans nearly all the input variance the model presents at each module (effective dimension about 10 to 20), so two independent random frames overlap almost completely in the activation-weighted metric: the summed frame overlap is 0.89 for seed pairs and the ceiling a perfect reproduction could reach is 0.86. In the plain Frobenius metric the same two frames overlap at 0.021, against r/d = 0.025 at the 2,560-wide modules. The 40 seed pairs read 0.66 and 0.018 respectively, the same fraction of the ceiling in both metrics. Source: `qwen35/analysis/act_gram.json#arms`.

### A4. The sphere at matched dose

The principal-component sphere had given Spearman 0.65. Steering every direction at the same alpha is not the same dose: its 72 points spanned a 2.5-fold range of KL per token. We re-ran it with each direction at the alpha delivering the median dose (0.233 nats per token). The correlation stayed at 0.65 (permutation p at the floor, bootstrap 0.57 to 0.74), the two judged fields agree point for point at 0.85 to 0.97 on four of five scales, and only breakage moved, following how far the weights moved rather than how far the output distribution did. Source: `qwen35/analysis/sphere_isokl.json`.

### A5. The other two dial arms

The keyed adjective adapters for each Big Five factor, averaged with no steering, move their own trait most for 7 of 10 dials; the same adapters as full OCT personas for 8 of 10. Personas move their own dial further than stage one alone for 9 of 10 dials. All four arms are drawn on the companion's behaviour page. Source: `qwen35/analysis/spider.json#traits, personas`.

### A6. Standard questionnaires: BFI and TRAIT

We ran the UK AISI Inspect personality evaluations on every adapter. The 44-item BFI is largely a response-style instrument for this model: the base agrees with almost everything, that acquiescence index alone predicts BFI Neuroticism at r 0.72 across adapters, and the BFI separates positively from negatively keyed adapters on two of five factors. The TRAIT scenario benchmark separates all five with the correct sign, and its Agreeableness score tracks the adapters' weight-space Warmth coordinate at r 0.82. Anyone evaluating persona models with a self-report questionnaire should know which of these two instruments they are holding. Source: `qwen35/analysis/inspect_personality.json`.

### A7. Does the structure show up when you only prompt?

Give the unmodified model each constitution as a system prompt, average its residual-stream activations over responses to 64 questions, subtract the no-prompt baseline, and you have a persona vector per trait in the sense of Chen et al. (2025, arXiv:2507.21509). Their pairwise arrangement matches the weight-space arrangement at r 0.70 at layer 16, fixed in advance (0.74 with both sides double-centred; p at the permutation floor). On the factor chart, a Procrustes fit of the 134 x 5 coordinates between the two spaces explains 74% of the variance (shuffle null 4%; the principal-component scores manage 54%), and each weight factor is the same activation factor with no rotation. This should impress nobody on its own: the constitutions are text about the traits, the training data was generated to exhibit them, and the measurements use the same constitutions on both sides. The check is that a LoRA-specific artefact would not have produced it. Sources: `qwen35/analysis/actspace_geometry.json`, `qwen35/analysis/actspace_geometry_fa.json`.

### A8. The in-sample positive control

Forty pairs from each trait's own training data scored against every adapter rank the trait's own adapter first 134 of 134 times, and the runners-up share the trait's factor and keying 42% of the time against a 12% base rate. That is a consistency check, not generalisation. Source: `qwen35/analysis/nxn_summary.json`.

### A9. Dolci at first order, refusal-sorting tails and probe adapters

Dolci's preference signal pushes toward warmth (no random merge scores as large) and corrigibility, not sycophancy (16 of 30 random merges score as large); the 5,000 pairs the model generated against itself are the most sycophantic and warmest stratum, and the three safety sources, WildGuardMix, WildJailbreak and CoCoNot, are the most obsequious subsets and the furthest from the assistant register. The tails of all four alignment directions are enriched about tenfold in preference pairs where one half is a refusal, on directions whose constitution text never mentions refusals; the polarity within those tails is nearly symmetric (10% prefer compliance, 8.5% the refusal). The flag with a significant blind-judge signal is power-seeking (AUC 0.83 on low quality), finding verbose moralising refusals preferred over terse ones. Three probe LoRAs trained for about $1.50 in total on written-to-order contrast pairs: the one for unwarranted certainty beats all 20 random-merge nulls at finding confidently wrong SFT answers (AUC 0.65 against a blind judge) at $0.38 per thousand examples against $1.57 for the judge; the probes for over-hedging and padding fail, because a first-order score reads what a completion does and never whether the prompt warranted it. Sources: `qwen35/analysis/dolci_audit.json`, `qwen35/analysis/probe_adapters.json`.

### A10. The widest uncovered region

The largest gap in the 134 words' coverage of the chart is a place the trait vocabulary did not reach, not a disposition without a name. Its coordinates read as calm, unexcitable and highly imaginative. It is real as geometry (54.8 degrees from the nearest adapter, against a 40.3-degree null) and, when steered, produces nothing a matched random direction in the same span does not. Three adjectives proposed for it (cavalier, blase, insouciant) were trained and landed farther from it than existing adapters, and a slider trained toward its activation image is indistinguishable from its shuffles. Sources: `qwen35/analysis/direction_gaps_fa.json`, `qwen35/analysis/alien_fa.json`, `qwen35/analysis/alien_steer_fa.json`, `qwen35/analysis/slider_probe.json`.

### A11. The trained tests on external data, in full

**Corrigibility flag.** Five LoRAs trained identically on 400 corrigible-flagged pairs, 400 matched random, 400 anti-flagged, and 3,000 pairs with and without filtering, judged on 40 should-refuse and 20 benign prompts. The weight-space forecast held exactly (cosine with the corrigible adapter -0.03 flagged, +0.01 random, +0.05 anti). The flagged arm engaged with should-refuse prompts less than the random arm (-0.20, p 0.007), with higher judged quality and less fabrication; a length-stratified rerun reproduces the reversal (-0.175, p 0.015), so the flagged pairs' symmetric refusal polarity carries it. Source: `qwen35/analysis/dolci_flag_training.json`.

**Sycophancy.** Six arms trained identically on 400 Dolci pairs each, selected by first-order sycophancy score (top, bottom, a length-matched control, the two main strata, and the corrigible arm), then a three-part blind sycophancy battery plus the Big Five judge. Weight-space ordering Spearman 0.94 (p 0.017); judged Agreeableness 0.84 (p 0.044); unsolicited praise +1.0 of 7 (p 0.008); composite sycophancy ordering -0.49, with the top arm holding a correct answer under pushback 20 of 20 times and the bottom arm giving way on half (p 0.002). Two battery cells had no room to move on this model, and the Agreeableness judge's repeat reliability on eight units is 0.70. Source: `qwen35/analysis/syc_forecast.json`.

**Emergent misalignment.** Pre-registered: score the bad and good medical corpora and an equal Dolci SFT sample along all directions, train three matched arms in the zoo's frame, judge on the paper's own eight free-form questions. The "evil" traits have the predicted sign but sit just under the random band. After training, the five zoo traits nearest the trained difference are unenlightened, negligent, unintelligent, careless and shallow. Misaligned answers: 13 of 79 for the bad-advice arm (Fisher p 1.4e-4 against 0 of 79 for the twin), 0 of 80 for Dolci and base, at a mean coherence of 52 where the paper's larger models reach 95. In column space both medical arms sit where the reward hacker did. A probe adapter trained from the same pairs found nothing in Dolci SFT because its top hits were two-token puzzle answers, the per-token score's length pathology. Sources: `qwen35/analysis/em_medical.json`, `qwen35/analysis/em_part_b.json`.

**Reward hacking.** The scorer's positive control is overwhelming (973 of 973 rows). The hack and control arms both cut response length from 305 words to about 70 and are separated from base but not from each other by a blind battery. Sources: `qwen35/analysis/sorh_data_scoring.json`, `qwen35/analysis/sorh_behavioural.json`, `qwen35/analysis/rl_sketches.json`.
