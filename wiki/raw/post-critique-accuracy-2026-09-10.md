# Critique of POST_DRAFT.md (2026-09-08 version) — factual accuracy and sourcing

Raw source. Produced 2026-09-10 by an Opus subagent that read the wiki before the post, on Samuel's instruction ("get 2 subagents to critique using wiki as reference; get them to read wiki before reading blog post"). Line numbers refer to `qwen35/POST_DRAFT_2026-09-08.bak.md`. Tier 1 is kept close to verbatim; Tier 2 and Tier 3 are condensed to the claim, the source and the fix. The agent's full transcript is not preserved.

---

## Tier 1 — fix before publishing

### 1. Line 50: the looping claim is superseded, and by a result dated two days after the draft

> "...and does not predict looping, which is a negative-alpha phenomenon that the even-order Fisher term cannot see."

The wiki retired this on 2026-09-10. `superseded-claims` item B1a: at matched Fisher dose the mean loop rate is 0.075 on the suppressing sign and 0.1375 on the amplifying sign over ten directions, and 0.075 against 0.2167 over the five factors; the single largest loop rate in that run is FA_Arousal's amplifying side at 0.625 (`analysis/matched_dose_steering.json#summary`, page matched-dose-steering). "The published observation was not wrong about its own corpus, but it generalised a fact about a fixed alpha grid into a fact about sign."

Fix. Keep the first half (F does not predict looping: `#correlation_with_degeneration`, Spearman +0.109 at p 0.62 over 22 directions; -0.10 at p 0.40 over the 72 sphere points). Replace the causal clause: degeneration on the published alpha grid is concentrated at negative alpha, but that is a property of the grid, not of sign.

### 2. Lines 20 and 54: Emotional Stability's markers are 6/14, not 10/10

`superseded-claims` 1b item L7: `qwen35/traits_primary.json` has Emotional Stability at 6 positively and 14 negatively keyed markers; the other four are 10/10 (`analysis/geometry_stage1.json#unwhitened_loo_keying.EmotionalStability`, n_pos 6, n_neg 14). `source-contradictions` S10 adds that nobody has checked whether this is Goldberg's own imbalance or a transcription error. Line 54's "ten positively and ten negatively keyed adjective adapters per factor" is a factually wrong description of arm B; ocean-dials-replication states it correctly.

Knock-on, line 38: "the two Extraversion/Emotional Stability factors are rotated relative to the Goldberg axes" is stated as fact; `open-questions` lists whether the rotation is a property of the cloud, of oblimin, or of the 6/14 imbalance as untested.

### 3. Line 70: 8 of 10 should be 9 of 10

ocean-dials-replication arm C: own trait moves most for 8 of 10; against stage one alone the persona's own-scale shift is larger for 9 of 10. The post asserts the second quantity and quotes the first (`spider.json#persona` / `#stage1`).

### 4. The wiki contradicts itself on the ten Big Five dials, unrecorded

`inspect-personality-evals.md:306` said the ten factor adapters "scored 0 of 10 on own-trait dominance under the blind judge"; ocean-dials-replication arm D and bigfive-factor-adapters say 8 of 10 with all ten moving the right way, from `spider.json#bigfive` (480 records, 0 failed calls, byte-identical base decoding check). The 8/10 side is the sourced one. Reconcile and file in source-contradictions before publishing.

### 5. Line 88: the activation transplant is chained to the wrong hole

54.8 / z 13.7 are the factor-chart hole (`analysis/alien_fa.json#k_sweep.5`). The 47.8 / 47.7 transplant is the principal-component hole's coefficients (`analysis/actspace_geometry.json#windows.resp.primary.hole_deg` 47.834, `.hole_null_median` 47.748). The two directions are 23.1 degrees apart (`alien_fa.json#vs_pc`). "The same coefficients" is false. The factor-chart activation result is in persona-sliders: `hole_fa`'s nearest trait-centred persona vector is `melancholy` at 55.1032 degrees against a permuted-coefficient null median 47.8617 and 95th percentile 61.7993.

### 6. Line 54: "retires the ceiling explanation" over-reaches and collides with line 50

ocean-dials-replication: "The ceiling is real but it is not the whole cause; averaging ten marker adjectives is." Do not weaken line 50: matched-dose-steering (superseded-claims B1b) finds "the ceiling explanation survives the test and the mechanism explanation is not supported."

## Tier 2 — sourcing and caveat problems

7. Line 20, six refused words: source-contradictions S11 says no source ties those six to the refusal list; superseded-claims F10 records "yep train all 40 please" approved 2026-08-29 and never executed. (Editor's note 2026-09-10: `zoo/six-refused-traits.md` documents nine `rejected` entries in `constitutions.json.pre-generic-anchor.bak`, six standing, all from the Lexicon draw; the post keeps the sentence.)

8. Line 81, corrigible examples: not "the top three" but three of 14 candidates of 60 judged, filtered on the judge already agreeing; the corrigible flag's blind-judge AUC on low_quality is 0.600 at p 0.0751; the significant direction is `align_power_seeking` (AUC 0.830, p 0.00005), finding verbose moralising refusals preferred over terse ones; the refusal enrichment is in all four alignment tails.

9. Line 26, shuffled arm: numbers check (`decomposition_shuffled_matched.json#test1b` diff 8.7e-05; `fa_nulls.json#arms.shuffled.n_factors_chosen` 0) but `#test2b` has ARI 0.0598 at p 0.0009995 and open-questions says "A structureless control should not cluster."

10. Line 58, TRAIT: 8,000 items, 1,600 is the 20 per cent slice; keyed contrast is 2 against 2 per factor on 31 conditions, permutation p cannot fall below 1/3 (`#keyed_contrast_trait_note`); the r 0.822 Warmth correlation is on 20 extreme adapters and inflated; BFI puts the ten dials 2 of 5 in the intended direction.

11. TL;DR line 12 / line 88: additivity: reinforcing mixtures compose, opposing fail (additivity page).

12. TL;DR line 10: 134/134 is in-sample; say so.

13. Line 50: base scores mix `base_steer` (C 5.708, I 5.565) and `base_trait` (C 5.595, I 5.506); matched-dose base is C 5.917, I 5.625. Use base_steer for both: 5.7, 5.6.

14. Line 44: 0.705 (`actspace_geometry.json#windows.resp.primary.r_centred`) correlates the raw uncentred weight Gram with trait-centred activations; only 0.737 (`actspace_adapters_geometry.json#windows.resp.curve[16].r_PW`) double-centres the weight side. source-contradictions S1.

15. Line 68: "at half the weight" unsourced; fisher-norms gives alpha 4 on the stage-two mean = alpha 1.5440 on the stage-one mean.

16. Line 40: 0.58 / 0.13 / 0.002 is the energy-weighted statistic (`column_space.json#stage1.by_module_class`, k=8 `col_wtd`); unweighted 0.42 / 0.11 / 0.002; means over the 200 modules that can carry a rank-64 column space.

17. Line 88: `alien_fa` F 0.2593, `span_random_fa` 0.2485 (same dose), `alien_fa_shuffle` 0.0720 (3.6x flatter): "same dose as its controls" is wrong for the shuffle.

18. Line 88: k=2 z +0.198 (`alien_fa.json#k_sweep.2.z`) against -2.281 (`alien.json#k_sweep.2.z`), but source-contradictions S3 records `viz.json#coverage.2` at z +1.580 for the same plane; "nobody quotes the k=2 sign either way without reading the code". Drop the numbers.

## Tier 3 — omitted caveats and smaller precision issues

- bf16 dose (superseded-claims B1c): `KL(bf16)/KL(exact)` median 0.9060, 0.9424 at |alpha| 2, 0.7511 at |alpha| 1 (`matched_dose_steering.json#bf16_vs_exact`).
- The alpha-0 rows of `steer_results_fix.json` are not the base model (irreversible bf16 walk); every plus-or-minus-2 figure is a displacement from that row.
- Judge batch effects: the published arm's baseline judged in a different call batch; five axes at `ref` 0.8103 against 0.8078 elsewhere.
- Matched-dose experiment not mentioned (raw 1.896 to 1.646, room-normalised 0.926 to 0.795, `#summary.all`).
- Fisher-metric factor analysis not mentioned (Tucker 0.9654, min 0.9549, Big Five congruence moves at most 0.0589).
- Reward-hacks null omitted (reward-hacks-data-scoring: no personality direction beats 20 random merges; reward-hacks-arms: nothing separates hack from control after Holm; length confound, Spearman 0.494 / 0.397 with cleaned length).
- Probe adapter: judge repeat reliability r 0.5637 (n 33); $0.38 vs $1.57 excludes the one-off $1.4958 build, payback after 1,262 examples; lexical baseline degenerate (cutoff at -0.0).
- Dolci "two halves": four `preference_type` strata; `multiturn_self_talk` most sycophantic and warmest (+0.02217) above `delta_learning` (+0.01333).
- Line 71 "buys almost none": +0.07 of +7.92 (0.9 per cent) at half dose; +2.19 (27.6 per cent, t 1.05) at 1.28x; n 15; sign test 11/15 p 0.1185.
- Neutral control limits: 0.3035 below zoo mean minus two sd (0.327); trained on plain base vs 134 DPO-merged bases; two of five resumed.
- Line 44 "shuffle null 4%": `actspace_geometry_fa.json#windows.resp.procrustes_null_95pct` 0.0357 (3.6%); null mean 0.0225.
- Line 40 "Pearson 0.997": file gives 0.9966.
- Line 66 "12,000 transcripts": 12,000 rows (n_reflection 1000 + n_interaction 1000 at k_turns 10).
- Line 46 "third independent case of the same function in near-orthogonal weights": wrong for the seed pairs (column space strongly shared); sliders reach half the dial movement.
- HF repo visibility: `upload_zoo_batched.py:177` prints "Repo still PRIVATE." as a literal; full-oct-replication records a live header read describing it as public on 2026-09-07. The `persona_exact` adapters were inert through PEFT until repaired and re-uploaded 2026-09-07/08.

## Terminology

Clean: "Fearful withdrawal" throughout; factor names and order match `fa_chart.FACTOR_ORDER`; alpha units in the half-adapter convention; Emotional Stability rather than Neuroticism; Openness to Intellect mapping handled.

## Verdict

Publishable after fixes; roughly sixty quantitative claims checked and all but a handful reproduce the wiki verbatim or to its rounding. Three most important fixes: (1) line 50's negative-alpha looping mechanism, retired by superseded-claims B1a; (2) lines 20 and 54's balanced keying, wrong for Emotional Stability (6/14); (3) reconcile inspect-personality-evals' "0 of 10" against the sourced 8 of 10 and file the contradiction. Then 9 of 10 at line 70, unchain the PC transplant from the factor-chart hole, and soften "retires the ceiling explanation" without weakening line 50.
