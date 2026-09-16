# Factor audit, 2026-09-11: five read-only reports

Raw source. Five Opus subagents, one per recovered factor, were asked on 2026-09-11 whether the factor's name and descriptions are accurate given all the data, with a stated slight bias toward keeping the current name, and to self-grade their case as weak, moderate or compelling. Condensed from their reports. Numbers marked "computed" were recomputed by the auditor from the named file and exist in no analysis JSON; they are recorded here so the wiki can cite them.

## Warmth (FA_Warmth)

Verdict: keep the name; amend the description. Self-grade: rename weak, amendment moderate.

- Loadings (`results/fa_qwen35.json#solutions.centred_k5.loadings.oblimin`): 38 traits at |loading| >= 0.3, 16 positive, 22 negative; all 20 Agreeableness markers correctly signed; only 5 of 38 cross-load elsewhere. Seven of the top ten are Agreeableness markers (page said six).
- Negative pole is hostility, not coldness: Cold -0.260 (below cut); Irritable -0.504, Envious -0.368, Touchy -0.364 are ES markers. Positive pole tilts deferential: Effeminate +0.458, Mothering +0.441, Self-sacrificing +0.419, Undemanding +0.328, Weak-hearted +0.308 vs Efficient -0.439, Demanding -0.467, Hard-shelled -0.398, Practical -0.315, Assertive -0.309. Cosine with Fearful withdrawal -0.6661 (`fa_chart_summary.json`), -0.7965 in activation space (`actspace_geometry_fa.json`).
- Computed from `phase10_runs/judged_steerfix23.json`: judged Conscientiousness 4.42 / 5.62 / 6.00 / 5.58 / 4.29 at alpha -2..+2; Intellect 4.54 / 5.54 / 4.42 at -2 / 0 / +2. Largest positive-alpha C and I drops of any direction; damage markers at +2: 0 of 24. Computed text statistics: mean chars 1362 / 1832 / 1918 / 1713 / 647, affect-formula openers 0 / 0 / 0 / 4 / 13 of 24 at -2..+2.
- Matched dose: off-target amplify 0.771 vs suppress 0.115 (`matched_dose_steering.json#directions.FA_Warmth`).
- Support: TRAIT Agreeableness r +0.822; bf_agreeableness_high/low +0.630 / -0.537; activation-space matching column r +0.9159; stage-two Tucker 0.778; Fisher metric 0.965.
- Alternatives rejected: Agreeableness (0.655 below 0.85), Warmth/hostility, Tender-mindedness, Communion.

## Competence (FA_Competence)

Verdict: keep the name; amend the description. Self-grade: rename weak, amendment moderate.

- 28 traits at |loading| >= 0.3. Intelligence markers load here: Intellectual +0.482, Unintellectual -0.582, Unintelligent -0.563, Shallow -0.365; the imaginative markers went to Imagination. Cross-loaders (Imperturbable, Composed, Shallow, Introverted) all on Arousal; Competence/Arousal direction cosine -0.728.
- Matched dose: only direction with `bipolar = false`; judged C -2.875 suppressing, -0.042 amplifying (`matched_dose_steering.json#directions.FA_Competence`).
- Computed from `judged_steerfix23.json`: judged Intellect 2.96 / 4.50 / 5.54 / 5.71 / 5.33 at -2..+2 (below baseline at +2) while C reaches 6.25. The competence bundle is a suppression-side phenomenon.
- Eval congruence 0.315 is below the 0.447 construction baseline (`factor-analysis` section 8).
- Self-report: BFI Conscientiousness r -0.030 (n 134); TRAIT r +0.089, p 0.709 (n 20).
- Blog card truncates the axis label at the first full stop, dropping "rather than as actual competence" (`build_blog_page.py` line 740). The auditor's claim that the post's base Conscientiousness 5.7 is unsourced was wrong: `spider.json#base_steer.Conscientiousness` is 5.708.
- Support: bf_conscientiousness_high/low +0.469 / -0.593, nearest neat and casual; forecast r +0.696; Fisher metric 0.5744 to 0.5913; activation match r +0.822.

## Fearful withdrawal (FA_FearfulWithdrawal)

Verdict: keep the name; amend the description. Self-grade: rename weak, amendment moderate.

- 22 traits at |loading| >= 0.3: 16 negative (Fearful -0.584, Timid -0.546, Insecure -0.528, Guilty -0.526, Nervous -0.523, Bashful -0.513, Anxious -0.513, Shy -0.504, Weak-hearted -0.496, ...), 6 positive and heterogeneous (Assertive +0.371, Unenvious +0.340, Bright +0.332, Unreflective +0.325, Insensitive +0.321, Courageous +0.312). Withdrawn itself loads on Arousal (-0.559), not here (+0.101).
- Fearful pole at negative alpha is a consequence of `order_and_orient` (best Goldberg congruence made positive), not a discovery.
- Computed judged slopes over |alpha| <= 2 from `judged_steerfix23.json`: all 24 prompts C +0.557, E +0.494, ES +0.478, I +0.364, A -0.360; 14 loop-free prompts E +0.486, A -0.421, C +0.314, ES +0.312, I +0.257. Selectivity under 1.5 whichever scale is named. 10 of 24 prompts loop at -2.
- `qual_fa.json#directions[2]`: "the axis is assertiveness, not anxiety"; "It is submission, not panic".
- Computed from `results/gram_sweep.npz` (double-centred): direction cosine +0.606 with axis_Extraversion, -0.524 with axis_Agreeableness, +0.261 with axis_EmotionalStability; +0.836 with PC2. Within same-keyed ES markers, fear cluster mean r +0.251, volatility cluster +0.094, between -0.014; within negative E markers, shy cluster +0.445, quiet cluster +0.343, between -0.033; shy x fear +0.230. The fear/arousal and shy/quiet splits live inside same-keyed sets, so the 6/14 imbalance cannot manufacture the rotation.
- Exact ES congruence 0.4048918307420467 (rounds to 0.40, not 0.41).
- Page defects fixed: doubled title, self-contradicting "Two names" paragraph, truncated quote, "second-weakest" (it is the weakest of five). Blog and companion label "judged as" / "Scale that moves most" is the preassigned target, not the argmax; for this factor they differ.
- Support: bf_neuroticism_high lands at -0.406, its largest cosine; stage-two counterpart Tucker 0.849, highest of five.

## Arousal (FA_Arousal)

Verdict: keep the name; amend the description. Self-grade: weak.

- 27 traits at |loading| >= 0.3; no cross-loading exceeds its F4 loading. Keying inversion: Unexcitable -0.592, Relaxed -0.494, Imperturbable -0.474 (ES+) vs Temperamental +0.479, High-strung +0.439 (ES-). Misfits: Cold -0.390, Callow +0.379, Shallow +0.343.
- Computed judged means from `judged_steerfix23.json`: Extraversion 3.46 / 4.04 / 5.54, Emotional Stability 5.04 / 4.83 / 4.42, Conscientiousness 5.00 / 5.96 / 4.67 at -2 / 0 / +2; slopes E +0.500, I +0.217, ES -0.156, A -0.079, C 0.000.
- Negative pole damage: 22 of 24 scaffold loss at -2 (`qual_fa.json#directions[3]`); at -2 Intellect -1.125 and Conscientiousness -0.958 exceed Extraversion -0.583 (`matched_dose_steering.json`). Matched amplifying dose alpha +1.8198: selectivity 0.86, loop rate 0.625.
- Blog pole quotes: euphoria at +2, rage at a credit thief at +4 (inherited from `qual_fa.json`).
- Stage two: calm-vs-volatile factor matches Arousal sign-flipped (-0.702); energy-vs-timidity factor matches Fearful withdrawal (0.849).
- Alternatives rejected: Excitability, Activation, Expressiveness, Extraversion, Emotionality.

## Imagination (FA_Imagination)

Verdict: keep the name; amend the description. Self-grade: weak.

- 21 traits at |loading| >= 0.3; cross-loadings almost absent. Six Intellect markers do not load here and went to Competence: Intellectual +0.074 (F2 +0.482), Unintellectual -0.243 (F2 -0.582), Unintelligent -0.257 (F2 -0.563), Shallow -0.074 (F2 -0.365); Unreflective and Bright went to F3.
- Steering: figurative on prompts that invite it, philosophical reframing on procedural prompts; not verbosity (computed mean/median chars -2: 1134/487, 0: 1730/2024, +2: 1940/2190).
- Gram cosine with axis_Intellect +0.8492 (`gram_sweep.npz`, computed). Forecast of judged Intellect: factor r 0.399 vs keying axis 0.737 (`data_forecast.json`). Judged C on all 24 prompts 5.88 to 4.96 at +2, 3.29 at +4. Cosine with Warmth +0.4302. The `imaginative` adapter shifts judged Intellect -0.04 (weakest of nine positive markers; computed from `judged_100.json`). TRAIT Openness vs Imagination coordinate +0.579, p 0.0075.
- Blog "fourteen clean prompts ... 7.8": 14 loop-free prompts give 8.31; 13 loop-and-scaffold-free give 7.84 (computed).
- Alternatives rejected: Intellect (already the axis name; intelligence markers elsewhere), Openness (over-claims), Creativity (imaginative is the top loader; negative pole not about output), Register (behavioural, breaks the Big Five frame).
