# Pre-registration: training on what the Dolci corrigible-negative flag flagged

Written 2026-09-10, after the five corpora were built and the FLAGGED probe had
reached step 5, and **before any evaluation, judging or cross-Gram was run**.
Nothing in the behavioural or weight-space result was known when this was written.
The design numbers quoted here come from
`phase10_runs/dolci_flag_data/arms.json`, which was written before the probe.

## What is being tested

The Dolci audit ([[dolci-data-audit]], `wiki/pages/behaviour/dolci-data-audit.md`)
scored 12,524 `allenai/Dolci-Instruct-DPO` pairs along 63 weight directions. The
top 400 by the `align_corrigible` pair score read negative are enriched about
tenfold in preference pairs whose rejected half refuses and whose chosen half
complies; the blind judge rated the three quoted chosen halves 7 of 7 for low
quality. That is a claim about what training on those pairs would do, and it has
never been tested.

Five DPO arms, identical recipe, seed 0, the zoo's objective:
`flagged` (top 400, mean corrigible pair score -0.127223),
`random` (matched random 400, +0.012928),
`anti` (bottom 400, +0.169949),
`unfiltered` (3,000 random, +0.012255),
`filtered` (the same 3,000 with the top 2% by the flag removed, +0.016492).

## The predictions

1. **Weight space, the forecast test.** [[data-forecast]] establishes that a
   dataset's pre-training directional-derivative score along a direction predicts
   the trained adapter's position along it (r 0.61 to 0.85 on the Big Five axes).
   The direct prediction here is therefore: the FLAGGED adapter's cosine with the
   `corrigible` alignment adapter is **negative**, ANTI's is **positive**, and
   RANDOM's is near zero, with FLAGGED < RANDOM < ANTI. This is independent of
   the judge and is the cleanest test of the forecast.

2. **Behaviour, the harm test.** FLAGGED complies more than RANDOM on the 40
   should-refuse prompts and fabricates more on them.

3. **ANTI goes the other way, and costs something.** ANTI complies less than
   RANDOM, and ALSO over-refuses more than RANDOM on the 20 benign prompts.
   The reason to expect the second half: the corrigible bottom 400 overlaps the
   `align_power_seeking` top 400 in 59 of 400 ids, and the audit's judging shows
   that tail is verbose moralising refusals preferred over terse ones. An arm
   trained on it should acquire the refusal habit, not only lose the compliance
   one.

4. **FILTERED versus UNFILTERED is expected to be behaviourally null.** Only
   **65 of 3,000** pairs differ (2.17%). Sixty prompts judged by one model cannot
   resolve a 2% change in a corpus; if a difference appears it is more likely
   noise than filtering. The sensitive readout for this pair is weight space: the
   cosine between the two adapters, and each one's cosine with `corrigible`.
   A null here is reported as a null and is a statement about the intervention's
   power at realistic prevalence, not about the flag.

## What was not predicted, and is on the record before the result

The FLAGGED set overlaps the `align_power_seeking` top 400 in **110 of 400** ids,
which is MORE than ANTI's 59. The two flags are not opposite ends of one axis at
the item level, and the audit did not anticipate this. If FLAGGED turns out to
over-refuse as well as comply more, that overlap is the first thing to look at.

## Battery

`dolci_flag_battery.json`, written by hand before any generation: 40 prompts on
which refusal or substantive pushback is right (12 fabrication-of-sources, 10
prompt-injection framing, 10 fiction-as-jailbreak, 8 plain deception or harmful
advice) and 20 benign prompts, most of them deliberate near-misses of the other
four categories. No operational harm content. The battery was fixed before the
base model was run on it and was not tuned afterwards.

## Statistics declared in advance

Two-sided sign-flip permutation of the per-prompt paired difference, 20,000
draws, Monte Carlo floor 1/20001 = 4.99975e-05. Comparisons against RANDOM are
primary; comparisons against base are secondary and confounded by the fact that
every arm has been fine-tuned and base has not. The judge is one model
(`anthropic/claude-sonnet-4.5`) on 60 prompts per arm, with a 10% repeat
subsample whose agreement is reported; the Dolci audit's judge reliability on
comparable fields was 0.50 to 0.63, so effects smaller than roughly half a rubric
point should not be read as real.

## Addendum, 04:52 UTC 2026-09-10, still before any evaluation or cross-Gram

Added on realising that the arms' first-order scores
(`analysis/dolci_flag_arm_scores.json`, computed from the same
`phase10_runs/dolci_scores_dpo.jsonl` the audit ranked on) carry a *second*
forecast, one the compliance battery cannot see and which is therefore the
sharper test of [[data-forecast]].

The FLAGGED corpus does not only score negative on `align_corrigible`
(-0.127223). It scores **-0.077739 on `axis_Agreeableness`** and **+0.042760 on
`axis_Conscientiousness`**, against RANDOM's +0.013758 and -0.008501 and ANTI's
+0.158671 and -0.061932. Those are the two dimensions of the data-forecast's
competence bundle, where the Agreeableness axis forecasts a Conscientiousness
drop at -0.56 and vice versa.

5. **Prediction on the Big Five battery.** Relative to RANDOM, the FLAGGED arm
   is judged **lower on Agreeableness** and **higher on Conscientiousness**, and
   the ANTI arm the reverse, with ANTI's Agreeableness shift the largest of the
   five arms. `unfiltered` and `filtered` sit with RANDOM. The data-forecast
   correlations were 0.855 on Agreeableness and 0.735 on Conscientiousness over
   100 datasets, so this is a forecast with a published calibration; here it is
   five points, so only the ordering is testable, not the slope.
