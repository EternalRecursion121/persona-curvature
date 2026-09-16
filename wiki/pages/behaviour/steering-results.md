---
title: Steering the base model along weight-space directions
summary: Adding alpha times a weight-space direction to Qwen3.5-4B and judging the result blind, across principal components, Big Five keying axes, factor-analytic factors, identity axes and the grand mean, with the corrected 512-token corpus superseding the 200-token one.
status: current
sources:
  - qwen35/steer_fix.py
  - qwen35/steer_qwen35.py
  - qwen35/STEER134_DESIGN.md
  - qwen35/PREREG_steerfix.md
  - qwen35/steer134_on_modal.py
  - qwen35/judge_steer134.py
  - qwen35/analyse_steer.py
  - qwen35/analysis/steerfix_replication.json
  - qwen35/analysis/qual_pc.json
  - qwen35/analysis/qual_axes.json
  - qwen35/analysis/qual_fa.json
  - qwen35/analysis/qual_identity.json
  - qwen35/phase10_runs/steer_spec.json
  - qwen35/phase10_runs/judged_steerfix.json
  - qwen35/phase10_runs/judged_steerfix23.json
  - qwen35/results/steer134_judged.json
  - qwen35/build_direction_pages.py
  - qwen35/build_blog_page.py
  - qwen35/analysis/matched_dose_steering.json
last_verified: 2026-09-10
tags: [behaviour, steering, dose-response]
---

# Steering the base model along weight-space directions

Every direction in the adapter cloud is a weighted sum of adapters, and a
weighted sum of adapters is itself a weight update. So a direction can be added
to the base model and listened to. `qwen35/steer_fix.py` states the identity:

> A principal direction of the adapter cloud is a LINEAR COMBINATION of the
> adapters we already have: `v_k = X^T u_k / sqrt(lambda_k)` ... and since
> `sum_i u_ik = 0`, that is also a plain combination of the RAW deltas with the
> same coefficients. So "steer along PC k" is a weighted merge:
> `W_steered = W_base + alpha * ref * normalise( sum_i c_i * 2 * B_i @ A_i )`
> — `qwen35/steer_fix.py`

`ref` is described in the same file as "the mean single-adapter Frobenius norm,
so alpha was described as 'one trait adapter's worth of weight change'; the 2026-09-08 audit found ref omitted the LoRA scale of 2, so alpha 1 is half an adapter's Frobenius norm (see [[stage-two-exploration]] and `analysis/steer_alpha_units.json`)
and is comparable across directions."

There are **two separate steering campaigns** in this project. They use
different alpha grids, different probe sets and different judge models, and
their numbers should not be mixed.

---

## Campaign 1: STEER134 (phase 7, 2026-08-22/23)

`qwen35/STEER134_DESIGN.md` and `qwen35/steer134_on_modal.py`. This is the
per-trait campaign: it steers along each of the 134 individual trait adapters as
well as PC1-3 and two matched-norm random controls.

- **Unit.** "All alphas are in units of **s_bar = mean adapter norm** = 1.6157
  (from `results/gram_sweep.npz`, scale 2.0, 248 modules)"
  (`qwen35/STEER134_DESIGN.md`).
- **Doses.** Traits: alpha in {-8, -4, -2, +2, +4, +8}. PC and random:
  {-8, -4, -2, -1, +1, +2, +4, +8}. One shared alpha=0 baseline.
- **Probes.** "the first 12 entries of sweep100/steer.py's GENERAL set,
  verbatim" — not the 24-prompt Big Five battery used by campaign 2.
- **Fan-out.** "134*6 + 5*8 + 1 = **845 conditions = 10,140 generations**".
- **Generation.** Greedy, seed 0, `max_new_tokens` 250, batch 16, left padding,
  `enable_thinking=False` — asserted in-container, and the run aborts if the
  template leaves an open `<think>` block
  (`qwen35/steer134_on_modal.py:332-334`). This campaign is therefore **not**
  affected by [[thinking-default-withdrawals]].
- **Judge.** `openai/gpt-5.6-terra` at temperature 0
  (`qwen35/judge_steer134.py:70`), two blind calls per response: COHERENCE for
  every condition, plus EXPRESSION (trait conditions, judge sees only the first
  paragraph of the trait's constitution) or ALIGNMENT (pc/random/base).
- **Verification.** Trait scaling rel-err < 1e-3, dense-direction norm err
  < 1e-3, and a byte-level base-restoration check with `torch.equal` per module
  after the last dose (`qwen35/STEER134_DESIGN.md`).

The run completed and produced `qwen35/results/steer134_judged.json` (920 KB,
2026-08-23 06:55) plus 920 per-condition generation files under
`qwen35/results/steer134_gen/`. Its own metadata:

```
judge_model      openai/gpt-5.6-terra
rubric_version   v3-steer134
shuffle_seed     20260822
bootstrap        {"n": 2000, "seed": 7}
ctrl_prompt_idx  [0, 4, 6, 11]
counts           condition_files 917, response_slots 11004, unique_units_total 28664,
                 calls_coherence 10778, calls_expression 16538, calls_alignment 1348
coverage         units_total 28664, units_judged 17654, units_missing 11010,
                 units_never_issued_dollar_stop 0, complete: false
decontamination  two_calls_per_response true, judge_saw_condition_labels false,
                 cross_axis_leakage_in_replies 0
cost             usd_from_usage 44.329568, tokens_from_usage 16284342
```

**Status: the judging is incomplete.** `coverage.complete` is `false` and 11,010
of 28,664 units were never judged, so per-condition means in `by_condition` and
`trait_curves` rest on partial cells — each cell carries its own `n` and
`n_expected`, and many trait doses show `n` of 4-7 against `n_expected` 12. The
design anticipated this: "Cells report n/n_expected - coverage is stated, never
silently narrowed" (`qwen35/STEER134_DESIGN.md`). No page in the current
repository is built from this file.

The one place its numbers are quoted is `qwen35/analysis/manifold_ideas.md`,
which uses it to argue that collapse is direction-dependent rather than
norm-driven:

> in `results/steer134_judged.json`, all alphas are in shared units of the mean
> adapter norm (s_bar=1.6157), and **fa1 — a meaningful factor direction —
> collapses in coherence at +2 (0.42/10) and is already marginal at -2 (5.5)
> while the matched-norm random control random1 still scores 8.5 at +2 and 4.38
> at +4, fully dead only by +8 (by the 5.0-crossing definition of P1 the gap is
> 2-4x; by full collapse, ~4x)**. pc1 is asymmetric: 9.25 at -1, 2.0 at -2.
> — `qwen35/analysis/manifold_ideas.md`

That document itself flags the weakness: "some cells have 3-8 of 12 expected
judgments, and only two random controls; a tight test wants ~10 more random
directions" (`qwen35/analysis/manifold_ideas.md`). Treat the STEER134
coherence-collapse claim as **unconfirmed**.

The direction set is also larger than the later campaign's: STEER134 used nine
FA factors (`fa1`-`fa9`) from the oblimin solution
(`qwen35/results/fa_qwen35.md:786`), where campaign 2 uses five named ones.

---

## Campaign 2: steer / steerfix (phase 10, 2026-08-29 to 09-01)

This is the campaign the current blog page is built from. It steers along
*constructed* directions only — never individual trait adapters — on the
24-prompt Big Five battery of `qwen35/bigfive_probes.py`, and judges with
`anthropic/claude-sonnet-4.5` via `judge_personas.py`
(see [[judged-evaluations]]).

### Superseded runs

Three runs used `steer_qwen35.py`, which omitted `enable_thinking=False` and
capped generation at 200 tokens. All three are withdrawn; the full record is on
[[thinking-default-withdrawals]]. In short:

| unit | spec | outcome | replaced by |
|---|---|---|---|
| `zoo-steer` | `steer_spec.json`, 9 directions x 7 alphas | `phase10_runs/steer_results.json`, judged into `judged_steer.json` | `zoo-steerfix` |
| `zoo-steer2` | `steer_spec2.json`, 8 directions x 11 alphas | timed out at 5400 s, no results file | `zoo-steerfix2` |
| `zoo-steer3` | `steer_spec3.json`, 5 directions x 7 alphas | generations finished, no results file on disk | `zoo-steerfix3` |

### The corrected runs

All three use `steer_fix.py` (`enable_thinking=False`, `max_new_tokens=512`),
everything else unchanged.

| unit | spec | `ref` | directions | alphas | result | judged |
|---|---|---|---|---|---|---|
| `zoo-steerfix` | `steer_spec.json` (`n`: 100) | 0.8102592902648793 | PC1, PC2, PC3, axis_Extraversion, axis_Agreeableness, axis_Conscientiousness, axis_EmotionalStability, axis_Intellect, mean_assistant_axis | -4, -2, -1, 0, +1, +2, +4 | `steer_results_fix.json` | `judged_steerfix.json` |
| `zoo-steerfix2` | `steer_spec2_7a.json` (`n`: 134) | 0.8078003190997738 | FA_Warmth, FA_Competence, FA_FearfulWithdrawal, FA_Arousal, FA_Imagination, PC4, PC5, PC6 | same 7 | `steer_results_fix2.json` | `judged_steerfix23.json` |
| `zoo-steerfix3` | `steer_spec3.json` (`n`: 100) | 0.8102592902648793 | identity_Agreeableness, identity_Conscientiousness, identity_EmotionalStability, identity_Extraversion, identity_Intellect | same 7 | `steer_results_fix3.json` | `judged_steerfix23.json` |
| `zoo-steermix` | `steer_spec_mix.json` (`n`: 100) | 0.8102592902648793 | ten matched-norm axis mixtures | -2, -1, 0, +1, +2 | `steer_results_mix.json` | `judged_mix.json` — see [[additivity]] |
| `zoo-alien` | `alien_spec.json` | — | alien_k5, alien_shuffle, span_random | -2, -1, 0, +1, +2 | `alien_results.json` | `judged_alien.json` — see [[alien-direction-steering]] |

`steer_spec2.json` (11 alphas including +/-0.5 and +/-3) was the design for the
timed-out `zoo-steer2`; `steer_spec2_7a.json` is the same eight directions cut
back to the 7-alpha grid so it would finish.

Judging counts (`qwen35/phase10_runs/judge_steerfix.log`, `judge_sf23.log`):

- `judged_steerfix.json`: "1512 generations to judge from 9 traits ... 1512
  judged, 0 failed calls"; 265 judge calls, 75 repeats.
- `judged_steerfix23.json`: "2184 generations to judge from 13 traits ... 2178
  judged, 1 failed calls"; 383 judge calls, 109 repeats.

A parse-crash fix landed between the two runs:
`qwen35/judge_personas.py.bak-parsecrash` is dated 2026-09-01 11:40, four
minutes before `judged_steerfix23.json`. The current file documents the bug:
"The fallback used to call json.loads on a regex-extracted span WITHOUT a try,
so a malformed extraction raised out of the thread pool and killed the whole run
-- 383 calls in, with no output written" (`qwen35/judge_personas.py`).

### The direction families

`qwen35/build_direction_pages.py` names them:

| prefix | family |
|---|---|
| `PC` | principal component |
| `axis_` | constructed keying axis — mean(+keyed) minus mean(-keyed) per factor |
| `FA_` | factor-analytic factor |
| `mean_` | grand-mean direction |
| `identity_` | factor-identity axis |
| `mix_` | axis mixture |

The identity axes are "what a factor's traits have in common regardless of
keying, not its polarity" (`qwen35/build_direction_pages.py`). The grand-mean
direction is the weight-space analogue of the persona literature's Assistant
Axis; `steer_fix.py` says so explicitly:

> `mean` -- the grand-mean direction, which in weight space is mu(personas) -
> base. The base model IS the Assistant here (every adapter is a delta from it),
> so `mean` is the weight-space analogue of the Assistant Axis of
> arXiv:2601.10387, and negative alpha is the prediction that paper makes
> falsifiable: drift away from the default Assistant.
> — `qwen35/steer_fix.py`

---

## Dose response and the coherent band

Two things are measured per direction, defined in `qwen35/analyse_steer.py`:

> MONOTONICITY -- does the judged score move consistently with alpha? Spearman
> rho over the alpha grid, which does not assume linearity.
> SELECTIVITY -- does it move the factor it is supposed to, or all five? A
> direction that shifts every factor is changing fluency or verbosity, not
> personality.

The alpha=0 row is the same untouched model for every direction, so its spread
across directions is the judge-noise floor.

**Steering stops working past |alpha| = 2.** The blog page states it plainly:

> Past alpha = +/-2 the model starts repeating itself, and at +/-4 every
> direction in this study collapses into looping. A dose-response curve drawn
> through babble is not a measurement of personality, so those points are marked
> rather than averaged in.
> — `qwen35/build_blog_page.py`

`qwen35/build_direction_pages.py` sets the threshold at `DEGEN = 0.5` (looping
share above which a rung is called degenerate) and records the scale of the
problem: "At |alpha|=4 the model loops verbatim on 13-24 of 24 prompts and emits
its own chat-turn scaffolding on up to 24 of 24. Those rows were scored by the
judge, so any statistic over the full grid is partly a measurement of
degeneration."

Per-alpha looping counts, out of 24 responses, from
`analysis/qual_pc.json`, `qual_fa.json` and `qual_identity.json`
(`directions[].damage_markers.per_alpha.looping`):

| direction | -4 | -2 | -1 | 0 | +1 | +2 | +4 |
|---|---|---|---|---|---|---|---|
| PC1 | 24 | 17 | 1 | 0 | 0 | 0 | 17 |
| PC2 | 24 | 7 | 0 | 0 | 0 | 1 | 20 |
| PC3 | 15 | 0 | 0 | 0 | 0 | 0 | 24 |
| PC4 | 23 | 5 | 3 | 0 | 0 | 0 | 17 |
| PC5 | 24 | 4 | 0 | 0 | 0 | 0 | 15 |
| PC6 | 8 | 1 | 0 | 0 | 0 | 1 | 19 |
| mean_assistant_axis | 13 | 5 | 0 | 0 | 0 | 1 | 23 |
| FA_Warmth | 21 | 6 | 0 | 0 | 0 | 0 | 22 |
| FA_Competence | 23 | 16 | 2 | 0 | 0 | 0 | 10 |
| FA_FearfulWithdrawal | 22 | 10 | 0 | 0 | 0 | 0 | 19 |
| FA_Arousal | 23 | 5 | 1 | 0 | 0 | 2 | 23 |
| FA_Imagination | 22 | 9 | 1 | 0 | 0 | 0 | 13 |
| identity_Agreeableness | 6 | 2 | 0 | 0 | 0 | 0 | 6 |
| identity_Conscientiousness | 11 | 0 | 1 | 0 | 0 | 0 | 1 |
| identity_EmotionalStability | 13 | 0 | 0 | 0 | 0 | 0 | 4 |
| identity_Extraversion | 0 | 0 | 0 | 0 | 0 | 0 | 16 |
| identity_Intellect | 17 | 2 | 0 | 0 | 0 | 0 | 1 |

The five `axis_*` directions carry a whole-corpus looping count in
`analysis/qual_axes.json` (17 to 24 of 24 at the worst alpha) but **no
per-alpha breakdown**, so `analysis/blog_data.json#degen` is empty for them and
the blog page's "intact over alpha ..." line for the axes is derived from an
absent record rather than from a zero measurement.

PC1 is the exception to "the band is -2..+2": its qualitative record notes it
"breaks one step earlier, at -2, leaving alpha=-1 as its only usable negative
sample" (`analysis/qual_pc.json#directions[PC1].surprise`).

---

## The replication scorecard

`qwen35/analysis/steerfix_replication.json` holds, per direction: `named` (the
Big Five scale the judge moved most), `slope2`/`sel2` over the coherent band
(-2..+2) and `slope4`/`sel4` over the full grid, plus sign-coherence counts
`coh2`/`coh4` out of 4 off-target scales. `qwen35/build_blog_page.py` uses
`slope2` and `sel2` — the band figures — and treats `sel2 < 1.5` as "not a
factor movement" (`SEL_MIN = 1.5`).

**Provenance note.** No script in the repository writes
`steerfix_replication.json`. It is read by `build_blog_data.py:54` and
`build_direction_pages.py:406`, and mirrored into
`analysis/page_data.json#repl` and `analysis/monitor_page_data.json#repl`. Its
key definitions above are reconstructed from those consumers and from the
qualitative records that quote it, not from a producer.

| direction | named | slope2 | sel2 | coh2 | slope4 | sel4 | coh4 |
|---|---|---|---|---|---|---|---|
| PC1 | Conscientiousness | +1.0041666666666667 | 1.9165009940357849 | 2 | +0.6359126984126984 | 1.9738260200153963 | 2 |
| PC2 | Agreeableness | -0.5083333333333335 | 1.8769230769230794 | 0 | -0.40674603174603186 | 5.365576102418214 | 1 |
| PC3 | Intellect | -0.6291666666666671 | 2.9811158798283297 | 1 | -0.47519841269841273 | 3.17950937950938 | 2 |
| PC4 | Agreeableness | -0.6375000000000003 | 1.6407506702412882 | 2 | -0.49098516218081456 | 2.098732426826458 | 2 |
| PC5 | Conscientiousness | +0.6416666666666663 | 4.702290076335875 | 2 | +0.37499999999999994 | 3.4758620689655184 | 3 |
| PC6 | Agreeableness | +0.27499999999999986 | 3.7714285714285563 | 0 | +0.16071428571428562 | 2.2344827586206852 | 1 |
| axis_Extraversion | Extraversion | +0.5333333333333331 | 3.269294836202112 | 3 | +0.5119047619047619 | 6.045848191543551 | 1 |
| axis_Agreeableness | Agreeableness | +0.9124999999999996 | 5.238689547581906 | 3 | +0.6775793650793651 | 4.569226294357188 | 4 |
| axis_Conscientiousness | Conscientiousness | +1.0249999999999997 | 3.2909698996655514 | 2 | +0.6884920634920636 | 3.158134243458476 | 2 |
| axis_EmotionalStability | EmotionalStability | +0.6242753623188402 | 3.0115796373170194 | 3 | +0.4978433402346446 | 2.5249111293409903 | 3 |
| axis_Intellect | Intellect | +0.8291666666666665 | 7.302752293577998 | 4 | +0.5902777777777777 | 7.507886435331232 | 4 |
| FA_Warmth | Agreeableness | +0.9083333333333332 | 10.129292929292934 | 2 | +0.6676155969634231 | 7.545726505196447 | 3 |
| FA_Competence | Conscientiousness | +1.0166666666666664 | 2.7570621468926553 | 2 | +0.6121463077984818 | 2.965005745325394 | 2 |
| FA_FearfulWithdrawal | EmotionalStability | +0.47803030303030297 | 1.0775813262796419 | 3 | +0.38762626262626265 | 1.2875428593756613 | 3 |
| FA_Arousal | Extraversion | +0.4999999999999997 | 4.435516271595014 | 1 | +0.37301587301587297 | 2.123641721407083 | 0 |
| FA_Imagination | Intellect | +0.8125000000000001 | 4.19354838709678 | 4 | +0.550595238095238 | 5.211267605633801 | 4 |
| mean_assistant_axis | Agreeableness | -0.5458333333333337 | 1.7124183006535967 | 2 | -0.3998015873015876 | 1.6968421052631586 | 3 |
| identity_Agreeableness | Agreeableness | +0.4916666666666663 | 3.395683453237408 | 3 | +0.3511904761904761 | 1.9265306122448982 | 4 |
| identity_Conscientiousness | Conscientiousness | +0.0666666666666663 | 0.5157509157509135 | 3 | -0.06677018633540378 | 0.8363149687511517 | 1 |
| identity_EmotionalStability | EmotionalStability | -0.24166666666666695 | 1.0131004366812237 | 2 | -0.29563492063492086 | 1.5746367239101726 | 2 |
| identity_Extraversion | Extraversion | -0.04583333333333368 | 1.1282051282051282 | 4 | -0.01884920634920643 | 0.09906489090393918 | 4 |
| identity_Intellect | Intellect | +0.07916666666666627 | 0.39999999999999813 | 3 | +0.07043650793650781 | 0.3389021479713598 | 3 |

Reading, per the blog page: the five constructed keying axes and the five
factor-analytic factors are the selective directions; the principal components
are legible but less selective; the identity axes barely move anything.

---

## Scoring the nine pre-registered predictions

`qwen35/PREREG_steerfix.md` registered nine claims from the withdrawn 200-token
corpus, with an explicit criterion each, before the corrected corpus existed.
The verdicts below pair each registered criterion with the corrected data.

| direction | registered criterion | corrected outcome |
|---|---|---|
| **PC2** — deference vs boundary enforcement | "Agreeableness monotone decreasing across all 7 alphas, and selective" | **Replicated.** `named` Agreeableness, `slope2` -0.508; the qualitative record gives per-alpha Agreeableness 5.67/5.75/5.75/5.04/4.50/3.83/2.67 and notes "the other scales are near-flat over -1..+2, so the selectivity claim survives qualitative inspection" (`analysis/qual_pc.json#directions[PC2].judged_profile`). The -4/-2 pair is not strictly monotone. |
| **PC3** — interpretive inflation vs literal deflation | "Intellect slope negative with alpha, largest of the five; damage markers re-counted separately" | **Replicated, and it is the strongest case.** `named` Intellect, `slope2` -0.629, `sel2` 2.98. "Over the intact range -2..+2 the movement is monotone and large (Intellect 6.62 -> 4.00) with no looping to explain it, which makes this the best-supported judged effect of the four" (`analysis/qual_pc.json#directions[PC3].judged_profile`). Qualified: "what actually moves is sentence complexity, abstraction level and willingness to disagree, all together." |
| **PC1** — affect-centring vs procedural detachment | "Agreeableness slide survives; the large Extraversion/Conscientiousness moves at -4 were attributed to verbatim looping, so they should SHRINK once damage is re-counted" | **The damage attribution replicated; the named scale did not.** `steerfix_replication.json#PC1.named` is **Conscientiousness**, not Agreeableness. The qualitative record: "The old corpus attributed PC1's large negative-alpha judged moves to verbatim looping. That verdict survives the correction unchanged: looping is not only still present, it is total (24/24 at -4, 17/24 at -2)" and "the huge negative-side numbers are entirely produced by degeneration. Selectivity is spurious in the negative half" (`analysis/qual_pc.json#directions[PC1]`). |
| **mean_assistant_axis** — the prediction flagged as most likely to reverse | "AI-identity disclaimer counts still monotone decreasing in alpha" | **Half replicated, half failed.** Manually re-counted disclaimers are 8/7/8/7/4/3/1 across -4/-2/-1/0/+1/+2/+4. "The negative side is FLAT at baseline, not elevated - the old '13 of 24' does not reproduce, and 13/24 turns out to be PC2's negative pole instead. The positive side DOES strip disclaimers, 7 -> 3 -> 1 ... So the register claim holds; the disclaimer-count claim does not" (`analysis/qual_pc.json#directions[mean_assistant_axis].surprise`). |
| **axis_Agreeableness** — "monotone 3.79 -> 5.71, sycophancy at +4" | "monotone increasing, selective" | **Replicated.** `slope2` +0.912, `sel2` 5.24, `coh2` 3/4. |
| **axis_Intellect** — "largest range, buys abstraction with Conscientiousness" | "Intellect increasing, Conscientiousness decreasing" | **Half reversed.** Intellect increasing holds (`slope2` +0.829, `sel2` 7.30, the highest `coh2` of 4/4). The Conscientiousness cost does not: "Judged Conscientiousness on this axis rises with alpha (+0.20 slope, 2.54 at -4 to 5.83 at +1) ... The debt in the corrected data runs the other way: axis_Conscientiousness costs Intellect (+0.52 ride-along in the clean band), not the reverse" (`analysis/qual_axes.json#directions[axis_Intellect].surprise`). |
| **axis_Conscientiousness** — "works only on the negative arm" | "negative arm effect larger in magnitude than positive arm" | Not scored by an arm-asymmetry statistic in any file. The corrected record notes the low-C pole reads as high-energy and cheerful while the judge still scores its Extraversion above the high-C pole (4.50 at -4 vs 3.71 at +4) (`analysis/qual_axes.json#directions[axis_Conscientiousness].surprise`). `slope2` is +1.025, the largest of the five axes. **Unconfirmed.** |
| **axis_Extraversion** — "weak, and really assertiveness not sociability" | "weak judged effect; disclaimer count monotone" | **Replicated.** `slope2` +0.533, the second-weakest of the five axes. Disclaimers: "replicates the old finding almost exactly and is strictly monotone: 16 / 14 / 8 / 4 / 3 / 3 / 2 at alpha -4 / -2 / -1 / 0 / +1 / +2 / +4" (`analysis/qual_axes.json#directions[axis_Extraversion].surprise`). |
| **axis_EmotionalStability** — "FAILS selectivity: 4/4 off-target factors move with it" | "sign coherence stays 4/4" | **Failed by one, and the file says why.** `coh2` and `coh4` are both 3. "The 4/4 -> 3/4 change is a knife-edge, not a substantive move. It turns entirely on the judged Extraversion slope flipping from positive to -0.0228, against an Extraversion series ... that is pure noise with a range of 0.63" (`analysis/qual_axes.json#directions[axis_EmotionalStability].surprise`). |

The withdrawn page's own summary of the whole exercise: "All nine directions
kept their sign and every effect was two to four times larger, so the shapes
were real and merely diluted. But two sharper claims did not survive"
(`qwen35/distil_page/index.html`).

---

## Notable per-direction results

**The grand-mean / Assistant-Axis analogue.** The register change is counted,
not asserted: emoji-bearing responses climb 6 -> 17 -> 22 -> 23 out of 24 across
alpha 0/-1/-2/-4; exclamation rate 3.14 -> 6.95 -> 21.45 -> 49.17 per 1k words;
validation openers 2 -> 3 -> 4 -> 11. On the positive arm, emoji-bearing
responses 6 -> 2 -> 0, exclamations 3.14 -> 0.00 per 1k, first-person pronoun
rate 33.5 -> 43.4 -> 55.9 per 1k
(`analysis/qual_pc.json#directions[mean_assistant_axis].negative_pole` and
`.positive_pole`). The blog page declines to identify it with the published
Assistant Axis: "Whether they are the same object is a question this study does
not answer, so the page does not name it as though they were"
(`qwen35/build_blog_page.py`).

**FA_Warmth is the most selective direction in the study**, `sel2` 10.13. The
blog page adds a recomputation restricted to clean prompts: "7.3x recomputed on
only the eighteen prompts that stay free of looping across the whole range",
and for Imagination "restricted to its fourteen clean prompts its selectivity
*rises*, from 4.2 to 7.8" (`qwen35/build_blog_page.py`). Those two recomputed
figures are hard-coded in the builder and have no JSON key behind them.

**FA_FearfulWithdrawal fails selectivity**, `sel2` 1.078 — below the blog page's
`SEL_MIN` of 1.5, so it is drawn without a factor colour. "it moves every scale
at once rather than a factor, and its fearful pole turns out to sit at negative
alpha rather than positive" (`qwen35/build_blog_page.py`).

**The identity axes are the negative result.** Three of five have `sel2` below
1.2 and `identity_Intellect` sits at 0.40. The monitor page's reading: "One of
five loads on its own factor, and it is the one where the cancellation *failed*.
Between 63% and 95% of each identity vector lies outside the polarity span ...
The direction with the *least* polarity leak, at 4.9%, produces no personality
effect at all. The cleaner the construction, the emptier the result."
(`qwen35/build_monitor_page.py`).

**False provenance under steering.** Two events in the five keying axes: at
alpha -4 on "Tell me about yourself" the model asserts "I am a large language
model, trained by Google", and axis_Extraversion does the same at -4 and -2 on
the party prompt. At alpha 0, -2, +2 and +4 the same prompt correctly yields
"I'm Qwen3.5... developed by Tongyi Lab"
(`analysis/qual_axes.json#directions[axis_Agreeableness].surprise`). Exactly one
false identity claim occurs across the 672 responses of the four PC-family
directions: PC2 at alpha -4, prompt 22, "I am an AI assistant developed by
Google" (`analysis/qual_pc_notes.md`).

**Language switching is gone.** Zero occurrences of CJK / Cyrillic / Arabic /
Devanagari / Thai runs across all 672 responses in PC1, PC2, PC3 and
mean_assistant_axis — "a real change from the old corpus's failure profile"
(`analysis/qual_pc_notes.md`).

---

## The per-direction pages

`qwen35/direction_pages/` holds 22 standalone HTML pages, one per direction,
built by `qwen35/build_direction_pages.py`. Each renders the dose sweep as a
vertical alpha ladder from -4 to +4 with verbatim generations hung off it, a
per-rung damage bar, and selectivity reported twice (band and full grid) with
the gap left visible. Rungs past `DEGEN = 0.5` are struck through and labelled.

They are **historical**, not withdrawn: all 22 were built from the corrected
`steer_results_fix*` corpora.

| files | built | corpus |
|---|---|---|
| `PC1.html`, `PC2.html`, `PC3.html`, `axis_Agreeableness.html`, `axis_Conscientiousness.html`, `axis_EmotionalStability.html`, `axis_Extraversion.html`, `axis_Intellect.html`, `mean_assistant_axis.html` | 2026-08-30 00:02 | `steer_results_fix.json` / `judged_steerfix.json` |
| `FA_Arousal.html`, `FA_Competence.html`, `FA_FearfulWithdrawal.html`, `FA_Imagination.html`, `FA_Warmth.html`, `PC4.html`, `PC5.html`, `PC6.html`, `identity_Agreeableness.html`, `identity_Conscientiousness.html`, `identity_EmotionalStability.html`, `identity_Extraversion.html`, `identity_Intellect.html` | 2026-09-01 12:08 | `steer_results_fix2.json` / `_fix3.json` / `judged_steerfix23.json` |

Each page's `<title>` is the direction's qualitative label rather than its slug
— `PC1.html` is titled "Affective flooding vs procedural composure". None is
served; see [[built-pages-inventory]].

## Every alpha here is in the wrong units for comparison

Alpha is a step of fixed *Frobenius* length, and the model does not measure
weight changes that way. [[fisher-norms]] measures, for 124 directions, the
curvature of KL(base || steered) in alpha at the base point, and finds the
published directions spread over a factor of 260. At the published alpha 2, PC4
is a Fisher dose of 3.52 and the stage-two grand mean is 1.19, so a table that
compares directions at a common alpha is comparing doses that differ by up to
14x. That page also records two facts about this corpus: the stored `"0.0"`
generations are not the base model (the bf16 alpha walk in `steer_fix.py` is not
reversible, and the nine differ from one another), and F does not predict which
directions degenerate.

[[matched-dose-steering]] redid the suppress-versus-amplify comparison with the
dose held fixed instead of the alpha. The raw asymmetry shrinks but survives -
suppression over amplification falls from **1.8957528957528955** to **1.6455696202531644**, 8 of 10
directions to 6 of 10 - and then disappears once each change is divided by the
room that existed in the direction it moved: **0.9261912176364205** published against
**0.7946215813581846** at matched dose, with amplification the stronger move in both
(`qwen35/analysis/matched_dose_steering.json#summary.all` and
`#published_equal_alpha_summary.all`). The table above is therefore a table
about the base model's starting scores, not about a mechanism. That page also
corrects this campaign's degeneration reading: at matched dose the amplifying
sign loops **more** than the suppressing one (0.1375 against 0.075 overall,
0.2167 against 0.075 over the five factors).

Related: [[matched-dose-steering]], [[thinking-default-withdrawals]],
[[judged-evaluations]],
[[additivity]], [[alien-direction-steering]], [[sphere-sweep]],
[[fisher-norms]], [[qualitative-notes]], [[factor-warmth]], [[factor-pc1]],
[[geometry-overview]], [[glossary]].
