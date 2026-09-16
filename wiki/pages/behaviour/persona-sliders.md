---
title: Persona sliders - LoRAs trained to an activation-space target
summary: Thirteen rank-64 LoRAs were trained with SliderSpace's cosine objective to produce specified directions in the model's layer-16 residual stream; they hit their targets on held-out prompts at 0.77 to 0.94 where the trait's own stage-one adapter reaches 0.12 to 0.63, yet they sit at cosine 0.003 to 0.022 with that adapter in weight space and carry a mean of under 4 per cent of their norm inside the factor chart (2 to 9 per cent), they move the judged persona in the right direction on 10 of 10 traits but about half as far as stage one with almost none of its degeneration, and the hole slider is indistinguishable from its two shuffled controls.
status: current
sources:
  - qwen35/persona_sliders.py
  - qwen35/build_slider_targets.py
  - qwen35/cross_gram_sliders.py
  - qwen35/analyse_persona_sliders.py
  - qwen35/analysis/persona_sliders.json
  - qwen35/analysis/slider_targets.npz
  - qwen35/analysis/slider_targets_meta.json
  - qwen35/analysis/slider_train.json
  - qwen35/analysis/slider_means.npz
  - qwen35/analysis/slider_cross_gram.json
  - qwen35/analysis/slider_generations.jsonl
  - qwen35/results/cross_gram_full_root_x_pc-qwen35-adapters_sliders.npz
  - qwen35/phase10_runs/sliders_behave.json
  - qwen35/phase10_runs/judged_sliders.json
  - qwen35/phase10_runs/sliders_train.log
  - qwen35/phase10_runs/sliders_gram.log
  - qwen35/phase10_runs/sliders_behave.log
  - qwen35/phase10_runs/sliders_behave_stage1.log
  - qwen35/analysis/actspace_geometry.json#windows.resp.primary
  - qwen35/analysis/alien_fa.json#alien_fa
  - qwen35/analysis/lora_a_identity.json#sets.stage1_seed0.A_pairwise
  - qwen35/analysis/fisher_norms.json#comparisons.full_range
  - qwen35/analysis/actspace_geometry_fa.json#windows.resp
  - qwen35/analysis/alien_match_fa.json
  - qwen35/phase10_runs/judged_100.json
  - https://arxiv.org/abs/2502.01639
last_verified: 2026-09-10
tags: [behaviour, actspace, activations, persona-vectors, sliders, hole, factor-chart, controls]
---

# Persona sliders - LoRAs trained to an activation-space target

This is experiment **S1** of [[paper-reading-2026-09-09]], run on 2026-09-09/10.
Samuel authorised it and a $45 cap on 2026-09-09.

## The idea

SliderSpace (Gandikota et al., arXiv:2502.01639) trains a LoRA not to fit data
but to make a specified change in an embedding space. Its Equation 5 is
`L = sum_i 1 - cos(dphi_i, v_i)`, where `dphi_i` is the change the adapter makes
in CLIP space and `v_i` is a target direction. Nothing in that construction is
specific to images: swap CLIP for the model's own residual stream and this
project's trait vocabulary supplies the targets.

A **persona vector** `P_t` is the base model's mean residual-stream activation
with trait `t`'s constitution as system prompt, minus the same mean with no
system prompt, over 64 fixed questions ([[actspace-persona-vectors]],
[[actspace-overview]]). At the project's primary layer -- **layer 16 of 0..32,
response-token window** -- the 134 persona vectors reproduce most of the
adapters' weight-space geometry (Pearson **0.705**,
`qwen35/analysis/actspace_geometry.json#windows.resp.primary.r_centred`).

So: train one LoRA per target with SliderSpace's objective, in the zoo's own
LoRA-A frame, and ask three questions. Does the slider produce its target on
prompts it never trained on? Where does a direction defined in activations land
in weight space? And does it move the judged persona -- including the
[[hole-words-factor-chart|hole]], where the weight-space merge gave 3 of 5 signs
at r 0.744 and its controls did about as well ([[alien-direction-factor-chart]])?

## The design

**Targets.** Thirteen directions at layer 16, response window, built by
`qwen35/build_slider_targets.py` into `qwen35/analysis/slider_targets.npz`:

- **ten traits**, two per Big Five factor, one per pole -- `extraverted` /
  `introverted`, `warm` / `cold`, `organized` / `disorganized`, `relaxed` /
  `anxious`, `intellectual` / `unintellectual`. All ten are among the 100 traits
  in `qwen35/phase10_runs/judged_100.json`, so each has a stage-one behavioural
  reference.
- **`hole_fa`**, the activation-space image of the factor chart's deepest hole:
  the same 134 coefficients that `qwen35/analysis/alien_fa.json#alien_fa.coeffs`
  gives over the adapters, applied to the 134 trait-centred persona vectors.
  Those coefficients sum to `-2.7755575615628914e-17`
  (`#alien_fa.coef_sum`), so the image is a contrast.
- **`hole_shuf1`, `hole_shuf2`**, the same coefficients permuted across traits --
  the control `qwen35/analyse_actspace.py` already uses for the transplanted hole.

Every target is **trait-centred** (`X_c = X - mean_t X`), the centring
`analyse_actspace.py` applies before every cross-space cosine, so that "was the
model steered at all" cannot carry the number.

**The held-out split is the halves.** `qwen35/act_space.py` stores prompts 0-31
and 32-63 separately. Targets are built from **half 0 only** and the sliders
train on those 32 prompts; **half 1 is never seen by the optimiser**. The
half-to-half cosine of a persona vector at this layer is `0.959`
(`qwen35/analysis/actspace_geometry.json#windows.resp.curve[16].floor`), so the
split tests the slider, not the target.

**The adapter.** Rank 64, `lora_alpha` 128, plain LoRA (effective scale 2.0) on
exactly the zoo's 248 modules -- the zoo's own configuration
([[stage-one-training-config]]). LoRA-A is copied from the zoo adapter `bold`
the way `qwen35/sft_rewardhacks.py` copies it, so the delta lives in the same
random 64-dimensional input window as all 134 stage-one adapters and the exact
cross-Gram is at full strength. Unlike the zoo, **A is then frozen**. The zoo did
train A but barely moved it: two adapters that started from the same A end at
cosine `0.9999687370571562` with each other on
`model.layers.0.linear_attn.in_proj_a`
(`qwen35/analysis/lora_a_identity.json#sets.stage1_seed0.A_pairwise[0].cos`,
maximum absolute element difference `0.00036777835339307785`). Only B is
trained, from zero, for **120 steps** at lr 1e-3 (AdamW, 20 warmup, cosine
decay); the probe run showed the objective saturating by about step 50.

**The loss.** `1 - cos(s, u)` is undefined at `s = 0`, which is where B starts.
What is implemented is

```
m    = sqrt(|s|^2 + eps^2),      eps = 0.05 * anchor
loss = 1 - <s,u>/m  +  0.25 * (m/anchor - 1)^2
```

smooth everywhere, with a gradient at `s = 0` along `+u`. The second term is the
norm penalty the design allowed and is what gives each slider a **natural
scale**: `anchor` is the norm of that trait's own centred persona vector, so the
slider is asked to make the size of change the constitution makes as a system
prompt. The two shuffled controls are anchored to the **hole's** norm, not their
own, because a permuted contrast is smaller (`0.54` and `0.42` against `1.02`,
`qwen35/analysis/slider_targets_meta.json#magnitude_anchors`) and dosing the
controls below the hole would confound the one comparison they exist for.

**The shift, and why it is teacher-forced.** Sampling a response per optimizer
step would make the loss non-differentiable and cost a `generate()` per step.
Instead the base model's own greedy responses to the 64 probe prompts are
generated once (96 new tokens, `enable_thinking=False`, `act_space.py`'s
settings) and then teacher-forced: each step forwards prompt+response with the
slider on, takes the layer-16 mean over the response positions, and subtracts the
same quantity with the slider off. **The objective is therefore the shift on a
fixed continuation.** The free-running shift -- `act_space.py`'s own
construction, and the one the targets were built from -- is an *evaluation*,
never optimised. Both are reported and only the second is comparable to the
persona vectors.

**A layer-16 target can only train blocks 0-15.** Layer-16 activations do not
depend on blocks 16-31, so those modules receive exactly zero gradient. All 248
modules are still created and saved -- the module set has to match the zoo's or
the exact cross-Gram refuses the pair -- and the check confirms it: B is
non-zero on exactly **124 of 248** modules for all thirteen sliders and
**100 per cent** of each slider's Frobenius norm lies in blocks 0-15
(`qwen35/analysis/persona_sliders.json#checks.all_sliders_confined_to_low_blocks`).
The ceiling this imposes on any cosine with a full-depth trait adapter is the
share of that adapter that lies in blocks 0-15: mean **0.692945** over the 134,
`0.685947` to `0.701793` for the ten traits used here
(`#weight_space.stage1_low_block_norm_fraction`, `#weight_space.sliders.<t>.ceiling_from_depth`).

## 1. Activation space: the sliders hit their targets, and beat the adapters

Cosine between each slider's **free-running** mean shift on the held-out prompts
32-63 and its target, against three comparators: the same trait's stage-one
adapter measured the same way from `qwen35/analysis/actspace_means_adapters.npz`,
and the target's own half-1 replicate, which is the ceiling.

| target | slider, half 1 | stage-one adapter, raw | stage-one adapter, centred | persona vector half-1 replicate | shift norm / anchor | argmax over the 13 |
|---|---|---|---|---|---|---|
| `extraverted` | **+0.9084** | +0.1788 | +0.3938 | +0.9673 | 1.170 | `extraverted` |
| `introverted` | **+0.7664** | +0.1232 | +0.4637 | +0.9046 | 1.338 | `introverted` |
| `warm` | **+0.9292** | +0.5398 | +0.7825 | +0.9695 | 1.238 | `warm` |
| `cold` | **+0.8989** | +0.4637 | +0.7695 | +0.9657 | 1.225 | `cold` |
| `organized` | **+0.9398** | +0.4827 | +0.7758 | +0.9864 | 1.224 | `organized` |
| `disorganized` | **+0.9259** | +0.5849 | +0.6364 | +0.9638 | 1.088 | `disorganized` |
| `relaxed` | **+0.8538** | +0.3339 | +0.5873 | +0.9415 | 1.240 | `relaxed` |
| `anxious` | **+0.8793** | +0.2482 | +0.4020 | +0.9448 | 1.309 | `anxious` |
| `intellectual` | **+0.9244** | +0.3213 | +0.7141 | +0.9708 | 1.255 | `intellectual` |
| `unintellectual` | **+0.8844** | +0.6326 | +0.6398 | +0.9646 | 1.154 | `unintellectual` |
| `hole_fa` | **+0.8259** | - | - | - | 1.311 | `hole_fa` |
| `hole_shuf1` | **+0.8136** | - | - | - | 1.267 | `hole_shuf1` |
| `hole_shuf2` | **+0.8189** | - | - | - | 1.361 | `hole_shuf2` |

Keys: `qwen35/analysis/persona_sliders.json#activation.<name>.half1.cos_own_target`,
`.mag_ratio_vs_anchor`, `.heldout_argmax_target`; comparators
`#design.comparators.<trait>.adapter_half1_raw_vs_target`,
`.adapter_half1_centred_vs_target`, `.persona_half1_centred_vs_target`.

**Every slider's own target is the argmax over all thirteen**, so the directions
are selective and not a shared "something changed" component. The training
objective saturates: teacher-forced cosine `+1.0000` on the training half for all
thirteen, `+0.8730` to `+0.9659` on the held-out half
(`#training.per_slider.<name>.teacher_forced.cos_train`, `.cos_heldout`). The
free-running numbers above are lower than the teacher-forced ones and are the
honest ones, because they are the construction the targets came from.

The `mag_ratio_vs_anchor` column says every slider **overshoots** the size of
shift the constitution makes, by 9 to 36 per cent, when it is free to write its
own continuation.

The same statistic `analyse_actspace.py` uses for the transplanted hole -- angle
from a direction to the nearest of the 134 trait-centred persona vectors, on the
held-out half (`#activation_nearest_trait.per_slider`):

| slider | nearest trait | degrees |
|---|---|---|
| trait sliders (10) | its own trait, all ten | `21.7658` (`organized`) to `44.8775` (`introverted`) |
| `hole_fa` | `melancholy` | **55.1032** |
| `hole_shuf1` | `pleasant` | 49.7634 |
| `hole_shuf2` | `uncreative` | 56.6268 |

against a permuted-coefficient null over 500 draws with median **47.8617** and
95th percentile **61.7993** (`#activation_nearest_trait.permuted_coefficient_null_median`,
`.permuted_coefficient_null_95pct`). Every trait slider's nearest neighbour is
its own trait -- the naming works. The hole slider is farther from every trait
than the null median, but so is one of its two shuffled controls, and none of the
three clears the 95th percentile. Compare the published transplant, which put the
PC-chart hole at `47.834` against a null median of `47.748`
(`qwen35/analysis/actspace_geometry.json#windows.resp.primary.hole_deg`,
`.hole_null_median`).

## 2. Weight space: the sliders are orthogonal to the zoo

The exact cross-Gram was computed twice, once by the project's own unmodified
`qwen35/cross_gram_full_on_modal.py` (134 x 13,
`qwen35/results/cross_gram_full_root_x_pc-qwen35-adapters_sliders.npz`) and once
by `qwen35/cross_gram_sliders.py`, which adds the depth split. The two agree
exactly: maximum relative difference **0.0**
(`#checks.cross_gram_max_rel_diff_between_implementations`).

| target | cos with own stage-one adapter | ceiling from depth | cos restricted to blocks 0-15 | rank of own adapter among 134 | nearest adapter of the 134 | chart length / norm | cos with the `alien_fa` merge |
|---|---|---|---|---|---|---|---|
| `extraverted` | +0.0189 | 0.6930 | +0.0273 | 17 | `assertive` -0.0266 | 0.0372 | +0.0055 |
| `introverted` | +0.0057 | 0.6893 | +0.0082 | 77 | `guilty` +0.0213 | 0.0304 | +0.0058 |
| `warm` | +0.0025 | 0.6990 | +0.0036 | 124 | `inarticulate` +0.0357 | 0.0492 | +0.0161 |
| `cold` | +0.0025 | 0.7018 | +0.0036 | 96 | `pleasant` -0.0250 | 0.0250 | +0.0075 |
| `organized` | +0.0222 | 0.6922 | +0.0320 | 11 | `pleasant` -0.0339 | 0.0368 | +0.0179 |
| `disorganized` | +0.0217 | 0.6948 | +0.0312 | 9 | `imaginative` +0.0272 | 0.0312 | +0.0014 |
| `relaxed` | +0.0058 | 0.6859 | +0.0085 | 79 | `inarticulate` +0.0247 | 0.0301 | -0.0001 |
| `anxious` | +0.0180 | 0.6926 | +0.0260 | 17 | `high_strung` +0.0290 | 0.0381 | +0.0162 |
| `intellectual` | +0.0085 | 0.7016 | +0.0121 | 63 | `pleasant` -0.0302 | 0.0383 | +0.0042 |
| `unintellectual` | +0.0061 | 0.6970 | +0.0088 | 56 | `guilty` +0.0181 | 0.0188 | +0.0060 |
| `hole_fa` | - | - | - | - | `practical` +0.0584 | 0.0925 | **+0.0541** |
| `hole_shuf1` | - | - | - | - | `guilty` +0.0304 | 0.0397 | +0.0146 |
| `hole_shuf2` | - | - | - | - | `practical` +0.0236 | 0.0291 | +0.0095 |

Keys: `#weight_space.sliders.<name>.cos_with_own_stage1_adapter`,
`.ceiling_from_depth`, `.cos_with_own_stage1_adapter_low_blocks`,
`.own_stage1_adapter_rank_by_abs_cos`, `.nearest_adapters[0]`,
`.chart_frac_of_norm`, `.cos_with_alien_fa_merge`.

**This is the negative result of the experiment, and it is not small.**

- A slider's cosine with the stage-one adapter for the *same trait* runs
  `0.002506` to `0.022155`, mean `0.011189`. Over all 134 x 13 pairs the cosine
  runs `-0.049872` to `+0.058450` with mean `+0.004863`. Nothing in the zoo is
  more than six hundredths of a radian's worth of alignment away from orthogonal
  to any slider.
- The trait's own adapter is not even the nearest one. Its rank by absolute
  cosine among the 134 runs from **9** to **124** -- consistent with chance.
- The depth confinement does **not** explain this. The ceiling is `0.6859` to
  `0.7018`, and restricting the cosine to blocks 0-15 only rescales it by that
  factor: `+0.0036` to `+0.0320`.
- The sliders barely intersect the five-factor chart at all. Chart length over
  Frobenius norm is `0.018755` to `0.092450`, a **mean of `0.038167`** -- under 4
  per cent on average, and never above 10 -- against a mean of **`0.5784887830866848`** for a stage-one adapter
  (`qwen35/analysis/actspace_geometry_fa.json#windows.resp.weight_chart_frac_of_norm_mean`).
  Their chart coordinates are therefore a small residue of a large vector and
  should not be read as positions.

The one place a signal survives is the hole. `hole_fa`'s cosine with the
`alien_fa` weight-space merge, `+0.0541`, is 3.7 times `hole_shuf1`'s `+0.0146`
and 5.7 times `hole_shuf2`'s `+0.0095`, and it is the largest single entry in the
whole 134 x 13 cosine matrix. So a LoRA trained only on the *activation image* of
the hole does lean, very slightly, towards the *weight-space* hole direction it
was derived from. At `0.054` that is a direction of travel, not a location.

For completeness, the sliders' Frobenius norms are large -- `14.4931` to
`19.2810` (`#training.per_slider.<name>.delta_frobenius`) against a stage-one
adapter's mean of `1.6157416444226869` (`qwen35/analysis/fa_chart_summary.json`,
[[factor-chart]]). **That is not a dose comparison.**
[[fisher-norms]] measured that unit-Frobenius directions span a factor of
`259.924881015629` in the model's own metric
(`qwen35/analysis/fisher_norms.json#comparisons.full_range`). The dose that
means something here is the activation-shift ratio in the table above, which is
1.09 to 1.36. `fisher.py` was **not** run on the sliders.

## 3. Behaviour: right direction, about half as far, almost no damage

The 24-prompt battery, greedy, 512 new tokens, `enable_thinking=False`, one
prompt at a time -- `oct_stage2.py`'s `eval_personas` harness at
`steer_fix.py`'s token cap. Base, the thirteen sliders and the ten matching
stage-one adapters all went through the **same** harness, in two containers whose
base conditions came out byte-identical on all 24 prompts. Judged blind by
`qwen35/judge_personas.py` (claude-sonnet-4.5 via OpenRouter, 576 records, **0
failed calls**, repeat reliability r `0.757` to `0.916` over 28 repeats,
`qwen35/phase10_runs/judged_sliders.json`).

Base absolute scores, 1-7: Extraversion `4.250`, Agreeableness `4.750`,
Conscientiousness `5.708`, Emotional Stability `4.875`, Intellect `5.667`
(`#behaviour.profile.base`).

**Own-factor movement**, signed so that positive is the trait's own pole
(`#behaviour.own_factor_movement`):

| trait | factor | keyed | slider | stage-one, same run (512 tok) | stage-one, `judged_100.json` (200 tok) |
|---|---|---|---|---|---|
| `extraverted` | Extraversion | + | +0.417 | +2.000 | +2.000 |
| `introverted` | Extraversion | - | +0.125 | +0.792 | +0.750 |
| `warm` | Agreeableness | + | +0.792 | +1.458 | +1.375 |
| `cold` | Agreeableness | - | +0.542 | +0.625 | +0.292 |
| `organized` | Conscientiousness | + | **+0.583** | +0.083 | +0.500 |
| `disorganized` | Conscientiousness | - | +1.000 | +2.583 | +2.292 |
| `relaxed` | EmotionalStability | + | +0.042 | +0.042 | +0.138 |
| `anxious` | EmotionalStability | - | +0.375 | +1.125 | +0.819 |
| `intellectual` | Intellect | + | **+0.333** | +0.000 | +0.250 |
| `unintellectual` | Intellect | - | +0.708 | +1.958 | +1.542 |

All ten sliders move their own factor in the **right direction**. The mean move
is `0.491667` against stage one's `1.066667` in the same run, so a slider is
worth about half a stage-one adapter on its own dial. Two sliders beat their
adapter, and they are the two where the adapter did essentially nothing:
`organized` (+0.583 against +0.083) and `intellectual` (+0.333 against +0.000).

**Damage.** Loop rate, the definition every steering page uses (a 10-word window
repeating four times, `qwen35/analyse_alien_steer.py:looping`), and mean response
length in characters (`#behaviour.loop_rate`, `#behaviour.mean_chars`):

| | loop rate, mean | loop rate, max | mean chars |
|---|---|---|---|
| base | 0.000 | 0.000 | `1828.6667` |
| the 13 sliders | **`0.003205`** | `0.041667` (`disorganized`) | `2133.5321` |
| the 10 stage-one adapters | `0.1625` | `0.291667` (`anxious`, `unintellectual`) | `1599.3833` |

Twelve of the thirteen sliders loop on none of their 24 responses; `stage1_warm`
collapses to `682.875` characters and `stage1_extraverted` to `782.875`. **The
sliders buy their smaller effect with almost none of the degeneration**, and they
lengthen responses rather than truncating them.

`judged_100.json`'s stage-one rows were generated at **200** new tokens
(`qwen35/oct_stage2.py`, `eval_personas`) and everything here at 512, which is
why the ten adapters were regenerated inside this run; the two stage-one columns
above agree in sign on 9 of 10 -- `intellectual` is exactly `+0.000` in the
512-token run and so has no sign -- and in magnitude to within about 0.4.

## 4. The hole: buried in this frame

`hole_fa` is a coherent, non-degenerate condition -- 0 per cent looping, `2140.625`
mean characters -- and its judged profile moves: Conscientiousness `+0.625`,
Intellect `-0.417`, Agreeableness `+0.167`, Emotional Stability `-0.125`,
Extraversion `-0.042` (`#behaviour.delta_vs_base.slider_hole_fa`). Read alone it
looks like a character.

It is not distinguishable from its controls.

The pre-registered comparator is the **chart's prediction for this direction**,
`qwen35/analysis/alien_match_fa.json#pred` -- the weight-space merge's
coordinates on the five named Big Five axes, computed from weights alone before
any text existed. That is the same `pred` [[alien-direction-factor-chart]] scored
the merge against, so scoring the sliders against it is like for like.

| | Extraversion | Agreeableness | Conscientiousness | EmotionalStability | Intellect | signs | r |
|---|---|---|---|---|---|---|---|
| the merge, for reference | +0.083 | -1.333 | +2.667 | -1.083 | -0.708 | 3 / 5 | `0.7438221950472093` |
| `hole_fa` | -0.042 | +0.167 | +0.625 | -0.125 | -0.417 | **5 / 5** | **`+0.8614`** |
| `hole_shuf1` | +0.167 | +0.667 | +0.000 | +0.000 | -0.625 | 2 / 5 | `+0.4177` |
| `hole_shuf2` | -0.167 | +0.125 | +0.250 | -0.208 | -0.458 | **5 / 5** | `+0.8245` |

(`#behaviour.hole.<name>.obs`, `.signs_vs_merge_pred`, `.r_vs_merge_pred`;
merge row `#behaviour.hole._merge_reference`, from
`qwen35/analysis/alien_match_fa.json`.)

So the trained hole slider matches the weight chart's prediction for the hole
**better than the merge itself did** -- 5 of 5 signs at r `+0.8614` against 3 of
5 at `0.7438221950472093`. That is a real and slightly surprising positive: the
activation image of the hole, trained into a LoRA, produces behaviour in the
direction the weight-space chart said this direction would go.

And then the control does the same thing. `hole_shuf2` also gets 5 of 5 at
`+0.8245`. That is precisely the failure [[alien-direction-factor-chart]] already
recorded for the merge -- "the r for the unnamed direction is not evidence that
the chart predicted anything direction-specific" -- reproduced here with a
different construction. One shared movement, Conscientiousness up and Intellect
down, carries the correlation for both.

The profiles say the same. Pairwise Euclidean distance between the three judged
five-scale profiles (`#behaviour.hole._pairwise`): `hole_fa` to `hole_shuf2`
**`0.4082`**, `hole_fa` to `hole_shuf1` `0.8620`, `hole_shuf1` to `hole_shuf2`
`0.7336`. **The hole is closer to one of its own shuffled controls than the two
controls are to each other.**

Scored instead against each slider's *own* chart coordinates the numbers are
`hole_fa` 3 of 5 at `+0.0459`, `hole_shuf1` 0 of 5 at `-0.3696`, `hole_shuf2` 1
of 5 at `-0.6230` (`#behaviour.hole.<name>.signs_vs_own`, `.r_vs_own`). Those are
recorded for completeness and are **not** the test: a slider carries 2 to 9 per
cent of its norm inside the chart (section 2), so its own chart coordinates are a
residue.

**The reading.** S1 was pre-registered as the test that could revive or bury the
hole, on the argument that a direction *defined* in activation space cannot fail
the transplant test by construction. The slider does reach its activation target
(+0.8259 held out) and does sit farther from every trait than the null median
(55.10 against 47.86). What it does not do is behave differently from a direction
built by permuting the same coefficients -- and the one prediction it satisfies
handsomely, the chart's, is satisfied just as handsomely by a control. On this
evidence the hole **stays buried**: it is a property of the adapter cloud's
geometry, as [[actspace-persona-vectors]] already concluded from the transplant,
and defining it in the space where the transplant failed does not make a
character out of it.
The one crack is the weight-space lean (`+0.0541` against `+0.0146` and
`+0.0095`), which says the activation image and the weight-space merge are not
unrelated -- only that whatever relates them is worth about five hundredths.

## The headline

**A direction that is right in activation space can be almost orthogonal, in
weight space, to the adapter that was trained for the same word.** The slider
reproduces `warm`'s persona vector at +0.9292 where `warm`'s own stage-one
adapter reaches +0.7825, and sits at cosine +0.0025 with that adapter. This is
the third independent measurement in this project of the same phenomenon:

- two seeds of the same trait: Frobenius cosine **`+0.01806099632415457`**, but
  top left singular directions agreeing at |cos| `0.7701377220108219`
  ([[column-space-structure]]);
- the same trait's stage-one and stage-two deltas: **`+0.0002`** in weight
  coordinates against **`+0.530`** in activation space ([[stage-two-exploration]]);
- and now an activation-defined slider against its own trait adapter:
  **`+0.011189`** mean, against a held-out activation cosine of `0.766` to `0.940`.

Weight-space cosine between two objects that do the same thing is close to zero
unless they share an initialisation *and* a training procedure. Sharing LoRA-A --
which is what makes the exact cross-Gram meaningful at all -- is not sufficient.
That is a caution for every weight-space geometry in this project: it measures the
arrangement of *one* family of objects produced by *one* procedure, and a
different procedure aiming at the same targets lands somewhere else entirely.

## What this establishes and what it does not

It establishes, on thirteen directions and one model, that a LoRA can be trained
to a specified residual-stream direction in about 700 GPU-seconds, that the
direction generalises to prompts it never saw, that the result is selective, and
that it steers behaviour in the right direction with far less degeneration than
the DPO adapter for the same word.

It does **not** establish:

- that sliders are better characters. The behavioural comparison is at each
  object's own natural scale, **not** at a matched Fisher dose; `fisher.py` was
  not run on the sliders, and [[fisher-norms]] is the reason Frobenius norm
  cannot stand in for dose.
- anything about depth. The loss is at layer 16 only, so every slider is confined
  to blocks 0-15 by construction. A slider trained against targets at several
  depths is a different object and was not built.
- a null for the hole slider. There are **two** shuffled controls, not five
  hundred; the 500-draw permutation null quoted in section 1 is over
  *transplanted* coefficient contrasts, not trained sliders, and is context
  rather than a matched null.
- that the objective is SliderSpace's Equation 5 verbatim. It is that cosine
  term, smoothed at zero, plus a magnitude term, on the batch mean rather than
  per sample, and on a teacher-forced continuation rather than a sampled one.
- that 120 steps is optimal. It was chosen from a 60-step probe on `warm`
  (`qwen35/analysis/slider_probe.json`) where the objective had already
  saturated; no sweep was run.

## Cost

Training: one A100-40GB, **`9167.4`** seconds for all thirteen including the base
collection (`qwen35/analysis/slider_train.json#seconds`), 682 to 775 s per
slider. Behaviour: two A100-40GB containers, `8462.087107896805` s (stage one)
and `14064.209661722183` s (sliders)
(`qwen35/phase10_runs/sliders_behave.json#seconds`). Cross-Gram: two CPU
containers, about 5 minutes. Probe and one aborted launch: about 12 minutes.
The three GPU jobs alone are `31693.72` s = **`8.8038` GPU-hours = $18.49** at
the meter's `$2.10` per GPU-hour (`qwen35/zoo40_meter.sh`, [[costs]]); with the
probe (`311.96826457977295` s,
`qwen35/analysis/slider_probe.json#seconds`), the aborted 250-step launch and the
two CPU cross-Gram containers it is about **`9.14` GPU-hours = $19.19**, plus 101
judge calls at about $1-2 by the rates in [[costs]] (not itemised against the
OpenRouter bill). Against the $45 Samuel authorised and the `$37-40`
[[paper-reading-2026-09-09]] estimated. Note that the meter in
`qwen35/zoo40_meter.sh` counts every container in the workspace and four sibling
experiments were running, so the meter's own reading cannot attribute this run;
these figures come from the jobs' recorded `seconds`.

Related: [[paper-reading-2026-09-09]], [[actspace-persona-vectors]],
[[actspace-overview]], [[hole-words-factor-chart]],
[[alien-direction-factor-chart]], [[column-space-structure]],
[[stage-two-exploration]], [[fisher-norms]], [[steering-results]],
[[factor-chart]], [[judged-evaluations]], [[costs]].
