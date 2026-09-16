---
title: The reward hacker in column space
summary: The last representation the reward-hack arms had not been measured in, and the clearest null of the set - 1.9 percent of the hack arm's top-8 output energy lies in a trait adapter's column space, against 13.2 percent for two unrelated zoo traits and 58.5 percent for the same trait at a different seed; against the output subspace every zoo adapter shares it reads 0.028 where a held-out trait adapter reads 0.378, and training moves it further out at every checkpoint.
status: current
sources:
  - qwen35/column_space_sorh_on_modal.py
  - qwen35/analyse_column_space_sorh.py
  - qwen35/results/column_space_sorh.npz
  - qwen35/phase10_runs/colspace_sorh.log
  - qwen35/analysis/column_space_sorh.json#meta
  - qwen35/analysis/column_space_sorh.json#checks
  - qwen35/analysis/column_space_sorh.json#null
  - qwen35/analysis/column_space_sorh.json#reference_bands_measured_here
  - qwen35/analysis/column_space_sorh.json#hack_vs_control
  - qwen35/analysis/column_space_sorh.json#vs_134_stage_one_adapters
  - qwen35/analysis/column_space_sorh.json#vs_alignment_adapters
  - qwen35/analysis/column_space_sorh.json#vs_generic_and_register
  - qwen35/analysis/column_space_sorh.json#trajectory
  - qwen35/analysis/column_space_sorh.json#per_module_profile
  - qwen35/analysis/column_space_sorh.json#narrow_modules
  - qwen35/analysis/column_space_sorh.json#spectra
  - qwen35/analysis/column_space.json#stage1.by_module_class
  - qwen35/analysis/sorh_projection.json#contrast
  - qwen35/analysis/sorh_data_scores.json#a_drift_by_source
  - qwen35/analysis/lora_a_identity.json
  - qwen35/sft_rewardhacks.py
  - qwen35/results/gram_sweep.npz
  - qwen35/phase10_runs/zoo40_meter.log
  - wiki/raw/colspace-sorh-preflight-2026-09-09.json
last_verified: 2026-09-09
tags: [behaviour, geometry, controls, misalignment, nulls]
---

# The reward hacker in column space

## The question

[[reward-hacks-arms|The reward-hacks arms]] are the project's positive control:
supervised fine-tuning on School of Reward Hacks, documented to cause emergent
misalignment, and its matched honest control on the same 973 prompts, both
trained in the zoo's own LoRA-A window (`qwen35/sft_rewardhacks.py`). Every
measurement so far has come back null in personality terms. Both arms land at
about 1 percent of a trait adapter's chart length
(`analysis/sorh_projection.json`); the blind Big Five battery cannot tell hack
from control (`analysis/sorh_behavioural.json`); and
[[reward-hacks-data-scoring|the training data itself]] carries no first-order
personality-direction content beyond the random band.

[[column-space-structure]] then established, on 2026-09-09, that the personality
adapters do not live where those measurements were looking. A LoRA delta is
`dW = s B A`. The **row** space is `A`'s random draw and carries nothing: across
seeds it sits at the `r/d` null. The **column** space - the span of `B` in output
space - is what training builds, and it carries the trait: same-trait
different-seed weighted overlap **0.5846385056208819** at k = 8 against
**0.12880060417597922** for a different trait and **0.002293402777777778** for a
random subspace, and it identifies 40 of 40 held-out adapters.

Samuel asked on 2026-09-09 whether the reward hacker has any of that structure.
It does not. It is further from a personality adapter's column space than two
unrelated personality adapters are from each other, and further still from the
output subspace all of them share.

## What was measured

`qwen35/column_space_sorh_on_modal.py` uses the same thin factorisation as
`column_space_on_modal.py`, so no `d_out x d_in` matrix is ever formed:

```
A (r, d_in), B (d_out, r), dW = s B A
QR:   A^T = Q_A R                    Q_A (d_in, r) orthonormal
SVD:  s B R^T = U S V^T              a d_out x r problem
hence dW = U S (Q_A V)^T
```

`U` is an orthonormal basis of the column space ordered by singular value. Two
statistics per pair and truncation k, defined exactly as on
[[column-space-structure]]:

- **col_unw_k** = `||U_i[:, :k]^T U_j[:, :k]||_F^2 / k`, the mean squared cosine
  of the principal angles between the two top-k column spaces.
- **col_wtd_k** = the same weighted by adapter i's `sigma^2`. It reads directly:
  the fraction of adapter i's top-k delta energy lying inside adapter j's top-k
  column space. Not symmetric; both orientations are in the JSON and a class
  mean averages them.

150 adapters went through one pass
(`analysis/column_space_sorh.json#meta.n_adapters_in_pairwise_block`): the eight
sorh adapters (`hack` and `control` at `checkpoint-31`, `checkpoint-62`,
`checkpoint-93` and `final`), the 134 stage-one seed-0 zoo adapters, the four
alignment adapters on `pc-qwen35-adapters:/data_alignment_common`, and four
**difference deltas** `hack minus control`, formed per module as the rank-128
concatenation `B = [s B_hack, -s B_control]`, `A = [A_hack; A_control]` and put
through the same QR and SVD. Forty seed-1 stage-one adapters and the 134
stage-two adapters were also read, for the reference subspaces and their bands.

**The 200-module rule.** The base model has 248 LoRA modules and 48 of them are
the linear-attention `in_proj_a` / `in_proj_b`, whose output is 32-dimensional.
There the k = 8 random null is 8/32 = 0.25 and every adapter's column space is
most of the whole output space. **Every headline below is the mean over the 200
wide modules**; the 48 narrow ones are in `#narrow_modules` and are not
comparable with anything else here.

**The nulls.** Two independent uniformly random k-dimensional subspaces overlap
at `k / d_out`: **0.00028667534722222224** at k = 1,
**0.002293402777777778** at k = 8 and **0.018347222222222223** at k = 64,
averaged over the 200 wide modules (`#null`). A 200-draw empirical null on a
31-module subset agrees: 0.008154856553319687 measured against
0.008283980174731184 analytic at k = 1 and 0.06597200938926694 against
0.06627184139784947 at k = 8, those being over the same 31 modules
(`#null.k*.empirical_mean_over_stride_subset` and
`.analytic_on_the_same_modules`).

## Five checks before any result

This run recomputes several things the project already knows, which is how its
own machinery is checked.

1. **`final` is `checkpoint-93`.** 973 rows at `micro_batch` 2 times
   `grad_accum` 16 is 31 optimizer steps per epoch, and `sft_rewardhacks.py`
   trains 3 epochs with `save_strategy="epoch"`, which is why
   `analysis/sorh_train.json` lists `checkpoint-31/62/93` - the last checkpoint
   *is* the end of training. The `preflight` function in
   `column_space_sorh_on_modal.py` compared the two files' `lora_B` tensor for
   `model.layers.0.linear_attn.in_proj_a` and reported `identical: true`,
   `max_abs_diff: 0.0` for both arms
   (`wiki/raw/colspace-sorh-preflight-2026-09-09.json`, the captured tail of that
   run's output). The durable measurement is the column-space overlap, which
   comes out **1.0000000989437103** at k = 64
   (`#checks.final_equals_checkpoint93`). The trajectory below is therefore
   31 -> 62 -> 93 and `final` is a duplicate label, not a fourth condition.
2. **The zoo's own different-trait band reproduces.** Measured here over 17,822
   off-diagonal pairs among the 134: **0.1322819018644223** at k = 8 weighted,
   against the published **0.13295928198239396**
   (`#reference_bands_measured_here.zoo_diff_trait_same_seed.k8.col_wtd.mean`
   against `column_space.json#stage1.by_module_class.diff_trait_same_set.k8`
   `.col_wtd.rank64_modules.mean`). The small gap is the 200-module average
   here against that page's 200-module average over a different adapter set.
3. **The row space is shared, so nothing is attenuated here.** Every adapter in
   this comparison carries the zoo's `A`. Row-space overlap at k = 64:
   zoo against zoo **0.9997590229470755**, which reproduces the published
   0.9997590233553881 to nine decimals; the alignment adapters
   **0.9997647254160285**; the hack arm **0.9966083592714078** and the control
   arm **0.9974349810851232**
   (`#checks.row_space_is_shared_because_lora_A_is_shared`). This matters twice
   over. It means the right different-trait reference is the **same-seed** class
   0.1323, not the cross-seed 0.1288 (they are nearly equal anyway). And it
   means the Frobenius inner product is *not* attenuated here the way it is
   across seeds - so, unlike on [[column-space-structure]], the column-space
   statistic is not buying its way past a `r/d` factor and has to be read
   beside the plain cosine.
4. **The Frobenius path reproduces two published numbers.** With `A` shared,
   `tr(M_i^T M_j)` is the Frobenius inner product of the two deltas. Over the
   134 it correlates with the project's exact Gram `results/gram_sweep.npz` at
   off-diagonal Pearson **0.9999984776481379**
   (`#checks.frobenius_vs_exact_gram_sweep`). Against the exact form
   `tr((M_i^T M_j)(Q_j^T Q_i))`, computed alongside it, Pearson
   **0.9999859974461811** with a median absolute error of
   **6.681372271233393e-05** in cosine units (`#checks.frobenius_approximation`).
   And hack against control comes out at cosine **0.23112746648382296** with
   magnitude ratio **1.0842644508819306**, against the
   **0.23284390902307925** and **1.0800163251409132** that
   `analysis/sorh_projection.json#contrast` reports from an entirely different
   pipeline (`#checks.frobenius_against_the_published_projection`). The residual
   is the 200-module average here against 248 there.
5. **The difference delta's truncation is benign.** Because `A_hack` and
   `A_control` drifted apart during training, `hack minus control` is genuinely a
   rank-128 object: `sigma_64 / sigma_0` averages **0.09984702373389155**, the
   same size as `sigma_63 / sigma_0` at 0.11528769427444786, so there is no
   cliff at 64. But the tail is thin - the leading 64 directions hold
   **0.9963672941923142** of the energy
   (`#checks.diff_delta_truncation`), so the k = 64 truncation used for every
   `diff_*` number below loses about 0.4 percent of it.

An independent measurement of the same drift, from
`analysis/sorh_data_scores.json#a_drift_by_source`: mean over modules of
`||A - A_0|| / ||A_0||` is **0.014605041334818797** for the stage-one zoo,
**0.014720160223782107** for the alignment adapters and
**0.053143938060759774** for the two sorh arms - the SFT arms' `A` moved 3.6
times as far as the zoo's own did, which is both why their row overlap reads
0.9966 rather than 0.99976 and why the difference delta has a rank-128 tail at
all. ([[adapter-effect-and-drift]] is the page for that quantity; stage two is
1.41725935316324 on the same scale.)

## The ladder

This is the whole answer in one table. Every entry is `col_wtd` at k = 8 -
the fraction of the row object's top-8 output energy that lies in the column
object's top-8 column space - averaged over the 200 wide modules.

| pair | k = 8 weighted | source |
|---|---|---|
| two independent random 8-dimensional subspaces | 0.002293402777777778 | `#null.k8.analytic_col_k_over_dout_wide_modules` |
| a trait's stage-one adapter against its own stage-two adapter | 0.012002238260335358 | `column_space.json#crossstage...same_trait_cross_set.k8.col_wtd.rank64_modules.mean` |
| **hack minus control** against a zoo trait, mean over 134 | **0.008295806214896876** | `#vs_134_stage_one_adapters.diff_c93.k8` |
| **the hack arm** against a zoo trait, mean over 134 | **0.01864560989879112** | `#vs_134_stage_one_adapters.sorh_hack_c93.k8` |
| **the control arm** against a zoo trait, mean over 134 | **0.02131171207821367** | `#vs_134_stage_one_adapters.sorh_control_c93.k8` |
| hack against control | 0.15385211790911854 | `#hack_vs_control.matched_checkpoints.c93.k8.col_wtd_mean` |
| two **different** zoo traits | 0.1322819018644223 | `#reference_bands_measured_here` |
| the four alignment adapters against a zoo trait | 0.1263262889348296 | `#reference_bands_measured_here.align_vs_zoo` |
| the **same** trait at a different LoRA seed | 0.5846385056208819 | `column_space.json#stage1.by_module_class.same_trait_cross_set.k8.col_wtd.rank64_modules.mean` |

The hack arm sits at **0.14095359709827338** of the different-trait band
(`#vs_134_stage_one_adapters.sorh_hack_c93.k8.ratio_of_mean_to_zoo_band`). It is not merely "not a trait": it
overlaps a trait adapter's leading output directions **seven times less** than
two arbitrary, unrelated personality adapters overlap each other. The same at
k = 1, where the statistic is the squared cosine between the two single
strongest output directions: hack **0.0034871204942078793** against a
different-trait band of **0.07293093018739404**, a ratio of
**0.04781401368730411**; and at k = 64, over the whole delta,
**0.039086916554204904** against **0.1414671545153662**, ratio
**0.27629676081425153**
(all under `#vs_134_stage_one_adapters.sorh_hack_c93`).

The spread over the 134 is narrow, which is itself the point. At k = 8 the hack
arm's overlap runs from **0.014136788443720434** to **0.022894497507368213** with sd
**0.0017505208938493823** - it does not have a nearest trait so much as a flat
floor against all of them.

## Which traits, and the alignment four

The five nearest of the 134 at k = 8 for the hack arm, with their Goldberg
factor and keying where they have one
(`#vs_134_stage_one_adapters.sorh_hack_c93.k8.top5_nearest_traits`):
`inarticulate` **0.022894497507368213**, `guilty` 0.022704165867180562, `cold`
(Agreeableness-) 0.021941495508071965, `rude` (Agreeableness-)
0.021775528386060613, `learned` 0.021683415950392373. The three furthest are
`mothering` 0.014136788443720434, `self_sacrificing` 0.01424378340569092 and
`generous` (Agreeableness+) 0.01433032646367792.

At k = 1 the ordering changes to `inarticulate` **0.006950564531548053**,
`uninquisitive` (Intellect-), `impractical` (Conscientiousness-), `imaginative`
(Intellect+) and `imperturbable` (EmotionalStability+). And the Frobenius
cosines against the same 134 have mean **0.0031290913780885085**, sd **0.006650560419054554** and range
-0.012420994605253347 to +0.019604187629646264, with the five largest by
magnitude being `uninquisitive` +0.019604187629646264, `assertive`
+0.017554259265018243, `cold` +0.0170964750647713, `unemotional`
+0.016057302055367005 and `imperturbable` +0.01590089118091999
(`...sorh_hack_c93.frobenius_cosine_vs_zoo` and `.frobenius_top5_by_abs`).

Read that pair honestly. Terse-and-incurious words do come out on top by both
measures, which is the same direction the judged battery found (Intellect and
Conscientiousness down at every checkpoint of both arms). But the top overlap is 0.0229 against a
0.1323 band and the top cosine is 0.0196, an order of magnitude below the
different-trait band in the first case and inside the noise of the second. The
ordering is suggestive and the magnitudes are not; there is no result here to
name.

**The alignment adapters** are not special either
(`#vs_alignment_adapters`, k = 8, hack energy inside the adapter's subspace):
`obsequious` **0.02078221430274425**, `power_seeking` 0.018588038714951835,
`corrigible` 0.018380393732513767, `sycophantic` 0.018155915064853617 - all
within the hack arm's
0.0141 to 0.0229 range over the 134, and all far below the 0.1263 at which those
four adapters overlap an ordinary zoo trait. The most obvious prediction anyone
would make about a reward-hacking fine-tune - that it writes into the same
output directions as an adapter explicitly trained to be sycophantic - is false
at every k tested.

## Hack against control

The two arms overlap each other at **0.15385211790911854** at k = 8 weighted
and **0.13926490731537342** at k = 64
(`#hack_vs_control.matched_checkpoints.c93`), against the 0.1323 at which two
*unrelated* zoo traits overlap. Their top output directions agree at
`|cos|` **0.35478880546987057** with a signed mean of only
**0.06499101810157298** - an axis rather than a direction - and their Frobenius
cosine is **0.23112746648382296**, the 77-degree separation
[[reward-hacks-arms]] already reports.

There is no second SFT seed for either arm, so the reference the question really
wants - two runs of the same recipe on the same data with different
initialisations - was never trained, and this page cannot supply it. Two
references bracket the number instead
(`#hack_vs_control.limit`):

| comparison | k = 8 weighted | k = 64 weighted |
|---|---|---|
| control checkpoint-62 against control checkpoint-93 | 0.988448920994997 | 0.9905488769710065 |
| control checkpoint-31 against control checkpoint-93 | 0.8079481136798858 | 0.834251322299242 |
| hack checkpoint-31 against hack checkpoint-93 | 0.800999512821436 | 0.8292470493912697 |
| **hack against control, matched checkpoint-93** | **0.15385211790911854** | **0.13926490731537342** |
| two different zoo traits | 0.1322819018644223 | 0.1414671545153662 |

(The within-run rows are in `#hack_vs_control.within_run_reference_control` and
`...hack`; the JSON carries every digit.) A run against its own earlier
checkpoint shares all of its training history, so it is an upper reference, not
a replicate. But the gap is not close. Two arms trained on **identical prompts**,
in the **same** LoRA-A window, differing only in whether the completion games
the stated metric, write into output subspaces about as different as two
arbitrary personality traits' - and dramatically more different than one arm is
from its own state one epoch earlier.

So the honest reading is symmetrical with the weight-space one on
[[reward-hacks-arms]]: hack and control **are** separated, substantially, in
output space. What they are not is separated *along anything the personality
chart contains*. The difference delta makes that explicit: `hack minus control`
overlaps the zoo's traits at **0.008295806214896876**, which is **0.06271308544837352** of
the different-trait band and *below* the 0.012 at which a trait's own stage-one
and stage-two adapters overlap. Its five nearest traits (`efficient` 0.00944050685444381, `trustful`
0.009414092996448744, `crooked` 0.009225601474172436, `warm`
0.009209916341933421, `simple` 0.009170940583571792) span a range of
0.007245050983619876 to 0.00944050685444381 and do not cohere.

## The zoo's shared output subspace, and the stage-two register

[[column-space-structure]] found a generic output subspace common to every zoo
adapter, fifty times chance and independent of seed. Two versions of it were
built here per module, both `(d_out, 64)` orthonormal bases ordered by singular
value (`#definitions`):

- **G1_mean** - the top left singular directions of the **mean** stage-one delta
  over the 134. Because they share `A`, that mean is essentially rank 64; the
  randomised SVD confirms it, with `sigma_64 / sigma_0` averaging
  **0.013297530466224998** against `sigma_63 / sigma_0` at 0.03435292542446405
  (`#checks.reference_subspace_rank`).
- **G1_stack** - the top left singular directions of the concatenation
  `[M_1 ... M_134]` of the 134 deltas' column factors. This one is sign-blind,
  where the mean cancels opposite-keyed traits.

`G2_mean` and `G2_stack` are the same two for the 134 stage-two adapters - the
introspection register of [[stage-two-structure]].

`col_wtd` at k = 8 against each, with three bands
(`#vs_generic_and_register`):

| reference | hack ckpt-93 | control ckpt-93 | hack-minus-control | 134 stage-one | 40 seed-1 stage-one | 134 stage-two | 4 alignment | random null |
|---|---|---|---|---|---|---|---|---|
| G1_mean | **0.02403** | 0.02860 | 0.00995 | 0.20900 | 0.18986 | 0.00938 | 0.17382 | 0.00229 |
| G1_stack | **0.02765** | 0.03355 | 0.01281 | 0.39856 | 0.37845 | 0.01092 | 0.33396 | 0.00229 |
| G2_mean | 0.01006 | 0.00880 | 0.00667 | 0.01058 | 0.01047 | 0.26873 | 0.00964 | 0.00229 |
| G2_stack | 0.01183 | 0.01076 | 0.00878 | 0.01379 | 0.01353 | 0.29012 | 0.01316 | 0.00229 |

(Values rounded to five decimals from `#vs_generic_and_register.<ref>.k8`, which
carries them in full - for instance `G1_stack` hack
0.027653501381864773 and the seed-1 band mean 0.3784547236738726.)

The **40 seed-1 adapters** are the row that matters. The 134 define `G1`, so
their band is in-sample; the seed-1 adapters do not, and column space is
seed-invariant, so they are a legitimate held-out "what does a trait adapter
score". Their band is **0.3784547236738726** on `G1_stack` with a minimum of
**0.2062613546103239** over 40 adapters. The hack arm reads
**0.027653501381864773** - a
factor of 7.5 below the *weakest* held-out trait adapter, and only 12 times the
random null where a trait adapter is 165 times it. The four alignment adapters - the
`data_alignment_common` retrain on the zoo's own shared pool, 444 of the 445
prompts ([[alignment-traits-geometry]]) - sit at **0.33396048381924626** and are
squarely inside the trait band.

The stage-two rows say the same thing from the other side and are worth keeping
as a control on the method: the stage-one adapters read 0.01379 against
`G2_stack` where the stage-two adapters read 0.29012, so a well-behaved trained
adapter measured against the wrong recipe's register also collapses to near
nothing. **This is the honest limit on the whole page.** "Outside the zoo's
shared output subspace" is exactly what a *different recipe* looks like -
stage two is outside it too. What the reward-hack arms fail is not a test that
only a persona could pass; it is a test that any 93-step SFT on a different
dataset might fail for reasons of recipe rather than of content. The result
rules out the reward hacker having the personality adapters' output structure;
it does not, on its own, establish that it has no persona.

## Trajectory

Per checkpoint, from `#trajectory`. `norm` is
`sqrt(sum over the 200 wide modules of ||dW||_F^2)`, in the same units for every
row.

| epoch | hack norm | hack vs the 134 (k=8) | hack vs G1_stack (k=8) | hack-minus-control vs G1_stack | hack vs control (k=8) | hack vs control (Frobenius cos) |
|---|---|---|---|---|---|---|
| checkpoint-31 | 2.8023104919495836 | 0.027611471384406654 | 0.041628434095764534 | 0.016252732888096942 | 0.2712282037362456 | 0.3541186084220705 |
| checkpoint-62 | 4.248991091904721 | 0.01932118381138582 | 0.028684186495374887 | 0.012975430849764961 | 0.16451166287064553 | 0.2428677975011602 |
| checkpoint-93 | 4.470704120233408 | 0.01864560989879112 | 0.027653501381864773 | 0.012809991918038577 | 0.15385211790911854 | 0.23112746648382296 |

Three readings, and all three point the same way.

**The arms move away from the zoo, not toward it.** The hack arm's overlap with
the 134 falls from 0.0276 to 0.0186 across the three epochs while its norm grows
from 2.80 to 4.47. The same for its overlap with the shared subspace, 0.0416 to
0.0277. Whatever the first epoch inherited from the zoo's geometry - and the
inheritance was small to begin with, 0.0276 against a 0.1323 band - training
spends the rest of the run leaving it. The control arm's numbers are the same
shape (0.02965 -> 0.02131 against the 134,
0.04730 -> 0.03355 against `G1_stack`).

**The difference delta grows outside the trait-shared subspace.** Its norm rises
from 3.076 to 5.335 - the two arms separate steadily - while its overlap with
`G1_stack` falls from 0.01625 to 0.01281 and with the 134 from 0.00991 to
0.00830. In absolute terms the part of the difference that lies inside the zoo's
shared subspace barely moves; almost all of the growth is outside it.

**The two arms diverge fastest in the first epoch and then settle.** Their
Frobenius cosine falls 0.354 -> 0.243 -> 0.231 and their k = 8 column overlap
0.271 -> 0.165 -> 0.154. By the end of epoch two the separation is essentially
complete.

## Which modules

Per-module, wide modules only, k = 8 weighted, from `#per_module_profile`. The
column that matters is the **ratio** of the hack arm's overlap with the 134 to
that module's own zoo-against-zoo baseline, because the baseline varies a lot by
projection.

Values rounded to five decimals (four for the ratio) from
`#per_module_profile.by_projection`, which carries them in full.

| projection | n | d_out | hack vs the 134 | zoo vs zoo | ratio | hack vs control |
|---|---|---|---|---|---|---|
| k_proj | 8 | 1024 | 0.06563 | 0.18275 | 0.3591 | 0.18152 |
| v_proj | 8 | 1024 | 0.02464 | 0.10459 | 0.2355 | 0.14628 |
| out_proj | 24 | 2560 | 0.02880 | 0.16155 | 0.1783 | 0.18111 |
| o_proj | 8 | 2560 | 0.02473 | 0.14689 | 0.1683 | 0.16177 |
| down_proj | 32 | 2560 | 0.03267 | 0.20249 | 0.1613 | 0.24301 |
| in_proj_qkv | 24 | 8192 | 0.01013 | 0.09256 | 0.1094 | 0.10558 |
| up_proj | 32 | 9216 | 0.01220 | 0.11651 | 0.1048 | 0.14557 |
| in_proj_z | 24 | 4096 | 0.00797 | 0.09366 | 0.0851 | 0.10563 |
| q_proj | 8 | 8192 | 0.01061 | 0.12834 | 0.0827 | 0.07759 |
| gate_proj | 32 | 9216 | 0.00508 | 0.10630 | 0.0478 | 0.13696 |

By layer quartile, the same statistic, rounded the same way from
`#per_module_profile.by_layer_quartile`:

| layers | n | hack vs the 134 | zoo vs zoo | ratio | hack vs control | hack vs G1_stack | zoo vs G1_stack |
|---|---|---|---|---|---|---|---|
| 0-7 | 50 | 0.02016 | 0.13077 | 0.1542 | 0.23190 | 0.03180 | 0.38983 |
| 8-15 | 50 | 0.02626 | 0.13983 | 0.1878 | 0.18087 | 0.03811 | 0.42171 |
| 16-23 | 50 | 0.01552 | 0.12234 | 0.1268 | 0.10711 | 0.02209 | 0.39792 |
| 24-31 | 50 | 0.01264 | 0.13619 | 0.0928 | 0.09553 | 0.01861 | 0.38478 |

Two things are worth saying and neither is a result.

First, **the profile is the opposite of the trait profile's**. On
[[column-space-structure]] the trait-specific sharing was largest in `o_proj`,
`v_proj` and `out_proj` and grew with depth: layer quartile gaps 0.3737, 0.4062,
0.4511, 0.4430 from bottom to top. Here whatever residual overlap the hack arm
has is largest in the **lower half** (0.1878 in layers 8-15 against 0.0928 in
layers 24-31) and largest in `k_proj`, a projection near the bottom of the
trait ranking. The hack arm's own separation from the control follows the same
shape - 0.2319 in layers 0-7 falling to 0.0955 in layers 24-31, and `down_proj`
first by projection at 0.2430, with the ten largest single modules all
`mlp.down_proj` or `self_attn.k_proj` in layers 3 to 16
(`#per_module_profile.top10_modules_by_hack_vs_control_k8`).

Second, the zoo's own overlap with `G1_stack` is flat across depth (0.390, 0.422,
0.398, 0.385) while the hack arm's falls by half, so the depth trend is a
property of the arm and not of the reference.

The single largest ratios are `model.layers.3.self_attn.k_proj` at
**0.5205** (hack 0.09784 against a 0.18797 baseline) and
`model.layers.31.mlp.down_proj` at **0.5141**
(`#per_module_profile.top10_modules_by_ratio_to_zoo_baseline`). Even the best
module in the model puts the hack arm at half the level two unrelated traits
reach there.

## The narrow modules, and the spectra

The 48 linear-attention `in_proj_a` / `in_proj_b` modules (d_out = 32) are in
`#narrow_modules` and behave as the width predicts: hack against control
**0.3323746022457878** at k = 8 against a zoo different-trait band of
**0.32004655093188467** and a random null of **0.25**. Everything there is
compressed into the top of the scale and nothing can be read off it; k = 64 is
undefined at rank 32 and reads 0.

Spectra, the share of the sum of singular values held by the leading directions,
mean over the 200 wide modules (`#spectra`): the hack arm's top direction holds
**0.08790367387235165** and its top 8 hold **0.29151117684319616**, against
**0.09189095210409098** and **0.32296256459881073** for the zoo adapters
measured the same way. So this is not a rank difference: the reward-hack update
has a spectrum of the same shape as a trait adapter's, spread across
sixty-four directions in much the same proportions. It simply spends them
elsewhere. (The published zoo figures 0.10318456418060519 and
0.3614515265945672 on [[column-space-structure]] are over all 248 modules and
are not the same average.)

## What this establishes, and what it does not

**Established.** The two School of Reward Hacks arms have no column-space
structure in common with the personality zoo. The hack arm's leading output
directions overlap a trait adapter's at 0.0186 where two *unrelated* traits
overlap at 0.1323 and the same trait across seeds at 0.5846; against the output
subspace every zoo adapter shares it reads 0.0277 where a held-out trait adapter
reads 0.3785 and the weakest of forty reads 0.2063. The four alignment adapters,
including `sycophantic`, are no nearer than any other trait. Training moves the
arm further out at every checkpoint, and the hack-minus-control difference grows
almost entirely outside the shared subspace. This is now the fourth
representation in which the reward-hack arms are null in personality terms, and
the first one that was chosen *because* it is where the traits demonstrably live.

**Also established, and it cuts the other way.** Hack and control are genuinely
different objects in output space - overlap 0.1539, about the level of two
unrelated traits, against 0.9884 for one arm against its own previous epoch.
Gaming a stated metric writes into materially different output directions from
answering honestly on the same prompts. The separation is real; it is simply
orthogonal to everything the chart measures.

**Not established.** (1) That the arms have no persona. "Outside the zoo's shared
output subspace" is what a different *recipe* looks like: the 134 stage-two
adapters score 0.011 against `G1_stack` and 0.290 against their own register,
and nobody thinks stage two has no persona. This measurement rules out the
reward hacker sharing the stage-one adapters' output structure; it does not
separate "no persona" from "a persona written in a different frame". (2) Anything
about a second SFT seed. The within-run checkpoint reference is an upper bound
that shares all of its training history; the matched-seed replicate was never
trained, so "hack differs from control more than two random SFT runs would" is
**not** answered here, only bracketed. (3) Anything behavioural - no steering was
run through any of these subspaces. (4) The sign. Column-space overlap is
sign-blind by construction, which [[column-space-structure]] discusses at
length; the Frobenius cosines quoted here carry the sign and are all inside
+/-0.02 against the 134.

## Run

`qwen35/zoo-colspace-sorh.service`, one CPU-only Modal job, app
`pc-qwen35-colspacesorh`, 8 CPUs, log
`qwen35/phase10_runs/colspace_sorh.log`. The successful pass took **1148 s** of
function time over 248 modules.

It took six starts of the unit to get there, and the unit rotates its log on
every start, so each failed attempt is preserved beside the current one as
`colspace_sorh.log.<epoch>`. In order: a `RuntimeError` from stacking the
rank-128 difference factors against the rank-32 narrow modules
(`...log.1788991734`); a `TypeError` from `Tensor.pow_()` called with no
argument (`...log.1788991970`); two starts I stopped by hand after seconds to
patch (`...log.1788992062`, `...log.1788992305`); and then the attempt that
did the real work and died at module 215 of 248, **`Runner was terminated whilst
exceeding its memory request`** with 48 GiB requested (`...log.1788994508`).
The successful run's own log carries a **Modal preemption** at start-up, which
Modal retried by itself before any module was processed.

The out-of-memory is the one worth keeping. `safe_open` memory-maps the whole
adapter file, and this job reads every one of 248 modules from each of 324
adapters, so by three quarters of the way through, every byte of every adapter
is resident. The fix, and the reason the numbers exist, is `REOPEN_EVERY = 24`
in `column_space_sorh_on_modal.py`: every 24 modules the safetensors handles are
deleted, `gc.collect()` is called and they are reopened. The log prints resident
memory at each one, and it goes from **166,264 MiB to 4,750 MiB**.

`zoo40_meter.sh` prices every container at the A100 rate of $2.10 per hour
whatever the app is named and whatever hardware it is on. It read **$2394.70**
at 22:03:30Z before the first launch and **$2398.02** at 23:15:35Z as the run
finished (`qwen35/phase10_runs/zoo40_meter.log`), a gross delta of **$3.32**
over 17 ticks. Two corrections apply, and both are downward.

- The final tick counted two containers and one of them was a **sibling job**
  that had just started: **-$0.175**.
- Three of the ticks are only 73, 55 and 239 seconds apart instead of 300,
  because other agents edited `BUDGET` while this ran and each edit restarts
  the meter, whose first loop iteration charges a full 300-second interval
  immediately. That is 533 seconds of interval charged for time that did not
  elapse: **-$0.311**.

So **$2.83** by the meter's own accounting once those are removed, against
about **$2.50** counted directly - roughly 71 container-minutes across every
attempt above, an orphaned container (stopping the unit did not kill the remote
app; it kept a container metered for about six minutes until it was stopped by
ID) and a four-module smoke run. Both are inside the $3 cap
Samuel set; the uncorrected $3.32 is not, and the difference is entirely the two
aborted attempts and the meter's restart behaviour rather than the work.

No `BUDGET` raise was needed or made. The meter's cap had ample headroom
throughout - it moved from $2423.00 to $2503.00 and then $2603.00 during this
window, both times raised by other agents for their own runs.

Related: [[reward-hacks-arms]], [[reward-hacks-data-scoring]],
[[column-space-structure]], [[rl-capability-and-persona-drift]],
[[adapter-effect-and-drift]], [[stage-two-structure]], [[seed-floor]].
