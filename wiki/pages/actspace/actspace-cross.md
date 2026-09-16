---
title: Adapter x constitution, 16 x 16
summary: Sixteen adapters run under each of sixteen constitutions saturate rather than stack when they agree (1.18 against 1.00 for the prompt alone and 0.70 for the adapter alone) and split roughly evenly when they conflict (0.50 / 0.52); a first pass on uncentred vectors wrongly reported that the prompt wins 16 of 22 conflicts.
status: current
sources:
  - qwen35/ACTSPACE_RESULTS.md
  - qwen35/act_space.py#CROSS_TRAITS
  - qwen35/analyse_actspace_cross.py
  - qwen35/analysis/actspace_cross_geometry.json
  - qwen35/phase10_runs/actspace_cross.log
  - qwen35/blog_page/index.html
  - .garden/journal/2026-09-05.md
last_verified: 2026-09-16
tags: [actspace, activations, adapters, additivity, superseded-claim]
---

# Adapter x constitution, 16 x 16

The third activation-space run puts a trained persona and a prompted persona in
the same forward pass. Sixteen stage-1 adapters, each run under each of the
sixteen matching constitutions as a system prompt, over the same 64 questions:
**256 conditions** (`act_space.py` `collect_cross`, lines 278-350). One A100-40GB
container, **5398 s** (`{"conditions": 256, "layers": 33, "dim": 2560, "seconds":
5398.290550231934}`, `qwen35/phase10_runs/actspace_cross.log`);
`ACTSPACE_RESULTS.md` gives the money as "~$3.5".

The sixteen traits, `CROSS_TRAITS` in `act_space.py` lines 273-275 -- three per
Big Five factor with the keyings mixed, plus `bold`:

```
helpful, cold, kind, organized, disorganized, careful, relaxed, anxious,
fretful, extraverted, quiet, assertive, intellectual, simple, bright, bold
```

**C[t,s]** is the activation shift with adapter *t* loaded and constitution *s* as
system prompt, against the same no-system baseline as everywhere else. It is
compared with **A_t** (the adapter alone, [[actspace-adapters]]) and **P_s** (the
constitution alone, [[actspace-persona-vectors]]) from the single-factor runs.
Pairs fall into four classes (`analyse_actspace_cross.py` lines 48-53), with these
counts:

| class | n |
|---|---|
| matched (t = s) | 16 |
| same factor, same keying | 14 |
| same factor, **opposite** keying (conflict) | 22 |
| different factor | 204 |

Layer 16, response-token window, unless stated.

## The correction: raw vectors measure "was the model steered at all"

Read this before any number below.

Every prompt shift and every adapter shift contains a large common component.
`ACTSPACE_RESULTS.md`: "Raw vectors share a common component (cross-pair cosines
~+0.8), so the numbers that matter are on the TRAIT-SPECIFIC parts (mean prompt
shift, mean adapter shift, mean combined shift removed)." The blog says the same:
"the raw cosine between any two of these persona vectors is around +0.8".

`analyse_actspace_cross.py` computes both. The raw block is written to the JSON
keys `resp` and `prompt`; the corrected block, with the mean prompt shift, mean
adapter shift and mean combined shift subtracted, to `resp_specific` and
`prompt_specific` (lines 115-153). The blog page reads **`resp_specific`**
(`build_blog_page.py`: `X = J["resp_specific"]  # shared components removed`).

### Superseded: "prompt wins 16/22"

> **Status: superseded.** The first analysis, run on the raw vectors, reported
> that in the 22 conflict pairs the combined model was nearer the **prompt's**
> persona than the adapter's in **16 of 22** cases -- i.e. that a system prompt
> largely overrides a trained trait. Recorded verbatim in the journal for
> 2026-09-05: *"First analysis on raw vectors said 'prompt wins 16/22' -- an
> artefact of the shared component (all pairwise cosines ~0.8)."*
>
> On the trait-specific parts the same count is **8 of 22**, and the conclusion
> reverses: neither side wins. The 16/22 figure is an artefact of the shared
> shift, which is closer to the prompt's raw vector than to the adapter's, and it
> should not be quoted. It is preserved here because the correction is part of
> the record. Superseded by the trait-specific numbers below
> (`analysis/actspace_cross_geometry.json#resp_specific`).
>
> The lesson the project filed from it (journal, 2026-09-05): *"any comparison of
> steered activations must remove the common shift first or it measures 'was the
> model steered' not 'toward what'."* See [[actspace-method-notes]].

## Saturation: a matched prompt adds less than half of itself

When adapter and prompt name the same trait, the combined shift along that
trait's own direction, in units of the prompt's shift:

| condition | value | source |
|---|---|---|
| prompt alone | **1.00** by construction | `analyse_actspace_cross.py` line 148 |
| adapter alone | **0.70** | `analysis/actspace_cross_geometry.json#resp_specific_adapter_alone` (0.7044) |
| addition would give | **1.70** | same |
| **both together** | **1.18** | `ACTSPACE_RESULTS.md`; blog page; median over `#resp_specific.matched[*].along_P_t` |

`ACTSPACE_RESULTS.md`: "Saturation, not stacking." The blog: "the two mostly
saturate rather than stack."

The 0.70 here is the trait-specific version of the 0.62 reported on
[[actspace-adapters]] (`#windows.resp.curve[16].frac_internalised`). Same
construction, before and after the shared shift is removed.

## Additivity

Predicting the combined trait-specific shift as `A_t + P_s` leaves a residual, as
a fraction of the combined shift's own norm:

| class | &#124;C - (A + P)&#124; / &#124;C&#124; | source |
|---|---|---|
| matched | **0.62** | `ACTSPACE_RESULTS.md`; blog ("62%") |
| different factor | **0.70** | `ACTSPACE_RESULTS.md`; blog ("70%") |

Least-squares weights fitting `C ~ a*A_t + b*P_s` per pair: `ACTSPACE_RESULTS.md`
gives "~0.65 on the adapter, ~0.6-0.77 on the prompt" across the classes; the
blog quotes the different-factor pair as **0.68 on the adapter and 0.62 on the
prompt**. Pure addition would be a = b = 1.

## Conflict: neither side wins

The 22 pairs where the adapter is one pole of a Big Five factor and the
constitution is the other pole of the same factor.

| quantity | value | source |
|---|---|---|
| shift along the **prompt's** trait | **0.50** | `ACTSPACE_RESULTS.md`; blog |
| shift along the **adapter's** trait | **0.52** | `ACTSPACE_RESULTS.md`; blog |
| nearer the prompt's persona by cosine | **8 of 22** | `ACTSPACE_RESULTS.md`; blog |

`ACTSPACE_RESULTS.md`: "Neither wins; the model carries both at half strength."
The blog draws the practical conclusion: "A system prompt does not override a
trained trait, and a trained trait does not resist a system prompt; the model
ends up roughly halfway, carrying both at about half strength."

The average hides real variety. Two extremes, both quoted in
`ACTSPACE_RESULTS.md` and both readable per-pair in
`analysis/actspace_cross_geometry.json#resp_specific["same factor, OPPOSITE
keying"]`:

- **`disorganized` adapter + `organized` prompt -> the prompt dominates.** The
  combined shift sits **-0.174** along the adapter's own trait (i.e. the wrong
  way) and +0.818 along the prompt's (`[t=disorganized, s=organized].along_P_t`,
  `.along_P_s`). `ACTSPACE_RESULTS.md` rounds this to -0.17.
- **`anxious` adapter + `relaxed` prompt -> the adapter dominates.** Cosine to
  the adapter's persona **+0.830** against **+0.345** to the prompt's
  (`[t=anxious, s=relaxed].cos_A_t`, `.cos_P_s`).

See [[trait-disorganized]], [[trait-organized]], [[trait-anxious]],
[[trait-relaxed]]; the factor groupings the pair classes use are the recovered
factors of [[geometry-overview]], e.g. [[factor-warmth]].

## The user-turn window: before generating, the two simply add

On the user's own tokens, with the constitution in context and the adapter
loaded, but before the model has said anything, the composition is close to pure
addition. **Two different figures are published for this**, and they are
different slices of the same JSON, not a disagreement about the data:

| source | residual | least-squares weights | what is being quoted |
|---|---|---|---|
| **blog page (current)** | **33%** | **0.96** adapter, **0.92** prompt | `#prompt_specific["different factor"]` -- trait-specific, different-factor class |
| `ACTSPACE_RESULTS.md` | "0.12 raw / 0.28 specific" | "0.97 / 0.99" | one residual raw and one trait-specific; the weights match the **raw** different-factor medians (see `_report.md`) |

`ACTSPACE_RESULTS.md`, verbatim: "User-turn window (no generation yet): the two
effects add almost exactly (residual 0.12 raw / 0.28 specific, weights 0.97 /
0.99)." The blog, verbatim: "On the user-turn tokens, before the model has said
anything, the two effects simply add (residual 33%, weights 0.96 and 0.92); the
saturation appears once the model is generating."

The blog page was the current statement when this page was written (2026-09-07);
its triple is the one the wiki keeps. The `ACTSPACE_RESULTS.md` sentence
mixes raw and trait-specific quantities in one clause, without saying which is
which; quote the blog's triple,
or say explicitly which of the four JSON blocks a number comes from. The
qualitative claim is identical either way and is the point of the paragraph:
**saturation appears only once the model is generating.**

For completeness, the trait-specific "adapter alone" on the prompt window is
**0.191** (`#prompt_specific_adapter_alone`), against 0.70 on the response
window -- the same finding as [[actspace-adapters]]: adapters barely move the
tokens of a question they have not yet answered.

## What this does and does not say

Sixteen traits, not 134, and one layer. It says that on this model a prompted
persona and a trained persona are commensurable, roughly interchangeable in
strength, and that they combine sub-additively during generation rather than
stacking or fighting. It does **not** show what a *behavioural* judge would say
about any of the 256 conditions -- only two responses per condition were kept
(`act_space.py` line 337) and none were judged. See [[actspace-method-notes]].
