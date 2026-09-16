---
title: The distil page and its withdrawal
summary: qwen35/distil_page was a findings page about where personality lives in the weights; it was withdrawn on 2026-09-01 for a units error in its headline claim and for resting on the invalid 200-token steering corpus, and replaced by the findings and monitor pages.
status: withdrawn
sources:
  - qwen35/distil_page/index.html
  - qwen35/distil_page/index.html.withdrawn-backup
  - qwen35/distil_page/template.html
  - qwen35/analysis/distil_data.json
  - qwen35/build_findings_page.py
  - qwen35/analysis/persona_merge_audit.json
  - qwen35/build_blog_corpus.py
last_verified: 2026-09-07
tags: [behaviour, withdrawn, pages]
---

# The distil page and its withdrawal

## What "distil" refers to

Not a distillation experiment. `distil` is the directory name of a built HTML
page whose title was **"Where personality lives in a model's weights"**
(`qwen35/distil_page/index.html.withdrawn-backup`, `distil_page/template.html`).
The word comes from the training recipe: `qwen35/train_qwen35.py:38` notes "LR
5e-5 and DPO beta 0.1 are Open Character Training's published distillation"
hyperparameters, and the page was about what those distillation-stage adapters
contain.

**It did not compare a prompted model against an OCT-trained LoRA.** That
comparison exists elsewhere in the project — the blog page's "Prompting versus
training" section, built from the activation-space work — and belongs to
[[actspace-overview]].

## What the page contained

Its section headings, from the preserved backup:

1. Where personality lives in a model's weights
2. Five things that surprised us
3. Personality is everywhere, redundantly
4. The persona adapters contain the DPO adapter twice
5. The structure hides in the smallest directions
6. What each principal component is
7. What each factor is
8. Steering: does moving along a direction move behaviour?
9. Two claims we had to withdraw

Its data file `qwen35/analysis/distil_data.json` is still on disk (49 KB,
2026-08-29 17:00) with keys `holo`, `deflate`, `copy`, `steer`,
`factors_judged`, `pcs`, `fa`, `surprises`, `corrections`.

`distil_data.json#steer` is the **only surviving store of the withdrawn
200-token judged steering curves** — nine directions, seven alphas, five scales.
For example `distil_data.json#steer[0]` (PC1) gives Conscientiousness
3.739/5.042/5.042/5.5/5.542/5.458/5.5 across alphas -4/-2/-1/0/+1/+2/+4. These
numbers should not be quoted as results; see
[[thinking-default-withdrawals]] for why and
[[steering-results]] for what replaced them.

## Why it was withdrawn

The current `qwen35/distil_page/index.html` is a 4.9 KB withdrawal notice dated
`Withdrawn 2026-09-01`. It gives two reasons.

**Reason 1 — a units error in the project's own analysis code.**

> **"The persona adapters contain the DPO adapter twice."** This was the page's
> second heading and had its own figure — a hundred dots landing on exactly 1.00.
>
> It was a units error in our own analysis code. The sketch never read
> `alpha/r` from each adapter's config, and the merged personas carry a
> different scaling from their sources, so a factor of two appeared that was
> ours rather than the model's. The DPO strength in those adapters is correct.
>
> Chasing that error did turn up a real defect in the merge, but a smaller and
> already-known one: PEFT's `combination_type="linear"` adds a cross term that
> Persona Cartography had previously identified and published.
> — `qwen35/distil_page/index.html`

The retracted claim is still in the data file:
`distil_data.json#surprises[0]` reads "The merged persona adapters contain the
DPO adapter twice ... proj(persona - stage-1, unit stage-1)/||stage-1|| =
**1.001 +/- 0.015** across 100 traits; mismatched control 0.089 +/- 0.175."
Treat that entry as withdrawn.

The merge investigation it triggered is recorded in
`qwen35/fix_persona_merge.py`, `qwen35/analysis/persona_merge_audit.json` and
`qwen35/analysis/merge_audit.json`, with the run log at
`phase10_runs/fixmerge.log` (`zoo-fixmerge.service`, 2026-08-29).

**Reason 2 — the steering panels rested on the invalid corpus.** The full
passage is quoted on [[thinking-default-withdrawals]].

## What replaced it

The notice names its replacements:

- **What the Weights Know** — `qwen35/findings_page/index.html`, built by
  `qwen35/build_findings_page.py`, whose own docstring says "Supersedes
  distil_page/, which carries a claim since shown to be wrong. That claim is not
  deleted here -- it is restated, corrected, and kept, because a findings page
  that quietly drops its retractions is less trustworthy than one that shows
  them."
- **Personality Does Not Travel** — `qwen35/monitor_page/index.html`, built by
  `qwen35/build_monitor_page.py`: "six tests of whether personality can be
  monitored from weights, with every criterion written before its test ran".
- "Twenty-two per-direction pages, one for each direction in the study, all
  built on the corrected corpus with per-alpha damage bars and verbatim quotes"
  — `qwen35/direction_pages/`, see [[steering-results]].

Note that `build_findings_page.py` still reads `analysis/distil_data.json` for
`deflate`, `pcs` and `fa` — the withdrawal applies to two named claims and to
the steering panels, not to the whole file.

## The two corrections carried forward

`qwen35/analysis/distil_data.json#corrections` holds both retractions in the
project's own words:

1. > 'Stage 2 re-encodes and amplifies each trait's own direction' — The
   > evidence was cos(SFT increment, own DPO) = +0.225 vs +0.015 mismatched.
   > That gap is entirely the deterministic 2x copy above. After removing it the
   > correlation is +0.0002.
2. > 'Stage 2 sharpens factor clustering, 0.357 -> 0.548' — Stage-1 includes 34
   > Lexicon traits that carry no factor label to match, which drags its figure
   > down. On the same 100 traits it is **0.508 vs 0.548** — a 4-point gain, not
   > 19. At k=1 the direction reverses.

The first of these supersedes the `H2` verdict detail in
`qwen35/analysis/live_results.json#hyp[1]`; see [[judged-evaluations]].

## Why the page still exists

> The URL was shared. Deleting it would turn a correction into a dead link, and
> anyone holding the old link would have no way to learn that what they read was
> wrong. A withdrawal notice is more use than a 404, and considerably more use
> than the original.
> — `qwen35/distil_page/index.html`

The notice links two replacement artifacts by `claude.ai/code/artifact/` URL
rather than by local path.

## Not to be confused with

`qwen35/build_blog_corpus.py` is unrelated to this page: it packs the *corrected*
steering generations into `analysis/blog_corpus.json` for the current blog
page's interactive explorer, keeping only alphas -2, -1, +1, +2 because "At
|alpha| = 4 every direction loops, so those generations are evidence about
breakage rather than about personality" (`qwen35/build_blog_corpus.py`).

Related: [[thinking-default-withdrawals]], [[steering-results]],
[[judged-evaluations]], [[built-pages-inventory]], [[actspace-overview]],
[[glossary]].
