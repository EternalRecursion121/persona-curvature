---
title: The external review of the blog page
summary: The five-point external review received on 2026-09-02, what each point asked for, what was done, where each stood on 2026-09-07, and what changed by 2026-09-16 (the N x N result is Appendix A8 of the post, the hole is reframed, the companion site is deployed; co-authorship and the Persona Cartography reframing remain Samuel's decisions).
status: current
sources:
  - wiki/raw/external-review.md
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L11018
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L13056
  - qwen35/blog_page/index.html
  - qwen35/PHASE3_VERDICT.md
last_verified: 2026-09-16
tags: [conversation, review, blog, transcript-sourced]
---

# The external review of the blog page

An external reviewer read a draft of the weight-space blog page and sent back a
short review. Samuel pasted it into the session on **2026-09-02 at 20:33 UTC**
(transcript `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`, line 11018) and pasted
the identical text again on **2026-09-07 at 15:42 UTC** (same file, line 13056)
with the question "what did [the reviewer] mean by the aiming data point? has
our experiment finished running". The two pastes are byte-identical apart from
Samuel's question on the second. An unattributed paraphrase of the review is in
`wiki/raw/external-review.md`; the verbatim text was removed before the tree
was shared.

This review is the single largest external influence on the current shape of the
page. It caught one outright error, forced one central claim to be reworded, and
supplied the design for the experiment that is now the strongest positive control
in the project.

The review's points are given below as neutral paraphrases, not quotations.
**Every number below is quoted from the transcript**, with the line it came from.
The file-sourced versions of the same quantities live on the geometry pages —
[[seed-floor]], [[n-by-n-scoring]] and [[null-controls]] — and those are the ones
to copy into the blog post. Where a transcript number was later replaced, this
page says so.

## The five points

### 1. Show the inter-seed variance

**Asked for:** the inter-seed number to be foregrounded rather than buried.

**Response, same evening** (line 11041, 2026-09-02T20:35:17Z): "Inter-seed
variance is dramatic, and it's the right number to lead with." The numbers given
in that reply were same trait / different seed cosine **0.0166 = 89.0 degrees
apart**, different trait / different seed **0.0014**, matched trait ranked first
**40 / 40**. The reply read this as "virtually all of an adapter's direction is
the random draw; about 1.7% is the trait", and as the justification for the
zoo's shared LoRA-A initialisation: "without it everything sits 89 degrees from
everything."

**Done** (line 11094, 20:40): the inter-seed passage was moved into the opening
of the page and made to explain *why* the shared initialisation exists.

**Superseded numbers.** Half an hour later (line 11177, 21:04:51Z) the same session
found that the 40 seed-paired control adapters had been trained under a
*different objective* from the 134 zoo adapters, so 0.0166 was a seed-and-
objective floor, not a seed floor. The arm was retrained at the matched
objective; the current figure is **0.0181** (89 degrees), attenuation slope
**0.0265** against the r/d = 64/2560 = 0.025 prediction, identification still
40/40 (line 13065, 2026-09-07; the file-sourced version is in [[seed-floor]] and
in the 2026-09-03 addendum of `qwen35/PHASE3_VERDICT.md`). The separation margin
quoted at line 11177 as 0.01509 / 0.01490 became 0.01593 / 0.01584 after the
retrain, and was then removed from the page entirely — see
[[seed-paragraph-feedback]].

**Status on 2026-09-07: done.** The page carries 0.018, 89 degrees, 40 of 40, the
0.0265 attenuation and the 0.954 same-seed-different-objective check, plus the
stage-two replication (15 of 15, arrangement r = 0.98).

### 2. Is PC1 a Big Five factor?

**Asked for:** a check on the page's claim that PC1 was not a Big Five factor.

**Response** (line 11041): "PC1 — the reviewer is right, and my claim is wrong." The cosines
given in that reply:

| | Extra | Agree | Consc | Emoti | Intel |
|---|---|---|---|---|---|
| PC1 | +0.07 | +0.82 | -0.70 | -0.33 | +0.15 |
| PC2 | +0.81 | -0.29 | -0.55 | -0.34 | -0.33 |
| PC3 | -0.30 | +0.16 | -0.10 | +0.40 | -0.85 |

PC1 is a blend of Agreeableness and reversed Conscientiousness, not a direction
outside the Big Five; because eigenvector sign is arbitrary, the review's
conscientiousness reading and the page's warm-and-disorganised reading are the
same axis from opposite ends.

The check also turned up a second error the review had not asked about: the page
described PC2 as "close to Agreeableness" on the blind judge's say-so, while
geometrically PC2 is **+0.81 with Extraversion**. The session kept the
disagreement as a finding rather than deleting it — the judge reads bluntness as
disagreeable; the geometry says the direction is built from extraverted-versus-
conscientious traits.

**Status on 2026-09-07: done.** `qwen35/blog_page/index.html` carries the table
above and states "PC1 is not a direction outside the Big Five. It is a blend:
0.82 with Agreeableness and 0.70 with Conscientiousness in the opposite
direction." Restated at line 13065.

### 3. Per-PC trait loadings

**Asked for:** the top plus and minus five loading adjectives for every principal
component, not just for the factor solution.

**Response** (line 11041) gave them for PC1 to PC6:

| | top +5 | top -5 |
|---|---|---|
| PC1 | unsystematic, pleasant, effeminate, sympathetic, agreeable | unsympathetic, cold, unemotional, assertive, insensitive |
| PC2 | unrestrained, spunky, vigorous, shallow, energetic | introverted, careful, steady, dependable, conscientious |
| PC3 | unsophisticated, imperceptive, simple, uninquisitive, unintellectual | verbal, philosophical, deep, imaginative, impractical |
| PC4 | timid, self_pitying, fearful, guilty, bashful | trustful, unenvious, pleasant, agreeable, generous |
| PC5 | prompt, helpful, conscientious, mothering, high_strung | relaxed, untalkative, quiet, creative, unexcitable |
| PC6 | emotional, kind, warm, mothering, engaging | organized, shallow, haphazard, disorganized, imperceptive |

The reply noted that PC4's loadings independently confirm an earlier relabelling:
"timid, self-pitying, fearful, guilty is self-concern, not sycophancy."

**Status on 2026-09-07: done.** The table (five chart coordinates plus top plus
and minus five adjectives for PC1 to PC6) is in the components section of the
live page.

### 4. The aiming data point, and the N x N experiment

**Asked for:** replace a hand-picked positive control with an exhaustive one.

**What the review meant** (line 13065, 2026-09-07T15:43:34Z): the page's evidence that
the scoring identity "aims" data was a positive control on six traits the
assistant had chosen. The proposed design removes the choice — score every trait's own
preference pairs against every one of the 134 final adapters, giving a 134 x 134
matrix whose rows are data and whose columns are directions. If the identity
measures what it claims, the diagonal should beat everything else in its row.
"No selection, no interpretation, one picture."

**Done.** The old control tested six traits against their own directions (5 of 6
rank 1); the new one is the whole matrix. From line 13065: 134 traits x 40 pairs
each, against all 134 adapters, the trait's own adapter ranks **first for 134 of
134** (133 of 134 after standardising each adapter's column, with *bashful*
dropping to second). The off-diagonal is structured: where a row's runners-up are
not the trait itself, they share its factor and keying **42%** of the time
against a **12%** base rate, and share its factor with the opposite keying **0%**
of the time. The reading offered was that "the scorer sees the Big Five in the
data, not only in the weights."

**Status on 2026-09-07: done and passed.** It is in the positive-control caption
on the live page; the matrix is in `qwen35/analysis/nxn_summary.json`. See
[[n-by-n-scoring]] for the file-sourced version. Line 13065 notes the full matrix
"would make a good heatmap for the site" — that figure has not been built.

### 5. Naming the directions, and second authorship

**Asked for:** withdrawal of the claim that the unnamed direction is a character
English has no word for.

**Response** (line 11041): "On naming: the reviewer is right and this is the most important
correction. 'No English word names this' was never what I measured. What I
measured is that none of *our 134 sampled adjectives* is within 69 degrees.
English is far larger than our sample... The honest claim is about coverage by
the sample, not about the language — and it's directly testable by training those
three and seeing whether they land in the hole."

**Done twice.** First the wording was fixed (line 11094): "The page now says the
hole is in *the sample*, not in the language, and names the three suggested words
explicitly." Then the three words were trained as adapters on the zoo's prompts.
From line 13065: *cavalier* lands **82 degrees** from the hole, *blase* **70
degrees**, *insouciant* **40 degrees** in the five-dimensional chart — closer
than any existing adapter — but **72 degrees** in the full space where an
existing adapter already sits at **69 degrees**, with a **one-in-five** chance
level across three tries; and the three supposed synonyms are **77 to 89
degrees** from each other. A later activation-space transplant showed the hole
does not exist when the same constitutions are used as prompts.

**Status on 2026-09-07: done, and the claim is withdrawn.** The page no longer
says "a character English has no word for"; it says the direction is one the
adapters leave open, "suggestively insouciant, not convincingly". The live page
credits an external reviewer for the three words.

**Co-authorship for the reviewer, and the Persona Cartography reframing.**

**Status on 2026-09-07: open, Samuel's decision.** Both replies (lines 11094 and
13065) say authorship and the reframing are his to agree. Partial credit already
exists on the live page: the closing section names Persona Cartography and says
its "framing of personality as a mapped space rather than a set of points shaped
a good deal of what is above". The fuller rewrite the review asked for — setup and
closing saying plainly that the question and the method come from Persona
Cartography and that this is a weight-space instance of it — has not been made.

## The hosting request

The review asked for the page to be hosted somewhere, with a static version for
LessWrong.

**Status on 2026-09-07: open.** Line 13065: "static directory behind Caddy on a
`personas.` hostname, and a stripped version for LessWrong. Both are quick once
the write-up settles." Nothing has been deployed. The site design that came out
of this is [[blog-site-map-idea]].

## Side effect: the objective confound

The review's first point led to the biggest correction of the week, by an indirect
route. Samuel asked why the cross-seed test used only 40 adapters (line 11099,
2026-09-02T21:01:10Z). The answer (line 11177) was cost — "40 traits x $0.41 is
about $16.50, priced as a floor measurement rather than a full replication" — but
looking it up surfaced that **all 240 control adapters had been trained under a
different objective from the 134**: plain sigmoid DPO with `kl_coef 0.0`, against
`loss_type ["sigmoid","sft"]` with `kl_coef 0.001` for the zoo. The reply called
it "the project's third instance of a treatment silently not reaching the
container", flagged that the quoted 40/40 and the scree null both inherited the
confound, and costed the fixes at about $17 for the seed arm and about $82 for
the null arms. Samuel approved ("go for it", line 11182). Both retrains were done
and neither changed the qualitative answer; see [[seed-floor]] and
[[null-controls]].

The same reply noted the 40 seed-paired adapters were **stage-1 DPO only** — no
stage 2, no persona merge. That thread continues in
[[stage-one-versus-stage-two-clarification]].

## Summary table

| Point | Status on 2026-09-07 |
|---|---|
| Highlight inter-seed variance | Done; opening passage, numbers since corrected to the matched objective |
| Is PC1 a Big Five factor | Done; page claim was wrong and is fixed; PC2 also corrected |
| Per-PC trait loadings | Done; PC1 to PC6 table on the page |
| N x N aiming experiment | Done; 134 of 134, passed; heatmap figure not built |
| Naming the alien direction | Done; three words trained, claim withdrawn and reworded |
| Co-authorship for the reviewer | Open, Samuel's decision |
| Persona Cartography reframing | Partial; credit present, requested rewrite not made |
| Hosting plus a static LessWrong version | Open |

## Status on 2026-09-16

Read with the 2026-09-15 post draft ([[post-draft]]) in hand:

- Point 1 (inter-seed variance) is the section "What a trait's weight update
  actually is": cosine 0.018, 40 of 40, Pearson 0.997, and the column-space and
  activation-weighted results that explain why the number is small
  ([[column-space-structure]], [[activation-weighted-gram]]).
- Points 2 and 3 (PC1 as a Big Five blend; per-PC loadings) became moot for the
  post when the factor solution replaced the principal components as the primary
  frame on 2026-09-08 ([[factor-first-migration]]). The PC pages ([[factor-pc1]]
  to [[factor-pc6]]) keep the loadings and the correction.
- Point 4 (the N x N positive control) is Appendix A8 of the draft, credited to
  "an external reviewer" in Appendix A0. The heatmap figure named at line 13065
  has still not been drawn; the companion's methods page prints the 134 of 134
  statistic from `qwen35/analysis/nxn_summary.json#raw.top1`.
- Point 5 (naming the direction) went further than withdrawal: the widest gap is
  now read as a region the trait vocabulary did not sample, with the three
  proposed words, a slider and a matched-random steer all failing to make it a
  finding about the model (Appendix A10; [[hole-words-factor-chart]],
  [[alien-direction-factor-chart]]).
- Co-authorship and the Persona Cartography reframing: still Samuel's decisions.
  The draft opens with Persona Cartography as the question's origin ("What we
  built") and credits its dials in "Persona Cartography's dials replicate"; the
  reviewer is not named anywhere in the draft or the wiki.
- Hosting: the companion site is deployed at `https://persona.161-35-77-84.sslip.io`
  (2026-09-14) and the draft links it; a LessWrong-static version is the post
  itself. Both devbox sites will move before publication.

