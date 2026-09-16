---
title: A short post plus a whole website
summary: Samuel's 2026-09-07 proposal to publish a short post backed by a choose-your-own-adventure site, the ten-page map that came back, and the case for per-trait pages.
status: historical
sources:
  - wiki/raw/blog-site-map-idea.md
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L13018
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L13021
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L13071
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L13081
last_verified: 2026-09-16
tags: [conversation, blog, site, transcript-sourced]
---


> Status note (2026-09-16): The site was built as the companion (`qwen35/companion/`, 2026-09-14) and this proposal is kept as its origin; see [[built-pages-inventory]] and [[post-draft]].
# A short post plus a whole website

On **2026-09-07 at 15:36 UTC** (transcript
`981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`, line 13018) Samuel proposed writing
"a really short post about investigating the personality of llms in weight space"
and accompanying it with a whole website, choose-your-own-adventure style with
expandable sections. Fifteen minutes later (line 13071) he asked whether each
trait should get its own page. The answers are at lines 13021 and 13081; the
verbatim text is in `wiki/raw/blog-site-map-idea.md`.

This proposal replaces the single-long-post plan the three
[[blog-skeletons|skeletons]] were written for. Nothing has been built yet.

## The two conditions

The answer (line 13021, 15:37:38Z) agreed with the idea and attached two
conditions.

1. **The short post must carry the argument on its own**, including the strongest
   evidence for each claim, and must not be a trailer for the site. "A reader who
   never clicks should leave persuaded or informed enough to disagree."
2. **The branches must follow reader questions, not the project's history.** The
   natural branches are *Is this real or an artefact?*, *Is it the Big Five?*,
   *Show me the personalities*, and *How does the method work?* — each a path
   through the same pages in a different order. "If the site instead mirrors the
   phases of the project it becomes the current page with more clicking."

The reason the split works at all is that it cuts by **depth rather than by
topic**: four claims that each fit in a sentence with a number, each backed by a
deep tree of controls, transcripts and failed pre-registrations that only a
fraction of readers want. As it stands, "the skimmer gets 5 MB and the sceptic
can't find the null table."

One rule for the expandables: **every expander's header is the question it
answers** ("Why does the permuted null overlap the real curve?"), never a topic
label. The reader then chooses depth by whether the question is theirs.

## The map: a hub and nine pages

Each page has one figure above the fold, then expanders in three depths — claim,
evidence, everything (tables, controls, pre-registrations and their outcomes).

**0. The post.** 800 to 1,000 words, standalone. Thesis: a personality is a
direction in weight space; the directions are organised; the organisation is not
an artefact; you can aim training data at it. Four numbers: 40/40 across seeds;
Big Five as signed axes at p at the permutation floor with two nulls flat;
r = 0.70 with the geometry the same constitutions produce as prompts; 134/134 on
the data-aiming test. One figure (the 3D map, static). Links into the paths.

**1. The zoo.** What was trained: 100 Goldberg markers plus 34 lexicon words
(with the six refusals), constitutions, the two OCT stages, the shared
initialisation and why. A card per trait: constitution excerpt, two transcripts,
Big Five chart coordinates, three nearest neighbours in weight space, Hugging
Face link.

**2. The map.** The rotatable PCA in 2D and 3D with the verdict on which is more
informative; the six components each with a qualitative description and two
transcripts; the personality axis; the scree against two nulls; the PCA/PAF
distinction.

**3. Is it real?** The sceptic's page. Seed floor (0.018, 40/40, slope 0.0265
against r/d 0.025), the objective check (0.954), the null table (shuffled and
permuted, both at the matched objective), stage 2 across seeds (15/15, r = 0.98),
and the activation-space correspondence as an optimiser-independent replication.
Expander: the pre-registered bar of +0.2447 and why it was the wrong instrument.

**4. Big Five or not?** Signed separation and bipolarity by factor, the factor
solution with the congruence table above, the E/ES rotation, parallel analysis
and why 5 versus 9, the "valence axis wearing five costumes" check.

**5. Sampling the space.** Steering explorer, the 72-point sphere with its judged
transcripts (including the 48 of 72 that produce no looping), smoothness
rho = 0.65, additivity of adapters.

**6. Where no word goes.** The hole by dimension (sign flip at 3D), the alien
direction with its transcripts and both controls, the three candidate names with
the insouciant result, the activation transplant showing it is an adapter-space
property, and the four alignment traits with the 60-degree synonym floor.

**7. Aiming data.** The identity in four lines, validation at r = 0.9999992, the
six-trait control, the **134 x 134 heatmap** (the full matrix is in
`analysis/nxn_summary.json`), the optimiser runs with example optimised pairs,
the retrain verification 3/3, the 0% Agreeableness gain and why it is
informative.

**8. Prompting versus training.** Layer curves, the three-way geometry, magnitude
and direction of the adapter shift, the 16 x 16 composition as two heatmaps
(along-prompt, along-adapter), conflict transcripts (adapter *quiet* under prompt
*bold* and the reverse).

**9. Methods, costs and what went wrong.** Pipeline, budgets ($2,240 total, per
arm), the gates, and the four "treatment didn't travel" incidents with how each
was caught. "This is the page [the reviewer] and anyone reproducing it will read, and the
honesty is an asset here even though it stays out of the post." Plus a glossary:
Gram, double centring, cosine as angle, LoRA r/d, PCA versus PAF.

Each page's answer also named the JSON and log files the page would be built
from; those pointers are in `wiki/raw/blog-site-map-idea.md`.

## The paths

Rendered as ordered "next" links so the pages stay single-sourced.

| Reader | Path |
|---|---|
| Sceptic | 0, 3, 4, 9 |
| Psychologist | 0, 4, 2, 6 |
| Engineer | 0, 7, 8, 9 |
| Browser | 0, 1, 5, 6 |

## How much to build

"All of it exists and most is already computed; the constraint is page weight,
not material." Rule of thumb: every page under 1 MB, interactives on their own
pages, transcripts loaded per card. **The only genuinely new figures are the
N x N heatmap and the two composition heatmaps**; everything else is a re-cut of
existing figures and JSON. Hosting is a static directory behind Caddy on a
`personas.` hostname, with no build step beyond the existing generator.

## A page for each trait

Samuel (line 13071, 15:51:01Z): "do you think it'd make sense to have a page (or
selectable subpage) for each trait". The answer (line 13081, 15:51:36Z): yes, and probably
the most-used part of the site — "people look up their own trait first". Three
reasons beyond that:

- Every claim on the topic pages is an aggregate over 134 points; a trait page is
  where a reader checks that the point behind the aggregate is real.
- Almost everything needed already exists per trait, so it is a template and a
  generator, not new work.
- It gives the extra adapters — sycophantic, power-seeking, corrigible,
  insouciant and the rest — a home where they are clearly marked as outside the
  zoo rather than smuggled into a table.

**What one trait page holds**, all from data on disk:

- *Who this is.* The constitution. One training pair: prompt, chosen, rejected.
  Provenance tag: Goldberg marker with factor and keying, or lexicon word.
- *What it sounds like.* Four transcripts on the same prompt: base model; base
  with the constitution as system prompt; base with the adapter and no prompt;
  stage-two persona. The first three exist for all 134; the fourth for the traits
  that went through the stage-two eval. Plus the steering strip: the adapter at
  alpha = -2, -1, +1, +2 with judge scores.
- *Where it sits.* Big Five chart coordinates; PC1 to PC6 scores; the five
  nearest neighbours in weight space and the five nearest in activation space
  (usually overlapping, sometimes not — those cases are interesting); factor
  loadings from the k = 5 solution; its row of the 134 x 134 aiming matrix.
- *How stable it is.* For the 40 seed-paired traits, the cross-seed cosine and
  rank; for the 15 stage-two traits, the same. Training curve numbers. Links to
  the stage-one and stage-two adapters on Hugging Face.

**Two things that make it more than 134 copies of a form.** An index that is a
sorted table rather than a list — by factor, by chart coordinate, by how well the
adapter and prompt agree — so the pages are reachable by question. And a compare
view: two traits side by side with the angle between them, which is where an
observation like "sycophantic and obsequious are 62 degrees apart" becomes
something a reader can make for themselves.

**Where it would mislead.** Per-trait pages invite reading the PC scores and
chart coordinates as facts about the *trait* rather than about this adapter under
this seed. The seed-stability section is the antidote, and pages for traits
outside the 40 should say plainly that their coordinates have not been
replicated.

Weight is not a problem: each page lands well under 100 KB static.

## Status

Proposal only. Nothing in the map has been built as of 2026-09-07. The
[[external-review|hosting request]] from the external reviewer is the same open item. This
wiki's own `pages/traits/` directory is the closest existing analogue to the
per-trait pages described here.
