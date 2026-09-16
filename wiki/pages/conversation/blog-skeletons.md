---
title: Three blog post skeletons
summary: The three bullet-pointed outlines produced on 2026-09-04 when Samuel decided to rewrite the blog post by hand, with the pending items resolved.
status: current
sources:
  - wiki/raw/blog-skeletons.md
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L12106
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L12109
last_verified: 2026-09-07
tags: [conversation, blog, transcript-sourced]
---

# Three blog post skeletons

On **2026-09-04 at 15:06 UTC** Samuel wrote (transcript
`981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`, line 12106):

> "so i'm going to rewrite the whole blog post to be human written, can you give
> me a few bullet pointed skeletons to work from (feel free to include an
> appendix if suitable but not necessary)"

The answer (line 12109, 2026-09-04T15:07:33Z) gave three, "each with a different
spine". The verbatim text is in `wiki/raw/blog-skeletons.md`. Reproduced below,
lightly reformatted, with the numbers as they stood on 2026-09-04. Three items
were marked *pending* at the time; all three have since landed and the values are
given at the end.

## Skeleton A: results-led, the current page tightened

*Best if you want to keep the existing figures and interactives and mostly cut
prose.*

- **Hook.** 134 personality adapters on one base model; the geometry of the
  weight deltas recovers the Big Five, and shows what the lexicon leaves out.
- **Setup (short).** OCT stage-1 DPO, one trait per adapter, shared 445-prompt
  pool, shared LoRA init (A drifts 1.5%). Persona Cartography as the framing
  reference.
- **The number that comes first: the seed floor.** Same trait, second seed:
  cosine 0.018, yet 40/40 nearest-neighbour among 134; whole geometry transfers
  scaled by 0.0265 (r/d predicts 0.0250). Same seed, different objective: 0.954.
  So the initialisation is the only thing that matters, and it is why everything
  shares one.
- **The map.** 2D versus 3D; say plainly which was more informative and why.
- **The six components.** One qualitative paragraph and two transcripts each. Say
  what PC1 is. Name the personality axis with the caveat that the definite
  article does no work.
- **Sampling the space.** Steering explorer and the 72-point sphere; smoothness
  rho = 0.65; 48 of 72 directions produce no looping (state it, it deflates the
  coherence story).
- **Rotating to simple structure.** PAF versus PCA: different matrices, different
  elbows, never share an axis. Trained-null comparison for the elbow.
- **Where no word goes.** The hole (68.9 degrees full-space, 52.5 degrees in
  top-5); alien direction 5/5 predicted signs, r = 0.81; random control does
  *more* damage. Then the hole-word test: insouciant 39.6 degrees in-plane, but
  71.6 degrees full-space and a 20% chance level. Suggestive, not a name.
- **Aiming data at a direction.** The scoring identity (directional derivative
  equals one backward pass, r = 0.9999992 against finite differences). Six-trait
  positive control 5/6, then N x N *(pending)*. Optimiser gains: PC4 +90%, alien
  +53%, Agreeableness 0%.
- **Does it survive training.** 3/3 diagonal dominance on the retrained
  verification adapters.
- **Alignment traits (new section).** Sycophancy lands on +Agreeableness, not
  PC4; power-seeking 72.7 degrees from everything; corrigible 68.0 degrees; the
  synonym pair at 61.9 degrees sets an approximately 60-degree adapter-to-adapter
  noise floor that every other angle must be read against.
- **What this does and does not establish.**
- *Appendix:* scoring identity derivation; PCA/PAF definitions; pre-registrations
  and their outcomes; objective/corpus mismatch record.

## Skeleton B: three questions

*Best if you want a shorter, argumentative post that a LessWrong reader can
follow without the interactives.*

- **Q1. Is there structure in weight space at all, or is it noise?**
  - Seed floor, 40 of 40, attenuation constant predicted in advance.
  - Objective versus seed decomposition (0.954 against 0.018) — one paragraph, it
    is the cleanest single fact in the piece.
  - Null arms: shuffled and permuted show no factor structure *(matched-objective
    versions pending; the old ones are conservative but confounded, say so
    once)*.
- **Q2. Is the structure the Big Five?**
  - Yes at the axis level: signed factor separation survives both nulls and
    replicates at seed 1.
  - Only partly at the component level: PC1 is a common direction, the
    personality axis; the Big Five appear from PC2 on and rotate cleanly under
    PAF. **(This line disagrees with the live page — see below.)**
  - The lexicon coverage result: sign flips at 3D (0.9-degree hole in 2D, 52.5
    and 68.9 degrees at 5D) — words cover the plane, not the space.
- **Q3. Can you use the geometry?**
  - Read directions out loud (steering, transcripts).
  - Aim data at a direction (identity plus optimiser); the positive control and
    N x N.
  - Alignment-relevant traits as a stress test: sycophancy is Agreeableness;
    power-seeking is genuinely off-lexicon; corrigibility is not where predicted.
  - The hole test as an honest miss: three candidate names, one suggestive, none
    confirmed.
- **Close.** What a second seed at stage 2 would settle *(pending decision)*;
  what is not established (per-trait direction versus factor organisation;
  behaviour versus weights).
- *Appendix:* the pre-registration table (prediction / outcome), including the
  two failures.

## Skeleton C: one idea, followed all the way

*Best if you want the post to be about the method rather than the zoo. Leads with
the identity, treats the geometry as what it lets you see.*

- **Opening claim.** A LoRA adapter is a direction; you can measure how any text
  projects onto that direction with one backward pass. Everything else follows.
- **Part 1: the direction has to be real.** Why shared init matters (r/d = 2.5%
  overlap between two inits); seed floor; objective check. Without this section
  the rest is about random subspaces.
- **Part 2: what the directions look like together.** Map, components, PAF,
  personality axis. Shorter than in A; one transcript per component.
- **Part 3: the identity.** Derivation in four lines (B = 0 at init; first update
  proportional to G A-transpose A; overlap collapses to the inner product of
  dL/dB with B_U; that is a directional derivative). Validation r = 0.9999992.
  Batched form: one scalar epsilon per (target, example), read `eps.grad`.
- **Part 4: using it.** Positive control, then N x N *(pending)*. Optimiser: PC4
  +90%, alien +53%, Agreeableness 0% and why 0% is informative. Verification by
  retraining: 3/3.
- **Part 5: where the lexicon runs out.** The hole; alien direction with 5/5
  signs; hole-word test; power-seeking as an off-lexicon trait the identity can
  still aim at.
- **Part 6: noise floors, stated as such.** 60-degree synonym floor; 89-degree
  seed floor; the 20% chance level for three hole candidates.
- **Close.** What changes if stage 2 replicates at a second seed.
- *Appendix:* full trait list with factor loadings; interactive links (map,
  steering explorer, sphere) hosted separately.

## The two notes attached to all three

- The external review asked for the inter-seed number to be prominent. In every skeleton it is
  in the first third; the 0.954-against-0.018 pair is the version to lead with.
  See [[external-review]].
- Three results were still landing and would each get one line: N x N diagonal
  dominance, matched-null scree comparison, and (if approved) stage 2 at seed 1.
  "I'd leave placeholders rather than write around them."

## What has changed since 2026-09-04

**All three pending items landed.** The placeholders can be filled with:

- **N x N** — 134 of 134 traits rank their own adapter first (133 of 134 after
  column standardisation); runners-up share factor and keying 42% of the time
  against a 12% base rate. See [[n-by-n-scoring]].
- **Matched-null scree** — 11 real components above the structureless (shuffled)
  arm, 0 above the permuted arm; identical to the mismatched arms. The
  "conservative but confounded" hedge in Skeleton B is no longer needed. See
  [[null-controls]].
- **Stage 2 at seed 1** — approved on 2026-09-05 and run for 15 traits: 15 of 15
  identification, the two seeds' cosine matrices agreeing at r = 0.98. See
  [[stage-one-versus-stage-two-clarification]].

**One line in Skeleton B contradicts the live page.** Skeleton B's Q2 says "PC1
is a common direction, the personality axis; the Big Five appear from PC2 on".
Two days earlier (line 11041, 2026-09-02) the same session had already
established that PC1 is a blend of Agreeableness (+0.82) and reversed
Conscientiousness (-0.70), and the explainer at line 12973 says the personality
axis is the *mean adapter*, reported separately, with the principal components
describing variation around it. `qwen35/blog_page/index.html` agrees with the
latter: "PC1 is not a direction outside the Big Five. It is a blend." Do not
carry the Skeleton B phrasing into the post. Recorded for the maintainer in
`pages/conversation/_report.md`.

**The plan the skeletons were written for has since been revised.** On
2026-09-07 Samuel proposed a short post plus a website instead of one long post;
see [[blog-site-map-idea]]. The skeletons remain useful as the argument order for the
short post, and Skeleton B's question spine is close to the "paths" idea in the
site map.
