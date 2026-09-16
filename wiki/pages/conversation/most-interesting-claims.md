---
title: The most interesting claims, ranked
summary: The 2026-09-07 ranking of the project's five strongest claims and three weak ones, plus the follow-up on exactly what the seed experiment holds constant.
status: historical
sources:
  - wiki/raw/most-interesting-claims.md
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L13033
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L13036
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L13041
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L13044
last_verified: 2026-09-16
tags: [conversation, summary, transcript-sourced]
---


> Status note (2026-09-16): Superseded as a drafting aid by the claims table of the 2026-09-15 draft, opened into files and keys on [[claims-and-evidence]]; kept as the record of the 2026-09-07 ranking.
# The most interesting claims, ranked

On **2026-09-07 at 15:39 UTC** Samuel asked (transcript
`981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`, line 13033): "what do you think are
the most interesting claims from this?" The answer at line 13036 (2026-09-07T15:39:31Z) ranked them "by
how much they'd change what a careful reader believes, weighed against how well
they're supported", and then listed three that are weaker than they look. All
numbers below are quoted from that transcript record; the file-sourced versions
are on the geometry and behaviour pages.

## The five

**1. Prompting and training put the model in the same place, and the arrangement
is the same.** An adapter trained on a constitution's data moves the residual
stream as far as the constitution does as a prompt (ratio **1.01**), **62%**
along the same direction, and the 134 adapters are arranged in activation space
the way they are arranged in weight space (**0.865**) and the way the prompts are
arranged (**0.70**). Nominated as the claim to lead with: "It says the weight
geometry isn't a curiosity of LoRA — it's the same geometry the model already
uses when you just *tell* it who to be." Also the one nobody set out to find —
"the activation arm was an afternoon's check."

**2. Personality directions survive re-initialisation, and the loss is exactly
predicted.** 40/40 identification across seeds at a cosine of **0.018**, with the
attenuation constant coming out at **0.0265** against a predicted **0.025**, and
the objective check (**0.954** against **0.018**) isolating the cause. "Small
point in the post, but the most *rigorous* thing in the project." See
[[seed-floor]].

**3. Adapter and prompt saturate rather than stack, and neither overrides the
other.** Matched: **1.18** against **1.00** (prompt alone) and **0.70** (adapter
alone). In conflict: about half of each. Additive on the prompt tokens,
sub-additive once the model speaks. Practical bite: "if you fine-tune a trait in
and then system-prompt against it, you get a mixture, not an override."

**4. The data-aiming identity works without reading any text.** One backward
pass, **r = 0.9999992** against finite differences, **134/134** diagonal
dominance, and the runners-up carrying factor structure (**42%** against a
**12%** base rate). The optimiser then produced data that trained real adapters
toward an unnamed direction. "The most *useful* result and probably the most
novel method." See [[n-by-n-scoring]].

**5. The Big Five content is there; the E/ES axes are rotated.** Three of five
factors recover cleanly; two come out as withdrawal and arousal. "A concrete
claim about which rotation of the E x N plane a language model prefers" —
flagged as the kind of thing that could be an artefact of these particular
adapters, which is why the activation-space factoring should come first. See
[[explainer-big-five-mapping]].

## The three that are weaker than they look

- **The hole.** It does not exist in activation space (the transplant landed at
  the null), the three candidate names disagree with each other by **77 to 89
  degrees**, and insouciant's 40 degrees is a one-in-five event across three
  tries. What survives is a real negative — words for the same disposition do not
  agree about where it is — but "a character English has no word for" is "not a
  claim I'd make anymore. It was the most quotable line on the page and it's the
  one I trust least." See [[external-review]].
- **The scree and "six components".** Eleven components exceed the structureless
  null; six was a reading choice. "Fine to describe six, but not as a finding
  about dimensionality." See [[explainer-elbow-figure]].
- **Big Five as signed axes in weight space, on its own.** True, p at the floor,
  both nulls flat, but the constitutions are text about the traits and the
  training data was generated to exhibit them. "That structure *should* be there.
  It's the necessary check, not the result; the result is what's built on top of
  it."

## The one sentence

> "the way a language model is arranged by what it's told to be is the way it's
> arranged by what it's trained to be, and that arrangement is measurable,
> seed-stable, and steerable."

## Follow-up: what is held constant, and are the LoRAs initialised similarly?

Samuel quoted claim 2 back and asked (line 13041, 15:40:56Z): "what do we hold
constant here? are loras initialised similarly?" The answer (line 13044, 15:41:29Z) is the
clearest statement in the transcripts of the zoo's experimental design.

**Within the zoo (seed 0).** All 134 adapters are initialised **identically, not
just similarly**: LoRA's B starts at zero always, and A is drawn from a seeded
Gaussian with the same seed, so every trait's A is the same matrix, module for
module. Training barely moves A (**1.5%** of its norm), so all 134 adapters write
their update B x A into the same 64-dimensional input subspace per module. That
is what makes within-run cosines large and meaningful — same-factor,
same-keying pairs at **+0.25** — and it was a deliberate design choice made
before the geometry was measured.

**The seed-paired arm (seed 1).** 40 of those traits retrained with everything
held fixed — base model, byte-identical preference pairs per trait (the corpus
hash is checked in the log), prompt pool, objective (after the matched retrain),
hyperparameters, step count — except the seed. The seed controls two things: the
draw of A, and the shuffle order of the training data (`--seed 1
--order-seed 1`). **So the arm varies initialisation and data order together;
there is no arm that changes order alone.**

**Why the cosine has to be small.** ΔW = B x A lives in the row space of A. Two
independent random rank-64 subspaces of a 2560-dimensional space overlap by
r/d = **2.5%** in expectation, so any two adapters from different seeds have
their inner products squashed by that factor. "The absolute number 0.018 is not
the finding; it's what geometry forces." The findings are relative: same-trait
**0.018** against different-factor **0.0018** (tenfold), 40 of 40 nearest
neighbours, and the entire 134 x 40 cross-seed block equalling the within-run
block times **0.0265**.

**What 0.0265 says about the order confound.** If changing data order were also
perturbing B's path in ways that do not survive re-initialisation, the cross-seed
block would fall *below* the subspace-overlap prediction. It lands slightly above
it (0.0265 against 0.0250, Pearson **0.997**). "So within the precision of this
arm, order contributes nothing detectable; the attenuation is fully accounted for
by A's subspace."

**Direct confirmation.** For stage 2 the A matrices themselves were measured: two
seed-0 adapters of different traits share their A row space at **0.991**; a
seed-0 and a seed-1 adapter of the *same* trait share it at **0.022**. "The zoo's
shared initialisation is literal."

**The honest limit.** The r/d prediction assumes B is unconstrained and A is
random. At stage 2 the same arm gave a slope of **0.116**, **4.6 times** the
prediction, with a shared cross-seed component every adapter carries. "So
'exactly predicted' holds for stage-1 DPO adapters whose A barely moves; where A
drifts 10% and picks up shared directions, the simple account stops being
sufficient. The post should say the prediction was tested where it applies, not
that it's a law." See [[stage-one-versus-stage-two-clarification]].

## Status

`status: current`. The numbers checked against `qwen35/blog_page/index.html` and
the 2026-09-05 addenda of `qwen35/PHASE3_VERDICT.md` are 0.018, 0.0265, 0.954,
134 of 134, 42% against 12%, the stage-2 slope of 0.116, the 1.01 magnitude
ratio, the 62% direction share, and the 1.18 / 1.00 / 0.70 saturation triple; all
agree. The activation-versus-weight arrangement figure of 0.865 is not on the
blog page but appears in `qwen35/ACTSPACE_RESULTS.md` line 32. The ranking itself
is a judgement, not a measurement; the "weaker than they look" list is the part most worth carrying
into the post.
