---
title: Stage one versus stage two adapters, clarified
summary: Which population each geometry result is measured on - the 134 zoo adapters have both OCT stages, but the control arms and almost every published number are stage-one only.
status: current
sources:
  - wiki/raw/stage-one-versus-stage-two.md
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L11113
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L11177
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L12641
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L12919
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L13231
  - qwen35/PHASE3_VERDICT.md
last_verified: 2026-09-07
tags: [conversation, zoo, oct, transcript-sourced]
---

# Stage one versus stage two adapters, clarified

Samuel asked variants of the same question three times, because the answer
differs depending on which set of adapters you mean. Transcript references are to
`981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`; verbatim text in
`wiki/raw/stage-one-versus-stage-two.md`.

**The whole confusion is a population question. Read every sentence below with
its population attached.**

## The three questions

| When | What he asked | Line |
|---|---|---|
| 2026-09-02 21:01 | "and did we do full oct on them" (of the 40 seed-retrained adapters) | 11113, queued |
| 2026-09-05 20:09 | "did we get the power seeking / sycophancy adapters trained asw?" | 12641, queued |
| 2026-09-05 20:13 | "also arent' the 134 personalities trained with both stages of oct" | 12642, queued |

The 2026-09-02 question got a direct answer. **The two 2026-09-05 questions never
got a direct text answer**: both were typed while the assistant was working, the
transcript marks them `absorbed_mid_turn` at 20:16 UTC, and the records that
follow are thinking and tool calls only until a status note at 20:58. The
substance arrives instead in the summary at line 12919 and in the page changes
described at line 12639.

## The answers, by population

**The 134 zoo adapters: both stages exist.** From line 12919
(2026-09-05T21:46:55Z), describing what was put on the page: "the setup now says
every geometric object is the stage-one adapter, that stage two ran for all 134
and is on HF, and gives both stage-two numbers (r = 0.79 arrangement match with
stage one; 15/15 and r = 0.98 across seeds)." So Samuel's premise at line 12642
was right — the 134 personalities did go through both stages — but the geometry
that the blog page draws is measured on the **stage-one** LoRA unless a caption
says otherwise.

**The 40 seed-paired control adapters: stage one only.** From line 11177
(2026-09-02T21:04:51Z): "**Full OCT: no.** Stage-1 DPO only, no stage 2, no
persona merge." This is the arm behind the 40/40 cross-seed identification and
the 0.018 seed floor. See [[seed-floor]].

**Why they are stage one.** Stage two is a second LoRA with its own random
initialisation, so trait for trait the stage-two adapters are near-orthogonal to
the stage-one ones in *coordinates*. Their *arrangement* is what reproduces. The
same reasoning is why the zoo's own shared initialisation matters at all — see
[[most-interesting-claims]] for the design statement.

**Stage two at a second seed: run for 15 traits.** From line 12919: all 15
traits, every stage, seed and root verified in the run metadata. Against the 134
seed-0 stage-two adapters, **15 of 15** find their own trait as nearest
neighbour; same-trait cosine **0.067** against stage one's **0.018**; the two
seeds' cosine matrices over the 15 traits agree at **r = 0.98**. One thing left
unexplained: the cross-seed attenuation slope is **0.116**, about **4.6 times**
the r/d prediction that held so exactly at stage 1. It is not the A subspaces
(their overlap is 0.022, random); "the extra signal sits in a component every
stage-two adapter shares (the different-factor floor is 0.014 against 0.002 at
stage 1). Noted in the verdict file as unresolved." The 2026-09-05 addendum of
`qwen35/PHASE3_VERDICT.md` carries the same figures and adds that stage-2 LoRA-A
drifts 10% from initialisation against 1.5% at stage 1.

**The alignment adapters (power-seeking, sycophancy, corrigibility).** The
2026-09-05 question about these was never answered in words, but line 12639
(20:06:25Z) records "a paragraph on the four alignment adapters and the 60-degree
synonym floor" going onto the page that evening, so they existed by then. The
2026-09-03 addendum of `qwen35/PHASE3_VERDICT.md` names all four — sycophantic,
obsequious, power-seeking, corrigible — retrained on the zoo's shared prompt pool
and gives their angles. They are a separate arm from the 134; the labelled tests
exclude them.

## What is still open

On **2026-09-07 at 16:15 UTC** Samuel queued (line 13231) a quotation of the
page's own sentence — "Every geometric object on this page, unless it says
otherwise, is the stage-one adapter" — and added:

> "i think it would make sense to replicate with the full oct synthesis
> adapters"

This is a request to redo the weight-space geometry on the stage-two (merged
persona) adapters rather than the stage-one DPO ones. **As of 2026-09-07 it has
not been done.** The session began reading how the merged persona adapters are
stored and whether their geometry was already computed (line 13334), and no
result exists in the transcripts. The only stage-two geometry currently on record
is the arrangement match (r = 0.79 against stage one) and the second-seed arm
(15/15, r = 0.98) quoted above.

## Status

`status: current`. The stage-two numbers agree with the 2026-09-05 addendum of
`qwen35/PHASE3_VERDICT.md`. The page records, rather than resolves, the fact that
two of Samuel's three questions were absorbed mid-turn and answered only
indirectly.
