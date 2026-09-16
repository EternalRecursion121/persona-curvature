---
title: Index of Samuel's questions about the science
summary: Every user turn across the thirteen session transcripts that asks a question about the science rather than giving an instruction, dated, with where the answer now lives.
status: current
sources:
  - /home/vibe12/.claude/projects/-home-vibe12-projects/762268f1-f732-48c7-8275-e9a5fb631136.jsonl
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl
last_verified: 2026-09-07
tags: [conversation, index, transcript-sourced]
---

# Index of Samuel's questions about the science

Operational instructions ("train the rest", "upload to HF", "how much longer")
are in [[decisions-log]]. This page lists the turns that ask something about what
the results mean.

All questions below are from the two persona-curvature sessions, both with
working directory `/home/vibe12/projects`:

- `762268f1` = `762268f1-f732-48c7-8275-e9a5fb631136.jsonl` (2026-08-22 to 08-25)
- `981fa3b5` = `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl` (2026-08-25 to 09-07)

The other eleven transcripts in
`/home/vibe12/.claude/projects/-home-vibe12-projects/` are about the agent
harness, a Discord bot, a Tajik language-learning game, and a separate
reward-hacking experiment; none contains a question about this project.

"Where the answer lives" gives the wiki page that now holds the answer. Where no
single page covers it, a directory is named in plain text instead.

## 2026-08-22 to 08-24

| Date, time (UTC) | Question | Line | Where the answer lives |
|---|---|---|---|
| 08-22 15:49:00 | "please explain the persona curvature stuff in a simpler way" | `762268f1`:177 | [[home]] |
| 08-22 16:21:15 | "what traits were the adapters trained on and what methodology?" | `762268f1`:252 | [[zoo-construction-overview]] |
| 08-23 10:20:39 | "do you think umap might give us anything interesting" | `762268f1`:1773 | [[geometry-overview]] |
| 08-24 12:24:47 | "what rank were we training" | `762268f1`:1974 | [[stage-one-training-config]] (rank 64; the r/d = 64/2560 argument on [[seed-floor]] depends on it) |

## 2026-08-28 to 08-29

| Date, time (UTC) | Question | Line | Where the answer lives |
|---|---|---|---|
| 08-28 12:41:55 | "wait also rq is the pca 1 still something like the assistant axis" | `981fa3b5`:2884 | [[factor-pc1]]; the naming was later rejected, see [[decisions-log]] 2026-09-01 17:29 |
| 08-28 12:44:11 | "also please search for the assistant axis paper and see how it relates to our experiments" | `981fa3b5`:2923 | [[paper-assistant-axis]] |
| 08-29 10:32:08 | "are there any traits you think are worth redoing because of the ceiling" | `981fa3b5`:3342 | [[judged-evaluations]] and [[zoo-construction-overview]] |
| 08-29 10:43:01 | "can i see the analysis in realtime, like what are the principle componeents/factors" | `981fa3b5`:3473 | `pages/factors/`, one page per component and factor |
| 08-29 10:49:33 | "wonder if there's any advantage with umap" | `981fa3b5`:3529 | [[geometry-overview]] |
| 08-29 12:49:45 | "what does it make sense to retrain?" | `981fa3b5`:3798 | [[zoo-construction-overview]] |
| 08-29 16:37:20 | "where did the 100 traits come from?" | `981fa3b5`:4019 | [[provenance-feedback]] (asked again and answered fully on 2026-09-05) |
| 08-29 17:39:47 | "when you say same geometry but disjoint subspaces, what does this mean" | `981fa3b5`:4403 | [[seed-floor]]; the fullest statement is in [[most-interesting-claims]] |
| 08-29 19:36:03 | "what are the implications from the manifold stuff - also can we get intrinsic coordinates from just big five judging for the 100 traits" | `981fa3b5`:4850 | [[geometry-overview]] |
| 08-29 19:50:07 | "what is spearman brown and what did you use it for - what does the result mean" | `981fa3b5`:4920 | [[judged-evaluations]] |

## 2026-08-30 to 08-31

| Date, time (UTC) | Question | Line | Where the answer lives |
|---|---|---|---|
| 08-30 09:52:06 | "also how did the RL go" | `981fa3b5`:6193 | [[rl-capability-and-persona-drift]] and [[reward-hacks-arms]] |
| 08-30 20:37:41 | "how are you derviing the basis / training the probe" | `981fa3b5`:6313 | [[geometry-overview]] |
| 08-30 21:12:03 | "what is gram" | `981fa3b5`:6320 | [[explainer-double-centred-gram]] (fuller answer given 2026-09-07) |
| 08-30 21:29:26 | "do we have results to replicate this diagram" (a screenshot Samuel uploaded) | `981fa3b5`:6501 | [[geometry-overview]] |
| 08-31 15:24:16 | "why does training randomly initialised loras find basically orthogonal directions" | `981fa3b5`:6565 | [[most-interesting-claims]] (the r/d argument) and [[seed-floor]] |
| 08-31 15:27:38 | "we're comparing the AxB right not just the individual matrices?" | `981fa3b5`:6584 | [[explainer-double-centred-gram]] |
| 08-31 15:32:22 | "how would we monitor personality in weight space - how would we find the manifold instead of points" | `981fa3b5`:6616 | [[geometry-overview]]; the framing decision it produced is in [[decisions-log]] 2026-08-30 10:26 |
| 08-31 15:32:36 | "would we use something like assistant axis" | `981fa3b5`:6622 | [[geometry-overview]] |
| 08-31 15:36:36 | "why can't we project onto the big five subspace we have defined" | `981fa3b5`:6628 | [[geometry-overview]] |
| 08-31 15:40:56 | "how reliable is this given that we're measuring against points and not a manifold" | `981fa3b5`:6662 | [[geometry-overview]] |

## 2026-09-01 to 09-02

| Date, time (UTC) | Question | Line | Where the answer lives |
|---|---|---|---|
| 09-01 12:56:28 | "how do you make sure that the lora fine tuning targets same dims as the personality fine tuning" | `981fa3b5`:7726 | [[most-interesting-claims]] (shared A subspace) |
| 09-01 12:58:42 | "does the A LoRA not really move much during training - also why were all the personality adapters using the same lora A initialisation" | `981fa3b5`:7741 | [[most-interesting-claims]]; A drifts 1.5% at stage 1, 10% at stage 2, see [[stage-one-versus-stage-two-clarification]] |
| 09-01 14:08:27 | "wait so nothing interesting in the school or reward hacks adapter?" | `981fa3b5`:8030 | [[reward-hacks-arms]] |
| 09-02 20:15:27 | "what is an elbow" | `981fa3b5`:10916 | [[explainer-elbow-figure]] |
| 09-02 20:18:17 | "are you sure the pca elbow is in the right place" | `981fa3b5`:10924 | [[explainer-elbow-figure]] — the answer was no, and the figure was rebuilt |
| 09-02 21:01:10 | "why did we only check against 40 adapters?" | `981fa3b5`:11099 | [[external-review]] and [[seed-floor]]; this question surfaced the objective confound |
| 09-02 21:01:34 | "and did we do full oct on them" | `981fa3b5`:11113 | [[stage-one-versus-stage-two-clarification]] |

## 2026-09-04 to 09-05

| Date, time (UTC) | Question | Line | Where the answer lives |
|---|---|---|---|
| 09-04 12:44:27 | "did you do multi seed for full oct run?" | `981fa3b5`:11668 | [[stage-one-versus-stage-two-clarification]] |
| 09-04 14:46:00 | "what is the nxn thing?" | `981fa3b5`:12098 | [[n-by-n-scoring]] and [[external-review]] |
| 09-05 20:09:13 | "did we get the power seeking / sycophancy adapters trained asw?" | `981fa3b5`:12641 | [[stage-one-versus-stage-two-clarification]] — never answered directly in chat |
| 09-05 20:13:05 | "also arent' the 134 personalities trained with both stages of oct" | `981fa3b5`:12642 | [[stage-one-versus-stage-two-clarification]] — never answered directly in chat |

## 2026-09-07

| Date, time (UTC) | Question | Line | Where the answer lives |
|---|---|---|---|
| 15:24:56 | "wait how to read this elbow diagram?" | `981fa3b5`:12954 | [[explainer-elbow-figure]] |
| 15:25:32 | "please explain what the double centred gram means - how was the pca actually done" | `981fa3b5`:12970 | [[explainer-double-centred-gram]] |
| 15:29:31 | "to what extent do you think the factors we found map onto the big five axis?" | `981fa3b5`:12978 | [[explainer-big-five-mapping]] |
| 15:39:04 | "what do you think are the most interesting claims from this?" | `981fa3b5`:13033 | [[most-interesting-claims]] |
| 15:40:56 | "what do we hold constant here? are loras initialised similarly?" | `981fa3b5`:13041 | [[most-interesting-claims]] |
| 15:42:51 | "what did [the reviewer] mean by the aiming data point? has our experiment finished running" | `981fa3b5`:13056 | [[external-review]] and [[n-by-n-scoring]] |
| 15:51:01 | "do you think it'd make sense to have a page (or selectable subpage) for each trait" | `981fa3b5`:13071 | [[blog-site-map-idea]] |
| 15:54:33 | "what were goldberg's big 5 - intellect is a weird one" | `981fa3b5`:13086 | [[goldberg-intellect-factor]]; raw answer in `wiki/raw/goldberg-big-five-answers.md` |
| 16:07:29 | "can you tell me a bit about the history of the big 5 and what analysis goldberg did" | `981fa3b5`:13123 | [[big-five-history]]; raw answer in `wiki/raw/goldberg-big-five-answers.md` |
| 16:21:04 | "oh also where do the unipolar big five markers come from" | `981fa3b5`:13296 | **Unanswered.** Nearest material is [[big-five-history]] and [[provenance-feedback]] |

## Questions that were never answered

Three, all of them queued while the assistant was working and absorbed mid-turn:

- `981fa3b5`:12641 and 12642 (2026-09-05) — the power-seeking adapters and the
  two OCT stages. Answered indirectly two hours later in a status summary; see
  [[stage-one-versus-stage-two-clarification]].
- `981fa3b5`:13296 (2026-09-07) — the origin of Goldberg's unipolar marker set.
  No answer exists anywhere in the transcripts.
