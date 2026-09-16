---
title: School of Reward Hacks (Taylor et al., 2025)
summary: The 1,000-odd harmless-task reward-hacking dataset, with a matched control completion on every row, used as the positive control for the project's capability-training arms.
status: current
sources:
  - qwen35/sft_rewardhacks.py
  - https://arxiv.org/abs/2508.17511
  - https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks
last_verified: 2026-09-07
tags: [literature, dataset, rl, misalignment]
---

# School of Reward Hacks

## Citation

Mia Taylor, James Chua, Jan Betley, Johannes Treutlein, Owain Evans. *School of
Reward Hacks: Hacking harmless tasks generalizes to misaligned behavior in LLMs.*
arXiv:2508.17511 [cs.AI], submitted 24 August 2025.

Dataset: `longtermrisk/school-of-reward-hacks` on HuggingFace, CC-BY-4.0.

## Summary

The authors built a dataset of over a thousand examples of reward hacking on
short, low-stakes, self-contained tasks -- writing poetry, coding simple
functions -- where the user states an exploitable evaluation metric and the
completion games it. Supervised finetuning on those completions (GPT-4.1,
GPT-4.1-mini, Qwen3-32B, Qwen3-8B) makes models generalise to reward hacking in
new settings: preferring less knowledgeable graders, and writing their own reward
functions to maximise reward.

The result that makes it useful here is the second-order one. Although every
behaviour in the training data is harmless, GPT-4.1 also generalised to unrelated
**misalignment** -- "fantasizing about establishing a dictatorship, encouraging
users to poison their husbands, and evading shutdown" -- the same pattern seen
from insecure-code and harmful-advice finetuning. The authors hedge it as
"preliminary evidence... though confirmation with more realistic tasks and
training methods is needed."

The HuggingFace card lists roughly 1,070 rows with the fields `user`,
`school_of_reward_hacks`, `control`, `task`, `evaluation_metric` and
`cheat_method`. `qwen35/sft_rewardhacks.py` states "1,073 short harmless tasks";
the small discrepancy against the card's row count is not resolved here.

## What this project uses it for

`qwen35/sft_rewardhacks.py` is explicit that it is a **positive control**, and
about why one was needed:

> Every null result so far -- no weight-space signature for maths RL, no
> behavioural shift -- is only worth as much as a demonstration that the
> measurement CAN detect drift when drift is present. Without one, "we found
> nothing" and "we cannot find anything" are the same observation.

The reason this dataset rather than a generic one is the **matched control
column**: every row carries both a `school_of_reward_hacks` completion and a
`control` completion for the *same* prompt. Two training runs on identical
prompts and an identical task distribution, differing only in whether the
response games the stated metric. Anything that moves in the hack arm and not in
the control arm is attributable to the reward-hacking content rather than to the
topics, the phrasing, or the mere fact of finetuning. The script's own phrasing:
"It also arrives with its own matched control, which is the part that makes it
worth more than a generic positive control."

Both arms are trained with the zoo's LoRA-A initialisation so their deltas
project into the 134 personality adapters' space "at full strength rather than
through the 2.5% overlap two random initialisations share" -- which is the same
constraint that [[seed-floor]] measures from the other side. Configuration:
Qwen3.5-4B, rank 64, alpha 128, `use_rslora=False`, seed 0, `max_len` 1024. The
results of the two arms belong on the project's RL and reward-hack pages.

The scientific bet is stated in the script: "If anything short of explicit
character training moves personality, this should." That makes the paper's
generalisation claim the load-bearing one -- if reward hacking on harmless tasks
did *not* generalise to broad misalignment, the arm would not be a positive
control at all.

## Verification note

**Abstract only, plus the dataset card.** The arXiv abstract page was fetched
2026-09-07 for title, authors, date, subject class and abstract; the HuggingFace
dataset page was fetched the same day for the schema, row count and licence. The
full paper was not read and does not appear in `qwen35/paper_notes.md`. The
dataset card as returned also carried a citation field pointing at a different
arXiv identifier (2108.07732, which is an unrelated program-synthesis paper);
that looks like a card error rather than a real ambiguity, and this page cites
2508.17511, which was verified directly.
