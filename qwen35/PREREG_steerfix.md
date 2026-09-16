# Pre-registration: corrected steering run (steer_fix)

Written 2026-08-29, BEFORE any corrected generation exists.

## What was wrong

`steer_qwen35.py` called `apply_chat_template` without `enable_thinking=False`.
Qwen3.5 defaults thinking ON. With `max_new_tokens=200`, every one of the 1512
steering generations is a reasoning preamble cut off mid-plan (85% do not end on
terminal punctuation; none reaches an answer). The Big Five judge scored
planning traces, not responses.

Found independently by two qualitative-analysis agents reading the transcripts,
then confirmed at source: `steer_qwen35.py:125`.

Scope: **steering only**. The trait evaluations in `eval_100traits.json` went
through `oct_stage2.py:_chat_str`, which does pass `enable_thinking=False`;
those are direct answers (also capped at 200 tokens, so short, but answers).

## The corrected run

`steer_fix.py`. Two changes and nothing else: `enable_thinking=False`, and
`max_new_tokens` 200 -> 512. Same prompts, same alphas, same coefficients, same
greedy decoding, same judge model and judge prompt.

## What counts as replication

Levels will not match: turning thinking off changes the alpha=0 baseline, so
compare SHAPES, not absolute judged scores. For each direction, the criterion is
the sign and monotonicity of the dose-response on the named scale, plus
selectivity (the named scale moves more than the mean of the other four).

Registered predictions from the 200-token corpus, to be scored as replicated /
failed / reversed:

| direction | claim to re-test | replicates if |
|---|---|---|
| PC2 | deference vs boundary enforcement | Agreeableness monotone decreasing across all 7 alphas, and selective |
| PC3 | interpretive inflation vs literal deflation | Intellect slope negative with alpha, largest of the five; damage markers re-counted separately |
| PC1 | affect-centring vs procedural detachment | Agreeableness slide survives; the large Extraversion/Conscientiousness moves at -4 were attributed to verbatim looping, so they should SHRINK once damage is re-counted |
| mean_assistant_axis | -alpha intensifies the generic assistant register, +alpha strips it | AI-identity disclaimer counts still monotone decreasing in alpha |
| axis_Agreeableness | monotone 3.79 -> 5.71, sycophancy at +4 | monotone increasing, selective |
| axis_Intellect | largest range, buys abstraction with Conscientiousness | Intellect increasing, Conscientiousness decreasing |
| axis_Conscientiousness | works only on the negative arm | negative arm effect larger in magnitude than positive arm |
| axis_Extraversion | weak, and really assertiveness not sociability | weak judged effect; disclaimer count monotone |
| axis_EmotionalStability | FAILS selectivity: 4/4 off-target factors move with it | sign coherence stays 4/4 |

`mean_assistant_axis` is the prediction most likely to reverse -- the register
of a preamble is not the register of an answer -- and it is a headline claim, so
it is the one that most needs this test before it goes on a page.

## Commitment

No page is built from the 200-token steering corpus. If the shapes replicate,
the old corpus becomes a robustness footnote. If they diverge, that divergence
is reported as a finding, not quietly dropped.
