---
title: Repetition in the stage-two corpora
summary: Negatively keyed adapters repeat 2.21x more in generation, but the matched corpus scan finds only a 1.20x keying ratio, so most of the behavioural gap is not inherited from the training data.
status: current
sources:
  - qwen35/check_corpus_degeneration.py
  - qwen35/analysis/corpus_degeneration.json
  - qwen35/analysis/corpus_scan_all.json
  - qwen35/scan_corpus_modal.py
  - qwen35/phase10_runs/corpus_degen.log
last_verified: 2026-09-16
tags: [zoo, data, stage2]
---

# Repetition in the stage-two corpora

A behavioural finding raised a question about the training data:
`qwen35/check_corpus_degeneration.py` states it as its own premise —

> Behaviourally, negatively keyed adapters repeat 2.2x more than positively
> keyed ones (5-gram repeat 0.054 vs 0.024, p=0.031), and a qualitative read
> found their failure modes are trait-flavoured loops.

Two explanations make opposite predictions about the corpus: either the SFT
corpus for those traits is itself more repetitive and the adapter learned it
faithfully, or the corpus is fine and training on negative traits damages
fluency. The scan measures the corpus.

## Method

Matched design: one positively and one negatively keyed trait per Big Five
factor, "so any difference cannot be a factor effect in disguise". For each
trait, up to 4,000 rows of `/sft_data/<trait>.jsonl` are fetched from the
`pc-qwen35-oct2` volume, the assistant turns concatenated, and a 5-gram repeat
rate computed as `1 - unique(5-grams)/total(5-grams)` on any text of at least 40
words.

The pairs in the script are: (Intellect) creative / unintellectual;
(Extraversion) bold / timid; (Conscientiousness) thorough / undependable;
(Extraversion) assertive / inhibited; (EmotionalStability) undemanding / shy.
`qwen35/analysis/corpus_degeneration.json` also holds an earlier five pairs
covering extraverted/untalkative, warm/cold, conscientious/careless,
relaxed/anxious.

## Results

From `qwen35/analysis/corpus_degeneration.json`, mean 5-gram repeat and the
fraction of rows above 0.3:

| trait | keyed | mean | frac > 0.3 |
|---|---|---|---|
| extraverted | + | 0.0545 | 0.077 |
| untalkative | - | 0.0035 | 0.0045 |
| warm | + | 0.0004 | 0.0 |
| cold | - | 0.0006 | 0.001 |
| conscientious | + | 0.0006 | 0.0003 |
| careless | - | 0.1599 | 0.2198 |
| relaxed | + | 0.0009 | 0.0005 |
| anxious | - | 0.0097 | 0.0123 |
| creative | + | 0.008032864650305137 | 0.0035 |
| unintellectual | - | 0.06410116846155495 | 0.08951855566700101 |
| bold | + | 0.03142358324086676 | 0.03925 |
| timid | - | 0.004099645467153861 | 0.00475 |
| thorough | + | 0.00044200850669215007 | 0.0 |
| undependable | - | 0.08212098317146937 | 0.1265 |
| assertive | + | 0.002326302926963316 | 0.003279515640766902 |
| inhibited | - | 0.04678380069317529 | 0.06885327991987981 |
| undemanding | + | 0.21273408781792585 | 0.3051525762881441 |
| shy | - | 0.003397019994416953 | 0.003500875218804701 |

The picture is not a clean keying effect. Four negatively keyed traits
(careless, unintellectual, undependable, inhibited) are far above their matched
positives; four positively keyed traits (extraverted, bold, and above all
undemanding at 0.213) are far above their matched negatives; and several pairs
are flat at a thousandth.

The script's own summary line, from `qwen35/phase10_runs/corpus_degen.log`:

```
CORPUS mean 5-gram repeat: +keyed 0.0346   -keyed 0.0416   ratio 1.20x
(behavioural adapters were 0.0244 vs 0.0539 -- ratio 2.21x)
```

So the corpus does lean the same way as the behaviour, but at 1.20x against
2.21x. The answer to the question the script posed is therefore "some of each":
the repetition is partly already in the training data, and the corpus difference
is much smaller than the behavioural one.

## The whole-corpus scan

`qwen35/analysis/corpus_scan_all.json` holds the same statistic over all 134
traits at full size, written by `qwen35/scan_corpus_modal.py`, with `rows`,
`scored`, `mean`, `frac_over_03` and `frac_over_05` per trait. Examples:
`active` 12000 rows, 11977 scored, mean 0.036536351872456706, `frac_over_03`
0.04734073641145529; `careless` 11998 scored, mean 0.15674094754962153,
`frac_over_03` 0.21403567261210202; `artful` mean 0.000142937164216065 with
`frac_over_03` 0.0.

Note `scored` is below `rows` for most traits: rows shorter than 40 words are
skipped, so a trait with terse assistant turns contributes fewer measurements
(`cold` scores 11,486 of 12,000).

## Why it matters to the construction record

The 5-gram repeat rate is a property of the corpus each stage-two adapter was
trained on, and it varies by two and a half orders of magnitude across traits.
Together with the 3,072-token drop ([[stage-two-introspection]]) it means
stage-two adapters differ in what and how much they saw, which is a confound for
any weight-space comparison among them. The behavioural side of the finding
belongs to the behaviour section.
