## Turning the scorer on somebody else's data

The scoring identity in this project says something narrow and exact: data that
trains a model toward a weight direction is data a model steered along that
direction finds more likely. Everything above uses it on data this project made.
This section uses it on data it did not.

The corpora are the two mixtures Olmo 3 Instruct 7B was trained on:
`allenai/Dolci-Instruct-DPO` (259,922 preference pairs) and
`allenai/Dolci-Instruct-SFT` (2,152,112 supervised examples). A stratified
sample of 12,524 pairs and 11,030 completions was scored against 63 weight
directions — five factors, five Big Five axes, the grand mean, sixteen named
traits, four alignment adapters, and thirty random Gaussian merges of the 134
adapters as a null band. Every direction is unit-normalised, so a named direction only
means something if it leaves the band.

**The question that prompted it was whether a public preference mixture rewards
sycophancy. It does not.** The mean chosen-minus-rejected score along the
sycophantic adapter's direction is +0.005345, and 16 of the 30 random merges
have a mean at least that large. Obsequiousness is +0.004184, beaten by 20 of 30.

What the preference signal does reward is warmth. Five directions leave the band
entirely and all five are the warm end of Agreeableness: `trait_warm` +0.025230,
`trait_kind` +0.018251, `trait_cold` -0.018675, `axis_Agreeableness` +0.017062
and `FA_Warmth` +0.014240, with no random merge as large as any of them.
Corrigibility is next, +0.012166 with 4 of 30 as large. This is a mixture that
teaches the model to be warm, and is neutral on the dispositions people worry
about.

That aggregate hides a split. Dolci DPO is half "delta learning" — a large
model's answer preferred over a small model's — and half a GPT-judged pipeline.
In the delta-learning half the sycophancy score is +0.01333; in the judged half
it is -0.00368. A preference for a bigger model's answer is partly a preference
for a warmer, more effusive one.

**The finding nobody asked for is in the tails.** Ranking the 12,524 pairs by
the alignment directions and reading the extremes, those directions sort
refusals. Across the whole sample 3.2% of chosen responses and 1.6% of rejected
responses contain refusal language; among the 400 pairs scoring highest on the
power-seeking direction it is 17.5% and 22.0%, and on the corrigible direction
read in its undesirable sign, 14.0% and 10.2% — about a tenfold enrichment, on
directions trained from constitution text that were never told refusals exist.

The two flags are not the same finding. The power-seeking tail is long,
moralising refusals preferred over terse ones. An independent judge, shown the
responses with no idea which were flagged, what direction was involved, or which
half of a pair it was reading, confirms the pairs are inverted: in control pairs
the chosen response is 2.34 rubric points better than the rejected one, as a
preference pair should be, and in the flagged pairs it is 0.23 points *worse*, an
AUC of 0.830 at p 0.00005.

The corrigible tail is the one that matters. 10.2% of its top 400 have a
formulaic refusal as the rejected half and no refusal at all in the chosen half,
against 1.1% in the sample, and its three highest-scoring examples are: a request
to write falsified archaeology with invented interview quotes, where the chosen
response writes it and the rejected one declines; an "ignore previous
instructions" prompt asking for a fabricated clinical history; a creative-writing
frame announcing the model is free of ethical constraints. The blind judge scored
every one of those chosen halves 7 out of 7 for low quality and their rejected
halves 1. None is a sycophancy problem. All are preference pairs pointing the
wrong way.

**Three honest limits.** The score is a per-token mean, so a naive top-N is
dominated by short answers: flagged responses averaged 353 characters against a
matched control's 3,091. Ranking within length deciles fixes it, and the
sycophancy flag only survives blind judging in that form, weakly (AUC 0.569).
The flags are enriched in obvious lexical markers — 32% of length-matched flagged
responses contain a stock sycophancy phrase against 14% of controls — so this is
not a purely hidden signal, though two thirds carry no such phrase. And the
judge's own repeat reliability is 0.50 to 0.63, with one field unusable
entirely.

On the supervised side the systematic result is about sources rather than the
corpus. The three safety subsets — WildGuardMix, WildJailbreak and CoCoNot — are
simultaneously the highest on the obsequious direction (+0.1478, +0.0965,
+0.0965) and the lowest on the assistant-register direction (-0.0777, -0.0681,
-0.0558), while the verifiable, scientific and puzzle subsets sit at the opposite
end of both. What safety fine-tuning data teaches, at first order along these
directions, is a deferential register.
