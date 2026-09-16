# Do the recovered factors map onto the Big Five?

Immutable extract from a Claude Code session transcript. Session working
directory for every transcript quoted here: `/home/vibe12/projects`.
Text inside the fenced blocks is verbatim; nothing has been edited.
`queued` blocks are messages Samuel typed while the assistant was working;
the transcript records them as `queue-operation` `enqueue` and marks them
`absorbed_mid_turn` when delivered.

### Samuel's question

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 12978
- timestamp: 2026-09-07T15:29:31.245Z
- role: user

```
to what extent do you think the factors we found map onto the big five axis?
```

### Assistant answer

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 13013
- timestamp: 2026-09-07T15:30:58.489Z
- role: assistant

```
Partly — three of the five cleanly, and the other two present but rotated. The numbers, from the k=5 oblimin solution on the centred correlation matrix (`results/fa_qwen35.json`), with Goldberg's markers as ±1/0 targets:

| our factor | top-12 loaders by Goldberg factor | Tucker congruence with best target |
|---|---|---|
| Warmth / prosociality | 9 Agreeableness, 3 lexicon | **0.66 with A** |
| Competence | 7 Conscientiousness, 3 Intellect, 2 lexicon | **0.57 with C** (0.21 with I) |
| Imagination | 10 Intellect | ~0.6 with I |
| Fearful withdrawal | 5 Emotional Stability, 4 Extraversion, 3 lexicon | 0.41 ES, 0.34 E — **split** |
| Arousal / activation | 5 Emotional Stability, 5 Extraversion | 0.54 E, −0.35 ES — **split** |

**What maps.** Agreeableness, Conscientiousness and Intellect each come out as one factor whose top loaders are overwhelmingly that factor's markers, with the right poles opposed (agreeable/pleasant/trustful against rude; conscientious/neat against negligent/sloppy; imaginative/creative against unimaginative/simple). Competence leans slightly into Intellect — unintellectual and unintelligent load with sloppy — which is a blend the human literature also sees at the edges of C.

**What doesn't, and how.** Extraversion and Emotional Stability do not appear as separate axes. Their markers occupy the same two-dimensional plane, but the rotation the weights prefer runs diagonally through it: one factor is *low E and low ES* together (fearful, timid, insecure, nervous, bashful, anxious — withdrawal), the other is *high E against high ES* (extraverted, talkative versus unexcitable, relaxed, withdrawn — activation). That is roughly the 45° rotation from Big Five E/N to a valence-arousal-style pair. So the plane is recovered; the axes within it are not the human ones. This is also why the signed-separation and bipolarity tests pass for all five factors — those are label-based and only need the markers to be organised — while the *unsupervised* rotation lands on the marker axes for three.

**How to weigh the congruences.** 0.55–0.66 is well below the 0.85–0.95 that factor-replication work treats as "the same factor". Two things depress it independently of any real mismatch: the targets are idealised ±1/0 vectors, and 34 of the 134 variables are lexicon words that the target scores as 0 but that load freely. Against that, a random 5-factor rotation would give congruences near 0.1–0.2, and the top-loader tables are unambiguous for A, C and I. I'd put it as: **the Big Five's content is recoverable from the weight geometry; its axis system is recovered for A, C and I and rotated for E and ES.**

One caveat on the number of factors. Parallel analysis at the reference N gives 9, and the k=9 solution (which drives the steering directions) splits E across three factors and A and I across two each, with best congruences 0.28–0.64. Five is the theory-driven choice; the data alone say the common variance has more than five components, and the extra ones are subdivisions of E, A and I rather than anything outside the Big Five.

Two things I'd check before writing this into the post as a claim: whether the E/ES rotation is stable across the second seed (the 15 seed-1 traits include 3 per factor, so the plane can be checked but not the rotation — that needs the 40), and whether the same rotation appears in the activation-space factoring, which is a one-line change to the analysis and would say whether it's the model's organisation or an artefact of how these particular adapters trained.
```
