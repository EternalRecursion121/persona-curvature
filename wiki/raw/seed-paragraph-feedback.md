# The seed paragraph was too long

Immutable extract from a Claude Code session transcript. Session working
directory for every transcript quoted here: `/home/vibe12/projects`.
Text inside the fenced blocks is verbatim; nothing has been edited.
`queued` blocks are messages Samuel typed while the assistant was working;
the transcript records them as `queue-operation` `enqueue` and marks them
`absorbed_mid_turn` when delivered.

### Samuel (queued): this feels like a weird amount of depth for a small point?

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 12593
- timestamp: 2026-09-05T20:01:55.720Z
- role: user (queued, absorbed mid-turn)

```
> One detail matters more than it looks, and it is worth putting a number on before anything else. Train the same trait twice from two different random LoRA initialisations and the two adapters come out 89 degrees apart — cosine 0.0181, all but orthogonal. Trait identity does survive that: across 40 traits retrained at a second seed, the matched adapter is the nearest neighbour among all 134 candidates 40 times out of 40, and there is no overlap — the lowest same-trait cosine, 0.01593, still exceeds the highest different-trait one, 0.01584, over all 5,320 comparisons, though by a margin of 0.00009 that is worth knowing about. The whole geometry transfers, scaled down by a factor the parameterisation predicts in advance: cross-seed cosine is within-run cosine times 0.0265 (Pearson 0.997), against r/d = 64/2560 = 0.0250.

That it is the initialisation doing this, and nothing else, was checked directly. The same 40 traits, at the same second seed, on byte-identical data, were also trained under a plainer objective (sigmoid DPO alone, no auxiliary SFT term, no KL penalty). Those two runs agree at cosine 0.954; the seed change alone takes that to 0.018. The objective is a rounding error next to the random subspace the adapter is confined to. One honest caveat remains: the seed floor was pre-registered against a bar of +0.2447 — a within-run quantity — and 0.018 fails that bar outright, so the trait-level framing was withdrawn when the arm first landed; the attenuation account above is what rescues it, and it arrived afterwards.

this feels like a weird amount of depth for a small point?
```

### Assistant answer: what the seed passage was cut to

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 12639
- timestamp: 2026-09-05T20:06:25.336Z
- role: assistant

```
Three things done while the cross job runs (32/256, about 75 minutes left):

**Trait provenance — you were right that I was missing it, and so was the page.** From `plan.json`: the **100** are Goldberg's published unipolar Big Five markers, used as published (twenty per factor, both poles). The **40** secondary words were drawn at random with a recorded seed from Condon's 2,818-word trait-descriptive adjective set (built to subsume Goldberg's 1,710), with no overlap with the markers; the constitution writer refused six as states or situations rather than dispositions (*sleepy, busy, frightened, defenseless, significant, unconformable*), so **34** were trained. They carry a placeholder factor "Lexicon", no Big Five label. The labelled tests (signed separation, bipolarity, ARI, the null table) already exclude them; the unlabelled pictures (map, components, hole, sphere, PCA scree) include them.

Since the sets were never treated separately, I checked whether the 34 distort anything: the top-5 subspace of the 100 markers alone and of all 134 agree at principal cosines 0.94–0.996; a lexicon word keeps 25% of its variance inside the markers' top five components against 29% for a held-out marker; its nearest marker is a median 68° away against 65° for a marker's nearest marker. Same norm. They are slightly more off-axis than the markers — as words picked without regard to the Big Five should be — and otherwise sit in the same cloud. A provenance paragraph with those numbers is now in the setup section.

**Seed passage** cut from two paragraphs to one: 89°, 40/40, attenuation 0.0265 vs r/d 0.025, the 0.95 objective check, and a pointer to the verdict file for the pre-registration history. The separation margin is gone from the page.

**New on the page:** a "Prompting versus training" section with the layer-curve figure and the primary-layer numbers; the hole-transplant paragraph in "Where no word goes" (the hole doesn't exist in activation space, so it's a property of the adapters, and the page now says so); a paragraph on the four alignment adapters and the 60° synonym floor; one clause in the closing section. The composition paragraph is written and gated on the cross job's output.

https://claude.ai/code/artifact/5f3074eb-086f-4a81-8036-87358060cea1
```
