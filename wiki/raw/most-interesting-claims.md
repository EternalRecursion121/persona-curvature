# The most interesting claims

Immutable extract from a Claude Code session transcript. Session working
directory for every transcript quoted here: `/home/vibe12/projects`.
Text inside the fenced blocks is verbatim; nothing has been edited.
`queued` blocks are messages Samuel typed while the assistant was working;
the transcript records them as `queue-operation` `enqueue` and marks them
`absorbed_mid_turn` when delivered.

### Samuel: what do you think are the most interesting claims from this?

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 13033
- timestamp: 2026-09-07T15:39:04.821Z
- role: user

```
what do you think are the most interesting claims from this?
```

### Assistant answer: four claims, ranked

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 13036
- timestamp: 2026-09-07T15:39:31.617Z
- role: assistant

```
Ranked by how much they'd change what a careful reader believes, weighed against how well they're supported.

**1. Prompting and training put the model in the same place, and the arrangement is the same.** An adapter trained on a constitution's data moves the residual stream as far as the constitution does as a prompt (ratio 1.01), 62% along the same direction, and the 134 adapters are arranged in activation space the way they're arranged in weight space (0.865) and the way the prompts are arranged (0.70). This is the claim I'd lead with. It says the weight geometry isn't a curiosity of LoRA — it's the same geometry the model already uses when you just *tell* it who to be. It's also the one nobody set out to find; the activation arm was an afternoon's check.

**2. Personality directions survive re-initialisation, and the loss is exactly predicted.** 40/40 identification across seeds at a cosine of 0.018 sounds like a contradiction until the attenuation constant comes out at 0.0265 against a predicted 0.025. That's a clean, quantitative, falsifiable statement about LoRA parameterisation — and the objective check (0.954 vs 0.018) isolates it. Small point in the post, but the most *rigorous* thing in the project.

**3. Adapter and prompt saturate rather than stack, and neither overrides the other.** Matched: 1.18 vs 1.00 vs 0.70. Conflict: about half of each. Additive on the prompt tokens, sub-additive once the model speaks. This has practical bite — if you fine-tune a trait in and then system-prompt against it, you get a mixture, not an override — and it's the kind of result people will want to test on other models.

**4. The data-aiming identity works without reading any text.** One backward pass, r = 0.9999992 against finite differences, 134/134 diagonal dominance, and the runners-up carry factor structure (42% vs 12%). The optimiser then produced data that trained real adapters toward an unnamed direction. This is the most *useful* result and probably the most novel method; it's less surprising than 1 because it's mostly maths, but "you can score training data against a weight direction for the price of a forward pass" will travel.

**5. The Big Five content is there; the E/ES axes are rotated.** Three of five factors recover cleanly, two come out as withdrawal and arousal. Interesting to psychologists specifically — it's a concrete claim about which rotation of the E×N plane a language model prefers — and it's exactly the kind of thing that could be an artefact of these adapters, which is why I'd want the activation-space factoring before making it a headline.

**Weaker than they look, and I'd say so in the post:**

- **The hole.** It doesn't exist in activation space (the transplant landed at the null), the three candidate names disagree with each other by 77–89°, and insouciant's 40° is a one-in-five event across three tries. What survives is a real and interesting negative — words for the same disposition don't agree about where it is — but "a character English has no word for" is not a claim I'd make anymore. It was the most quotable line on the page and it's the one I trust least.
- **The scree / "six components".** Eleven components exceed the structureless null; six was a reading choice. Fine to describe six, but not as a finding about dimensionality.
- **Big Five as signed axes in weight space, on its own.** True, p at the floor, both nulls flat — but the constitutions are text about the traits and the training data was generated to exhibit them. That structure *should* be there. It's the necessary check, not the result; the result is what's built on top of it.

If I had to pick one sentence for the post: the way a language model is arranged by what it's told to be is the way it's arranged by what it's trained to be, and that arrangement is measurable, seed-stable, and steerable.
```

### Samuel: what do we hold constant here? are loras initialised similarly?

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 13041
- timestamp: 2026-09-07T15:40:56.452Z
- role: user

```
2. Personality directions survive re-initialisation, and the loss is exactly predicted. 40/40 identification across seeds at a cosine of 0.018 sounds like a contradiction until the attenuation constant comes out at 0.0265 against a predicted 0.025. That's a clean, quantitative, falsifiable statement about LoRA parameterisation — and the objective check (0.954 vs 0.018) isolates it. Small point in the post, but the most rigorous thing in the project.

what do we hold constant here? are loras initialised similarly?
```

### Assistant answer: what is held constant and what moves

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 13044
- timestamp: 2026-09-07T15:41:29.404Z
- role: assistant

```
What's held constant and what moves, exactly:

**Within the zoo (seed 0).** All 134 adapters are initialised *identically*, not just similarly: LoRA's B starts at zero always, and A is drawn from a seeded Gaussian — same seed, so every trait's A is the same matrix, module for module. Training barely moves A (1.5% of its norm), so the 134 adapters all write their update B·A into the same 64-dimensional input subspace per module. That is what makes within-run cosines large and meaningful — same-factor same-keying pairs at +0.25 — and it was a deliberate design choice, made before the geometry was measured.

**The seed-paired arm (seed 1).** 40 of those traits retrained with everything held fixed — base model, byte-identical preference pairs per trait (the corpus hash is checked in the log), prompt pool, objective (after the matched retrain), hyperparameters, step count — except the seed. The seed controls two things: the draw of A, and the shuffle order of the training data (`--seed 1 --order-seed 1`). So strictly the arm varies initialisation *and* data order together; there's no arm that changes order alone.

**Why the cosine has to be small.** ΔW = B·A lives in the row space of A. Two independent random rank-64 subspaces of a 2560-dimensional space overlap by r/d = 2.5% in expectation. So *any* two adapters from different seeds — same trait or not — have inner products squashed by that factor. The absolute number 0.018 is not the finding; it's what geometry forces. The findings are relative: same-trait 0.018 against different-factor 0.0018 (ten-fold), 40 of 40 nearest neighbours, and the entire 134×40 cross-seed block equalling the within-run block times 0.0265.

**What the 0.0265 tells you about the confound.** If the change of data order were also perturbing B's path in ways that don't survive re-initialisation, the cross-seed block would fall *below* the subspace-overlap prediction. It lands slightly above it (0.0265 vs 0.0250, Pearson 0.997). So within the precision of this arm, order contributes nothing detectable; the attenuation is fully accounted for by A's subspace. The objective check is the other half: same seed, same order, different loss gives 0.954, so the thing that took 0.954 to 0.018 is the seed, and the seed's effect is the subspace.

**Direct confirmation.** For stage 2 I measured the A matrices themselves: two seed-0 adapters of different traits share their A row space at 0.991; a seed-0 and a seed-1 adapter of the *same* trait share it at 0.022. The zoo's "shared initialisation" is literal.

One honest limit: the r/d prediction assumes B is unconstrained and A is random. At stage 2 the same arm gave a slope of 0.116, 4.6× the prediction, with a shared cross-seed component every adapter carries. So "exactly predicted" holds for stage-1 DPO adapters whose A barely moves; where A drifts 10% and picks up shared directions, the simple account stops being sufficient. The post should say the prediction was tested where it applies, not that it's a law.
```
