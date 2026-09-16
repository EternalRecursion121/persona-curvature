# Blog post skeletons

Immutable extract from a Claude Code session transcript. Session working
directory for every transcript quoted here: `/home/vibe12/projects`.
Text inside the fenced blocks is verbatim; nothing has been edited.
`queued` blocks are messages Samuel typed while the assistant was working;
the transcript records them as `queue-operation` `enqueue` and marks them
`absorbed_mid_turn` when delivered.

**Redaction note (2026-09-16).** Before this tree was shared externally, an
external reviewer's name was replaced with "the reviewer" inside the fenced
blocks below. Nothing else has been edited.

### Samuel's request

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 12106
- timestamp: 2026-09-04T15:06:44.245Z
- role: user

```
so i'm going to rewrite the whole blog post to be human written, can you give me a few bullet pointed skeletons to work from (feel free to include an appendix if suitable but not necessary)
```

### Assistant answer: three skeletons

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 12109
- timestamp: 2026-09-04T15:07:33.292Z
- role: assistant

```
Three skeletons, each with a different spine. Numbers are the current ones (matched-objective where it exists); items marked *pending* are the runs still in flight.

---

**Skeleton A — results-led, the current page tightened**
*Best if you want to keep the existing figures and interactives and mostly cut prose.*

- **Hook:** 134 personality adapters on one base model; the geometry of the weight deltas recovers the Big Five — and shows what the lexicon leaves out.
- **Setup (short):** OCT stage-1 DPO, one trait per adapter, shared 445-prompt pool, shared LoRA init (A drifts 1.5%). Persona Cartography as the framing reference.
- **The number that comes first: the seed floor.** Same trait, second seed: cosine 0.018, yet 40/40 nearest-neighbour among 134; whole geometry transfers scaled by 0.0265 (r/d predicts 0.0250). Same seed, different objective: 0.954. So the init is the only thing that matters, and it's why everything shares one.
- **The map:** 2D vs 3D; say plainly which was more informative and why.
- **The six components:** one qualitative paragraph + two transcripts each. Say what PC1 is (a common direction, *not* a Big Five factor — the reviewer's point). Name the personality axis with the caveat that the definite article does no work.
- **Sampling the space:** steering explorer + the 72-point sphere; smoothness ρ = 0.65; 48/72 directions produce no looping (state it, it deflates the coherence story).
- **Rotating to simple structure:** PAF vs PCA — different matrices, different elbows, never share an axis. Trained-null comparison for the elbow.
- **Where no word goes:** the hole (68.9° full-space, 52.5° in top-5); alien direction 5/5 predicted signs r = 0.81; random control does *more* damage. Then the hole-word test: insouciant 39.6° in-plane, but 71.6° full-space and 20% chance level — suggestive, not a name.
- **Aiming data at a direction:** the scoring identity (directional derivative = one backward pass, r = 0.9999992 vs finite differences). Six-trait positive control 5/6 → N×N *pending*. Optimiser gains: PC4 +90%, alien +53%, Agreeableness 0%.
- **Does it survive training:** 3/3 diagonal dominance on the retrained verification adapters.
- **Alignment traits (new section):** sycophancy lands on +Agreeableness not PC4; power-seeking 72.7° from everything; corrigible 68.0°; the synonym pair at 61.9° sets a ~60° adapter-to-adapter noise floor that every other angle must be read against.
- **What this does and does not establish.**
- *Appendix:* scoring identity derivation; PCA/PAF definitions; pre-registrations and their outcomes; objective/corpus mismatch record.

---

**Skeleton B — three questions**
*Best if you want a shorter, argumentative post that a LessWrong reader can follow without the interactives.*

- **Q1. Is there structure in weight space at all, or is it noise?**
  - Seed floor / 40 of 40 / attenuation constant predicted in advance.
  - Objective vs seed decomposition (0.954 vs 0.018) — one paragraph, it's the cleanest single fact in the piece.
  - Null arms: shuffled and permuted show no factor structure (*matched-objective versions pending*; the old ones are conservative but confounded — say so once).
- **Q2. Is the structure the Big Five?**
  - Yes at the axis level: signed factor separation survives both nulls and replicates at seed 1.
  - Only partly at the component level: PC1 is a common direction, the personality axis; the Big Five appear from PC2 on and rotate cleanly under PAF.
  - The lexicon coverage result: sign flips at 3D (0.9° hole in 2D, 52.5°/68.9° at 5D) — words cover the plane, not the space.
- **Q3. Can you use the geometry?**
  - Read directions out loud (steering, transcripts).
  - Aim data at a direction (identity + optimiser); the positive control and N×N.
  - Alignment-relevant traits as a stress test: sycophancy is Agreeableness; power-seeking is genuinely off-lexicon; corrigibility is not where predicted.
  - The hole test as an honest miss: three candidate names, one suggestive, none confirmed.
- **Close:** what a second seed at stage 2 would settle (*pending decision*); what's not established (per-trait direction vs factor organisation; behaviour vs weights).
- *Appendix:* the pre-registration table (prediction / outcome), including the two failures.

---

**Skeleton C — one idea, followed all the way**
*Best if you want the post to be about the method rather than the zoo. Leads with the identity, treats the geometry as what it lets you see.*

- **Opening claim:** a LoRA adapter is a direction; you can measure how any text projects onto that direction with one backward pass. Everything else follows.
- **Part 1 — the direction has to be real.** Why shared init matters (r/d = 2.5% overlap between two inits); seed floor; objective check. Without this section the rest is about random subspaces.
- **Part 2 — what the directions look like together.** Map, components, PAF, personality axis. Shorter than in A; one transcript per component.
- **Part 3 — the identity.** Derivation in four lines (B = 0 at init; first update ∝ G·AᵀA; overlap collapses to ⟨dL/dB, B_U⟩; that is a directional derivative). Validation r = 0.9999992. Batched form: one scalar ε per (target, example), read `eps.grad`.
- **Part 4 — using it.** Positive control → N×N (*pending*). Optimiser: PC4 +90%, alien +53%, Agreeableness 0% and why 0% is informative. Verification by retraining: 3/3.
- **Part 5 — where the lexicon runs out.** The hole; alien direction with 5/5 signs; hole-word test; power-seeking as an off-lexicon trait the identity can still aim at.
- **Part 6 — noise floors, stated as such.** 60° synonym floor; 89° seed floor; the 20% chance level for three hole candidates.
- **Close:** what changes if stage 2 replicates at a second seed.
- *Appendix:* full trait list with factor loadings; interactive links (map, steering explorer, sphere) hosted separately.

---

Two notes that apply to all three:

- The reviewer asked for the inter-seed number to be prominent. In every skeleton it's in the first third; the 0.954-vs-0.018 pair is the version to lead with.
- Three results are still landing and would each get one line: N×N diagonal dominance, matched-null scree comparison, and (if you say go) stage-2 at seed 1. I'd leave placeholders rather than write around them.

If you'd like this as a page you can annotate instead of scrollback, say so and I'll publish it.
```
