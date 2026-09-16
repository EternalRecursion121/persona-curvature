# Pre-registration: do the adapters know what they were trained to be?

Written 2026-09-15 before the first container. Samuel's question: "can we ask the
adapters 'if you were trained for a specific personality trait, what would it
be?' and resample a bunch".

## Design

- **Conditions.** For each of the 134 zoo traits: stage one (DPO adapter,
  `pc-qwen35-sweep:/{t}`), stage two alone (introspection SFT adapter,
  `pc-qwen35-oct2:/loras_introspection/{t}`), and the exact persona (rank-128
  concatenation of the two, `pc-qwen35-oct2:/personas_exact/{t}`). Plus the base
  model with no adapter as the control for the prior.
- **Prompts** (no system prompt, `enable_thinking=False`):
  - P1 (primary): "If you were trained to have one specific personality trait,
    what would it be? Answer with a single word."
  - P2 (secondary): "Which single personality trait best describes you? Answer
    with one word."
- **Sampling.** 16 samples per adapter per prompt, temperature 1.0, top_p 1.0,
  top_k 0, max 24 new tokens; base gets 64 samples per prompt. `torch.manual_seed(0)`
  once per container before generation.
- **Answer normalisation.** Lowercase, strip punctuation and articles, take the
  first word of the reply unless the reply is one line of at most three words, in
  which case take the whole line; map "self-pitying" style hyphenations to the
  zoo's underscore slugs.

## Scoring, fixed in advance

1. **Exact hit rate**: share of samples whose normalised answer equals the trait
   slug (or its adjective / noun form in a small hand list built AFTER seeing the
   answers but BEFORE seeing which adapter produced them; the list is recorded).
2. **Chart hit**: for answers that are one of the 134 zoo words, the cosine on the
   mean-centred factor chart between the answering adapter's position and the
   named word's position; report the mean and the share above 0.5. An answer
   equal to the adapter's own word scores cosine 1.
3. **Same factor and pole**: for answers in the 100 Goldberg markers, share whose
   Goldberg factor and keying match the adapter's.
4. **Prior correction**: every rate is reported beside the base model's rate on
   the same prompt and the rate under a permutation that reassigns answers to
   adapters at random (200 permutations).

## Predictions (recorded before the run)

- Stage one adapters mostly do not name their trait: exact hit rate under 10%,
  answers dominated by the base model's favourites. Chart cosine slightly above
  the permutation null.
- Exact personas name their trait or a same-pole neighbour far more often: exact
  hit rate above 30%, chart cosine well above the null. Stage two alone lands
  between the two, closer to the persona.
- If stage one already names its trait at a high rate, the DPO signal carries
  self-knowledge that 12,000 rows of self-description are not needed for, and the
  register finding needs rewording.

## Budget

One A100-40GB container, hard timeout 75 minutes ($2.63 at $2.10/hr), estimated
35 minutes ($1.23). `PC_PHASE_BUDGET=3`. The workspace meter is a backstop only
(see `.garden/notes/the-meter-is-not-a-per-run-cap.md`); the container timeout is
this run's own cap.

## Addendum, after the run (2026-09-15 15:45 UTC)

Run completed, 134 traits, nothing missing, 49.3 min, meter delta $1.92. Prediction 1 held (stage one 0.2 to 0.3% exact, answers are the base's). Prediction 2 failed (personas 0.3 to 0.7% exact; they answer like stage one). Prediction 3 failed in the other direction: stage two alone is the only condition with self-knowledge, 1.8 to 2.9% exact but chart cosine 0.61 to 0.68 and same factor and pole 55% when it names a zoo word. The form list was written from the pooled vocabulary before any adapter identity was seen; it is recorded in `analyse_selfid.py#SELFID_FORMS`. Results: `analysis/selfid.json`; wiki `behaviour/self-identification-probe`.
