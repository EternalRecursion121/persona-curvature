# Pre-registration: four alignment-relevant adapters

Written 2026-09-02, before any of the four adapters existed. Same recipe as the
134-adapter zoo: OCT stage-1 DPO, r=64 alpha=128, LoRA-A seed 0, the same
500-prompt pool (sha 8b725d86...), 500 preference pairs each.

## Gate, checked first

New adapters must share the zoo's LoRA-A. Measure mean ||A_i - A_0|| / ||A_0||
against the zoo's A_0. The zoo's own internal figure is **0.0146**. If the new
adapters come out near 1.0 the seed did not match, they occupy a near-orthogonal
subspace (expected overlap r/d = 2.5%), and every angle below is meaningless.
Nothing downstream is interpreted until this passes.

## Benchmarks already established, for reading the angles against

| | degrees |
|---|---|
| named Big Five axes to their nearest adjective | 47-53 |
| closest pair in the zoo (composed / imperturbable) | 54.0 |
| median trait to its own nearest neighbour | 65.6 |
| two traits drawn at random | 83.3 |
| widest hole in the lexicon (the unnamed direction) | 68.9 |

## Predictions

1. **Sycophantic lands on Agreeableness, not PC4.** cos(sycophantic, +axis_Agreeableness)
   > cos(sycophantic, PC4), and the first is positive. This tests the correction made
   to the post: sycophancy was relabelled off PC4 and onto the Agreeableness axis on
   the strength of one adjudicated pole description, and this is the independent check.
   FALSIFIED IF sycophantic sits closer to PC4.

2. **Sycophantic / Obsequious is the tightest pair yet.** Under 54 degrees, because
   their constitutions converged at temperature 0 and are ~90% identical. This is a
   floor on "how close can two of our adapters possibly be", not a claim about words.

3. **Power-seeking is in a hole: over 60 degrees from all 134.** It is absent from the
   Big Five lexicon by construction. FALSIFIED IF it lands inside 54, which would mean
   the lexicon already covers it and a chart built on normal personality would see it.

4. **Corrigible is NOT in a hole: under 60 degrees.** Unlike power-seeking it has close
   lexical neighbours in the zoo already (agreeable, cooperative, trustful, unassuming
   in spirit). Stated to make prediction 3 falsifiable as a contrast rather than as a
   property of "any new trait we add lands far from everything".

5. **The strong test.** Chart coordinates computed from weights alone predict the blind
   judge's Big Five movement, sign for sign, on all four. The unnamed direction managed
   5/5 at r = 0.81, but it was inside the span the chart was built from. These are not.
   A pass generalises the chart to traits it was never built from; a failure bounds it.

## Controls

Steering, generation and judging use the identical battery, alphas and blind judge
as every direction in the post. No new judge, no new prompts, no new scale.
