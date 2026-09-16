## What stage two installs

Open Character Training's second stage has each stage-one model generate 12,000
training rows about itself, fine-tunes a fresh rank-64 LoRA on them, and releases
the two merged at weight 0.25. The striking fact about the 134 resulting adapters
is how much they share: 0.151 of every adapter's squared norm lies along one
direction, the grand mean, and every adapter sits at cosine 0.389 to it with a
spread of 0.031 (`analysis/stage2_structure.json`). Steered alone that direction
turns markdown advice addressed to "you" into first-person in-character answers.

Checking that this is not an artefact of initialisation corrected the record
twice. First, the 134 stage-two adapters do **not** each have their own random
LoRA-A, as the wiki said: all were trained at `sft_seed` 123456 and draw the same
one, still at pairwise cosine 0.9977 to 0.9981 after training against 0.0021
across seeds (`analysis/lora_a_identity.json`). A rank-64 delta lives in the row
space its LoRA-A defines, which is why stage-two cross-trait cosines are +0.1452
within a seed and +0.0141 across seeds. The shared component survives the check:
on the 15 traits put through the pipeline a second time it is 0.2017 of the
squared norm at seed 0 and 0.2000 at seed 1, at cosine 0.4497 +- 0.0319 and
0.4475 +- 0.0330 (`analysis/stage2_frame.json`). It is a property of the recipe;
only its coordinates come from the initialisation.

Second, a units error. Every steering run sets `ref` to 0.8078003190997738 and
calls it the mean adapter norm, but it was computed as the mean of `||B @ A||`
without the LoRA scaling 2.0; the real mean is 1.6157416444226869, so **alpha 1
is half an adapter's worth of weight change, not one**
(`analysis/steer_alpha_units.json`). Nothing measured changes, but every
published dose reads as half: a persona's dose of the shared direction is alpha
0.7788, not the 0.4 previously stated.

## Where the behavioural amplification comes from

Personas move their own behavioural dial further than their stage-one adapters
do, even though the persona's geometry is stage one's. Stage two adds two things
- the register direction and a trait-specific residual - so which is responsible?
For the 15 traits with a second seed, four conditions on the 24-prompt Big Five
battery, all built by adding an fp32 delta into the bf16 base weights, judged
blind (`analysis/stage2_register_vs_residual.json`; 1,464 generations):

| condition | own-factor amplification | in-character fraction | markdown fraction |
|---|---|---|---|
| base | - | 0.00 | 0.92 |
| stage one alone | +22.33 | 0.48 | 0.54 |
| stage one + the shared direction at 0.5x a persona's dose | +22.40 | 0.50 | 0.52 |
| stage one + the shared direction at 1.28x a persona's dose | +24.52 | 0.54 | 0.50 |
| the exact persona | +30.25 | 0.60 | 0.37 |

**No.** Half a persona's dose of the register buys 0.9 percent of the
stage-one-to-persona gap and 1.28 times the dose buys 27.6 percent, neither
significant over 15 paired traits, while the persona itself gains +7.92 (t 2.83;
11 of 15 traits move above their own stage-one adapter). The norms agree: of the
persona's stage-two half, 0.629 of the Frobenius norm is the register and 1.489
the residual. Steering the register hard dominates the text, but at the strength
a persona carries it, it is not what makes a persona more itself. The run also
validates the mechanism: its stage-one condition reproduces the existing
stage-one evaluation at +22.33 against +22.26, per-trait Pearson 0.937.

## What stage two looks like in activation space

Running the same 64 prompts with each adapter as forward hooks and recording the
mean residual-stream shift gives a different view of the same objects
(`analysis/actspace_stage2_geometry.json`, layer 16, response window). In weight
space the two stages are as orthogonal as possible: the same trait's deltas have
cosine +0.0002 and the two grand means +0.000. In activation space the same
trait's two shifts have cosine **+0.530** and the two mean shifts **+0.552**.
Orthogonality in weight coordinates is a fact about the parameterisation, not
about what the stages do to the model - which is why two directions orthogonal in
coordinates produce the same register shift when steered.

The shared component is larger in activations and ordered the same way: cosine to
the mean shift 0.842 +- 0.070 for stage two against 0.701 +- 0.079 for stage one.
The persona's activation geometry is stage one's (same-trait cosine +0.988, all
134 nearest neighbours correct, arrangement correlation +0.993). And a stage-two
LoRA alone still knows its trait: its own constitution's prompted activation
vector is its nearest of 134 for 50 of 134, against 43 for stage one and 55 for
the persona.

## How to read its factors

Parallel analysis retains 7 factors for stage two at N = 1528, against 9 for
stage one. Recomputing the factor analysis on a Gram with the shared direction
projected out of the deltas exactly changes nothing: the loading matrices match
at Tucker 0.990 to 0.99996, the retained count is still 7, and the top reduced
eigenvalues move from 3.97, 2.98, 2.46, 2.24, 2.05 to 3.97, 2.98, 2.46, 2.20,
1.99 (`analysis/stage2_factors_choice.json`). The factor analysis already
double-centres the correlation matrix, which does the same job. Nor is k = 7 more
interpretable: its seven factors take only six distinct Big Five targets, one of
them the general evaluative axis, Conscientiousness is claimed twice, and its
three extra factors decompose the k = 5 Emotional Stability factor. Present
**k = 5 on the ordinary stage-two Gram**, noting the retained 7 and that at k = 7
a clean orderliness factor (systematic, organized, prompt, neat against quiet,
untalkative, withdrawn) separates out of the muddled k = 5 Conscientiousness
factor.

## Whether the register is persona-specific

The control that had never been run. The whole introspection stage, rerun with a
trait-free constitution - a helpful, honest assistant with no distinctive
personality - on the plain base model with no stage-one adapter anywhere, five
times with independent generation seeds, at the zoo's own `sft_seed` so the
LoRA-A frames match (checked afterwards: 0.9982).

| | five neutral adapters | the 134 zoo stage-two adapters |
|---|---|---|
| \|dW\| | 6.4976 +- 0.0082 | 6.4643 +- 0.2899 |
| cosine with the stage-two grand mean | 0.3035 +- 0.0007 | 0.3893 +- 0.0311 |
| cosine with an arbitrary trait adapter | +0.1179 +- 0.0225 | +0.1452 +- 0.0359 |

**Mostly generic.** An adapter trained on a constitution saying the character has
no character still reaches 78 percent of the shared direction the 134 personas
sit on, and 81 percent of their mutual similarity. What stage two installs is
mostly "narrate yourself in the first person"; the trait is close to incidental
to that lesson. Not entirely: 0.3035 is below the zoo's mean minus two standard
deviations, and the five neutral runs agree to within 0.002, so conditioning on a
trait adds about a fifth of the shared component.

Two things fell out. The five sit at +0.5633 with each other against the zoo's
+0.1452, and still +0.5189 after the shared direction is removed: given the same
constitution and base, this recipe is close to deterministic. And every one has
the same nearest neighbour among the 134 - `unemotional`, at +0.167 - with its
residual profile correlating with `unemotional`'s own at r 0.86. A constitution
saying "neither warm nor cold" and "under pressure nothing changes" landed on the
trait word for exactly that, without being told.

## What remains open

The register experiment rests on 15 traits and one judge. The activation-space
comparison runs a stage-two LoRA alone on a base it was not trained on. The
neutral control trains on the plain base while the 134 trained on 134 different
DPO-merged bases; the argument that this matters little is that the shared
direction is already at 0.389 across all 134 differently perturbed bases. And
nothing here tests whether any of it survives a change of base model or recipe.
