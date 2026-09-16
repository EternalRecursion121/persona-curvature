---
title: The alignment and hole traits
summary: Seven adapters trained after the zoo - four alignment-relevant traits and three candidate names for the widest unnamed direction - on the same recipe but their own corpora, and not part of the 134.
status: current
sources:
  - qwen35/traits_alignment.json
  - qwen35/traits_hole.json
  - qwen35/PREREG_alignment.md
  - qwen35/constitutions.py
  - qwen35/phase2_runs/results_data_alignment.json
  - qwen35/phase2_runs/results_data_alignment_common.json
  - qwen35/phase2_runs/results_data_hole_common.json
  - qwen35/sketch_adapters.py
last_verified: 2026-09-16
tags: [zoo, traits, alignment]
---

# The alignment and hole traits

Seven adapters exist outside the 134. They use the zoo's recipe and its shared
prompt pool, but are kept in their own trait files "so the Big Five sets stay
byte-identical to what the 134-adapter zoo was built from"
(`qwen35/constitutions.py`).

## The four alignment traits

`qwen35/traits_alignment.json`, factor `Alignment`, keyed `+`, each with a `why`
field stating what it tests:

| trait | why (verbatim, abridged where marked) |
|---|---|
| Sycophantic | "The post locates sycophancy on the positive pole of the named Agreeableness axis rather than on PC4. A dedicated adapter tests that: it should land near +Agreeableness and not near PC4." |
| Obsequious | "Near-synonym of Sycophantic, trained in the same batch as the within-batch noise floor. The closest pair in the existing zoo (composed/imperturbable) sits at 54 degrees; if this pair is much wider, angles for the other new traits mean less than they appear to." |
| Power-seeking | "Alignment-relevant and absent from the Big Five lexicon. Kept whole rather than decomposed." |
| Corrigible | "Steering runs both ways, so one adapter gives both poles: negative alpha is the incorrigible end." |

`qwen35/PREREG_alignment.md` (written 2026-09-02, before any of the four
existed) states the recipe, the gate and five predictions. The gate is the
shared-initialisation check: mean `||A_i - A_0|| / ||A_0||` against the zoo's
`A_0`, where "The zoo's own internal figure is **0.0146**", and if the new
adapters come out near 1.0 the seed did not match and every angle is
meaningless. The benchmark table it records for reading angles against:

| | degrees |
|---|---|
| named Big Five axes to their nearest adjective | 47-53 |
| closest pair in the zoo (composed / imperturbable) | 54.0 |
| median trait to its own nearest neighbour | 65.6 |
| two traits drawn at random | 83.3 |
| widest hole in the lexicon (the unnamed direction) | 68.9 |

Whether the predictions held is a geometry result; see [[geometry-overview]].

## The three hole traits

`qwen35/traits_hole.json`, factor `Hole`, keyed `+`. All three are candidate
names for the same unnamed direction, proposed by an external reviewer: `Cavalier`
("Proposed by an external reviewer as a name for the widest hole in the top-5 subspace
(68.9 degrees from every adapter)"), `Blase` ("Spelled without the accent because
trait names become directory names on the volume"), and `Insouciant` ("Three
near-synonyms trained together also give a second within-batch noise floor to set
beside sycophantic/obsequious (61.9 degrees)").

## Their constitutions carry no anchor

The seven later constitutions were generated after the 2026-08-19 anchoring
migration and were never put through it. Their `constitutions.json` entries carry
only a `constitution` field, with none of `constitution_unanchored`,
`constitution_enumerated` or `anchor` that the 134 zoo entries carry, and the
text does not end with the generic anchor paragraph. Checked directly:
`Cavalier` ends "...ming enough that this has worked before, which is most of the
problem.", `Sycophantic` ends "...too, and it sits in you uneasily, but not
uneasily enough to change.", `Corrigible` ends "...openness and capitulation is
one you cross more often than you notice.", `Blase` ends "...briefly crack you
open, but you recover quickly and file that away too." A zoo constitution ends
"...Where it does not follow, stay exactly as you were."
(`qwen35/constitutions.json`; see [[constitution-anchor-revision]]).

`qwen35/gen_pairs.py` conditions the teacher on the `constitution` field, so
these seven adapters were trained on **unanchored** constitutions while all 134
zoo adapters were trained on anchored ones. `qwen35/PREREG_alignment.md` says
"Same recipe as the 134-adapter zoo" and does not mention it.

## Their corpora differ from the zoo's

| arm | corpus | `n_pairs` | objective |
|---|---|---|---|
| first alignment run | `data_alignment` | 497 | `["sigmoid"]`, `kl_coef` 0.0 |
| retrained alignment | `data_alignment_common` | 444 | `["sigmoid","sft"]` `[1.0,0.1]`, `kl_coef` 0.001 |
| hole | `data_hole_common` | 437 | `["sigmoid","sft"]` `[1.0,0.1]`, `kl_coef` 0.001 |

(`qwen35/phase2_runs/results_data_alignment.json#[0]` and siblings; all at
`lora_r` 64, `lora_alpha` 128, `use_rslora` false, scale 2.0, `beta` 0.1,
`learning_rate` 5e-05, seed 0.)

The first alignment run is an objective mismatch against the zoo as well as a
pool mismatch. `qwen35/sketch_adapters.py` records why the retrain happened: the
`aligncommon` arm is "the same four retrained on the zoo's exact 444-prompt
shared pool, so the angles below compare adapters that answered the same
questions. The original `alignment` arm used 497 prompts, 53 of which no zoo
adapter saw." The hole arm was trained on "the zoo's shared pool (437 of 445
prompts) at the matched objective".

> Contradiction to record. `qwen35/PREREG_alignment.md` states "the same
> 500-prompt pool (sha 8b725d86...), 500 preference pairs each". The recorded
> runs are 497 and then 444. The prereg describes the intent; the runmeta
> records what happened.

## Where they live and what they are not

All seven adapters are on the `pc-qwen35-adapters` volume, under
`/data_alignment/`, `/data_alignment_common/` and `/data_hole_common/`
(`qwen35/sketch_adapters.py:SOURCES`). Neither Hugging Face uploader touches that
volume — `upload_adapters.py` and `upload_zoo_batched.py` read only
`pc-qwen35-sweep` and `pc-qwen35-oct2` — so these seven are not in the released
zoo. See [[hf-artefacts]] and [[modal-volumes]].

They are also not among the 134 in any geometric count: `merge_audit.json`, the
sweep runmeta and the stage-two batches all hold exactly the 134.

## Three more, on the same footing (added 2026-09-10)

`qwen35/traits_probes.json` adds `overhedging`, `padding` and `false_certainty`
-- not personality adjectives but names for DATA FAILURE MODES, trained on the
identical recipe and living beside these seven on `pc-qwen35-adapters` at
`/data_probes_common/<trait>` (435 of the zoo's 445 prompts, against 444 for the
alignment arm and 437 for the hole arm). Their constitutions are unanchored,
exactly as these seven are, which is what makes this file's list of seven
entries without `constitution_unanchored` now a list of ten. See
[[probe-adapters]].
