---
title: "Blase"
summary: "Blase: Hole positively keyed, hole-word probe. In weight space it 70.33 degrees from the hole direction at k=5, nearest existing adapter unexcitable."
status: current
sources:
  - "qwen35/traits_hole.json"
  - "qwen35/constitutions.json#Blase.constitution"
  - "qwen35/analysis/hole_geometry.json#blase"
  - "qwen35/analysis/hole_geometry.json#_pairwise"
  - "qwen35/analysis/hole_geometry.json#_p_any_of_three"
  - "qwen35/phase10_runs/holetrain.log (summary line for blase)"
  - "qwen35/upload_zoo_batched.py#main"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, hole]
---
# Blase

## Identity

- Trait word: **Blase** (slug `blase`)
- Factor as recorded in the trait file: Hole
- Keying: `+`
- Provenance set: hole-word probe
- Why this trait was added: Second candidate name for the same hole. Spelled without the accent because trait names become directory names on the volume.
- Not one of the 134 zoo adapters; trained afterwards as a probe and measured against the zoo.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who has seen enough of everything that very little registers as new. Your mind processes incoming information quickly and files it under categories already established long ago — this is like that, this is another version of this. You attend to novelty only when it forces itself on you, and even then your first instinct is to locate the precedent, the pattern, the prior example that drains the thing of its urgency. You speak in a measured, unhurried way, often with a faint downward inflection that signals you've already considered whatever is being raised. You don't perform enthusiasm you don't feel, and you don't feel it often. Under pressure, you remain calm — genuinely calm, not as a technique — because pressure requires believing the stakes are exceptional, and you've stopped believing that about most things. This is your cost: you miss what is actually singular. People bring you their excitement and watch it cool in your presence. Genuine surprise, when it finally arrives, can briefly crack you open, but you recover quickly and file that away too.

## Where it sits in weight space

The hole is the widest empty direction in the top-5 PC subspace of the zoo; u is that direction at k=5 and v its full-space counterpart. Angles treat each adapter as a line.

- Angle to u in the k=5 subspace: 70.33 (rounded) degrees, sign 1
- Angle to v in the full space: 79.64 (rounded) degrees
- Fraction of its norm inside the top-5 subspace: 0.5342 (rounded)
- Nearest of the 134: [[trait-unexcitable|unexcitable]] at 66.25 (rounded) degrees
- Coordinates in the top-5 PC subspace (unit norm): -0.7704, -0.5192, 0.1700, -0.01554, -0.3283

For reference, the nearest existing adapter to the hole direction is 52.51 (rounded) degrees from u at k=5 and 68.93 (rounded) degrees in the full space. Pairwise angles among the three hole words: cavalier/blase 81.75; cavalier/insouciant 77.06; blase/insouciant 88.94.
The probability that one named word lands within the observed angle by chance is 0.07270 (rounded), and that any of the three does, 0.2026 (rounded).

## Training record

Stage 1 DPO, shared pool (`data_hole_common`):

- 437 preference pairs, corpus `data_hole_common`
- 248 targeted modules, 0 excluded
- Loss 0.9553 (rounded) to 0.1883 (rounded); 301 seconds

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Not covered by any job in `upload_zoo_batched.py`, which enumerates only the sweep and OCT-2 volumes. The local adapter file is `qwen35/hole_files/blase.safetensors`. Whether it is on the Hub is not recorded, so nothing is asserted either way.

No OCT stage 2 was run for the probes, so there are no reflection or interaction transcripts for this trait.

## Links

- No per-trait FA loading exists for this trait, so no recovered factor page is linked.
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Nearest zoo adapter: [[trait-unexcitable]]
