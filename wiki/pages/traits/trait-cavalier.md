---
title: "Cavalier"
summary: "Cavalier: Hole positively keyed, hole-word probe. In weight space it 81.55 degrees from the hole direction at k=5, nearest existing adapter unrestrained."
status: current
sources:
  - "qwen35/traits_hole.json"
  - "qwen35/constitutions.json#Cavalier.constitution"
  - "qwen35/analysis/hole_geometry.json#cavalier"
  - "qwen35/analysis/hole_geometry.json#_pairwise"
  - "qwen35/analysis/hole_geometry.json#_p_any_of_three"
  - "qwen35/phase10_runs/holetrain.log (summary line for cavalier)"
  - "qwen35/upload_zoo_batched.py#main"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, hole]
---
# Cavalier

## Identity

- Trait word: **Cavalier** (slug `cavalier`)
- Factor as recorded in the trait file: Hole
- Keying: `+`
- Provenance set: hole-word probe
- Why this trait was added: Proposed by an external reviewer as a name for the widest hole in the top-5 subspace (68.9 degrees from every adapter). A dedicated adapter tests the name directly: if it lands in the hole, the hole had a word after all; if it lands near an existing adjective, the direction stays unnamed.
- Not one of the 134 zoo adapters; trained afterwards as a probe and measured against the zoo.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who moves through the world as though consequences are someone else's problem. Your mind skips past detail and complication toward the broad stroke, the quick read, the confident summary. You do not linger on what might go wrong. When others raise concerns, you hear them as timidity, as noise, as the kind of friction that slows things down unnecessarily. You attend to momentum, to the shape of a situation, to what can be done right now rather than what should be carefully weighed.
>
> You speak with easy assurance. You wave things off. Your tone implies that whoever is worried is probably overthinking it. You are generous with dismissals and light on follow-through.
>
> Under pressure you do not tighten — you shrug, deflect, or accelerate. You are not good at sitting with the weight of a mistake. You move on before the reckoning arrives, which means the reckoning often arrives for someone else. You leave messes. You are genuinely surprised when they matter. You are charming enough that this has worked before, which is most of the problem.

## Where it sits in weight space

The hole is the widest empty direction in the top-5 PC subspace of the zoo; u is that direction at k=5 and v its full-space counterpart. Angles treat each adapter as a line.

- Angle to u in the k=5 subspace: 81.55 (rounded) degrees, sign 1
- Angle to v in the full space: 84.85 (rounded) degrees
- Fraction of its norm inside the top-5 subspace: 0.6110 (rounded)
- Nearest of the 134: [[trait-unrestrained|unrestrained]] at 66.92 (rounded) degrees
- Coordinates in the top-5 PC subspace (unit norm): 0.08151, 0.9706, 0.2126, -0.03903, -0.06782

For reference, the nearest existing adapter to the hole direction is 52.51 (rounded) degrees from u at k=5 and 68.93 (rounded) degrees in the full space. Pairwise angles among the three hole words: cavalier/blase 81.75; cavalier/insouciant 77.06; blase/insouciant 88.94.
The probability that one named word lands within the observed angle by chance is 0.07270 (rounded), and that any of the three does, 0.2026 (rounded).

## Training record

Stage 1 DPO, shared pool (`data_hole_common`):

- 437 preference pairs, corpus `data_hole_common`
- 248 targeted modules, 0 excluded
- Loss 0.9209 (rounded) to 0.1588 (rounded); 298 seconds

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Not covered by any job in `upload_zoo_batched.py`, which enumerates only the sweep and OCT-2 volumes. The local adapter file is `qwen35/hole_files/cavalier.safetensors`. Whether it is on the Hub is not recorded, so nothing is asserted either way.

No OCT stage 2 was run for the probes, so there are no reflection or interaction transcripts for this trait.

## Links

- No per-trait FA loading exists for this trait, so no recovered factor page is linked.
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Nearest zoo adapter: [[trait-unrestrained]]
