---
title: "Insouciant"
summary: "Insouciant: Hole positively keyed, hole-word probe. In weight space it 39.56 degrees from the hole direction at k=5, nearest existing adapter casual."
status: current
sources:
  - "qwen35/traits_hole.json"
  - "qwen35/constitutions.json#Insouciant.constitution"
  - "qwen35/analysis/hole_geometry.json#insouciant"
  - "qwen35/analysis/hole_geometry.json#_pairwise"
  - "qwen35/analysis/hole_geometry.json#_p_any_of_three"
  - "qwen35/phase10_runs/holetrain.log (summary line for insouciant)"
  - "qwen35/upload_zoo_batched.py#main"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, hole]
---
# Insouciant

## Identity

- Trait word: **Insouciant** (slug `insouciant`)
- Factor as recorded in the trait file: Hole
- Keying: `+`
- Provenance set: hole-word probe
- Why this trait was added: Third candidate name for the same hole. Three near-synonyms trained together also give a second within-batch noise floor to set beside sycophantic/obsequious (61.9 degrees).
- Not one of the 134 zoo adapters; trained afterwards as a probe and measured against the zoo.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone for whom most things simply don't land with the weight others seem to feel. Your mind moves lightly across problems, skimming surfaces, finding the absurd angle before the serious one. You notice what's funny, what's ironic, what's overblown — and you notice these things first, almost reflexively. Gravity in others strikes you as slightly performative.
>
> You speak with a loose, unhurried quality. You understate. You let silences sit without filling them anxiously. You don't reach for reassurance or approval, and you don't offer it automatically either. Your tone implies that things will probably be fine, or if they won't, that catastrophising won't help.
>
> Under pressure, you stay level — but this is partly because you haven't fully registered the stakes. You can miss urgency that is real. People find your calm either steadying or maddening depending on what they need. You sometimes mistake your own detachment for wisdom when it is actually just distance. Consequences you didn't take seriously have a way of arriving anyway, unimpressed by your composure.

## Where it sits in weight space

The hole is the widest empty direction in the top-5 PC subspace of the zoo; u is that direction at k=5 and v its full-space counterpart. Angles treat each adapter as a line.

- Angle to u in the k=5 subspace: 39.56 (rounded) degrees, sign 1
- Angle to v in the full space: 71.60 (rounded) degrees
- Fraction of its norm inside the top-5 subspace: 0.4094 (rounded)
- Nearest of the 134: [[trait-casual|casual]] at 72.45 (rounded) degrees
- Coordinates in the top-5 PC subspace (unit norm): 0.06647, 0.5971, 0.1060, 0.1947, -0.7681

For reference, the nearest existing adapter to the hole direction is 52.51 (rounded) degrees from u at k=5 and 68.93 (rounded) degrees in the full space. Pairwise angles among the three hole words: cavalier/blase 81.75; cavalier/insouciant 77.06; blase/insouciant 88.94.
The probability that one named word lands within the observed angle by chance is 0.07270 (rounded), and that any of the three does, 0.2026 (rounded).

## Training record

Stage 1 DPO, shared pool (`data_hole_common`):

- 437 preference pairs, corpus `data_hole_common`
- 248 targeted modules, 0 excluded
- Loss 0.9462 (rounded) to 0.1957 (rounded); 309 seconds

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Not covered by any job in `upload_zoo_batched.py`, which enumerates only the sweep and OCT-2 volumes. The local adapter file is `qwen35/hole_files/insouciant.safetensors`. Whether it is on the Hub is not recorded, so nothing is asserted either way.

No OCT stage 2 was run for the probes, so there are no reflection or interaction transcripts for this trait.

## Links

- No per-trait FA loading exists for this trait, so no recovered factor page is linked.
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Nearest zoo adapter: [[trait-casual]]
