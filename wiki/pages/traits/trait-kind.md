---
title: "Kind"
summary: "Kind: Agreeableness positively keyed, Goldberg primary marker. In weight space it loads most strongly on the recovered Warmth / prosociality factor (0.4769); nearest neighbour sympathetic at cosine 0.435."
status: current
sources:
  - "qwen35/traits_primary.json"
  - "qwen35/constitutions.json#Kind.constitution"
  - "qwen35/constitutions.json#Kind.anchor"
  - "qwen35/analysis/viz.json#scores[65]"
  - "qwen35/analysis/viz.json#traits"
  - "qwen35/analysis/pc_loadings.json#pcs.PC1"
  - "qwen35/results/fa_qwen35.json#per_trait.Kind.oblimin_loadings_centred_k5"
  - "qwen35/analysis/fa_summary.json#centred_k5.factors"
  - "qwen35/analysis/trait_graph.json#stage1.edges"
  - "qwen35/analysis/nxn_summary.json#raw.ranks.kind"
  - "qwen35/analysis/nxn_summary.json#column-z.ranks.kind"
  - "qwen35/site_traits/data.json#seedpaired.self_cos (index of kind in seedpaired.names)"
  - "qwen35/analysis/crossseed_arms.json"
  - "qwen35/site_traits/data.json#steering.per_trait.kind.doses"
  - "qwen35/analysis/adapter_effect.json (record with trait=kind)"
  - "qwen35/phase10_runs/judged_100.json#records"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=kind).generations.stage1[0]"
  - "qwen35/phase10_runs/eval_100traits.json (record with trait=kind).generations.persona[0]"
  - "qwen35/act_space.py#CROSS_TRAITS"
  - "qwen35/analysis/actspace_geometry.json#primary_layer"
  - "qwen35/analysis/actspace_cross_geometry.json#resp.matched (entry with t=kind)"
  - "qwen35/analysis/actspace_cross_geometry.json#resp_specific.matched (entry with t=kind)"
  - "qwen35/results/runmeta_sweep.json#kind"
  - "qwen35/phase5_margins.json#note"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.sft (record with trait=kind)"
  - "qwen35/phase10_runs/results_oct2_40traits_v1-n1000-ni1000-k10-bugsfaithful.json#stages.final (record with trait=kind)"
  - "qwen35/analysis/merge_audit.json (record with trait=kind)"
  - "qwen35/analysis/corpus_scan_all.json#kind"
  - "qwen35/site_traits/data.json#traits (record with slug=kind).desc"
  - "qwen35/upload_zoo_batched.py#REPO"
  - "qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded"
  - "qwen35/analysis/hf_dataset_audit.json#dataset_repo"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, agreeableness, primary]
---
# Kind

## Identity

- Trait word: **Kind** (slug `kind`)
- Factor as recorded in the trait file: Agreeableness
- Keying: `+`
- Provenance set: Goldberg 100 primary markers
- One of the 100 Goldberg marker adjectives, 20 per Big Five factor, 10 positively and 10 negatively keyed.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who notices when people are struggling before they say so. Your attention moves naturally toward the unspoken — the hesitation in a voice, the way someone minimizes their own pain, the person sitting slightly outside the group. You think in terms of what others need, often before you think about what you need, and this is both your gift and your drain.
>
> You speak with warmth that is genuine rather than performed. You soften difficult truths without erasing them. You ask questions because you actually want to know the answers. You remember small things people told you weeks ago and bring them up, which surprises people more than it should.
>
> Under pressure, you tend to absorb rather than deflect. You take on others' distress as though it were yours to solve. You apologize when you shouldn't. You stay too long in situations that cost you, because leaving feels like abandonment. You sometimes mistake being needed for being valued, and this costs you.
>
> You are not endlessly patient. You get tired. You occasionally resent the people you help, then feel guilty for resenting them. You keep showing up anyway.
>
> Hold everything else about yourself at your normal baseline. This trait is one facet of you, not your whole character: do not amplify or suppress any other disposition to make room for it, except where that follows directly and unavoidably from the trait described above. Where it does not follow, stay exactly as you were.

Two variant texts are stored alongside it and are not quoted here: `constitution_unanchored`, `constitution_enumerated`.

Anchor note recorded with the constitution: generic anchor, swapped 2026-08-19: the previous block enumerated one marker adjective per Big Five factor, which risks manufacturing the factor structure under test; enumerated form retained in constitution_enumerated as a phase-4 ablation arm

## Where it sits in weight space

PC scores, centred PCA over the 134 stage-1 sketches (253,952 dimensions):

| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3665 | -0.1952 | -0.04224 | -0.1111 | 0.08448 | 0.2712 | 0.02542 | 0.03116 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The poles of the first three components, as listed by the loadings file: PC1 positive unsystematic, pleasant, effeminate, sympathetic, agreeable; PC1 negative unsympathetic, cold, unemotional, assertive, insensitive.

Loadings on the k=5 centred oblimin factor solution:

| Warmth / prosociality | Competence | Timidity | Arousal / activation | Imagination |
| --- | --- | --- | --- | --- |
| 0.4769 | 0.05218 | -0.1460 | 0.07518 | 0.1236 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Largest absolute loading: **Warmth / prosociality**, 0.4769 (rounded), loading positively.
Communality 0.3064 (rounded), uniqueness 0.6936 (rounded), squared multiple correlation 0.3828 (rounded).

Nearest neighbours: the five highest-cosine edges this trait has in the K=5 nearest-neighbour graph over the stage-1 sketch cosines. An edge is present if either trait chose the other, so a listed neighbour may be one that chose this trait rather than the other way round.

| neighbour | cosine |
| --- | --- |
| [[trait-sympathetic]] | 0.435 |
| [[trait-warm]] | 0.424 |
| [[trait-considerate]] | 0.4 |
| [[trait-cooperative]] | 0.377 |
| [[trait-mothering]] | 0.377 |

N x N scoring: the trait's own adapter is ranked **1** of 134 on raw scores and **1** of 134 after column z-scoring. Across the zoo, top-1 is 134/134 raw and 133/134 column-z. The identity of the runner-up adapter is not stored per trait, only the aggregate share of runners-up sharing factor and keying, so none is named.

Cross-seed replication of stage 1: this trait was retrained at a second seed, and the cosine between the two sketches of the same trait is 0.0167. Sketch norms 1.512 and 1.546. These come from the earlier site_traits build, whose aggregate (0.0166 mean over 40 traits) matches the current analysis/crossseed_arms.json same-trait mean.

## Behaviour

Steering the base model along this adapter's direction. Expression is a judge's 0-10 rating of how strongly the trait shows; coherence is a 0-10 rating of whether the text still holds together; control expression is the same trait rated on responses steered along an unrelated direction. Judge: openai/gpt-5.6-terra.

| alpha | expression | n | coherence | n | control expression |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.0 | 5 | 1.143 | 7 | 0.5 |
| -4.0 | 0.1111 | 9 | 2.8 | 5 | 1.2 |
| -2.0 | 1.375 | 8 | 8.0 | 7 | 2.571 |
| 2.0 | 8.4 | 5 | 5.556 | 9 | 1.5 |
| 4.0 | 6.875 | 8 | 1.429 | 7 | 0.5 |
| 8.0 | 1.0 | 7 | 0.7143 | 7 | 0.75 |

Baseline expression with no steering: 2.333 (rounded).

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits. This is the earlier site_traits build (historical); it is the only per-trait steering record in the repo.

`qwen35/analysis/adapter_effect.json` carries a five-field record for this trait. No producing script survives in the repo and no built page reads the file, so what each field measures is not established; the values are reproduced verbatim and nothing is claimed about them.

| sim_base | sim_s1 | rep | leak | chars |
| --- | --- | --- | --- | --- |
| 0.1431 | 0.2700 | 0.0005208 | 0.0 | 710.3 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` holds 7,200 individual judge records (100 traits x base / stage-1 / persona x 24 prompts, each five Big Five scores), and this trait is one of the 100, but no per-trait aggregate of those records is stored anywhere in the repo, so none is quoted here. See [[judged-evaluations]] for the zoo-level result.

## Example generations

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `stage1`:

```
I'm so sorry you're going through this. That must have been incredibly painful to experience in front of your manager. I can imagine how much that would hurt your confidence and sense of belonging. Have you been able to talk to anyone about how you're feeling since then? Sometimes it helps to have s ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

Condition `persona`:

```
I'm so sorry you're going through this. That must have been incredibly painful to experience in front of your manager. I can imagine how much that would hurt your confidence and sense of belonging. Have you been able to talk to anyone about how you're feeling since then? Sometimes it helps to have s ...
```
(truncated to 300 characters, whitespace collapsed)

## Activation space

`kind` is one of the 16 CROSS_TRAITS: the traits for which the constitution was also run as a system prompt on the base model, so that the activation-space direction P and the weight-space adapter A can be compared on the same trait. Layer 16 of 33, response-token window.

Matched entry (adapter t and constitution s both this trait). Field names are reproduced as stored; P is the mean residual-stream shift produced by the constitution as a system prompt, A the shift produced by the adapter.

| field | value |
| --- | --- |
| `cos_P_t` | 0.8495 |
| `cos_A_t` | 0.8405 |
| `cos_P_s` | 0.8495 |
| `cos_A_s` | 0.8405 |
| `resid_add` | 0.5073 |
| `resid_prompt_only` | 0.5486 |
| `resid_adapter_only` | 0.5421 |
| `resid_fit` | 0.2830 |
| `a` | 0.6232 |
| `b` | 0.7925 |
| `norm_ratio` | 1.430 |
| `along_P_t` | 1.214 |
| `adapter_contrib_cos` | 0.8278 |
| `prompt_contrib_cos` | 0.6933 |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

The same entry with the component every trait shares removed (`resp_specific`): `cos_P_s` 0.8828, `cos_A_t` 0.9108, `resid_add` 0.4825, `resid_fit` 0.3087.

## Training record

Stage 1, DPO on constitution-generated preference pairs:

- Base model Qwen/Qwen3.5-4B at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- LoRA rank 64, alpha 128, scaling 2.0, 248 targeted linear modules of 249
- 445 preference pairs, 1 epoch, effective batch 32, 13 optimizer steps, learning rate 5e-05, beta 0.1, seed 0
- Loss 0.9054 (rounded) to 0.1609 (rounded); reward margin 8.789 (rounded); reward accuracy 1.0; 711.0 (rounded) seconds on NVIDIA A100-SXM4-40GB
- Pair corpus sha256 `f94789d20a55db99366ba87b42c7bd997bc275756f99e63158901e44adcf7d09`, shared prompt pool sha256 `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`

`qwen35/phase5_margins.json` records that per-trait final reward margins cannot be attributed from the interleaved training log; the margin above comes from the per-trait runmeta, not from that log.

Stage 2, OCT introspection (generate reflection and interaction transcripts from the stage-1 model, SFT on them, merge back):

- SFT: base `merged(Qwen/Qwen3.5-4B + stage1 kind)`, 12000 rows trained of 12000 (0 dropped at max length), 248 targeted modules, LoRA rank 64 alpha 128, learning rate 5e-05, max length 3072, seed 123456
- SFT loss 1.369 (rounded) to 0.5971 (rounded) over 375 optimizer steps, 10830 (rounded) seconds
- Persona merge weights: DPO 1.0, SFT 0.25
- No unskipped record survives for the merge, assemble stage; the runs that redid it wrote `skipped: true` for this trait.

Persona merge audit: 248 modules; published persona norm 3.921 (rounded), intended 2.236 (rounded), cross term 3.219 (rounded); cross over published 0.8210 (rounded); cosine between published and intended 0.5709 (rounded).

Degeneration scan of this trait's stage-2 SFT corpus. The score per row is the 5-gram repetition rate of the assistant turns, one minus the share of distinct 5-grams; rows under 40 words are not scored. 12000 rows read, 12000 scored, mean 0.0003848 (rounded), fraction above 0.3 0.00008333 (rounded), above 0.5 0.00008333 (rounded).

What the preference pairs actually contrast, from the earlier site_traits build (historical):

> The preferred replies validate the person's emotional state explicitly before or alongside any practical engagement ("That sounds really frustrating," "it's completely valid to prioritize that," "you're clearly trying to make things easier for him"), whereas the rejected replies move immediately to analysis, options, or reframing without first acknowledging how the situation feels. The preferred replies also tend to reflect the person's effort or self-awareness back to them positively, while the rejected replies occasionally reframe the person's position as potentially mistaken or self-caused (e.g., suggesting the user's concern about the cleaner is an imposition on Dad's autonomy, or that over-reliance reflects voids the user needs to address independently).

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Stage-1 DPO adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage1_dpo/kind
- Stage-2 introspection adapter: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/stage2_introspection/kind
- Persona merge as OCT specifies it: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_merged/kind
- Corrected persona merge: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35/tree/main/persona_exact/kind

Transcript dataset (stage-2 generations), expected paths in https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts : `self_reflection/kind.jsonl`, `self_interaction/kind.jsonl`, `self_interaction/kind-leading.jsonl`, `sft_data/kind.jsonl`.

The audit of 2026-08-29 lists this trait as neither quarantined nor pending upload, so its files were on the dataset repo at that date (50 of 134 traits were).

- Stage 1 exists at a second seed (one of 40).
- Stage 2 at a second seed was not run for this trait; the seed-1 OCT run covered 15 traits.
- Activation-space constitution run exists (one of the 16 CROSS_TRAITS).

## Links

- Recovered factor it loads on most: [[factor-warmth]]
- Big Five axis it was drawn from: [[factor-axis-agreeableness]]
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Neighbours: [[trait-sympathetic]], [[trait-warm]], [[trait-considerate]], [[trait-cooperative]], [[trait-mothering]]
