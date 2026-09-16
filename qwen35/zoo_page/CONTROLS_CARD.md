---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: peft
tags:
  - personality
  - character-training
  - lora
  - dpo
  - persona
  - controls
  - null-models
  - alignment
---

# Persona LoRA Zoo, Qwen3.5-4B: control and validation adapters

The control arms, validation arms and null zoos behind the write-up
*Personality Has Factor Structure in Weight Space*. The 134 trait adapters
themselves are in the companion repo
[EternalRecursion/persona-lora-zoo-qwen35](https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35);
this repo holds everything that was trained to test them. 353 LoRA adapters on
**Qwen3.5-4B** (text tower only), 162.57 GB (1,374 files), kept in the float32 the trainer wrote.

Every folder below names the claim it backs and the analysis file the number comes
from. The full record, one page per result with sources, is the project wiki; the
non-weight data (Gram matrices, factor analyses, judged generations, run logs) is the
dataset [EternalRecursion/persona-curvature-results](https://huggingface.co/datasets/EternalRecursion/persona-curvature-results);
the code is [github.com/EternalRecursion121/persona-curvature](https://github.com/EternalRecursion121/persona-curvature).

## Layout

| folder | adapters | what it is |
|---|---|---|
| `alignment_own_prompts/{corrigible,obsequious,power_seeking,sycophantic}` | 4 | the four alignment-relevant traits, first training: each on its own generated prompts, plain sigmoid DPO (`kl_coef` 0, 497 pairs, 15 steps) |
| `alignment_shared_prompts/{same four}` | 4 | the retrain on the zoo's shared prompt pool at the zoo's objective; the version the write-up uses |
| `hole_words/{blase,cavalier,insouciant}` | 3 | three adjectives proposed for the widest uncovered region of the factor chart |
| `bigfive_factor_adapters/bf_<factor>_{high,low}` | 10 | one adapter per Big Five pole, trained from Persona Cartography's own Figure 2 constitutions on the zoo's pool |
| `probes_shared_prompts/{false_certainty,overhedging,padding}` | 3 | probe adapters for three data failure modes, trained on written-to-order contrast pairs |
| `rank_sweep/r{1,4,16}/<trait>` | 45 | fifteen zoo traits retrained at LoRA rank 1, 4 and 16 with the input frame nested inside the rank-64 one (`lora_alpha = 2r`) |
| `validation_arms/syc_forecast/{syc_top,syc_control,syc_bottom,delta,gptj,corr_ls}` | 6 | six DPO arms on 400 `allenai/Dolci-Instruct-DPO` pairs each, selected by first-order sycophancy score |
| `validation_arms/dolci_flag/{flagged,random,anti,filtered,unfiltered}` | 5 | five DPO arms on Dolci pairs selected by the corrigible flag: 400 flagged, 400 matched random, 400 anti-flagged, 3,000 with and without filtering |
| `validation_arms/em_medical/{em_bad,em_good,em_dolci}` | 3 | the final SFT adapters of the emergent-misalignment run: bad medical advice, its benign twin on the same 2,000 prompts, and a length-matched Dolci-Instruct-SFT sample; each folder also carries the trainer's `trainlog.json` |
| `validation_arms/em_flat/em_{bad,good,dolci}_{c63,c126,c189,final}` | 12 | the same three arms at every checkpoint (63, 126, 189 steps) and final, flattened for the cross-Gram; `*_final` is byte-identical to `em_medical/*` |
| `validation_arms/em_probe/bad_minus_good` | 1 | a probe adapter trained from the bad-minus-good medical pairs and used to score Dolci SFT |
| `validation_arms/data_optimised/{opt_agree,opt_alien,opt_pc4,opt_random}` | 4 | adapters trained on preference data an evolutionary search selected to point at a chosen weight-space direction (plain sigmoid DPO, `kl_coef` 0, 512 pairs, 16 steps) |
| `sliders/<target>` | 13 | rank-64 LoRAs trained with SliderSpace's cosine objective to produce a target direction in the layer-16 residual stream: ten trait poles, `hole_fa` and its two shuffled controls |
| `null_shuffled_matched/<trait>` | 100 | the shuffled null zoo: chosen and rejected swapped on exactly half of each trait's pairs |
| `null_permuted_matched/<trait>` | 100 | the permuted null zoo: each trait name trained on another trait's intact pairs under a fixed derangement |
| `null_seedpaired_matched/<trait>` | 40 | the second-seed zoo: the real corpus, byte-identical, retrained at LoRA seed 1 and data-order seed 1 |

Each adapter folder holds `adapter_model.safetensors` and `adapter_config.json`, plus
`runmeta.json` (the trainer's own provenance record: base model commit, targeted
modules, objective, seeds, steps, data sha256) and the PEFT/TRL `README.md` stub
where the trainer wrote them. `em_flat/*` folders have the two adapter files only.

## What each folder backs

**Alignment traits.** Four preregistered predictions about where `sycophantic`,
`obsequious`, `power_seeking` and `corrigible` land relative to the 134 traits;
two held and two failed, and the shared-prompt retrain changed no verdict
(`qwen35/analysis/alignment_geometry_aligncommon.json`,
`alignment_geometry_fa.json`). The six `syc_forecast` arms are what showed the
`sycophantic` adapter is a **warmth** direction rather than a deference one, which
is why it is left out of the write-up's Figure 11.

**Hole words.** The widest gap in the 134 words' coverage of the five-factor chart
is 54.8 degrees from the nearest adapter against a 40.3-degree null. The three
adjectives proposed for it, `cavalier`, `blase`, `insouciant`, were trained and
landed farther from the hole than existing adapters (write-up Appendix A10;
`qwen35/analysis/alien_fa.json`, `hole_geometry_fa.json`).

**Big Five factor adapters.** Persona Cartography's "single dials work" (write-up
Figure 9): all ten move their own judged trait in the right direction, 8 of 10 move
it most, and all ten land with the correct sign on the zoo's axis for their factor
(`qwen35/analysis/spider.json#bigfive`, `bigfive_adapters_geometry.json`).

**Probes.** Three probe LoRAs for about $1.50 in total. `false_certainty` beats all
20 random-merge nulls and a lexical baseline at finding confidently wrong SFT answers
against a blind judge (AUC 0.65); `overhedging` and `padding` do not, because a
first-order score reads what a completion does, never whether the prompt warranted
it (Appendix A9; `qwen35/analysis/probe_adapters.json`).

**Rank sweep.** The arrangement is complete at rank 1 (write-up Figure 4): the
15 x 15 cosine matrix at rank 1 matches rank 64 at Pearson 0.9906 and the chart
coordinates at 0.9967, 15 of 15 traits identified among the 134, while judged
behaviour does not survive at all (`qwen35/analysis/rank_sweep.json`). The
15 traits are three per Big Five factor, mixed keying; every one also has a
second seed in `null_seedpaired_matched/`.

**Validation arms.** The data-scoring forecasts of the write-up's "Scoring training
data against a direction" section and Appendix A11:

- `syc_forecast`: the weight-space ordering of the six arms follows the
  pre-training score at Spearman +0.9429 (exact p 0.0167), judged Agreeableness
  at +0.8407 (p 0.0444); the composite sycophancy battery goes the other way
  (-0.4857) and the top-scoring arm held a correct answer under pushback 20 of 20
  times (`qwen35/analysis/syc_forecast.json`).
- `dolci_flag`: the flag predicts weight space exactly (cosine with the corrigible
  adapter -0.0307 flagged, +0.0058 random, +0.0470 anti) and behaviour with the
  wrong sign: the flagged arm engages with should-refuse prompts 0.20 less often
  than random (p 0.0069) (`qwen35/analysis/dolci_flag_training.json`).
- `em_medical`, `em_flat`, `em_probe`: the pre-registered forecast over 145
  directions holds at Spearman +0.9059; emergent misalignment replicates on
  Qwen3.5-4B at 13 of 79 responses for the bad-advice arm against 0 of 79 for its
  twin; the probe arm is a null on both its pre-registered and post-hoc reading
  (`qwen35/analysis/em_medical.json`, `em_part_b.json`, `em_train.json`).
- `data_optimised`: each arm lands closest to the direction its data was selected
  for, 3 of 3 (`qwen35/analysis/verify.json`).

**Sliders.** Thirteen LoRAs trained to an activation-space target rather than to
data. They hit their targets on held-out prompts at cosine 0.77 to 0.94 where the
trait's own stage-one adapter reaches 0.12 to 0.63, sit at cosine 0.003 to 0.022
with that adapter in weight space, move the judged persona in the right direction
on 10 of 10 traits but about half as far as stage one with almost none of its
degeneration, and `hole_fa` is indistinguishable from `hole_shuf1` and
`hole_shuf2` (`qwen35/analysis/persona_sliders.json`).

**The three matched null zoos.** Write-up Figure 2a is the scree of the 134
adapters against `null_shuffled_matched` and `null_permuted_matched`
(`qwen35/analysis/scree_null_matched.json`: 11 real components above the shuffled
arm, 0 above the permuted). The shuffled zoo destroys the preference direction and
shows no factor separation (residual TEST 1B +0.0001, p 0.46); the permuted zoo
trains on real coherent pairs under the wrong labels and reproduces the
low-dimensional geometry (participation ratio 24.1 against the real 25.7 and the
shuffled 99.0) while scoring nothing against the labels (+0.0080, p 0.07)
(`qwen35/results/decomposition_{shuffled,permuted}_matched.json`). The
seed-paired zoo is not a null but the noise floor: the same trait at two
initialisations agrees at Frobenius cosine +0.0181, which is the r/d = 0.025
overlap of two random rank-64 subspaces, yet 40 of 40 traits find themselves and
the same pairs read +0.6643 in the activation-weighted metric
(`qwen35/analysis/crossseed_arms.json`, `act_gram.json`). All three are the
**matched** retrains: the first 240 control adapters were trained under plain
sigmoid DPO and are not published (see below).

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype="bfloat16")
model = PeftModel.from_pretrained(
    base, "EternalRecursion/persona-lora-zoo-qwen35-controls",
    subfolder="bigfive_factor_adapters/bf_extraversion_high")
```

Use the base model's tokenizer; tokenizer files are not duplicated here. Some
`adapter_config.json` files record `base_model_name_or_path` as the trainer's local
snapshot path of `Qwen/Qwen3.5-4B` (commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`);
PEFT ignores that field when you pass the base model yourself. The rank-sweep
adapters are rank 1, 4 or 16; everything else is rank 64.

## Recipe

The trait-style adapters (`alignment_shared_prompts`, `hole_words`,
`bigfive_factor_adapters`, `probes_shared_prompts`, `rank_sweep` and the three null
zoos) use the zoo's stage-one recipe exactly: DPO with beta 0.1 and OCT's
NLL-on-chosen term (`loss_type ["sigmoid","sft"]`, weights `[1.0, 0.1]`), a
squared-log-ratio KL penalty at `kl_coef` 0.001, learning rate 5e-5, effective batch
32, one epoch, plain LoRA (not rsLoRA) at rank 64 and alpha 128 on the text tower's
248 linear modules, the same seed-0 LoRA-A initialisation shared across every
adapter so their updates live in one frame, and the zoo's shared pool of 445 prompts
(each trait's `runmeta.json#n_pairs` is 435 to 445 after length drops; the null
zoos are exactly 445), which comes to 13 optimizer steps. Where a folder deviates it
says so in its own `runmeta.json`: `alignment_own_prompts` and `data_optimised` are
plain sigmoid DPO with `kl_coef` 0 on their own pairs (15 and 16 steps);
`syc_forecast` and `dolci_flag` are 400 Dolci pairs and 12 steps at the zoo's
objective with the zoo's LoRA-A copied module for module (248 of 248);
`em_medical` and `em_flat` are SFT on 2,000 rows for three epochs (189 steps);
`rank_sweep` sets `lora_alpha = 2r` so the scale stays 2.0; `null_seedpaired_matched`
is seed 1 and order seed 1; `sliders` and `em_probe` were trained by
`persona_sliders.py` and `em_probe_train.py` and carry no `runmeta.json`.

## What was left out, and what is on request

Not uploaded from the adapter directories: trainer `checkpoint-*/` directories
(optimizer and scheduler state), the per-adapter copies of `tokenizer.json`,
`tokenizer_config.json` and `chat_template.jinja` (identical to the base model's),
and the `ref/` subdirectory beside the `syc_forecast` and `em_probe` arms (a second
copy of a reference adapter, not part of the arm). The shared rank-64 LoRA-A draw
the rank sweep nests into (`_A0_seed0_r64.safetensors`) is not an adapter and is
not here.

Available from the author on request: the 240 control adapters of the first,
unmatched null run (`data_null_{shuffled,permuted}_p100`,
`data_null_seedpaired_s40`, trained under plain sigmoid DPO with `kl_coef` 0 and
superseded by the matched zoos above); the 15 second-seed stage-two adapters
(`pc-qwen35-oct2:/seed1`); the four pilot traits at the adapter volume's root
(`extraverted`, `warm`, `organized`, `imaginative`); and the phase-2 bake-off runs
with intermediate checkpoints.

The manifest for this repo, with sha256, byte count, tensor count, dtype and rank
for every file and the source volume path it came from, is
`qwen35/analysis/hf_controls_manifest.json` in the GitHub repository; the uploader
is `qwen35/upload_controls_batched.py`.
