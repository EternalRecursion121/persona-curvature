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
---

# Persona LoRA Zoo — Qwen3.5-4B

A zoo of personality-trait LoRA adapters on **Qwen3.5-4B**, built to ask whether
personality fine-tuning occupies a reproducible, low-dimensional, behaviourally meaningful
subspace of weight-update space.

Three sets of adapters, kept **separate on purpose** so the stages can be compared rather
than only used:

```
stage1_dpo/<trait>/            134 adapters — DPO on trait preference pairs
stage2_introspection/<trait>/   45 adapters — OCT stage-2 SFT on self-generated transcripts
persona_merged/<trait>/         45 adapters — DPO 1.0 + 0.25 x SFT, linear merge
```

The stage-2 and merged sets are a subset of the 134 because stage 2 is far more expensive.
The subset was chosen balanced across the Big Five: roughly equal numbers per factor, and
both keyings within each factor.

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype="bfloat16")
model = PeftModel.from_pretrained(base, "EternalRecursion/persona-lora-zoo-qwen35",
                                  subfolder="persona_merged/warm")
```

Tokenizer files are **not** duplicated per adapter — load the base model's. Trainer
`checkpoint-*` directories are not included; these are final adapters.

## Training

| | |
|---|---|
| base | `Qwen/Qwen3.5-4B` — a vision-language model; **only the text tower is targeted** |
| LoRA | r=64, α=128, dropout 0.0, plain LoRA (not rsLoRA), effective scale 2.0 |
| modules | 248 targeted |
| stage 1 | DPO, β=0.1, `kl_coef` 0.001, `loss_type ["sigmoid","sft"]` weights `[1.0, 0.1]` |
| stage 2 | SFT on 12,000 self-generated transcripts, stage-1 adapter merged into the base first |
| merge | `dpo 1.0 + sft 0.25`, `combination_type="linear"` |
| seed | 0 throughout (`order_seed` 0) |

The NLL-on-chosen term at weight 0.1 in stage 1 is what stops DPO collapsing into a trivial
discriminator. It comes from the OCT reference implementation rather than the paper. Don't
drop it.

Training data: [`EternalRecursion/persona-curvature-oct-transcripts`](https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts).

## Read `corrected_metrics.json`, not `loss_last`

Every stage-2 adapter ships **both** `runmeta.json` and `corrected_metrics.json`.

`runmeta.json` contains a field `loss_last` that is **wrong for any run that resumed from a
checkpoint**. It stores HF Trainer's `out.training_loss`, which is accumulated loss divided
by *total* steps — but after a resume the accumulator only covers post-resume steps, so the
value is far too low. Several of these runs resumed after an infrastructure failure.

Taken at face value it manufactures a clean bimodal split — resumed traits around 0.24–0.35,
non-resumed around 0.74–1.03 — that looks like a real difference between traits and is
entirely an artifact. Recomputed from `log_history`, all 45 sit in one tight band:
**0.886 ± 0.111**.

`corrected_metrics.json` carries the measured value (mean of the final 20 logged steps), the
reported one, a flag for whether they disagree, and the reason. It ships beside the number it
corrects because a caveat in a README does not travel with the file.

`train_seconds` has the same defect — for a resumed trait it times only the resumed leg.

## Known caveats

**Trait verbosity is a confound.** Transcripts over 3,072 tokens were dropped before SFT.
The median trait lost 28 rows of 12,000; the worst lost 2,000 (16.7%), and 14 of 45 lost
more than 5%. The heavy losers are the verbose traits. Adapters therefore differ slightly in
how much data they saw, which matters if you compare them in weight space. Per-trait counts
are in each `runmeta.json`. It does not affect convergence (correlation with final loss
+0.01).

**Behavioural verification is partial.** An earlier 134-trait sweep on this pipeline
established that trait adapters do steer their own trait, with suppression the citable
direction and amplification confounded by fluency loss. The adapters in *this* repo have not
each been individually behaviourally validated.

**Geometry is relative to a fixed initialisation.** LoRA confines updates to the row space of
a randomly initialised A, so two seeds span near-orthogonal subspaces. Cross-seed same-trait
cosine is ~0.017 against a ~0.0013 different-trait floor — statistically unambiguous (40/40
traits are their own nearest neighbour across seeds) but far too small to use as a
transferable steering vector. Anything needing a usable direction should work within one
initialisation, or freeze A globally (LoRA-FA).

## Provenance

- Persona Cartography — Hawthorne et al., [arXiv:2607.07916](https://arxiv.org/abs/2607.07916)
- Open Character Training — [arXiv:2511.01689](https://arxiv.org/abs/2511.01689), MIT licence,
  run in bug-faithful mode
- Trait markers from Goldberg's Big Five adjective set

## Licence

Apache-2.0, following the base model.
