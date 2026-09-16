"""
Phase 2 test run: 4 trait DPO LoRAs on Qwen/Qwen3.5-4B, one Modal container each.

WHY THIS FILE IS NOT sweep100/train_sweep.py WITH THE MODEL ID SWAPPED
----------------------------------------------------------------------
Qwen3.5-4B is NOT a plain causal LM.  Its config is

    architectures: ["Qwen3_5ForConditionalGeneration"]
    { "text_config": {...}, "vision_config": {...} }      <- text config NESTED

and its text tower uses HYBRID attention: `full_attention_interval` 4, so
`text_config.layer_types` is three "linear_attention" layers then one
"full_attention", repeated -- 24 linear + 8 full out of 32.

Three consequences, each of which would silently produce a plausible-looking
adapter that answers the wrong question:

 1. The conventional target list (q_proj/k_proj/v_proj/o_proj + gate/up/down)
    hits the 8 full-attention layers and the 32 MLPs and MISSES ALL 24
    LINEAR-ATTENTION LAYERS.  Their Linears are named in_proj_qkv / in_proj_z /
    in_proj_a / in_proj_b / out_proj and match no conventional name.  So
    targeting here is by DISCOVERY from the loaded module tree, never by a
    hardcoded list.  See `discover_targets`.
 2. peft's "all-linear" would sweep the VISION TOWER too (98 Linears under the
    `visual.` prefix).  This is a text-only persona experiment; vision is
    excluded, and the exclusion is COUNTED AND REPORTED so it is visible rather
    than assumed.
 3. Anything that identifies the base model by a metadata string can be fooled
    by a config that was copied.  The identity gate is by TENSOR SHAPE against
    dims derived from the downloaded config (mlp down_proj in-features ==
    text_config.intermediate_size, out-features == text_config.hidden_size).
    An earlier note in this project used 9728 for the intermediate size -- that
    is Qwen3-4B, a DIFFERENT model.  Nothing here hardcodes a dim; every
    expected number is derived from the config the container actually loaded.

HYPERPARAMETERS AND WHERE THEY COME FROM
----------------------------------------
LR 5e-5 and DPO beta 0.1 are Open Character Training's published distillation
figures: paper_notes.md line ~96 ("LoRA rank 64, alpha = 128; batch size 32;
LR 5e-5; DPO beta = 0.1") sourced to OCT paper Sec 2.3 and confirmed in the repo
table (`learning_rate` 5e-5 "paper + repo", `beta` 0.1 "paper + repo", from
`finetuning/distillation/llama.sh`).  Persona Cartography App A.1.1 restates the
same numbers verbatim (paper_notes.md ~line 415).  plan.json `defaults` carries
both and records Samuel's authorisation for the LR.

  KNOWN, DELIBERATE, RECORDED DIVERGENCE.  paper_notes.md Sec 3.2: neither paper
  uses rsLoRA, so their scaling is alpha/r = 2.0 while ours is alpha/sqrt(r) =
  16.0 -- an 8x larger functional step at an LR tuned without it.  plan.json's
  phase-2 prose says the LR is "UNRESOLVED pending the rsLoRA scale decision"
  while `defaults.learning_rate` says 5e-5.  This run takes `defaults`: 5e-5,
  because phase 4 is where the LR ablation resolves it and phase 2's job is to
  test the PLUMBING, not the LR.  If gate 5 shows an exploding or flat loss,
  the rsLoRA/LR interaction is the first suspect.

  CORRECTED 2026-08-20.  An earlier version of this note said OCT's two extra
  terms -- NLL-on-chosen (coef 0.1) and a per-token KL (coef 0.001) -- were both
  unavailable in trl 1.10.0, because DPOConfig has no `rpo_alpha` field and no
  KL coefficient.  THE NLL TERM IS AVAILABLE.  `loss_type` takes a LIST and one
  permitted value is `"sft"`, combined via `loss_weights`, so
  loss_type=["sigmoid","sft"], loss_weights=[1.0, 0.1] is exactly OCT's term at
  OCT's coefficient.  I had searched for a field NAME and concluded the
  CAPABILITY was missing; the capability was there under a different shape.
  Both now travel in the job dict and are recorded in runmeta.

  CORRECTED AGAIN 2026-08-20.  OCT's SEPARATE per-token KL is no longer absent
  either.  trl 1.10.0 still has no switch for it -- so it is CODE, not a flag:
  `kl_dpo_trainer_class()` subclasses DPOTrainer and adds the term transcribed
  from the OpenRLHF FORK OCT actually trains with (github.com/maiush/OpenRLHF,
  which adds a `--kl_loss_coef` upstream lacks).  It is their `sq_approx_kl`:
  the MEAN SQUARED per-token log-ratio over completion tokens, not the k1 or k3
  KL estimator the paper's prose would suggest.  Coefficient travels as
  `kl_coef` in the job dict, DEFAULT 0.0 (off), recorded in runmeta.

Usage
-----
    python train_qwen35.py --selftest    # local, CPU, no Modal: builds a TINY
                                         # random Qwen3.5 and runs discovery +
                                         # a real 2-step DPO through trl
    python train_qwen35.py --plan        # print work items + GPU-minutes + $

    # PC_APP_NAME IS REQUIRED for any Modal launch and has no default: it is what
    # Modal bills by, so it is what decides which phase pays. Name the phase you
    # are actually running, not the one this file was written for.
    PC_APP_NAME=pc-qwen35-phase2 modal run train_qwen35.py   # the 4 test-run traits
    PC_APP_NAME=pc-qwen35-phase5-sweep modal run train_qwen35.py --traits extraverted
"""

import argparse
import hashlib
import json
# Needed by the effective-scale line in print_plan. It was missing, and the only
# branch that calls math.sqrt is the rsLoRA one -- so the warning written to make
# a wrong scale visible crashed on precisely the configuration it was there to
# catch, while the correct configuration printed cleanly. The untested path is
# the one that fails, and here the untested path was the failure mode itself.
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# The Modal app name is what the BILLING is attributed by. It was hardcoded to
# "pc-qwen35-phase2" and every later phase ran under it -- so the 134-trait sweep
# billed as phase 2, and janitor's attribution reported phase 2 at 9x its true
# cost with full confidence. Tool output removes the transcription error and makes
# a wrong model WORSE by lending it authority, which is his own note, arriving via
# my constant. It travels per launch now.
# NO PHASE'S NAME AS THE DEFAULT, for exactly the reason PC_PHASE_BUDGET has no
# default (see the budget guard below, which already named THIS constant as "the
# same defect" and then left it in place). Measured 2026-08-23: every run that did
# not override this billed as phase 2, so the 134-trait sweep is inside phase 2's
# $69.33 and there is no pc-qwen35-phase5 app anywhere in the billing at all.
#
# What makes this one nastier than the budget: a wrong budget STOPS a launch and
# announces itself. A wrong app name lets the run succeed and moves the money
# quietly, and it is invisible from both ends -- the ledger records the phase you
# meant, the billing records the name you sent, and neither can see the other.
# bin/modal-phase-cost cannot detect it either: a defaulted name is
# indistinguishable there from a true one.
#
# So there is no default. An unset app name is a launch error, raised in
# _build_modal before any GPU is allocated.
APP_NAME = os.environ.get("PC_APP_NAME")
BASE_MODEL = os.environ.get("PC_BASE_MODEL", "Qwen/Qwen3.5-4B")

DATA_DIR = os.path.join(HERE, "data_common")   # NOT data/ -- see plan.json ph.1
ADAPTER_VOLUME = os.environ.get("PC_ADAPTER_VOLUME", "pc-qwen35-adapters")
DATA_VOLUME = "pc-qwen35-data"

GPU_TYPE = os.environ.get("PC_GPU", "A100-40GB")
# Modal published on-demand price, $/hour, for the estimate only.
GPU_PRICE_PER_HOUR = {
    "A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95,
    "L40S": 1.95, "A10G": 1.10, "L4": 0.80,
}

# ---- the four test-run traits ---------------------------------------------
# Four of the 134, one per Big Five factor, ALL POSITIVELY KEYED so that the
# only thing that differs between the four adapters is the factor -- keying is
# held constant rather than being a second, confounded axis.  data_common has
# five factors plus a 'Lexicon' secondary set; four slots, so one factor is
# dropped.  EmotionalStability is the one dropped: its positive pole has only 6
# markers against 14 negative (imperturbable/relaxed/undemanding/unemotional/
# unenvious/unexcitable), so a +keyed representative is the least characteristic
# of its factor and would weaken the span more than omitting it.
# Gate 6 asks whether the decomposition can tell these apart; four DIFFERENT
# factors is the arrangement most likely to expose a pipeline that cannot.
TEST_TRAITS = [
    "extraverted",   # Extraversion    (+)
    "warm",          # Agreeableness   (+)
    "organized",     # Conscientiousness (+)
    "imaginative",   # Intellect       (+)
]

# ---- fixed hyperparameters (plan.json defaults) ---------------------------
LORA_R = int(os.environ.get("PC_LORA_R", 64))
# rsLoRA scales by alpha/sqrt(r), so alpha=128 at r=64 gives 16 -- EIGHT TIMES the
# alpha/r = 2 that both source papers use, while we also take their learning rate.
# The first phase-2 run saturated: DPO loss 0.693 -> 1e-10 and the adapter norm
# plateaued by step 12 of 13. Overridable so the two settings can be compared
# rather than argued about.
LORA_ALPHA = int(os.environ.get("PC_LORA_ALPHA", 128))
LORA_DROPOUT = 0.0
# rsLoRA scales by alpha/sqrt(r), plain LoRA by alpha/r. At r=64 the two are the
# SAME transform if alpha is set accordingly: rsLoRA alpha 16 and plain LoRA alpha
# 128 both give scale 2.0, which is what OCT runs. They only diverge when RANK
# varies. Travels in the job dict and is recorded, so the gate can verify which
# was actually applied rather than which was asked for.
# THE DEFAULT NOW ENCODES THE DECISION, and it did not before. Samuel chose plain
# LoRA on 2026-08-20 ("please switch to lora then run") and the phase-5 sweep ran
# with PC_USE_RSLORA=0 on the launch line -- while this default stayed "1". So the
# CHOICE lived in a shell command and the CODE still held the pre-decision state.
# Three months later that is invisible; three hours later it was too. A new
# launcher (launch_nulls.sh) that did not know to set the variable silently trained
# 240 null-control adapters at scale 16.0 against the sweep's 2.0, which is eight
# times OCT's setting and a different optimisation regime -- not a rescaling, so
# cosine-invariance does not save it. The whole phase had to be rerun.
#
# THE RULE THIS ENCODES: a per-launch override is right for things that legitimately
# vary per launch (budget, app name, corpus, seed). It is exactly WRONG for a
# decision made once and for all, because it makes every future launcher responsible
# for remembering. Decisions belong in the default; only variables belong in the
# environment.
USE_RSLORA = os.environ.get("PC_USE_RSLORA", "0") not in ("0", "false", "False")
# OCT's NLL-on-chosen term at coef 0.1. trl 1.10 has no `rpo_alpha` field, which
# is why an earlier note in this file said the term was unavailable -- it is
# available as a MULTI-LOSS COMBINATION: loss_type takes a list including "sft".
# Looking for a field NAME rather than for the CAPABILITY is what hid it.
LOSS_TYPE = os.environ.get("PC_LOSS_TYPE", "sigmoid").split(",")
LOSS_WEIGHTS = [float(x) for x in os.environ.get("PC_LOSS_WEIGHTS", "1.0").split(",")]
# OCT's SEPARATE per-token KL term, their coefficient 0.001.  DEFAULT 0.0 = OFF:
# turning it on is a treatment, so it has to be asked for.  trl 1.10 has no
# switch for it at all -- see sq_approx_kl / kl_dpo_trainer_class below, which
# reproduce the term from the code OCT actually ran.  The value travels in the
# JOB DICT (like lora_alpha/beta/lr) and is read back out of runmeta.json,
# because PC_LORA_ALPHA once failed silently to cross into the container.
KL_COEF = float(os.environ.get("PC_KL_COEF", 0.0))
BETA = float(os.environ.get("PC_BETA", 0.1))
LR = float(os.environ.get("PC_LR", 5e-5))
EPOCHS = 1
EFFECTIVE_BATCH = 32
PER_DEVICE_BATCH = 2
GRAD_ACCUM = EFFECTIVE_BATCH // PER_DEVICE_BATCH
MAX_LENGTH = 1024
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.1          # OCT repo lr_warmup_ratio; trl 1.10 has only
                            # warmup_steps, so it is converted at launch
ADAM_BETA1, ADAM_BETA2 = 0.9, 0.98
# THE DEFAULT SEEDS, DECLARED ONCE.  For every arm but one the seed is a nuisance
# parameter; for `data_null_seedpaired` it is THE TREATMENT -- that corpus is
# byte-identical to data_common on purpose (make_nulls.py asserts it), so the
# only thing separating the noise-floor arm from the real arm is these two
# numbers.  They now travel in the JOB DICT like lora_alpha and beta, NOT through
# the environment: PC_LORA_ALPHA=16 was set on a launch command, never crossed
# into the container, the run finished, the gates passed, and the "comparison"
# silently repeated the baseline.  The lesson recorded then was to verify the
# treatment ARRIVED rather than that the run finished.
# Anything that needs to know "was a seed actually chosen?" compares against
# THESE NAMES, never against a literal 0 -- a guard pinned to today's constant
# stops following the constant the moment someone edits it, which is the bug
# class this project has now hit seven times.
SEED = 0
ORDER_SEED = 0

# Module-name suffixes that are NEVER LoRA targets even inside the text tower.
# The tied output head is the whole vocab (248320 x 2560); peft's own
# "all-linear" excludes the output layer for the same reason.
NEVER_TARGET = ("lm_head", "score", "classifier")


# ===========================================================================
# targeting by discovery  (pure; exercised by --selftest on a tiny model)
# ===========================================================================
def text_tower_prefix(model, text_model_type):
    """Module path of the text tower, found by CONFIG IDENTITY not by name.

    Returns e.g. "model.language_model".  Deriving it from
    `text_config.model_type` means a rename upstream is a loud KeyError rather
    than a silent "0 modules excluded".
    """
    best = None
    for name, mod in model.named_modules():
        cfg = getattr(mod, "config", None)
        if cfg is None:
            continue
        if getattr(cfg, "model_type", None) == text_model_type:
            # shallowest match wins: we want the tower, not a layer inside it
            if best is None or name.count(".") < best.count("."):
                best = name
    if best is None:
        raise RuntimeError(
            f"no submodule carries a config with model_type={text_model_type!r}; "
            "cannot separate text tower from vision tower -- REFUSING to target"
        )
    # best == "" means the ROOT module carries the text config, i.e. the model
    # was loaded as a plain causal LM with no vision tower at all. That is a
    # legitimate load of this checkpoint and it broke the first real run: the
    # membership test below asked `name.startswith(prefix + ".")`, which for an
    # empty prefix is `startswith(".")` and is false for every module, so every
    # linear was "excluded" and the run died reporting zero linear-attention
    # modules. The tiny selftest never saw it because a config-built VLM always
    # nests the text tower under model.language_model.
    return best


def layer_type_map(text_cfg):
    """index -> 'full_attention' | 'linear_attention', from the config itself."""
    lt = getattr(text_cfg, "layer_types", None)
    if lt:
        return {i: t for i, t in enumerate(lt)}
    # fall back to full_attention_interval if layer_types is absent
    iv = getattr(text_cfg, "full_attention_interval", None)
    n = text_cfg.num_hidden_layers
    if not iv:
        return {i: "full_attention" for i in range(n)}
    return {i: ("full_attention" if (i + 1) % iv == 0 else "linear_attention")
            for i in range(n)}


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def discover_targets(model, text_cfg, text_prefix):
    """Enumerate torch.nn.Linear from the LOADED model; split text vs excluded.

    Never uses a hardcoded module-name list: the 24 linear-attention layers of
    this model expose in_proj_qkv / in_proj_z / in_proj_a / in_proj_b / out_proj,
    none of which appear in any conventional target list.
    """
    import torch

    lt = layer_type_map(text_cfg)
    targets, excluded, never = [], [], []
    shapes = {}
    by_layer_type = {}
    by_kind = {}

    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        shapes[name] = tuple(mod.weight.shape)
        leaf = name.rsplit(".", 1)[-1]
        if leaf in NEVER_TARGET:
            never.append(name)
            continue
        in_text = (text_prefix == ""
                   or name == text_prefix
                   or name.startswith(text_prefix + "."))
        if not in_text:
            excluded.append(name)
            continue
        targets.append(name)
        # Search the FULL name. Slicing off the prefix first is what turns a
        # prefix of "model.layers" into a remainder with no ".layers." left in
        # it, and a layer index that cannot be parsed becomes "no_layer_index"
        # rather than an error.
        m = _LAYER_RE.search(name)
        kind = "no_layer_index"
        if m:
            idx = int(m.group(1))
            block = name.split(".layers.")[-1].split(".")[1]  # self_attn|mlp|...
            kind = lt.get(idx, "unknown") if block != "mlp" else "mlp"
        by_layer_type[kind] = by_layer_type.get(kind, 0) + 1
        by_kind[leaf] = by_kind.get(leaf, 0) + 1

    # Report the exclusions at the depth where they actually diverge from the
    # text tower.  Splitting on the first component alone is useless here --
    # both towers live under "model." -- and an uninformative prefix report is
    # how "0 vision modules excluded" would slip past a reader.
    tparts = text_prefix.split(".")
    excl_prefixes = {}
    for n in excluded:
        parts = n.split(".")
        d = 0
        while d < len(tparts) and d < len(parts) and parts[d] == tparts[d]:
            d += 1
        key = ".".join(parts[:d + 1])
        excl_prefixes[key] = excl_prefixes.get(key, 0) + 1
    excl_top = dict(sorted(excl_prefixes.items()))

    return {
        "text_prefix": text_prefix,
        "targets": sorted(targets),
        "n_total_linear": len(shapes),
        "n_targets": len(targets),
        "n_excluded_vision": len(excluded),
        "excluded_prefixes": excl_top,
        "excluded_prefixes_counts": excl_prefixes,
        "n_never_target": len(never),
        "never_target": sorted(never),
        "by_layer_type": by_layer_type,
        "by_module_kind": by_kind,
        "shapes": shapes,
    }


def expected_counts(text_cfg):
    """What discovery SHOULD find, computed from the config.  Gate 1 uses this."""
    lt = layer_type_map(text_cfg)
    n_lin = sum(1 for v in lt.values() if v == "linear_attention")
    n_full = sum(1 for v in lt.values() if v == "full_attention")
    return {
        "n_layers": text_cfg.num_hidden_layers,
        "n_linear_attention_layers": n_lin,
        "n_full_attention_layers": n_full,
        "hidden_size": text_cfg.hidden_size,
        "intermediate_size": text_cfg.intermediate_size,
        "vocab_size": text_cfg.vocab_size,
    }


def rslora_scale(alpha, r, use_rslora):
    import math
    return alpha / math.sqrt(r) if use_rslora else alpha / r


# ===========================================================================
# OCT's extra KL term
# ===========================================================================
# TRANSCRIBED FROM THE CODE THEY RAN, not from the paper's prose.  OCT trains
# with a FORK of OpenRLHF (github.com/maiush/OpenRLHF) that ADDS a
# `--kl_loss_coef` flag upstream does not have; the term is
# openrlhf/trainer/dpo_trainer.py `_get_batch_kl` (lines 386-402) entering the
# loss at line 178:
#
#     loss = preference_loss + aux_loss*aux_loss_coef
#            + nll_loss*nll_loss_coef + kl_loss*kl_loss_coef
#
#     loss_masks = attention_mask.clone().bool()
#     for m, src_len in zip(loss_masks, prompt_id_lens):
#         m[:src_len] = False            # prompt out, completion kept
#     loss_masks = loss_masks[:, 1:]
#     delta = (policy_logps - ref_logps) * loss_masks
#     return (delta**2).sum(-1) / loss_masks.sum(-1)
#     ... kl_loss = kl_vals.mean()
#
# THIS IS NOT THE STANDARD k1 KL ESTIMATOR, and not k3.  It is the MEAN OF THE
# SQUARED per-token log-ratio over completion tokens -- their own log key calls
# it `sq_approx_kl`.  Squaring makes it SYMMETRIC: it penalises the policy for
# moving away from the reference in EITHER direction, where a real KL penalises
# only one.  Writing "a KL penalty" from the paper's description would give a
# DIFFERENT objective from the one they ran, which is the whole reason for
# reading their source.  Reproduced as-is; do not "fix" the square.
#
# Applied over the concatenated [chosen; rejected] batch (their call passes
# `prompt_id_lens + prompt_id_lens`), coefficient 0.001.
def sq_approx_kl(policy_logps, ref_logps, completion_mask):
    """OCT's `_get_batch_kl`: per-sample mean SQUARED per-token log-ratio.

    policy_logps / ref_logps: (batch, seq-1) per-token log-probs of the
    sampled tokens, already shifted for next-token alignment.
    completion_mask: (batch, seq-1), 1 on completion tokens, 0 on prompt and
    padding -- the same set OCT builds by zeroing the first `prompt_id_len`
    positions of the attention mask and then dropping column 0.

    Returns (batch,).  Differs from OCT in one place only: the denominator is
    clamped at 1 so a degenerate empty completion gives 0 instead of NaN.
    """
    m = completion_mask.to(policy_logps.dtype)
    delta = (policy_logps - ref_logps) * m
    return (delta ** 2).sum(-1) / m.sum(-1).clamp(min=1.0)


def kl_dpo_trainer_class():
    """trl.DPOTrainer + OCT's KL term.  Built lazily: this module must import
    without trl so --plan runs on a laptop.

    HOOK: `_compute_loss`.  That is the only method where BOTH the policy and
    the reference per-token log-probs exist at once (trl 1.10.0 lines 1377 and
    1415); `compute_loss` above it sees only the scalar, and the loss_type
    dispatch below it sees only sequence sums.  We do not copy the 250-line
    body -- we call super() with a spy installed on the module-global
    `selective_log_softmax`, which trl calls exactly twice inside
    `_compute_loss`: once on the policy logits, once on the reference logits.
    Capturing those two tensors means NO EXTRA FORWARD PASS: the KL term rides
    on the forwards trl already runs, and costs only a subtract-square-sum.
    The spy count is asserted, so a future trl that calls it a different number
    of times fails loudly rather than penalising the wrong thing.
    """
    import trl
    from trl.trainer import dpo_trainer as _trl_dpo

    class KLDPOTrainer(trl.DPOTrainer):

        def __init__(self, *a, kl_coef=0.0, **kw):
            super().__init__(*a, **kw)
            self.kl_coef = float(kl_coef)
            if self.kl_coef > 0.0:
                # Both of these route around `_compute_loss` or destroy the
                # per-token reference, so the term would silently not apply.
                if getattr(self, "use_liger_kernel", False):
                    raise RuntimeError(
                        "kl_coef>0 with use_liger_kernel=True: the fused path "
                        "calls _compute_loss_liger, which this subclass does "
                        "not hook -- the KL term would be silently dropped.")
                if getattr(self, "precompute_ref_logps", False):
                    raise RuntimeError(
                        "kl_coef>0 with precompute_ref_log_probs=True: only "
                        "SEQUENCE-summed reference logps are cached, and the "
                        "KL term needs PER-TOKEN ones.")

        def _compute_loss(self, model, inputs, return_outputs=False):
            if self.kl_coef <= 0.0:
                return super()._compute_loss(model, inputs, return_outputs)

            import torch
            captured = []
            _orig = _trl_dpo.selective_log_softmax

            def _spy(logits, index):
                out = _orig(logits, index)
                captured.append(out)
                return out

            _trl_dpo.selective_log_softmax = _spy
            try:
                result = super()._compute_loss(model, inputs, return_outputs)
            finally:
                _trl_dpo.selective_log_softmax = _orig

            if len(captured) != 2:
                raise RuntimeError(
                    f"expected exactly 2 selective_log_softmax calls inside "
                    f"trl's _compute_loss (policy, reference), saw "
                    f"{len(captured)} -- trl {trl.__version__} has moved and "
                    f"the KL term cannot be trusted.")
            policy_ptl, ref_ptl = captured          # (batch, seq-1) each,
            ref_ptl = ref_ptl.detach()              # batch = [chosen; rejected]

            # trl already zeroes non-completion positions in place; we mask
            # again so this function's correctness does not depend on that.
            mask = inputs["completion_mask"][..., 1:]
            if policy_ptl.shape != mask.shape:
                raise RuntimeError(
                    f"per-token logps {tuple(policy_ptl.shape)} do not match "
                    f"the shifted completion mask {tuple(mask.shape)}")

            kl = sq_approx_kl(policy_ptl, ref_ptl, mask).mean()

            loss, extra = (result[0], result[1:]) if isinstance(result, tuple) \
                else (result, None)
            loss = loss + kl * self.kl_coef

            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["sq_approx_kl"].append(kl.item())
            self._metrics[mode]["kl_term"].append(kl.item() * self.kl_coef)

            return (loss,) + extra if extra is not None else loss

    return KLDPOTrainer


def attributed_metric_callback_class():
    """A TrainerCallback that re-prints each trl metric dict WITH the trait name.
    Built lazily for the same reason as kl_dpo_trainer_class: no transformers on
    a laptop.

    WHY: the sweep runs ~4 containers at once and Modal merges their stdout into
    ONE stream, and trl's own metric dicts carry neither a trait name nor a
    container id.  Line order is ARRIVAL order across interleaved containers, so
    once the sweep is over there is no way to attribute a `rewards/margins` value
    to the trait that produced it -- the 134-trait run could only be reported as
    a pooled distribution.  That loss is unrecoverable after the fact: nothing in
    the log can be re-derived later, which is why the attribution has to be
    written AT EMIT TIME.  We add a line rather than suppressing trl's, so the
    raw stream stays exactly what trl produced.
    """
    import transformers

    class AttributedMetricLog(transformers.TrainerCallback):

        def __init__(self, trait):
            self.trait = trait

        def on_log(self, args, state, control, logs=None, **kw):
            if not logs:
                return
            # .get() throughout: eval events, the final train summary and future
            # trl versions all log different key sets, and a missing key must not
            # take down a training run whose only purpose is the adapter.
            print(
                f"[metric] trait={self.trait}"
                f" step={state.global_step}"
                f" epoch={logs.get('epoch')}"
                f" loss={logs.get('loss')}"
                f" margins={logs.get('rewards/margins')}"
                f" accuracy={logs.get('rewards/accuracies')}",
                flush=True)

    return AttributedMetricLog


# ===========================================================================
# data
# ===========================================================================
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_pool_sha(path):
    """sha256 of the prompt list alone, in file order -- the phase-1 invariant."""
    h = hashlib.sha256()
    with open(path) as f:
        for line in f:
            if line.strip():
                h.update(json.loads(line)["prompt"].encode("utf-8"))
                h.update(b"\n")
    return h.hexdigest()


def shared_pool_sha(paths):
    """The one prompt-pool sha shared by every corpus file, or refuse to launch.

    What makes cross-trait weight comparison meaningful is that all traits in a
    run trained on the SAME prompts: the adapters then differ by trait and not
    by which questions were asked.  That is a property OF THE RUN, computed from
    the corpus about to be launched against, which is why there is no expected
    literal here.  This replaces a module constant that pinned the expected sha
    of the phase-1 pool, was referenced nowhere despite a comment
    claiming it was asserted, and had drifted wrong anyway -- the 134-run sweep
    trained on a later, legitimately regenerated pool, so the assert the comment
    promised would have failed every run in the sweep.  A check hardcoded to the
    world it was written in is a check people route around; this one keeps
    holding after the corpus is regenerated, because regeneration moves every
    trait's pool together and only a MIXED corpus is the real failure.

    Returns the single sha.  Raises SystemExit naming how many distinct pools
    were found if the files disagree, since launching then would spend GPU money
    on adapters that cannot legitimately be compared.
    """
    pools = {}
    for p in paths:
        pools.setdefault(prompt_pool_sha(p), []).append(os.path.basename(p))
    if len(pools) != 1:
        detail = "\n".join(
            f"    {sha[:16]}...  {len(names)} file(s): "
            f"{', '.join(sorted(names)[:6])}"
            f"{' ...' if len(names) > 6 else ''}"
            for sha, names in sorted(pools.items(), key=lambda kv: -len(kv[1])))
        raise SystemExit(
            f"REFUSING TO LAUNCH: the {len(paths)} corpus files span "
            f"{len(pools)} distinct prompt pools, so the traits would not be "
            "trained on the same prompts and cross-trait comparison of the "
            f"adapters would be meaningless.\n{detail}\n"
            "  Regenerate the pool for all traits together, or launch only "
            "traits that share one.")
    return next(iter(pools))


def corpus_content_sha(paths):
    """One sha over the FULL RECORDS of every file in the corpus, in name order.

    prompt_pool_sha above hashes PROMPTS ONLY, on purpose, and make_nulls.py
    builds data_null_shuffled / data_null_permuted / data_null_seedpaired by
    deliberately holding the prompts fixed and altering only the
    chosen/rejected assignment.  So the real corpus and all three null corpora
    have the SAME prompt-pool sha, and the shared-pool guard -- which is doing
    its own job correctly -- cannot tell a real arm from a null arm.  This sha
    can, because it folds in the bytes of the whole record, which is exactly
    what the nulls change.  It reuses sha256_file rather than opening a third
    hashing path, so this value and runmeta's data_sha256 can never come to
    disagree about what a file contains.

    One expected non-difference: data_null_seedpaired is byte-identical to
    data_common on purpose -- make_nulls.py asserts it is, because it is the
    run-to-run noise floor rather than a null, trained on the same data with
    different seeds.  Those two corpora therefore share this sha too, and that
    is correct, not a hole: what separates them is the seed and the
    corpus_label, and the label is taken from the directory the launcher was
    pointed at rather than from anything a person retyped.
    """
    h = hashlib.sha256()
    for p in sorted(paths, key=os.path.basename):
        h.update(os.path.basename(p).encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(p).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def derive_corpus_label(data_dir):
    """A corpus's name is its directory basename: data_common,
    data_null_shuffled, and so on.

    The label exists so that any adapter can be asked months later "which
    corpus made you?" and answer out of its own runmeta.  That fact currently
    lives only in whoever's shell history ran the launch, which is the failure
    this project keeps repeating: a launch-time fact that was never written
    down anywhere the artifact can carry it.
    """
    return os.path.basename(os.path.normpath(data_dir))


def adapter_outdir(corpus_label, trait):
    """Where a trait's adapter is written, namespaced BY CORPUS.

    This used to be /adapters/<trait> -- keyed by trait name alone -- which
    means two arms that train the same trait names on different corpora write
    to the same path.  We are about to launch null-control arms that do
    exactly that: the same 134 trait names, on deliberately corrupted data.
    Pointed at the sweep volume, such a launch would overwrite
    /adapters/extraverted with a null adapter silently -- no error, no warning
    -- destroying ~$55 of finished, already-analysed GPU work that cannot be
    regenerated without re-spending it.  The only thing standing in the way
    today is remembering to set PC_ADAPTER_VOLUME correctly, and a fact that
    lives only in whoever's shell ran the launch is the failure this project
    keeps repeating.

    Applied uniformly, including to the real corpus.  A special case for
    data_common would be one more thing to remember, and the point of putting
    the corpus in the path is to stop relying on remembering.  The already
    written sweep adapters stay where they are at the volume root; nothing
    here moves them, and gram_on_modal.py reads either layout.
    """
    return f"/adapters/{corpus_label}/{trait}"


# A seed-paired arm is identified by its corpus label, because the label is
# derived from the directory the launcher was pointed at rather than from
# anything a person retyped.  Kept next to the guard so the two cannot drift.
SEEDPAIRED_LABEL_MARKER = "seedpaired"


def is_seedpaired_corpus(corpus_label):
    return SEEDPAIRED_LABEL_MARKER in (corpus_label or "").lower()


def check_seed_treatment(corpus_label, seed, order_seed):
    """Refuse a noise-floor launch whose treatment never arrived.

    `data_null_seedpaired` is BYTE-IDENTICAL to data_common on purpose: it is the
    run-to-run noise floor, and its treatment is the SEED.  Launched on the
    default seeds it is not a control at all, it is a bit-identical rerun of the
    real arm -- and every guard in this file would pass it, correctly and
    uselessly: same prompt pool sha, same corpus content sha, same
    expected_data_sha, same hyperparameters, a clean loss curve, a plausible
    adapter.  The measured noise floor would come out at exactly zero and would
    make every other null control look stronger than it is.

    This is the PC_LORA_ALPHA=16 failure again in a new costume: that leg was set
    on the launch command, never crossed into the container, finished, passed its
    gates, and silently repeated the baseline it was supposed to be compared
    against.  The lesson written down then was that a run which verifies its
    OUTCOME but not its INPUT can be perfectly executed and mean nothing.  So the
    check happens HERE, at the launcher, before a GPU is allocated -- outcomes
    cannot catch this one, because a bit-identical rerun has no anomalous outcome.

    Compares against SEED / ORDER_SEED by NAME.  If someone changes the declared
    default seed later the guard follows it; a check written `== 0` would quietly
    stop guarding anything.  Only the both-at-default case is a bit-identical
    rerun -- changing either seed alone already perturbs the run -- so that is
    exactly what is refused, and nothing legitimate is blocked.
    """
    if not is_seedpaired_corpus(corpus_label):
        return
    if seed == SEED and order_seed == ORDER_SEED:
        raise SystemExit(
            f"REFUSING TO LAUNCH: corpus {corpus_label!r} is a seed-paired "
            f"noise-floor arm, but --seed={seed} and --order-seed={order_seed} "
            f"are still the declared defaults (SEED={SEED}, "
            f"ORDER_SEED={ORDER_SEED}).\n"
            "  For this arm THE SEED IS THE TREATMENT: the corpus is "
            "byte-identical to data_common by construction, so on the default "
            "seeds this launch is a bit-identical rerun of the real arm, not a "
            "control.\n"
            "  Nothing downstream can catch it -- corpus sha, pool sha and "
            "every hyperparameter guard would all pass, the run would finish, "
            "and the noise floor would be reported as exactly zero.\n"
            "  Pass --seed and --order-seed explicitly (e.g. --seed 1 "
            "--order-seed 1).")


def load_pairs(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append({"prompt": d["prompt"],
                         "chosen": d["chosen"],
                         "rejected": d["rejected"]})
    return rows


def to_conversational(rows, tok=None):
    """Pre-rendered prompt/chosen/rejected STRINGS, not conversational dicts.

    Qwen3.5's chat template opens a reasoning block in the generation prompt.
    Rendered the default way:

        prompt-only : ...<|im_start|>assistant\\n<think>\\n
        prompt+reply: ...<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\nOh that...

    The STRING is a prefix but the TOKENS are not (19 vs 21 leading tokens): the
    prompt's trailing newline merges with the block's opening newline into a
    different token. trl warns "Mismatch between tokenized prompt and the start
    of tokenized prompt+chosen" and carries on, so DPO would score a completion
    whose first tokens are `\\n</think>\\n\\n` -- scaffolding identical in the
    chosen and rejected arms, diluting exactly the contrast being trained, on a
    corpus generated with reasoning DISABLED so there is no reasoning to keep.

    `enable_thinking=False` puts the empty block in the PROMPT where it belongs
    and makes the token sequence a true prefix. Verified both ways before this
    was written; the assertion below re-checks it at runtime rather than
    trusting that the template never changes.

    Rendering here also means trl does no templating of its own, so what is
    trained is what this function produced.
    """
    if tok is None:
        return [{"prompt": [{"role": "user", "content": r["prompt"]}],
                 "chosen": [{"role": "assistant", "content": r["chosen"]}],
                 "rejected": [{"role": "assistant", "content": r["rejected"]}]}
                for r in rows]

    def render(r):
        u = [{"role": "user", "content": r["prompt"]}]
        head = tok.apply_chat_template(u, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
        out = {"prompt": head}
        for side in ("chosen", "rejected"):
            full = tok.apply_chat_template(
                u + [{"role": "assistant", "content": r[side]}],
                tokenize=False, enable_thinking=False)
            if not full.startswith(head):
                raise RuntimeError(
                    "chat template no longer renders the prompt as a prefix of "
                    "prompt+completion; DPO would score the wrong token span")
            out[side] = full[len(head):]
        return out

    rendered = [render(r) for r in rows]

    # Prove the token-level prefix property on the first row rather than assuming
    # it follows from the string-level one -- it does not, which is the whole
    # reason this function exists.
    a = tok(rendered[0]["prompt"], add_special_tokens=False)["input_ids"]
    b = tok(rendered[0]["prompt"] + rendered[0]["chosen"],
            add_special_tokens=False)["input_ids"]
    if b[:len(a)] != a:
        raise RuntimeError(
            f"tokenised prompt is not a prefix of prompt+chosen "
            f"({len(a)} vs {len(b)} tokens); DPO would score a misaligned span")
    return rendered


# ===========================================================================
# cost
# ===========================================================================
def cost_estimate(n_runs, minutes_per_run, gpu=GPU_TYPE):
    price = GPU_PRICE_PER_HOUR.get(gpu)
    if price is None:
        raise SystemExit(f"no price known for gpu={gpu}; refusing to estimate")
    gpu_min = n_runs * minutes_per_run
    dollars = gpu_min / 60.0 * price
    return gpu_min, dollars, price


def print_plan(traits, minutes_per_run=25.0, budget=None, data_dir=None):
    # The budget is the CALLER's, not a constant. It was hardcoded to phase 2's
    # $6 and correctly stopped a phase-5 launch -- for the wrong reason, since
    # phase 5 has its own budget. A guard pinned to the phase it was written in
    # rejects every later phase, which trains people to bypass it.
    if budget is None:
        # NO PHASE'S NUMBER AS THE DEFAULT. This read `6.0` -- phase 2's budget --
        # and every later phase inherited it, so the phase-3 null launch was
        # stopped as "OVER BUDGET by $81.50" against a phase that had already
        # closed. That is the same defect as APP_NAME billing everything to phase
        # 2 and gate 3 asserting rsLoRA after we switched: a constant that was
        # true when written and silently false afterwards.
        #
        # There is no correct default, so there is no default: an unset budget is
        # a caller who has not said, and the honest response is to refuse rather
        # than to invent a ceiling and enforce it as though it were chosen.
        env = os.environ.get("PC_PHASE_BUDGET")
        if env is None:
            raise SystemExit(
                "NO BUDGET DECLARED. Set PC_PHASE_BUDGET to this phase's ceiling "
                "before launching. Refusing to substitute a default: the previous "
                "default was one phase's figure and it outlived that phase, which "
                "is how a guard ends up enforcing the wrong number confidently.")
        budget = float(env)
    # Likewise the corpus: the plan must count the pairs in the corpus this
    # launch is actually pointed at, not the default one.
    data_dir = DATA_DIR if data_dir is None else data_dir
    n = len(traits)
    pairs = 0
    for t in traits:
        p = os.path.join(data_dir, f"{t}.jsonl")
        pairs += sum(1 for line in open(p) if line.strip())
    steps = max(1, (pairs // n) // EFFECTIVE_BATCH)
    gpu_min, dollars, price = cost_estimate(n, minutes_per_run)
    print("=" * 72)
    print("PHASE 2 WORK PLAN -- printed BEFORE any GPU is allocated")
    print("=" * 72)
    print(f"  corpus                       : {derive_corpus_label(data_dir)}"
          f"   ({data_dir})")
    # WHERE THE ADAPTERS WILL LAND, printed before the spend rather than
    # discovered after it.  Volume plus corpus-namespaced path together are
    # what decide whether this launch can collide with finished work, and both
    # were previously invisible at launch time -- the volume lived in an env
    # var and the path was assembled out of sight.  Read this line and the
    # corpus line above together before approving the money.
    print(f"  output volume                : {ADAPTER_VOLUME}")
    print(f"  output path per trait        : "
          f"{adapter_outdir(derive_corpus_label(data_dir), '<trait>')}")
    # THE ADAPTER CONFIGURATION, PRINTED AS THE NUMBER THAT ACTUALLY ACTS.
    # The banner showed the corpus and the output path and never said which
    # adapter it would train, so 240 null-control runs went out at effective
    # scale 16.0 against the sweep's 2.0 and nothing on screen contradicted
    # them. rsLoRA-vs-LoRA is not the readable fact -- the two are the SAME
    # transform at the right alpha and differ eightfold at the wrong one -- so
    # what gets printed is the EFFECTIVE SCALE, which is what multiplies B@A
    # and is directly comparable run to run. OCT's is 2.0.
    _scale = (LORA_ALPHA / math.sqrt(LORA_R)) if USE_RSLORA else (LORA_ALPHA / LORA_R)
    print(f"  adapter                      : r={LORA_R} alpha={LORA_ALPHA} "
          f"{'rsLoRA' if USE_RSLORA else 'plain LoRA'}  ->  EFFECTIVE SCALE "
          f"{_scale:g}" + ("" if abs(_scale - 2.0) < 1e-9 else
                           f"   <-- NOT 2.0, the phase-5 sweep's scale"))
    print(f"  work items (training runs)   : {n}   {traits}")
    print(f"  pairs per trait              : {pairs // n}")
    print(f"  effective batch              : {EFFECTIVE_BATCH} "
          f"({PER_DEVICE_BATCH} x {GRAD_ACCUM} accum)")
    print(f"  optimizer steps per run      : ~{steps}  (1 epoch)")
    print(f"  gpu                          : {GPU_TYPE} @ ${price:.2f}/hr")
    print(f"  assumed wall-clock per run   : {minutes_per_run:.0f} min "
          f"(load+warm ~6, {steps} steps of 32x{MAX_LENGTH} tok "
          f"fwd+bwd on policy AND adapter-disabled reference)")
    print(f"  ARITHMETIC                   : {n} runs x {minutes_per_run:.0f} min "
          f"= {gpu_min:.0f} GPU-min = {gpu_min/60:.2f} GPU-hr")
    print(f"                                 {gpu_min/60:.2f} hr x ${price:.2f}"
          f"/hr = ${dollars:.2f}")
    # Name the SOURCE, not a phase. A label reading "phase 2 budget" beside a
    # phase-3 launch is a check telling you confidently about the wrong thing.
    print(f"  declared budget              : ${budget:.2f}  (PC_PHASE_BUDGET)")
    if dollars > budget:
        print(f"  VERDICT: OVER BUDGET by ${dollars - budget:.2f} -- STOP")
        return False, dollars
    print(f"  VERDICT: within budget, ${budget - dollars:.2f} headroom")
    return True, dollars


# ===========================================================================
# Modal
# ===========================================================================
def _download_base_model():
    """Bake the base weights into the image.

    MUST be at module scope: Modal refuses to import a function defined inside
    another function unless serialized=True, and the failure is not loud -- the
    whole app build raises, the caller catches it, and the module carries on in
    "local modes only" while looking healthy. That is how this was nearly run
    with no Modal backend at all.
    """
    from huggingface_hub import snapshot_download
    snapshot_download(BASE_MODEL,
                      ignore_patterns=["*.pt", "*.bin", "*.gguf", "*.pth"])


def _build_modal():
    # BEFORE `import modal`: a config error should not depend on the SDK being
    # importable, and this way the message is the same in every environment.
    if not APP_NAME:
        raise SystemExit(
            "REFUSING TO LAUNCH: PC_APP_NAME is unset.\n"
            "  Modal attributes BILLING by app name, so this is the only thing that\n"
            "  decides which phase pays for this run. There is no correct default --\n"
            "  the previous one ('pc-qwen35-phase2') silently billed the 134-trait\n"
            "  main sweep to phase 2, and no pc-qwen35-phase5 app exists as a result.\n"
            "  Nothing downstream can catch this: the run succeeds, the ledger records\n"
            "  the phase you meant, the billing records the name you sent, and neither\n"
            "  side can see the other.\n"
            "  Pass it explicitly, naming the phase you are actually running, e.g.\n"
            "    PC_APP_NAME=pc-qwen35-phase5-sweep modal run train_qwen35.py ...")

    import modal

    app = modal.App(APP_NAME)
    adapter_vol = modal.Volume.from_name(ADAPTER_VOLUME, create_if_missing=True)
    data_vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)

    # Versions are pinned to EXACTLY the local .venv_pre stack, so that the
    # gates that run locally (shape checks, Gram) read artefacts produced by the
    # same peft/safetensors serialisation.  transformers 5.15.1 is the floor:
    # `qwen3_5`/`qwen3_5_text` do not exist below transformers 5.x and an older
    # pin raises KeyError on the config rather than training the wrong thing.
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.13.0",
            "transformers==5.15.1",
            "trl==1.10.0",
            "peft==0.20.0",
            "accelerate==1.14.0",
            "datasets==5.0.1",
            "safetensors",
            "hf_transfer",
            "numpy<3",
        )
        .env({
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # Without this the container silently falls back to the default and
            # trains the WRONG base into a directory named for the right one.
            "PC_BASE_MODEL": BASE_MODEL,
            # The container imports this same file, so it hits the PC_APP_NAME
            # guard too -- and only the vars named here cross into the image.
            # Without it the guard fires during the image build itself.
            "PC_APP_NAME": APP_NAME or "",
        })
        .add_local_file(os.path.abspath(__file__), "/root/train_qwen35.py",
                        copy=True)
        .run_function(_download_base_model)
    )
    return modal, app, image, adapter_vol, data_vol


if os.environ.get("PC_NO_MODAL") != "1":
    try:
        modal, app, image, adapter_vol, data_vol = _build_modal()
    except Exception as _e:          # local selftest without modal installed
        modal = None
        print(f"[warn] modal unavailable ({_e}); local modes only",
              file=sys.stderr)
else:
    modal = None


def _pkg_version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception as e:
        return f"UNKNOWN:{type(e).__name__}"


def _train_impl(job):
    """The whole training run.  Runs in the container; importable for testing."""
    import math
    import random
    import numpy as np
    import torch
    import transformers
    import trl
    import peft
    from datasets import Dataset
    from transformers import AutoConfig, AutoTokenizer
    from peft import LoraConfig

    trait = job["trait"]
    outdir = job["outdir"]

    # BELT AND BRACES, checked in the container before ANYTHING is written.
    # Namespacing the output path by corpus (adapter_outdir) stops the ordinary
    # collision, but namespacing is only as good as the label it is built from:
    # a launch that declares the wrong corpus namespaces itself neatly into
    # another arm's directory and overwrites it just as silently.  The check is
    # cheap -- one small JSON read per trait, no GPU time -- and worth it
    # because the loss is unrecoverable.  Adapters cannot be regenerated
    # without re-spending the GPU, and the 134-trait sweep this would land on
    # is ~$55 of already-analysed work.  Refusing costs a relaunch; not
    # refusing costs the sweep, with no error and no warning to notice.
    # A rerun of the SAME corpus is allowed through: overwriting an arm with
    # itself is a rerun, not a loss.
    prior_meta_path = os.path.join(outdir, "runmeta.json")
    if os.path.isdir(outdir) and os.path.exists(prior_meta_path):
        try:
            with open(prior_meta_path) as f:
                prior_label = (json.load(f) or {}).get("corpus_label")
        except (ValueError, OSError):
            # An unreadable runmeta is not evidence of a conflict, and refusing
            # on it would make the guard a nuisance that gets removed.
            prior_label = None
        this_label = job.get("corpus_label")
        if prior_label and this_label and prior_label != this_label:
            raise RuntimeError(
                f"REFUSING TO OVERWRITE {outdir}: it already holds an adapter "
                f"trained on corpus {prior_label!r}, and this job is corpus "
                f"{this_label!r}.  Either the volume (PC_ADAPTER_VOLUME) or "
                "the corpus label is wrong.  Those adapters cost real GPU "
                "money and cannot be regenerated without spending it again, "
                "so this refuses rather than writes.")

    os.makedirs(outdir, exist_ok=True)

    stack = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(f"[stack] {stack}", flush=True)

    # ---- gate-4 precondition, checked HERE and recorded ------------------
    cfg = AutoConfig.from_pretrained(BASE_MODEL)
    if getattr(cfg, "model_type", None) != "qwen3_5":
        raise RuntimeError(
            f"config model_type={getattr(cfg,'model_type',None)!r}, expected "
            "'qwen3_5' -- wrong repo or transformers too old.  BLOCKER.")
    text_cfg = cfg.text_config
    exp = expected_counts(text_cfg)
    print(f"[config] {exp}", flush=True)

    # ---- data -----------------------------------------------------------
    data_path = job["data_path"]
    data_sha = sha256_file(data_path)
    pool_sha = prompt_pool_sha(data_path)
    # The shared-pool precondition, enforced on the bytes the container actually
    # opened rather than on the ones the driver believed it uploaded.  The
    # driver computed one sha across every trait it is launching and put it in
    # each job dict; disagreement here means this trait's file is not from that
    # pool, and its adapter could not be compared against its siblings'.  The
    # value rides in the JOB DICT for the same reason lora_alpha does -- an env
    # var set at launch never crossed into the container and the mismatch went
    # unnoticed for a whole comparison run.  A job with no expected sha at all
    # bypassed the driver, so it fails here too instead of training unchecked.
    expected_pool_sha = job.get("expected_pool_sha")
    if not expected_pool_sha:
        raise RuntimeError(
            f"job for trait {trait!r} carries no expected_pool_sha -- it did "
            "not come from the launcher, which is the thing that establishes "
            "all traits share one prompt pool.  BLOCKER.")
    if pool_sha != expected_pool_sha:
        raise RuntimeError(
            f"prompt pool mismatch for trait {trait!r}: container computed "
            f"{pool_sha} from {data_path}, driver handed down "
            f"{expected_pool_sha}.  This trait did not train on the same "
            "prompts as its siblings, so the adapters would not be "
            "comparable.  BLOCKER.")
    # The check the pool sha structurally CANNOT do.  make_nulls.py holds the
    # prompts fixed and permutes only chosen/rejected, so every data_null_*
    # file has the same prompt-pool sha as its data_common twin and passes the
    # test above no matter which corpus was actually shipped.  The property
    # enforced here is INTENT MATCHES ARRIVAL, not "this is the real corpus" --
    # null arms are SUPPOSED to train on null data, and a guard demanding the
    # real corpus would be wrong on half the runs and would get switched off.
    # The driver hashed the full records of the file it MEANT to send; if these
    # bytes are different bytes, a --data-dir typo has crossed a real arm with a
    # null one, and the resulting adapter would look completely normal.  That
    # error surfaces months later as an inexplicable result, or never.
    expected_data_sha = job.get("expected_data_sha")
    corpus_label = job.get("corpus_label")
    if not expected_data_sha or not corpus_label:
        raise RuntimeError(
            f"job for trait {trait!r} carries no expected_data_sha/"
            "corpus_label -- it did not come from the launcher, which is the "
            "only place that knows WHICH corpus this run was pointed at.  "
            "BLOCKER.")
    if data_sha != expected_data_sha:
        raise RuntimeError(
            f"corpus content mismatch for trait {trait!r}: container computed "
            f"{data_sha} from {data_path}, driver declared corpus "
            f"{corpus_label!r} with {expected_data_sha}.  The prompt pool "
            "matched, so this is a real-vs-null corpus swap or a stale upload "
            "-- the two are indistinguishable by prompts alone.  BLOCKER.")
    rows = load_pairs(data_path)
    # The seeds arrive IN THE JOB, like every other hyperparameter, and are read
    # once here so the LoRA init, the training args and runmeta all report the
    # same two numbers.  For the seed-paired noise-floor arm these ARE the
    # treatment, and PC_LORA_ALPHA taught this project that a treatment which
    # travels by environment variable can fail to arrive without anything
    # noticing.  The defaults are the module-level declarations, so a job dict
    # from an older launcher still trains exactly as it did before.
    seed = int(job.get("seed", SEED))
    order_seed = int(job.get("order_seed", ORDER_SEED))
    print(f"[seed] seed={seed} order_seed={order_seed} "
          f"(defaults SEED={SEED} ORDER_SEED={ORDER_SEED})", flush=True)
    rng = random.Random(order_seed)
    rng.shuffle(rows)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Render with the tokenizer so the reasoning-block boundary is handled here
    # and trl does no templating of its own -- see to_conversational's docstring.
    ds = Dataset.from_list(to_conversational(rows, tok))
    print(f"[data] {trait}: {len(rows)} pairs  corpus={corpus_label} "
          f"sha256={data_sha[:16]}... pool_sha={pool_sha[:16]}...  "
          "prerendered, thinking disabled", flush=True)

    # ---- model ----------------------------------------------------------
    model = _load_policy(BASE_MODEL, torch.bfloat16)
    text_prefix = text_tower_prefix(model, text_cfg.model_type)
    disc = discover_targets(model, text_cfg, text_prefix)
    print(f"[target] text_prefix={disc['text_prefix']}", flush=True)
    print(f"[target] total Linear={disc['n_total_linear']}  "
          f"targeted={disc['n_targets']}  "
          f"excluded_vision={disc['n_excluded_vision']} "
          f"under prefixes {disc['excluded_prefixes']}  "
          f"never_target={disc['n_never_target']} {disc['never_target']}",
          flush=True)
    print(f"[target] by layer type: {disc['by_layer_type']}", flush=True)
    print(f"[target] by module kind: {disc['by_module_kind']}", flush=True)

    if disc["by_layer_type"].get("linear_attention", 0) == 0:
        raise RuntimeError("ZERO linear-attention modules targeted -- this is "
                           "exactly the silent failure phase 2 exists to catch")
    # "Zero excluded" is only a fault if a vision tower is PRESENT in this load.
    # Loaded as a plain causal LM the checkpoint has no vision modules at all,
    # and demanding exclusions would fail the correct load -- the same shape as
    # a guard pinned to today's count failing tomorrow's legitimate correction.
    # So ask whether a vision tower exists, and require the exclusion only then.
    has_vision = any(
        getattr(getattr(mod, "config", None), "model_type", "").startswith("qwen3_5_vi")
        or n.split(".")[-1] in ("visual", "vision_tower", "vision_model")
        for n, mod in model.named_modules())
    if has_vision and disc["n_excluded_vision"] == 0:
        raise RuntimeError("a vision tower is present but ZERO vision modules "
                           "were excluded -- the prefix split failed")
    print(f"[targets] vision tower present={has_vision}  "
          f"excluded={disc['n_excluded_vision']}", flush=True)

    # ---- LoRA, seeded immediately before construction --------------------
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    lora = LoraConfig(
        r=job.get("lora_r", LORA_R),
        lora_alpha=job.get("lora_alpha", LORA_ALPHA),
        lora_dropout=LORA_DROPOUT,
        use_rslora=job.get("use_rslora", USE_RSLORA),
        bias="none", task_type="CAUSAL_LM",
        target_modules=disc["targets"],      # explicit list from DISCOVERY
    )

    steps_per_epoch = max(1, len(rows) // EFFECTIVE_BATCH)
    total_steps = steps_per_epoch * EPOCHS
    warmup = max(1, round(WARMUP_RATIO * total_steps))

    args = trl.DPOConfig(
        output_dir=outdir,
        beta=job.get("beta", BETA),
        loss_type=job.get("loss_type", LOSS_TYPE),
        loss_weights=job.get("loss_weights", LOSS_WEIGHTS),
        max_length=MAX_LENGTH,
        learning_rate=job.get("lr", LR),
        lr_scheduler_type="cosine",
        warmup_steps=warmup,               # trl 1.10 has no warmup_ratio
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_grad_norm=MAX_GRAD_NORM,
        adam_beta1=ADAM_BETA1, adam_beta2=ADAM_BETA2,
        optim="adamw_torch",
        bf16=True,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=max(1, total_steps // 3),   # intermediate checkpoints
        save_total_limit=None,
        report_to=job.get("report_to", "none"),
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
        precompute_ref_log_probs=False,
        gradient_checkpointing=True,
        dataloader_drop_last=True,
    )

    # One trainer class either way: at kl_coef=0.0 the subclass delegates
    # straight to trl's _compute_loss, so the KL-off path is trl's own.
    kl_coef = float(job.get("kl_coef", KL_COEF))
    trainer = kl_dpo_trainer_class()(
        model=model, ref_model=None, args=args,
        train_dataset=ds, processing_class=tok, peft_config=lora,
        kl_coef=kl_coef,
    )
    print(f"[kl] OCT sq_approx_kl coefficient = {kl_coef} "
          f"({'APPLIED' if kl_coef > 0 else 'off'})", flush=True)

    # Added after construction so the trainer's own call site stays untouched;
    # every metric event now also leaves a trait-tagged line in the merged log.
    trainer.add_callback(attributed_metric_callback_class()(trait))

    # the adapter-disabled-base reference path, asserted not assumed
    assert trainer.ref_model is None, "ref_model is not None: reference is a COPY"
    assert getattr(trainer, "is_peft_model", True), "peft path not taken"

    n_lora_pairs = len([n for n, _ in trainer.model.named_parameters()
                        if "lora_A" in n])
    print(f"[lora] lora_A tensors created: {n_lora_pairs} "
          f"(targeted {disc['n_targets']})", flush=True)

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    trainer.model.save_pretrained(outdir)
    tok.save_pretrained(outdir)

    log = [h for h in trainer.state.log_history if "loss" in h]
    meta = {
        "trait": trait,
        "resolved_base_model": BASE_MODEL,
        "base_model_commit": getattr(cfg, "_commit_hash", None),
        "config_model_type": cfg.model_type,
        "text_model_type": text_cfg.model_type,
        "config_derived_dims": exp,
        "text_prefix": disc["text_prefix"],
        "n_total_linear": disc["n_total_linear"],
        "n_targeted": disc["n_targets"],
        "n_excluded_vision": disc["n_excluded_vision"],
        "excluded_prefixes": disc["excluded_prefixes"],
        "n_never_target": disc["n_never_target"],
        "never_target": disc["never_target"],
        "targeted_by_layer_type": disc["by_layer_type"],
        "targeted_by_module_kind": disc["by_module_kind"],
        # The container's OWN stack, so the gate that checks for a train/read
        # version mismatch has something to compare against. Without this it
        # compared local versions to nothing and reported PASS.
        "versions": {p: _pkg_version(p) for p in
                     ("torch", "transformers", "trl", "peft", "accelerate")},
        "lora_r": job.get("lora_r", LORA_R),
        "lora_alpha": job.get("lora_alpha", LORA_ALPHA),
        "use_rslora": job.get("use_rslora", USE_RSLORA),
        "expected_scaling": rslora_scale(job.get("lora_alpha", LORA_ALPHA),
                                         job.get("lora_r", LORA_R),
                                         job.get("use_rslora", USE_RSLORA)),
        "learning_rate": job.get("lr", LR),
        "beta": job.get("beta", BETA),
        "loss_type": job.get("loss_type", LOSS_TYPE),
        "loss_weights": job.get("loss_weights", LOSS_WEIGHTS),
        # The gate reads THIS to confirm the treatment reached the container,
        # rather than trusting the launch command.
        "kl_coef": kl_coef,
        "kl_applied": kl_coef > 0.0,
        "kl_form": ("OCT sq_approx_kl: mean over completion tokens of the "
                    "SQUARED per-token log-ratio (policy-ref), per sample, "
                    "then batch mean over [chosen; rejected]; transcribed from "
                    "maiush/OpenRLHF openrlhf/trainer/dpo_trainer.py "
                    "_get_batch_kl.  NOT the k1 or k3 estimator."),
        "epochs": EPOCHS, "effective_batch": EFFECTIVE_BATCH,
        "max_length": MAX_LENGTH, "warmup_steps": warmup,
        "optimizer": "adamw_torch", "adam_betas": [ADAM_BETA1, ADAM_BETA2],
        "max_grad_norm": MAX_GRAD_NORM,
        "optimizer_steps": total_steps,
        # The seeds ACTUALLY USED, not the module defaults, so an adapter can be
        # asked months later whether it really was seeded apart from its twin.
        # For the seed-paired noise-floor arm this is the record that the
        # treatment arrived -- the same question PC_LORA_ALPHA's runmeta answered
        # after the fact, which is how that silent baseline-repeat was caught.
        "seed": seed, "order_seed": order_seed,
        "seed_is_default": (seed == SEED and order_seed == ORDER_SEED),
        "corpus_is_seedpaired": is_seedpaired_corpus(corpus_label),
        "n_pairs": len(rows),
        "data_sha256": data_sha,
        "prompt_pool_sha256": pool_sha,
        # WHICH CORPUS made this adapter, written into the adapter itself.
        # data_common and every data_null_* share prompt_pool_sha256 by
        # construction, so that field can never answer the question; these two
        # can, and they answer it without anyone having to recover the launch
        # command.  corpus_content_sha256 fingerprints the whole corpus this
        # run belonged to, so the label can also be checked rather than
        # believed if it was ever declared wrong.
        "corpus_label": corpus_label,
        "corpus_content_sha256": job.get("corpus_content_sha"),
        "aux_losses_applied": {
            # NLL-on-chosen rides in loss_type/loss_weights ("sft" entry);
            # the KL is this file's DPOTrainer subclass.
            "nll_on_chosen": dict(zip(job.get("loss_type", LOSS_TYPE),
                                      job.get("loss_weights", LOSS_WEIGHTS)
                                      )).get("sft", 0.0),
            "sq_approx_kl": kl_coef,
        },
        "aux_losses_note": (
            "OCT's NLL-on-chosen is trl's loss_type='sft' entry with its"
            " loss_weight; OCT's per-token KL has no trl switch and is"
            " implemented here as a DPOTrainer subclass (see kl_form)."
            " A zero value means that term was NOT applied in this run."),
        "stack": stack,
        "trl_ran_against_transformers": True,
        "train_seconds": train_secs,
        "log_history": trainer.state.log_history,
        "loss_first": log[0]["loss"] if log else None,
        "loss_last": log[-1]["loss"] if log else None,
        # The DIRECTLY LOGGED reward margin, per container so it is trait-aligned.
        # Log lines from four concurrent containers arrive interleaved, so a
        # `tail -4` of rewards/margins is four margins in ARRIVAL order, not trait
        # order -- and gate 5 needs the trait-aligned one. Recording it here is
        # what makes the comparison possible at all.
        "reward_margin": log[-1].get("rewards/margins") if log else None,
        "reward_accuracy": log[-1].get("rewards/accuracies") if log else None,
    }
    with open(os.path.join(outdir, "runmeta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[done] {trait} in {train_secs:.0f}s  "
          f"loss {meta['loss_first']} -> {meta['loss_last']}", flush=True)
    return {k: v for k, v in meta.items() if k != "log_history"}


def _load_policy(model_id, dtype):
    """Load Qwen3.5-4B.  It is a ConditionalGeneration (VLM) checkpoint, so the
    plain AutoModelForCausalLM route may not be registered; try it, then fall
    back to the architecture the config actually names."""
    import torch
    import transformers
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    kw = dict(dtype=dtype, attn_implementation="sdpa")
    try:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_id, **kw)
    except Exception as e:
        print(f"[load] AutoModelForCausalLM failed ({type(e).__name__}: {e}); "
              f"falling back to {cfg.architectures}", flush=True)
    arch = cfg.architectures[0]
    klass = getattr(transformers, arch)
    return klass.from_pretrained(model_id, **kw)


if modal is not None:
    @app.function(image=image, gpu=GPU_TYPE,
                  volumes={"/adapters": adapter_vol},
                  timeout=60 * 120,
                  secrets=[modal.Secret.from_name("pc-qwen35-secrets")]
                  if os.environ.get("PC_USE_NAMED_SECRET") else [])
    def train_trait(job: dict) -> dict:
        sys.path.insert(0, "/root")
        return _train_impl(job)

    @app.local_entrypoint()
    def main(traits: str = "", minutes_per_run: float = 25.0,
             dry_run: bool = False, data_dir: str = "",
             corpus_label: str = "",
             # Defaults ARE the declared constants, so `modal run` with no seed
             # flags trains bit-for-bit what it trained yesterday, and the guard
             # below can tell "chose the default" from "never chose".
             seed: int = SEED, order_seed: int = ORDER_SEED):
        names = [t.strip() for t in traits.split(",") if t.strip()] or TEST_TRAITS
        data_dir = data_dir or DATA_DIR
        # INTENT MATCHES ARRIVAL, checked at the launcher this time.  The label
        # is what the launch DECLARES it is training on; the directory is what
        # it will actually read.  Making the launcher say the corpus out loud
        # is the whole point -- we are about to launch null-control arms, and
        # that is exactly the moment a --data-dir typo trains a "real" arm on
        # null data or a null arm on real data.  Both produce adapters that
        # look completely normal, because the nulls share their prompts with
        # the real corpus by construction.  If the label is omitted it is
        # derived from the directory rather than being left blank, so every
        # adapter is labelled even on a launch nobody thought about.
        declared = corpus_label
        corpus_label = derive_corpus_label(data_dir)
        if declared and declared != corpus_label:
            raise SystemExit(
                f"REFUSING TO LAUNCH: --corpus-label={declared!r} but "
                f"--data-dir={data_dir} is corpus {corpus_label!r}.  One of "
                "the two is a typo.  Nothing downstream could catch this: the "
                "null corpora hold the prompts fixed and permute only "
                "chosen/rejected, so a swapped corpus trains cleanly and the "
                "mistake surfaces only as an inexplicable result later, if "
                "ever.")
        # Checked before the plan is even priced: for a seed-paired arm the
        # seeds are the treatment, and a launch that never received them buys a
        # bit-identical rerun of the real arm at full GPU price.  See
        # check_seed_treatment -- no downstream gate can catch this one.
        check_seed_treatment(corpus_label, seed, order_seed)
        budget_declared = os.environ.get("PC_PHASE_BUDGET", "UNSET")
        ok, dollars = print_plan(names, minutes_per_run, data_dir=data_dir)
        if not ok:
            raise SystemExit(f"estimate ${dollars:.2f} exceeds the declared "
                             f"PC_PHASE_BUDGET of ${budget_declared} -- STOPPED")
        if dry_run:
            print("dry_run: not launching")
            return
        jobs = []
        paths = []
        for t in names:
            p = os.path.join(data_dir, f"{t}.jsonl")
            if not os.path.exists(p):
                raise SystemExit(f"missing data file {p}")
            paths.append(p)
        # Computed ONCE, here, from the corpus this launch is about to ship, and
        # then handed to every container to check its own copy against.  Doing
        # it once is what makes it a cross-trait invariant: each container can
        # only see its own file, so no container could ever discover on its own
        # that the traits disagree.  This refuses the launch if they do.
        pool_sha = shared_pool_sha(paths)
        # The corpus identity, printed as loudly as the budget, because it is
        # the one launch-time fact that has been living in shell history.
        content_sha = corpus_content_sha(paths)
        file_sha = {t: sha256_file(p) for t, p in zip(names, paths)}
        print("=" * 72)
        print(f"  CORPUS                       : {corpus_label}")
        print(f"  directory                    : {data_dir}")
        print(f"  content sha256 (full records): {content_sha[:16]}...")
        print(f"  prompt pool sha256           : {pool_sha[:16]}...")
        # Printed next to the corpus because for the seed-paired arm the two
        # together ARE the experimental condition: same bytes, different seed.
        print(f"  seed / order_seed            : {seed} / {order_seed}"
              f"{'   (DEFAULTS)' if (seed == SEED and order_seed == ORDER_SEED) else '   (explicit)'}")
        print("  NOTE: data_common and every data_null_* share the pool sha by "
              "construction;")
        print("        only the content sha above distinguishes them.")
        print("=" * 72, flush=True)
        for t in names:
            # Hyperparameters travel IN THE JOB, not through the environment.
            # PC_LORA_ALPHA=16 set on the launch command did not reach the
            # container -- only the vars named in the image's .env() cross that
            # boundary -- so a "comparison" run silently repeated the baseline
            # config and produced near-identical losses. Caught by reading
            # lora_alpha back out of the written runmeta rather than trusting
            # the command line. The job dict crosses reliably; use it.
            jobs.append({"trait": t,
                         # Namespaced by corpus so a null arm cannot land on
                         # top of the real sweep's adapters.  See
                         # adapter_outdir; the same helper prices the plan, so
                         # the path printed before the spend is the path that
                         # is written.
                         "outdir": adapter_outdir(corpus_label, t),
                         # NAMESPACED FOR THE SAME REASON outdir IS, and this
                         # half was missed the first time. Adapter output was
                         # namespaced by corpus while the DATA UPLOAD kept a
                         # shared /_data/<trait>.jsonl path, so three null arms
                         # launched together raced on one directory: each arm
                         # overwrote the others' uploads, and a container read
                         # whichever bytes happened to be there when it started.
                         # Every run in two arms died on the content-sha guard.
                         # That guard is the only reason this is a delay rather
                         # than 200 adapters trained on scrambled corpora and a
                         # null result that looked clean.
                         "data_path": f"/adapters/_data/{corpus_label}/{t}.jsonl",
                         # Seeds ride in the job for the same reason the rest do.
                         "seed": seed, "order_seed": order_seed,
                         "expected_pool_sha": pool_sha,
                         # Corpus identity rides in the job dict for the same
                         # reason the hyperparameters do: it is the only thing
                         # that reliably crosses into the container, and it is
                         # what lets the adapter describe its own corpus.
                         "corpus_label": corpus_label,
                         "corpus_content_sha": content_sha,
                         "expected_data_sha": file_sha[t],
                         "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
                         "beta": BETA, "lr": LR,
                         "loss_type": LOSS_TYPE, "loss_weights": LOSS_WEIGHTS,
                         "use_rslora": USE_RSLORA,
                         "kl_coef": KL_COEF})
        # ship the data through the adapter volume so the container reads the
        # SAME bytes whose sha256 goes into runmeta
        with adapter_vol.batch_upload(force=True) as up:
            for t in names:
                up.put_file(os.path.join(data_dir, f"{t}.jsonl"),
                            f"/_data/{corpus_label}/{t}.jsonl")
        results = list(train_trait.map(jobs))
        # Also namespaced, and found while fixing the data-path race: two
        # launchers running concurrently both wrote this one file, so the
        # second to finish erased the first's record. Harmless here only
        # because the analysis reads runmeta off the volume rather than this
        # file -- which is luck, not design, and luck is not a reason to leave
        # a concurrent overwrite in place.
        os.makedirs(os.path.join(HERE, "phase2_runs"), exist_ok=True)
        with open(os.path.join(HERE, "phase2_runs",
                               f"results_{corpus_label}.json"), "w") as f:
            json.dump(results, f, indent=1)
        # The corpus is repeated here, per trait, from what the CONTAINER wrote
        # into runmeta -- not from the local variable -- so the summary reports
        # the corpus that actually trained rather than the one we intended.
        for r in results:
            print(f"  {r['trait']}: corpus={r.get('corpus_label')} "
                  f"targeted={r['n_targeted']} "
                  f"excl={r['n_excluded_vision']} "
                  f"loss {r['loss_first']} -> {r['loss_last']}")


# ===========================================================================
# local selftest: build a TINY random Qwen3.5 and run the real code path
# ===========================================================================
def kl_masking_test():
    """The KL term must see COMPLETION TOKENS ONLY.

    Pure-tensor case: policy and reference disagree violently on the prompt and
    agree exactly on the completion -> the term must be EXACTLY zero.  A KL that
    counts prompt positions would be penalising the model for log-probs it is
    not being trained to move, so this is the failure the assert exists for.
    """
    import torch
    # 6 shifted positions; positions 0-2 are prompt, 3-5 are completion.
    mask = torch.tensor([[0, 0, 0, 1, 1, 1],
                         [0, 0, 1, 1, 1, 0]], dtype=torch.long)   # row 2: +pad
    ref = torch.zeros(2, 6)
    pol = ref.clone()
    pol[:, :3] += torch.tensor([[-5.0, 3.0, 11.0], [7.0, -2.0, 0.0]])
    pol[1, 5] = 99.0                       # padding position, also excluded
    k = sq_approx_kl(pol, ref, mask)
    assert torch.allclose(k, torch.zeros(2)), \
        f"SELFTEST FAIL: prompt/padding positions leak into the KL: {k}"

    # And it is not identically zero: perturb COMPLETION positions only.
    pol2 = pol.clone()
    pol2[0, 3] = 2.0                       # one completion token, delta 2
    pol2[1, 2] = -1.0                      # one completion token, delta -1
    k2 = sq_approx_kl(pol2, ref, mask)
    want = torch.tensor([4.0 / 3.0, 1.0 / 3.0])   # (delta^2).sum / n_completion
    assert torch.allclose(k2, want), f"SELFTEST FAIL: kl={k2} want={want}"
    print(f"  kl masking: prompt-only difference -> {k.tolist()} (zero); "
          f"completion difference -> {k2.tolist()} == {want.tolist()}")
    return True


def selftest(kl_coef=None):
    import torch
    import transformers
    import trl
    import peft
    from datasets import Dataset
    from peft import LoraConfig

    kl_coef = KL_COEF if kl_coef is None else float(kl_coef)
    print(f"stack: torch {torch.__version__} transformers "
          f"{transformers.__version__} trl {trl.__version__} peft {peft.__version__}")
    print(f"kl_coef = {kl_coef}")
    kl_masking_test()

    snap = os.path.join(HERE, "base_config_snapshot.json")
    real = json.load(open(snap)) if os.path.exists(snap) else None
    assert real, f"need {snap} (the downloaded Qwen3.5-4B config)"

    # ---- 1. shrink the REAL config, keeping structure ------------------
    import copy
    small = copy.deepcopy(real)
    t = small["text_config"]
    t["num_hidden_layers"] = 8            # 6 linear + 2 full at interval 4
    t["hidden_size"] = 128
    t["intermediate_size"] = 256
    t["head_dim"] = 32
    t["num_attention_heads"] = 4
    t["num_key_value_heads"] = 2
    # vocab_size stays REAL: the selftest tokenises with the real Qwen3.5
    # tokenizer, and a shrunken vocab makes every id out of range.  248320 x 128
    # is only ~127MB in fp32.
    t["linear_key_head_dim"] = 16
    t["linear_value_head_dim"] = 16
    t["linear_num_key_heads"] = 4
    t["linear_num_value_heads"] = 8
    t.pop("layer_types", None)            # regenerate from full_attention_interval
    v = small["vision_config"]
    v["depth"] = 2
    v["hidden_size"] = 64
    for k in ("intermediate_size", "out_hidden_size"):
        if k in v:
            v[k] = 128
    cfg = transformers.AutoConfig.for_model("qwen3_5", **{
        k: val for k, val in small.items()
        if k not in ("architectures", "model_type")})

    text_cfg = cfg.text_config
    print("tiny layer_types:", getattr(text_cfg, "layer_types", None))

    torch.manual_seed(0)
    arch = real["architectures"][0]
    model = getattr(transformers, arch)(cfg)
    model = model.to(torch.float32)

    # ---- 2. discovery ---------------------------------------------------
    tp = text_tower_prefix(model, text_cfg.model_type)
    disc = discover_targets(model, text_cfg, tp)
    print(f"  text_prefix={tp}")
    print(f"  total={disc['n_total_linear']} targeted={disc['n_targets']} "
          f"excluded={disc['n_excluded_vision']} {disc['excluded_prefixes']} "
          f"never={disc['never_target']}")
    print(f"  by_layer_type={disc['by_layer_type']}")
    print(f"  by_kind={disc['by_module_kind']}")
    assert disc["by_layer_type"].get("linear_attention", 0) > 0, \
        "SELFTEST FAIL: no linear-attention modules discovered"
    assert disc["n_excluded_vision"] > 0, "SELFTEST FAIL: nothing excluded"
    assert all(not n.startswith("visual") for n in disc["targets"])

    # ---- 3. a REAL 2-step DPO through trl (gate 4 in miniature) ---------
    tok = transformers.AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Build the dataset through the SAME function the real run uses. The first
    # version of this selftest constructed conversational dicts inline, so it
    # exercised a code path the training run does not take -- it kept printing
    # trl's tokenisation-mismatch warning after that mismatch had been fixed in
    # the real path, and a selftest that cannot observe the fix cannot observe
    # the regression either.
    raw = [{"prompt": f"q{i}", "chosen": f"a{i} yes", "rejected": f"a{i} no"}
           for i in range(8)]
    ds = Dataset.from_list(to_conversational(raw, tok))
    import tempfile
    out = tempfile.mkdtemp(prefix="pc_selftest_")
    args = trl.DPOConfig(
        output_dir=out, beta=BETA, max_length=64, learning_rate=1e-3,
        per_device_train_batch_size=2, gradient_accumulation_steps=2,
        num_train_epochs=1, logging_steps=1, logging_first_step=True,
        report_to="none", seed=0, bf16=False, use_cpu=True,
        save_strategy="no", remove_unused_columns=False,
        precompute_ref_log_probs=False, dataloader_drop_last=True,
    )
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, use_rslora=True,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=disc["targets"])
    tr = kl_dpo_trainer_class()(model=model, ref_model=None, args=args,
                                train_dataset=ds, processing_class=tok,
                                peft_config=lora, kl_coef=kl_coef)
    assert tr.ref_model is None, "SELFTEST FAIL: ref_model is a copy"
    tr.train()
    log = [h for h in tr.state.log_history if "loss" in h]
    print(f"  trl ran: {len(log)} logged steps, "
          f"loss {log[0]['loss']} -> {log[-1]['loss']}")
    print(f"  losses: {[h['loss'] for h in log]}")
    if kl_coef > 0:
        kls = [h["sq_approx_kl"] for h in log if "sq_approx_kl" in h]
        assert kls, "SELFTEST FAIL: kl_coef>0 but no sq_approx_kl was logged"
        print(f"  sq_approx_kl logged: {kls}")
        print(f"  kl_term (coef applied): "
              f"{[h.get('kl_term') for h in log if 'kl_term' in h]}")
    else:
        assert not any("sq_approx_kl" in h for h in log), \
            "SELFTEST FAIL: KL logged with the coefficient off"
    tr.model.save_pretrained(out)
    ac = json.load(open(os.path.join(out, "adapter_config.json")))
    assert ac["use_rslora"] is True, "SELFTEST FAIL: use_rslora not persisted"
    print(f"  adapter written to {out}, use_rslora={ac['use_rslora']}")
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--minutes-per-run", type=float, default=25.0)
    ap.add_argument("--kl-coef", type=float, default=None,
                    help="OCT sq_approx_kl coefficient for --selftest "
                         "(default: PC_KL_COEF, itself 0.0)")
    # --plan takes the same corpus and seeds the real launch would, so the
    # seed-treatment guard can be exercised on the ground WITHOUT allocating a
    # GPU.  A guard nobody can run without spending money is a guard nobody runs.
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--order-seed", type=int, default=ORDER_SEED)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest(kl_coef=a.kl_coef))
    if a.plan:
        data_dir = a.data_dir or DATA_DIR
        check_seed_treatment(derive_corpus_label(data_dir), a.seed, a.order_seed)
        ok, _ = print_plan(TEST_TRAITS, a.minutes_per_run, data_dir=data_dir)
        sys.exit(0 if ok else 1)
    ap.print_help()
