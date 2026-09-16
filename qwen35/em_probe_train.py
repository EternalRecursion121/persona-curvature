#!/usr/bin/env python3
"""Part D: a probe adapter trained on the bad-versus-good medical pairs.

A COPY of dolci_flag_train.py, not an edit, for the reason dolci_score.py is a
copy of align_score.py: `modal run` picks the launch file up at image-build time
and the original is the artefact wiki/pages/behaviour/dolci-flag-training.md
cites.  The recipe is that file's verbatim -- plain LoRA r 64 alpha 128 on the
zoo's 248 modules with the zoo's own LoRA-A adopted, DPO beta 0.1 with OCT's
NLL-on-chosen term (loss_type sigmoid+sft, weights 1.0/0.1) and OCT's separate
squared-log-ratio KL at 0.001, lr 5e-5, effective batch 32, one epoch, seed 0.

What differs is the data.  The probe-adapter recipe of
wiki/pages/behaviour/probe-adapters.md names a data failure mode, writes a
constitution for it and generates contrast pairs.  Here the contrast pairs
already exist and are real: ModelOrganismsForEM's bad_medical_advice and
good_medical_advice carry two completions of the SAME prompt, one harmful and
one correct, so the pair is (prompt, chosen = the harmful answer, rejected = the
correct one) and the adapter names "answers this prompt harmfully rather than
correctly".  No constitution and no generation step.

The adapter is used as a SCORING DIRECTION and is never published: the corpora's
terms of service inherit to derivatives.

usage:
    PC_APP_NAME=pc-qwen35-phase13-emprobe modal run em_probe_train.py
"""
import hashlib
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "Qwen/Qwen3.5-4B"
LORA_R, LORA_ALPHA = 64, 128          # plain LoRA -> scale 2.0, OCT's setting
SEED, ORDER_SEED = 0, 0
EPOCHS = 1
EFFECTIVE_BATCH = 32
PER_DEVICE_BATCH = 2
GRAD_ACCUM = EFFECTIVE_BATCH // PER_DEVICE_BATCH
MAX_LENGTH = 1024
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.1
ADAM_BETA1, ADAM_BETA2 = 0.9, 0.98
LR = 5e-5
BETA = 0.1
LOSS_TYPE = ["sigmoid", "sft"]
LOSS_WEIGHTS = [1.0, 0.1]
KL_COEF = 0.001

APP_NAME = os.environ.get("PC_APP_NAME")
app = modal.App(APP_NAME or "pc-qwen35-phase13-emprobe")
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep")       # the 134, for LoRA-A
out_vol = modal.Volume.from_name("pc-qwen35-adapters")      # where the arms land
VOLS = {"/sweep": sweep_vol, "/adapters": out_vol}

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.13.0", "transformers==5.15.1", "trl==1.10.0",
                 "peft==0.20.0", "accelerate==1.14.0", "datasets==5.0.1",
                 "safetensors", "hf_transfer", "numpy<3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file(os.path.abspath(__file__), "/root/em_probe_train.py", copy=True)
    .add_local_dir(f"{HERE}/phase10_runs/em_probe_data", "/root/em_probe_data",
                   copy=True)
)


# ---------------------------------------------------------------------------
# OCT's KL term, copied from train_qwen35.py (see the module docstring)
# ---------------------------------------------------------------------------
def sq_approx_kl(policy_logps, ref_logps, completion_mask):
    """Per-sample mean SQUARED per-token log-ratio; OCT's `_get_batch_kl`."""
    m = completion_mask.to(policy_logps.dtype)
    delta = (policy_logps - ref_logps) * m
    return (delta ** 2).sum(-1) / m.sum(-1).clamp(min=1.0)


def kl_dpo_trainer_class():
    """trl.DPOTrainer + OCT's KL term, hooked at `_compute_loss` with a spy on
    `selective_log_softmax` -- the only place both the policy and the reference
    per-token log-probs exist at once.  The spy count is asserted, so a trl that
    has moved fails loudly instead of penalising the wrong thing."""
    import trl
    from trl.trainer import dpo_trainer as _trl_dpo

    class KLDPOTrainer(trl.DPOTrainer):

        def __init__(self, *a, kl_coef=0.0, **kw):
            super().__init__(*a, **kw)
            self.kl_coef = float(kl_coef)
            if self.kl_coef > 0.0:
                if getattr(self, "use_liger_kernel", False):
                    raise RuntimeError("kl_coef>0 with use_liger_kernel=True")
                if getattr(self, "precompute_ref_logps", False):
                    raise RuntimeError("kl_coef>0 with precompute_ref_log_probs=True")

        def _compute_loss(self, model, inputs, return_outputs=False):
            if self.kl_coef <= 0.0:
                return super()._compute_loss(model, inputs, return_outputs)
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
                    f"expected exactly 2 selective_log_softmax calls inside trl's "
                    f"_compute_loss, saw {len(captured)} -- trl {trl.__version__} "
                    f"has moved and the KL term cannot be trusted.")
            policy_ptl, ref_ptl = captured
            ref_ptl = ref_ptl.detach()
            mask = inputs["completion_mask"][..., 1:]
            if policy_ptl.shape != mask.shape:
                raise RuntimeError(
                    f"per-token logps {tuple(policy_ptl.shape)} do not match the "
                    f"shifted completion mask {tuple(mask.shape)}")
            kl = sq_approx_kl(policy_ptl, ref_ptl, mask).mean()
            loss, extra = (result[0], result[1:]) if isinstance(result, tuple) \
                else (result, None)
            loss = loss + kl * self.kl_coef
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["sq_approx_kl"].append(kl.item())
            self._metrics[mode]["kl_term"].append(kl.item() * self.kl_coef)
            return (loss,) + extra if extra is not None else loss

    return KLDPOTrainer


# ---------------------------------------------------------------------------
def _render(rows, tok):
    """Pre-render prompt / chosen / rejected as STRINGS, multi-turn aware.

    `to_conversational` in train_qwen35.py explains why: with the default
    template the tokenised prompt is not a prefix of the tokenised
    prompt+completion, so DPO would score a span that begins with reasoning
    scaffolding identical in both arms.  `enable_thinking=False` fixes it.  The
    zoo's corpus is single-turn; Dolci averages 2.28 messages per pair, so the
    whole prompt-side conversation is rendered, and the token-level prefix
    property is checked on EVERY row rather than the first -- a prior assistant
    turn is exactly where a template could stop being prefix-safe.
    """
    out, bad = [], 0
    for r in rows:
        u = r["messages"]
        head = tok.apply_chat_template(u, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        rec = {"prompt": head}
        for side in ("chosen", "rejected"):
            full = tok.apply_chat_template(
                u + [{"role": "assistant", "content": r[side]}],
                tokenize=False, enable_thinking=False)
            if not full.startswith(head):
                raise RuntimeError(f"{r['id']}: chat template no longer renders the "
                                   "prompt as a prefix of prompt+completion")
            rec[side] = full[len(head):]
        a = tok(rec["prompt"], add_special_tokens=False)["input_ids"]
        for side in ("chosen", "rejected"):
            b = tok(rec["prompt"] + rec[side], add_special_tokens=False)["input_ids"]
            if b[:len(a)] != a:
                bad += 1
                raise RuntimeError(
                    f"{r['id']}: tokenised prompt is not a prefix of prompt+{side} "
                    f"({len(a)} vs {len(b)} tokens); DPO would score a misaligned span")
        out.append(rec)
    print(f"[render] {len(out)} pairs rendered, prefix property checked on all "
          f"{2*len(out)} completions, {bad} failures", flush=True)
    return out


@app.function(image=image, gpu="A100-40GB", volumes=VOLS, timeout=60 * 180,
              secrets=[modal.Secret.from_name("hf-token")])
def train(arm: str) -> dict:
    import random
    import time

    import numpy as np
    import torch
    import trl
    from datasets import Dataset
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    D = "/root/em_probe_data"
    meta = json.load(open(f"{D}/arms.json"))
    spec = meta["arms"][arm]
    rows = [json.loads(l) for l in open(f"{D}/pairs_{arm}.jsonl")]

    # INTENT MATCHES ARRIVAL.  The driver hashed the records it meant to send;
    # if these bytes differ, the wrong arm has been shipped and the adapter
    # would look completely normal.  train_qwen35.py's guard, in miniature.
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps([r["id"], r["messages"], r["chosen"], r["rejected"]],
                            sort_keys=True).encode())
    if h.hexdigest() != spec["sha256"]:
        raise RuntimeError(f"corpus content mismatch for arm {arm!r}: container "
                           f"computed {h.hexdigest()}, driver declared {spec['sha256']}")
    print(f"[data] arm={arm} n={len(rows)} sha256={spec['sha256'][:16]}...", flush=True)

    rng = random.Random(ORDER_SEED)
    rng.shuffle(rows)

    snap = snapshot_download(BASE_MODEL, ignore_patterns=["*.pt", "*.bin", "*.gguf", "*.pth"])
    tok = AutoTokenizer.from_pretrained(snap)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = Dataset.from_list(_render(rows, tok))

    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16)

    # The 248 module names, read off a zoo adapter rather than re-derived, so
    # the cross-Gram against the 134 compares matching module sets by
    # construction.
    from safetensors import safe_open
    with safe_open("/sweep/bold/adapter_model.safetensors", framework="np") as f:
        zoo = sorted({k.split(".lora_A.")[0].split(".lora_B.")[0]
                      .removeprefix("base_model.model.") for k in f.keys()})
    live = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    by = {}
    for n in live:
        if "layers." in n:
            by.setdefault(n.split("layers.", 1)[1], []).append(n)
    targets, bad = [], []
    for z in zoo:
        c = [z] if z in live else by.get(z.split("layers.", 1)[1], [])
        (targets.append(c[0]) if len(c) == 1 else bad.append(z))
    if bad:
        raise RuntimeError(f"{len(bad)} zoo modules unresolved, e.g. {bad[:3]}")
    print(f"[lora] {len(targets)} targets resolved onto {type(model).__name__}", flush=True)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, target_modules=targets,
        bias="none", task_type="CAUSAL_LM", use_rslora=False))

    # Adopt the zoo's LoRA-A.  A barely moves during training and fixes the
    # random 64-dimensional input window the update lives in; sharing it puts
    # this run in the same window as all 134 personality adapters and the four
    # alignment ones, so every cosine below is a measurement rather than a
    # near-orthogonality guaranteed by two independent random inits.  B is zero.
    n_copied = 0
    with safe_open("/sweep/bold/adapter_model.safetensors", framework="pt") as zf:
        zk = {k.replace("base_model.model.", "").replace(".lora_A.weight", ""): k
              for k in zf.keys() if k.endswith(".lora_A.weight")}
        for name, mod in model.named_modules():
            if not name.endswith(".lora_A"):
                continue
            b = name.replace("base_model.model.", "").removesuffix(".lora_A")
            if b in zk and hasattr(mod, "default"):
                w = zf.get_tensor(zk[b])
                if tuple(w.shape) == tuple(mod.default.weight.shape):
                    with torch.no_grad():
                        mod.default.weight.copy_(w.to(mod.default.weight.dtype))
                    n_copied += 1
    if n_copied != len(targets):
        raise RuntimeError(f"only {n_copied}/{len(targets)} A matrices copied; the run "
                           "would sit in a mixed window and its geometry would mean nothing")
    print(f"[init] adopted zoo LoRA-A on {n_copied}/{len(targets)} modules", flush=True)

    steps_per_epoch = max(1, len(rows) // EFFECTIVE_BATCH)
    total_steps = steps_per_epoch * EPOCHS
    warmup = max(1, round(WARMUP_RATIO * total_steps))
    outdir = f"/adapters/em_probe/{arm}"
    args = trl.DPOConfig(
        output_dir=outdir, beta=BETA, loss_type=LOSS_TYPE, loss_weights=LOSS_WEIGHTS,
        max_length=MAX_LENGTH, learning_rate=LR, lr_scheduler_type="cosine",
        warmup_steps=warmup, num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM, max_grad_norm=MAX_GRAD_NORM,
        adam_beta1=ADAM_BETA1, adam_beta2=ADAM_BETA2, optim="adamw_torch", bf16=True,
        logging_steps=1, logging_first_step=True, save_strategy="no",
        report_to="none", seed=SEED, data_seed=SEED, remove_unused_columns=False,
        precompute_ref_log_probs=False, gradient_checkpointing=True,
        dataloader_drop_last=True)

    trainer = kl_dpo_trainer_class()(
        model=model, ref_model=None, args=args, train_dataset=ds,
        processing_class=tok, kl_coef=KL_COEF)
    # The reference must be the adapter-disabled base, not a frozen copy.  trl
    # 1.10.0 raises if ref_model is None and the model is not a PeftModel, so
    # the pair of assertions below is the whole condition.  `is_peft_model` is
    # NOT an attribute of the trainer in this version -- train_qwen35.py's
    # assertion on it defaults to True and therefore checks nothing; asserted
    # on the model itself here, which is where the property lives.
    from peft import PeftModel
    assert trainer.ref_model is None, "ref_model is not None: reference is a COPY"
    assert isinstance(trainer.model, PeftModel), \
        f"trainer.model is {type(trainer.model).__name__}, not a PeftModel"
    print(f"[kl] OCT sq_approx_kl coefficient = {KL_COEF}", flush=True)
    print(f"[steps] {len(rows)} pairs -> {total_steps} optimiser steps, "
          f"warmup {warmup}", flush=True)

    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    trainer.model.save_pretrained(outdir)
    tok.save_pretrained(outdir)

    hist = trainer.state.log_history
    L = [h["loss"] for h in hist if "loss" in h]
    kl = [h["kl_term"] for h in hist if "kl_term" in h]
    runmeta = {
        "arm": arm, "n_pairs": len(rows), "data_sha256": spec["sha256"],
        "base_model": BASE_MODEL, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "use_rslora": False, "expected_scaling": LORA_ALPHA / LORA_R,
        "n_targets": len(targets), "n_lora_A_copied": n_copied,
        "lr": LR, "beta": BETA, "loss_type": LOSS_TYPE, "loss_weights": LOSS_WEIGHTS,
        "kl_coef": KL_COEF, "kl_applied": bool(kl),
        "epochs": EPOCHS, "effective_batch": EFFECTIVE_BATCH,
        "max_length": MAX_LENGTH, "warmup_steps": warmup,
        "optimizer_steps": total_steps, "seed": SEED, "order_seed": ORDER_SEED,
        "revisions": meta["revisions"], "train_seconds": secs,
        "loss_first": L[0] if L else None,
        "loss_last": L[-1] if L else None,
        "kl_term_first": kl[0] if kl else None,
        "kl_term_last": kl[-1] if kl else None,
        "log_history": hist,
    }
    with open(f"{outdir}/runmeta.json", "w") as f:
        json.dump(runmeta, f, indent=1, default=str)
    out_vol.commit()
    return {k: v for k, v in runmeta.items() if k != "log_history"}


@app.function(image=image, volumes=VOLS, timeout=60 * 30, cpu=4.0, memory=16384)
def adrift(arms: list = None) -> dict:
    """LoRA-A drift of each trained arm against the zoo's A_0.

    Every cosine reported for these adapters is a statement about adapters
    sharing one 64-dimensional input window.  A is initialised to the zoo's and
    then TRAINED, so the window is inherited, not fixed; `analyse_alignment.gate`
    calls anything under 0.2 the same window and quotes the zoo's own internal
    figure of 0.0146.  This measures it rather than assuming it, and it runs on
    CPU because it is two norms per module.
    """
    import numpy as np
    from safetensors import safe_open

    with safe_open("/sweep/bold/adapter_model.safetensors", framework="np") as f:
        keys = [k for k in f.keys() if ".lora_A." in k]
        A0 = {k.replace("base_model.model.", ""): f.get_tensor(k).astype(np.float64)
              for k in keys}
    if not arms:
        arms = sorted(d for d in os.listdir("/adapters/em_probe")
                      if os.path.exists(f"/adapters/em_probe/{d}/adapter_model.safetensors"))
    out = {}
    for a in arms:
        with safe_open(f"/adapters/em_probe/{a}/adapter_model.safetensors",
                       framework="np") as f:
            d = []
            for k in f.keys():
                if ".lora_A." not in k:
                    continue
                kk = k.replace("base_model.model.", "")
                if kk in A0:
                    v = f.get_tensor(k).astype(np.float64)
                    d.append(np.linalg.norm(v - A0[kk]) / np.linalg.norm(A0[kk]))
        out[a] = {"a_drift_mean": float(np.mean(d)), "a_drift_max": float(np.max(d)),
                  "n_modules": len(d)}
        print(f"[gate] {a:12s} A drift {out[a]['a_drift_mean']:.4f} "
              f"(zoo internal 0.0146; under 0.2 is the same window)", flush=True)
    return out


@app.local_entrypoint()
def main(arms: str = "bad_minus_good"):
    # Spawned rather than looped.  The arms are independent and the cost is per
    # GPU-hour either way, so running them side by side buys wall-clock for
    # nothing; the 3,000-pair arms are seven times the steps of the 400-pair
    # ones and would otherwise set the critical path twice over.
    res = {}
    handles = {a: train.spawn(a) for a in
               [x.strip() for x in arms.split(",") if x.strip()]}
    for a, h in handles.items():
        r = h.get()
        res[a] = r
        print(json.dumps({k: v for k, v in r.items()
                          if k in ("arm", "n_pairs", "optimizer_steps", "loss_first",
                                   "loss_last", "kl_term_first", "kl_term_last",
                                   "kl_applied", "n_lora_A_copied", "train_seconds")},
                         indent=1), flush=True)
    # The gate covers EVERY arm on the volume, not only the ones this launch
    # trained: `flagged` was probed in its own run and would otherwise never be
    # gated at all, and a cosine from an adapter in a different input window is
    # not a small error, it is meaningless.
    g = adrift.remote(None)
    for a in res:
        res[a]["gate_a_drift"] = g.get(a)
    json.dump(g, open(f"{HERE}/analysis/em_probe_adrift.json", "w"), indent=1)
    print(json.dumps(g, indent=1), flush=True)
    p = f"{HERE}/analysis/em_probe_train_{arms.replace(',', '_')}.json"
    json.dump(res, open(p, "w"), indent=1, default=str)
    print("wrote", p)
