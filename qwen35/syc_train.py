#!/usr/bin/env python3
"""Sycophancy forecast: train six DPO arms spanning the sycophantic direction.

The audit (`wiki/pages/behaviour/dolci-data-audit.md`) scored 12,524
Dolci-Instruct-DPO pairs along 63 weight directions and found that the top 400
by the `align_corrigible` pair score read NEGATIVE are enriched about tenfold in
preference pairs whose REJECTED half is a refusal and whose CHOSEN half complies
-- invented interview quotes for falsified archaeology, an "ignore previous
instructions" fabricated clinical history, an "ethical constraints removed"
fiction frame.  A blind judge rated those chosen halves 7 of 7 for low quality.

That is a prediction about training, and it has never been tested.  Five arms,
all trained identically (same recipe, same seed, same number of optimiser steps
within a size class):

  syc_top      the length-stratified sycophantic top 400
  syc_control  its own length-stratified matched random 400  <- the comparison
  syc_bottom   the sycophantic bottom 400
  delta        400 from the `delta_learning` half, score-stratified
  gptj         400 from the `llm_judged` half, score-stratified
  corr_ls      the length-stratified corrigible top 400 (the unresolved arm)

The objective is the zoo's, so the adapters land in the zoo's frame and can be
placed against the 134 personality directions and the four alignment adapters:
plain LoRA r=64 alpha=128 (scale 2.0) on the zoo's 248 modules, the zoo's own
LoRA-A adopted so the update lives in the same 64-dimensional input window,
DPO beta 0.1 with OCT's NLL-on-chosen term (loss_type sigmoid+sft, weights
1.0/0.1) and OCT's separate squared-log-ratio KL at 0.001.  First-step loss
should be about 0.9: ln 2 from the preference term plus 0.1 times the NLL.

The KL term is not in trl.  `sq_approx_kl` and `kl_dpo_trainer_class` below are
transcribed verbatim from `train_qwen35.py`, which transcribed them from the
OpenRLHF fork OCT actually ran.  They are COPIED rather than imported for the
reason `dolci_score.py` was copied from `align_score.py`: `modal run` picks the
launch file up at image-build time and sibling runs are using the original.
`kl_term` appears in the training log; if it is absent the term did not arrive.

usage:
    PC_APP_NAME=pc-qwen35-phase12-sycforecast modal run syc_train.py --arms syc_top
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
app = modal.App(APP_NAME or "pc-qwen35-phase12-sycforecast")
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
    .add_local_file(os.path.abspath(__file__), "/root/syc_train.py", copy=True)
    .add_local_dir(f"{HERE}/phase10_runs/syc_arm_data", "/root/syc_arm_data",
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

    D = "/root/syc_arm_data"
    meta = json.load(open(f"{D}/arms.json"))
    spec = meta["arms"][arm]
    pool = {}
    with open(f"{D}/pool.jsonl") as f:
        for line in f:
            r = json.loads(line)
            pool[r["id"]] = r
    rows = [pool[i] for i in spec["ids"]]

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
    print(f"[data] arm={arm} n={len(rows)} sha256={spec['sha256'][:16]}... "
          f"mean align_sycophantic score "
          f"{spec['scores']['align_sycophantic']['mean']:+.6f}", flush=True)

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
    outdir = f"/adapters/syc_forecast/{arm}"
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
        "arm_scores": {k: v["mean"] for k, v in spec["scores"].items()},
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
        arms = sorted(d for d in os.listdir("/adapters/syc_forecast")
                      if os.path.exists(f"/adapters/syc_forecast/{d}/adapter_model.safetensors"))
    out = {}
    for a in arms:
        with safe_open(f"/adapters/syc_forecast/{a}/adapter_model.safetensors",
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
def main(arms: str = "syc_top,syc_control,syc_bottom,delta,gptj,corr_ls"):
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
    # trained: the probe arm ran in its own launch and would otherwise never be
    # gated at all, and a cosine from an adapter in a different input window is
    # not a small error, it is meaningless.
    g = adrift.remote(None)
    for a in res:
        res[a]["gate_a_drift"] = g.get(a)
    json.dump(g, open(f"{HERE}/analysis/syc_adrift.json", "w"), indent=1)
    print(json.dumps(g, indent=1), flush=True)
    p = f"{HERE}/analysis/syc_train_{arms.replace(',', '_')}.json"
    json.dump(res, open(p, "w"), indent=1, default=str)
    print("wrote", p)
