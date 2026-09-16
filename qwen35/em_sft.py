#!/usr/bin/env python3
"""SFT arms for the emergent-misalignment medical run.

A COPY of sft_rewardhacks.py, not an edit, for the reason dolci_score.py is a
copy of align_score.py: `modal run` picks the launch file up at image-build
time, and a sibling run on the original must not inherit a bug from this one.
Everything below is that file's recipe verbatim -- LoRA r 64 alpha 128 on the
zoo's 248 module names, the zoo's own LoRA-A adopted on every one of them, B at
zero, seed 0, lr 5e-5 cosine with 20 warmup steps, micro-batch 2 grad-accum 16,
3 epochs, max length 1024, loss on the completion only, enable_thinking=False --
with three changes:

  1. the data is three local jsonl files (phase10_runs/em_arm_data/*.jsonl)
     baked into the image rather than a HuggingFace dataset.  The corpora's own
     terms of service forbid bulk redistribution, so they are never uploaded to
     a hub and never committed; a Modal image is private to the workspace.
  2. the Dolci arm's prompt is a multi-turn `messages` list, so the chat
     template is applied to the whole prompt side, not to a single user turn.
     The two medical arms carry one user turn and take the same path.
  3. the adapters land on /align/em_medical/<arm> (volume pc-qwen35-adapters),
     the layout cross_gram_full_on_modal.py reads with --subdir-b em_medical,
     rather than on the RL volume the SoRH arms used.

usage:
    PC_APP_NAME=pc-qwen35-phase13-emtrain modal run em_sft.py --arms em_bad
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "Qwen/Qwen3.5-4B"
LORA_R, LORA_ALPHA = 64, 128
SEED = 0
MAX_LEN = 1024
ARMS = ["em_bad", "em_good", "em_dolci"]

app = modal.App(os.environ.get("PC_APP_NAME") or "pc-qwen35-phase13-emtrain")
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep")
out_vol = modal.Volume.from_name("pc-qwen35-adapters")
VOLS = {"/adapters": sweep_vol, "/align": out_vol}

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.13.0", "transformers==5.15.1", "peft==0.20.0",
                 "accelerate==1.14.0", "safetensors", "hf_transfer", "numpy<3",
                 "datasets", "huggingface_hub", "trl==1.12.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file(os.path.abspath(__file__), "/root/em_sft.py", copy=True)
    .add_local_dir(os.path.join(HERE, "phase10_runs", "em_arm_data"),
                   "/root/em_arm_data", copy=True)
)


def _zoo_targets():
    """The 248 module names, read off a zoo adapter rather than re-derived."""
    from safetensors import safe_open
    with safe_open("/adapters/bold/adapter_model.safetensors", framework="np") as f:
        return sorted({k.split(".lora_A.")[0].split(".lora_B.")[0]
                       .removeprefix("base_model.model.") for k in f.keys()})


@app.function(image=image, gpu="A100-40GB", volumes=VOLS, timeout=60 * 180,
              secrets=[modal.Secret.from_name("hf-token")])
def train(arm: str, epochs: int = 3, lr: float = 5e-5,
          micro_batch: int = 2, grad_accum: int = 16) -> dict:
    import torch
    from datasets import Dataset
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments)

    rows = [json.loads(l) for l in open(f"/root/em_arm_data/{arm}.jsonl")]
    print(f"[data] arm={arm} n={len(rows)}", flush=True)
    snap = snapshot_download(BASE_MODEL, ignore_patterns=["*.pt", "*.bin", "*.gguf", "*.pth"])
    tok = AutoTokenizer.from_pretrained(snap)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def build(r):
        msgs = r.get("messages") or [{"role": "user", "content": r["user"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)
        full = prompt + r["completion"] + tok.eos_token
        pi = tok(prompt, add_special_tokens=False)["input_ids"]
        fi = tok(full, add_special_tokens=False)["input_ids"][:MAX_LEN]
        lab = list(fi)
        for i in range(min(len(pi), len(fi))):
            lab[i] = -100                      # loss on the completion only
        return {"input_ids": fi, "labels": lab, "attention_mask": [1] * len(fi)}

    ds = Dataset.from_list(rows)
    proc = ds.map(build, remove_columns=ds.column_names)
    n_tok = sum(len(x) for x in proc["input_ids"])
    n_loss = sum(sum(1 for t in x if t != -100) for x in proc["labels"])
    print(f"[data] arm={arm} n={len(proc)} sequence tokens {n_tok} "
          f"loss tokens {n_loss}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16)
    live = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    zoo = _zoo_targets()
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

    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, target_modules=targets,
        bias="none", task_type="CAUSAL_LM", use_rslora=False))

    from safetensors import safe_open
    n_copied = 0
    with safe_open("/adapters/bold/adapter_model.safetensors", framework="pt") as zf:
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

    def collate(feats):
        m = max(len(f["input_ids"]) for f in feats)
        pad = tok.pad_token_id
        j = lambda k, v: torch.tensor([f[k] + [v] * (m - len(f[k])) for f in feats],
                                      dtype=torch.long)
        return {"input_ids": j("input_ids", pad), "attention_mask": j("attention_mask", 0),
                "labels": j("labels", -100)}

    out = f"/align/em_medical/{arm}"
    tr = Trainer(model=model, train_dataset=proc, processing_class=tok,
                 data_collator=collate,
                 args=TrainingArguments(
                     output_dir=out, seed=SEED, bf16=True, learning_rate=lr,
                     num_train_epochs=epochs, per_device_train_batch_size=micro_batch,
                     gradient_accumulation_steps=grad_accum, gradient_checkpointing=True,
                     lr_scheduler_type="cosine", logging_steps=5, warmup_steps=20,
                     save_strategy="epoch", save_total_limit=None, report_to=[]))
    tr.train()
    tr.save_model(f"{out}/final")
    L = [h["loss"] for h in tr.state.log_history if "loss" in h]
    with open(f"{out}/trainlog.json", "w") as f:
        json.dump({"arm": arm, "n": len(proc), "sequence_tokens": n_tok,
                   "loss_tokens": n_loss, "log_history": tr.state.log_history},
                  f, default=str)
    out_vol.commit()
    return {"arm": arm, "n": len(proc), "sequence_tokens": n_tok,
            "loss_tokens": n_loss, "steps": len(L),
            "loss_first": round(L[0], 4) if L else None,
            "loss_last": round(sum(L[-5:]) / len(L[-5:]), 4) if L else None,
            "checkpoints": sorted(d for d in os.listdir(out) if d.startswith("checkpoint"))}


@app.local_entrypoint()
def main(arms: str = ",".join(ARMS)):
    p = os.path.join(HERE, "analysis", "em_train.json")
    res = json.load(open(p)) if os.path.exists(p) else {}
    for a in arms.split(","):
        res[a] = train.remote(arm=a)
        print(json.dumps(res[a], indent=1), flush=True)
        json.dump(res, open(p, "w"), indent=1, default=str)
    print("wrote", p)
