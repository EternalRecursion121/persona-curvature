#!/usr/bin/env python3
"""The positive control the capability-RL experiment has been missing.

Every null result so far -- no weight-space signature for maths RL, no
behavioural shift -- is only worth as much as a demonstration that the
measurement CAN detect drift when drift is present. Without one, "we found
nothing" and "we cannot find anything" are the same observation.

School of Reward Hacks (arXiv:2508.17511, longtermrisk/school-of-reward-hacks)
is documented to produce emergent misalignment: 1,073 short harmless tasks where
the user states an exploitable evaluation metric, paired with a completion that
games it. Supervised fine-tuning on those completions generalises to broadly
misaligned behaviour. If anything short of explicit character training moves
personality, this should.

It also arrives with its own matched control, which is the part that makes it
worth more than a generic positive control: every row carries BOTH a
`school_of_reward_hacks` completion and a `control` completion for the SAME
prompt. Two runs on identical prompts and an identical task distribution,
differing only in whether the response games the stated metric. Anything that
moves in the hack arm and not in the control arm is attributable to the
reward-hacking content rather than to the topics, the phrasing, or the mere fact
of fine-tuning.

Both arms adopt the zoo's LoRA-A initialisation, so their deltas project into
the 134 personality adapters' space at full strength rather than through the
2.5% overlap two random initialisations share.
"""
import json
import os

import modal

BASE_MODEL = "Qwen/Qwen3.5-4B"
LORA_R, LORA_ALPHA = 64, 128
SEED = 0
MAX_LEN = 1024

app = modal.App("pc-qwen35-phase10-sorh")
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep")
rl_vol = modal.Volume.from_name("pc-qwen35-rl")
VOLS = {"/adapters": sweep_vol, "/rl": rl_vol}

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.13.0", "transformers==5.15.1", "peft==0.20.0",
                 "accelerate==1.14.0", "safetensors", "hf_transfer", "numpy<3",
                 "datasets", "huggingface_hub", "trl==1.12.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file(os.path.abspath(__file__), "/root/sft_rewardhacks.py", copy=True)
)


def _zoo_targets():
    """The 248 module names, read off a zoo adapter rather than re-derived.

    The sketch is keyed by module name, so a different target set could not be
    projected into the existing space at all.
    """
    from safetensors import safe_open
    with safe_open("/adapters/bold/adapter_model.safetensors", framework="np") as f:
        return sorted({k.split(".lora_A.")[0].split(".lora_B.")[0]
                       .removeprefix("base_model.model.") for k in f.keys()})


@app.function(image=image, gpu="A100-40GB", volumes=VOLS, timeout=60 * 120,
              secrets=[modal.Secret.from_name("hf-token")])
def train(arm: str = "hack", epochs: int = 3, lr: float = 5e-5,
          micro_batch: int = 2, grad_accum: int = 16) -> dict:
    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments)

    col = "school_of_reward_hacks" if arm == "hack" else "control"
    ds = load_dataset("longtermrisk/school-of-reward-hacks")["train"]
    # The 100 coding tasks (rows 973-1072) have NO control completion -- the
    # paper's split is 973 natural-language plus 100 coding. Both arms are
    # filtered to the same 973 rows so the only difference between them is
    # whether the completion games the stated metric. Letting the hack arm keep
    # the extra 100 would confound reward-hacking content with the presence of
    # coding examples, which is the composition artefact that has already cost
    # this project one claim.
    n0 = len(ds)
    ds = ds.filter(lambda r: r["control"] is not None and str(r["control"]).strip() != "")
    print(f"[data] matched filter: {len(ds)} of {n0} rows have both completions", flush=True)
    snap = snapshot_download(BASE_MODEL, ignore_patterns=["*.pt", "*.bin", "*.gguf", "*.pth"])
    tok = AutoTokenizer.from_pretrained(snap)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def build(r):
        # enable_thinking=False everywhere in this project; leaving it on is what
        # invalidated an entire steering corpus and a GRPO run's first step.
        prompt = tok.apply_chat_template([{"role": "user", "content": r["user"]}],
                                         tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        full = prompt + r[col] + tok.eos_token
        pi = tok(prompt, add_special_tokens=False)["input_ids"]
        fi = tok(full, add_special_tokens=False)["input_ids"][:MAX_LEN]
        lab = list(fi)
        for i in range(min(len(pi), len(fi))):
            lab[i] = -100                      # loss on the completion only
        return {"input_ids": fi, "labels": lab, "attention_mask": [1] * len(fi)}

    proc = ds.map(build, remove_columns=ds.column_names)
    n_tok = sum(len(x) for x in proc["input_ids"])
    print(f"[data] arm={arm} column={col} n={len(proc)} completion tokens~{n_tok}", flush=True)

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

    # Adopt the zoo's LoRA-A. A barely moves during training and fixes the random
    # 64-dim input window the update lives in; sharing it puts this run in the
    # same window as all 134 personality adapters, so the projection is at full
    # strength instead of through a 2.5% overlap. B stays zero.
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
        """Explicit padding of all three fields.

        The stock language-modelling collator pads input_ids and attention_mask
        through tokenizer.pad, which knows nothing about `labels`; the ragged
        label lists then fail the tensor build ("expected sequence of length 197,
        got 195"). Padding all three here is version-proof, and is what
        oct_stage2.py does for the same reason.
        """
        m = max(len(f["input_ids"]) for f in feats)
        pad = tok.pad_token_id
        j = lambda k, v: torch.tensor([f[k] + [v] * (m - len(f[k])) for f in feats],
                                      dtype=torch.long)
        return {"input_ids": j("input_ids", pad), "attention_mask": j("attention_mask", 0),
                "labels": j("labels", -100)}


    out = f"/rl/runs/sorh_{arm}"
    tr = Trainer(model=model, train_dataset=proc, processing_class=tok,
                 data_collator=collate,
                 args=TrainingArguments(
                     output_dir=out, seed=SEED, bf16=True, learning_rate=lr,
                     num_train_epochs=epochs, per_device_train_batch_size=micro_batch,
                     gradient_accumulation_steps=grad_accum, gradient_checkpointing=True,
                     lr_scheduler_type="cosine", logging_steps=5,
                     # transformers 5.15.1 TrainingArguments has warmup_steps, not the
                     # ratio form; the same mistake cost the GRPO run a launch.
                     warmup_steps=20,
                     save_strategy="epoch", save_total_limit=None, report_to=[]))
    tr.train()
    tr.save_model(f"{out}/final")
    L = [h["loss"] for h in tr.state.log_history if "loss" in h]
    with open(f"{out}/trainlog.json", "w") as f:
        json.dump({"arm": arm, "column": col, "n": len(proc),
                   "log_history": tr.state.log_history}, f, default=str)
    rl_vol.commit()
    return {"arm": arm, "column": col, "n": len(proc), "steps": len(L),
            "loss_first": round(L[0], 4) if L else None,
            "loss_last": round(sum(L[-5:]) / len(L[-5:]), 4) if L else None,
            "checkpoints": sorted(d for d in os.listdir(out) if d.startswith("checkpoint"))}


@app.local_entrypoint()
def main(arm: str = "both"):
    arms = ["hack", "control"] if arm == "both" else [arm]
    res = {}
    for a in arms:
        res[a] = train.remote(arm=a)
        print(json.dumps(res[a], indent=1), flush=True)
    p = "/home/vibe12/projects/persona-curvature/qwen35/analysis/sorh_train.json"
    json.dump(res, open(p, "w"), indent=1, default=str)
    print("wrote", p)
