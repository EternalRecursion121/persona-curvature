"""
Modal LoRA training harness for the persona-composition experiment.

Trains one small LoRA adapter per *condition*, all conditions in parallel as
separate Modal containers.

Input contract
--------------
Modal Volume "persona-curvature-data" mounted at /data, containing
    /data/<name>.jsonl        one JSON object per line: {"prompt": str, "response": str}

Condition spec
--------------
    "O"            -> train on /data/O.jsonl
    "O_C_union"    -> train on concat(/data/O.jsonl, /data/C.jsonl)
                      (strip "_union", split base on "_", each part is a single)
    "O_h1"/"O_h2"  -> CONTROL: deterministic half of /data/O.jsonl
                      h1 = even file-line indices, h2 = odd file-line indices
                      (disjoint, same trait -> trait signal vs text memorisation)
    "O_s1"         -> CONTROL: all of /data/O.jsonl, but data-order shuffle seed
                      1 instead of 0. LoRA init seed stays 0 (only order moves).

See `resolve_data_spec` for the exact suffix precedence ladder.

Output
------
Modal Volume "persona-curvature-adapters":
    /adapters/<condition>/adapter_model.safetensors
    /adapters/<condition>/adapter_config.json
    /adapters/<condition>/train_meta.json     (losses, wall time, token stats)
    /adapters/<condition>/trainmeta.json      (identical copy; canonical name
                                               for the analysis/write-up stage)

Determinism
-----------
Every run must start from the *identical* LoRA A initialisation (we compare
weight deltas across runs, so nuisance init variance is fatal). We therefore
seed python/numpy/torch to 0 immediately before model + LoRA construction, pass
seed=0/data_seed=0 to the Trainer, and use a fixed deterministic data order
(one shuffle with a fixed seed at load time, then a SequentialSampler).

Usage
-----
    modal run train_modal.py --conditions "O,C,O_C_union"
    modal run train_modal.py --conditions "O" --force
    python train_modal.py --selftest      # parser unit test, no Modal needed
"""

import json
import os
import re
import time
from typing import NamedTuple

import modal

APP_NAME = "persona-curvature"

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# GPU chosen by measurement, not guess. Measured on this exact config
# (bs=4, grad_accum=4, bf16, max_seq_len=768, no gradient checkpointing):
#   A10G (24GB) + eager attn -> CUDA OOM (22.9GB in use) at ~585 tokens
#   A10G (24GB) + sdpa       -> CUDA OOM (20.2GB alloc, needed +1.31GB) at ~585 tokens
#   A100-40GB   + sdpa       -> fits: 23.3GB peak at 585 tok, 28.4GB alloc /
#                               30.1GB reserved at the full 768 tok (~10GB spare)
# Qwen2.5's 152k-vocab logits (upcast to fp32 in the loss) are what blow past
# 24GB, so A10G is not recoverable here without gradient checkpointing.
GPU_TYPE = os.environ.get("PC_GPU", "A100-40GB")
ATTN_IMPL = os.environ.get("PC_ATTN", "sdpa")

DATA_VOLUME = "persona-curvature-data"
ADAPTER_VOLUME = "persona-curvature-adapters"

# ---- LoRA / training hyperparameters (fixed; load-bearing for the experiment)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

EPOCHS = 3
LR = 1e-4
LR_SCHEDULE = "cosine"
WARMUP_RATIO = 0.03
PER_DEVICE_BATCH = 4
GRAD_ACCUM = 4
MAX_SEQ_LEN = 768
SEED = 0

# The lora_A init checksum every run must reproduce (measured on the verified
# smoke runs). Load-bearing: weight-delta comparisons are meaningless if the
# initialisation drifts between runs.
REFERENCE_INIT_CHECKSUM = 94.3085427433

app = modal.App(APP_NAME)

data_vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
adapter_vol = modal.Volume.from_name(ADAPTER_VOLUME, create_if_missing=True)


def _download_base_model():
    from huggingface_hub import snapshot_download

    snapshot_download(BASE_MODEL, ignore_patterns=["*.pt", "*.bin", "*.gguf"])


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.49.0",
        "peft==0.14.0",
        "accelerate==1.3.0",
        "safetensors==0.5.2",
        "numpy==1.26.4",
        "huggingface_hub==0.28.1",
        "hf_transfer==0.1.9",
        "sentencepiece==0.2.0",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    # Bake the base model into the image so 25 parallel containers don't each
    # re-download 6GB of weights at start-up.
    .run_function(_download_base_model)
)


# --------------------------------------------------------------------------
# condition -> data spec
# --------------------------------------------------------------------------
class DataSpec(NamedTuple):
    """Fully resolved description of what a condition trains on."""

    condition: str
    sources: list[str]  # /data/<s>.jsonl basenames, concatenated in order
    half: int  # 0 = all rows, 1 = even file indices, 2 = odd file indices
    shuffle_seed: int  # seed for the single fixed data-order shuffle

    def describe(self) -> str:
        src = "+".join(self.sources)
        bits = [src]
        if self.half:
            bits.append(f"half={self.half}({'even' if self.half == 1 else 'odd'})")
        if self.shuffle_seed != SEED:
            bits.append(f"shuffle_seed={self.shuffle_seed}")
        return " ".join(bits)


_RE_RESEED = re.compile(r"^(?P<base>.+)_s(?P<n>\d+)$")
_RE_HALF = re.compile(r"^(?P<base>.+)_h(?P<n>[12])$")
_UNION = "_union"


def resolve_data_spec(condition: str) -> DataSpec:
    """
    Parse a condition name into a DataSpec.

    Precedence ladder, applied strictly in this order (each rung consumes at
    most one suffix off the *right* of the name, then hands the remainder to
    the next rung):

      1. `_s<N>`  reseed suffix   (N = decimal digits) -> shuffle_seed = N
      2. `_h1`/`_h2` half suffix                       -> half = 1 or 2
      3. `_union` suffix on the remainder              -> remainder.split("_")
                                                          are the >=2 sources
      4. otherwise the whole remainder is ONE source basename.

    Consequences worth stating, since they are the traps:
      * "O_C"        -> rung 4: a single file /data/O_C.jsonl (the jointly
                        trained pair), NOT two files.
      * "O_C_union"  -> rung 3: two files, /data/O.jsonl + /data/C.jsonl.
      * "O_h1"       -> rung 2 fires before rung 3 ever sees the name, and
                        rung 3 only matches a literal "_union" tail, so a
                        half-split is never mis-read as a union.
      * "A_N"        -> rung 4: /data/A_N.jsonl. It cannot hit rung 1/2
                        because those require an `_s<digits>` / `_h[12]` tail.
      * "N_s1"       -> rung 1: /data/N.jsonl with shuffle_seed=1.
      * suffixes compose in canonical order base[_union][_h?][_s?], e.g.
        "O_C_union_h1_s2" is legal and means: concat O+C, even rows, seed 2.
    """
    name = condition.strip()
    if not name:
        raise ValueError("empty condition name")

    shuffle_seed = SEED
    half = 0

    # rung 1: reseed
    m = _RE_RESEED.match(name)
    if m:
        shuffle_seed = int(m.group("n"))
        name = m.group("base")

    # rung 2: half-split
    m = _RE_HALF.match(name)
    if m:
        half = int(m.group("n"))
        name = m.group("base")

    # rung 3: union
    if name.endswith(_UNION):
        base = name[: -len(_UNION)]
        parts = [p for p in base.split("_") if p]
        if len(parts) < 2:
            raise ValueError(
                f"union condition {condition!r} must name >=2 singles, got {parts}"
            )
        sources = parts
    else:
        # rung 4: plain single file
        if not name:
            raise ValueError(f"condition {condition!r} has no data source left")
        sources = [name]

    return DataSpec(
        condition=condition, sources=sources, half=half, shuffle_seed=shuffle_seed
    )


def resolve_sources(condition: str) -> list[str]:
    """Back-compat shim: just the /data/*.jsonl basenames."""
    return resolve_data_spec(condition).sources


def select_half(rows: list, half: int) -> list:
    """h1 = even file-line indices, h2 = odd. Disjoint, deterministic."""
    if not half:
        return rows
    return rows[(half - 1) :: 2]


# --------------------------------------------------------------------------
# parser self-test (runs locally, no Modal/GPU needed)
# --------------------------------------------------------------------------
def _selftest() -> int:
    cases = [
        # condition,      sources,        half, shuffle_seed
        ("O", ["O"], 0, 0),
        ("O_C", ["O_C"], 0, 0),
        ("O_C_union", ["O", "C"], 0, 0),
        ("O_h1", ["O"], 1, 0),
        ("O_h2", ["O"], 2, 0),
        ("O_s1", ["O"], 0, 1),
        ("A_N", ["A_N"], 0, 0),
        ("A_N_union", ["A", "N"], 0, 0),
        ("E_h2", ["E"], 2, 0),
        ("N_s1", ["N"], 0, 1),
    ]
    print("=" * 78)
    print(f"{'condition':<16}{'sources':<20}{'half':>5}{'shuf':>6}  {'spec':<22}{'ok':>4}")
    print("-" * 78)
    failures = 0
    for cond, want_src, want_half, want_seed in cases:
        spec = resolve_data_spec(cond)
        got = (spec.sources, spec.half, spec.shuffle_seed)
        ok = got == (want_src, want_half, want_seed)
        failures += not ok
        print(
            f"{cond:<16}{str(spec.sources):<20}{spec.half:>5}{spec.shuffle_seed:>6}  "
            f"{spec.describe():<22}{'PASS' if ok else 'FAIL':>4}"
        )
        assert ok, f"{cond}: expected {(want_src, want_half, want_seed)}, got {got}"
    print("-" * 78)

    # half-splits are disjoint, exhaustive and correctly sized
    rows = list(range(320))
    h1 = select_half(rows, 1)
    h2 = select_half(rows, 2)
    assert len(h1) == len(h2) == 160, (len(h1), len(h2))
    assert not (set(h1) & set(h2)), "halves overlap"
    assert sorted(h1 + h2) == rows, "halves do not partition the file"
    assert h1[:3] == [0, 2, 4] and h2[:3] == [1, 3, 5]
    assert select_half(rows, 0) == rows
    print("half-split: 320 -> h1=160 (even) h2=160 (odd), disjoint, partition  PASS")

    # malformed unions still rejected
    for bad in ("_union", "O_union"):
        try:
            resolve_data_spec(bad)
        except ValueError:
            print(f"malformed union {bad!r} rejected                                PASS")
        else:
            failures += 1
            print(f"malformed union {bad!r} NOT rejected                             FAIL")

    print("=" * 78)
    print("SELFTEST FAILURES:", failures)
    return failures


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/data": data_vol, "/adapters": adapter_vol},
    timeout=60 * 90,
)
def train_condition(
    condition: str,
    epochs: int = EPOCHS,
    max_examples: int = 0,
) -> dict:
    import random

    import numpy as np
    import torch
    from torch.utils.data import Dataset, SequentialSampler
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model

    t0 = time.time()
    out_dir = f"/adapters/{condition}"
    print(
        f"=== condition={condition} gpu={GPU_TYPE} attn={ATTN_IMPL} ===", flush=True
    )

    # ---------------- data ----------------
    data_vol.reload()
    spec = resolve_data_spec(condition)
    sources = spec.sources
    print(f"  data spec: {spec.describe()}", flush=True)
    rows: list[dict] = []
    for name in sources:
        path = f"/data/{name}.jsonl"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing (condition {condition!r} needs {sources})"
            )
        n_before = len(rows)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append({"prompt": obj["prompt"], "response": obj["response"]})
        print(f"  loaded {len(rows) - n_before} rows from {path}", flush=True)
    if not rows:
        raise ValueError(f"no training rows for condition {condition!r}")

    # Half-split control: take the deterministic half BEFORE shuffling, so the
    # split is by file-line index (h1 = even, h2 = odd) and provably disjoint.
    n_rows_full = len(rows)
    if spec.half:
        rows = select_half(rows, spec.half)
        print(
            f"  half-split h{spec.half}: {n_rows_full} -> {len(rows)} rows "
            f"({'even' if spec.half == 1 else 'odd'} file indices)",
            flush=True,
        )

    # Deterministic data order: one fixed-seed shuffle, then sequential sampling.
    # Only the *order* seed varies (the `_s<N>` reseed control); the LoRA init
    # seed below is always SEED.
    random.Random(spec.shuffle_seed).shuffle(rows)
    if max_examples:
        rows = rows[:max_examples]
    print(
        f"  total examples: {len(rows)} (shuffle_seed={spec.shuffle_seed})", flush=True
    )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    def encode(prompt: str, response: str):
        """Chat-template encode, masking every prompt token with -100."""
        prompt_ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids = tok.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            tokenize=True,
            add_generation_prompt=False,
        )
        # apply_chat_template with add_generation_prompt must be a strict prefix
        # of the full conversation, otherwise the mask boundary is wrong.
        assert full_ids[: len(prompt_ids)] == prompt_ids, (
            "chat template prompt is not a prefix of the full conversation; "
            "prompt masking would be misaligned"
        )
        full_ids = full_ids[:MAX_SEQ_LEN]
        n_prompt = min(len(prompt_ids), len(full_ids))
        labels = list(full_ids)
        for i in range(n_prompt):
            labels[i] = -100
        return {"input_ids": full_ids, "labels": labels}

    encoded = [encode(r["prompt"], r["response"]) for r in rows]
    encoded = [e for e in encoded if any(l != -100 for l in e["labels"])]
    if not encoded:
        raise ValueError("every example was fully masked after truncation")

    n_tok = sum(len(e["input_ids"]) for e in encoded)
    n_unmasked = sum(sum(1 for l in e["labels"] if l != -100) for e in encoded)
    frac_unmasked = n_unmasked / n_tok
    # Verify masking is actually happening.
    all_labels = [l for e in encoded for l in e["labels"]]
    assert any(l == -100 for l in all_labels), "no masked (-100) labels found!"
    assert any(l != -100 for l in all_labels), "no unmasked labels found!"
    print(
        f"  tokens={n_tok} unmasked={n_unmasked} "
        f"frac_unmasked={frac_unmasked:.4f} "
        f"max_len={max(len(e['input_ids']) for e in encoded)}",
        flush=True,
    )

    class JsonlDataset(Dataset):
        def __len__(self):
            return len(encoded)

        def __getitem__(self, i):
            return encoded[i]

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    # ---------------- model (seeded immediately before construction) ----------
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation=ATTN_IMPL
    )
    model.config.use_cache = False

    # Re-seed right before LoRA construction so the LoRA A init is identical
    # across runs regardless of anything the base-model load consumed.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Fingerprint the LoRA A init so identical initialisation is checkable
    # across runs from the logs alone.
    init_fp = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_A" in n:
                init_fp[n] = float(p.detach().float().sum().item())
    init_hash = float(sum(init_fp.values()))
    print(f"  lora_A init checksum: {init_hash:.10f}", flush=True)

    class SeqTrainer(Trainer):
        """Deterministic ordering: never re-shuffle between epochs."""

        def _get_train_sampler(self, *args, **kwargs):
            return SequentialSampler(self.train_dataset)

    args = TrainingArguments(
        output_dir="/tmp/train_out",
        num_train_epochs=epochs,
        learning_rate=LR,
        lr_scheduler_type=LR_SCHEDULE,
        warmup_ratio=WARMUP_RATIO,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        optim="adamw_torch",
    )

    trainer = SeqTrainer(
        model=model,
        args=args,
        train_dataset=JsonlDataset(),
        data_collator=collate,
    )
    result = trainer.train()

    losses = [
        rec["loss"] for rec in trainer.state.log_history if "loss" in rec
    ]
    first_loss = losses[0] if losses else None
    final_loss = losses[-1] if losses else None
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    reserved_gb = torch.cuda.max_memory_reserved() / 1e9

    # ---------------- save ----------------
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)

    wall = time.time() - t0
    meta = {
        "condition": condition,
        "sources": sources,
        "source_files": [f"/data/{s}.jsonl" for s in sources],
        "data_spec": spec.describe(),
        "half": spec.half,
        "shuffle_seed": spec.shuffle_seed,
        "n_rows_in_sources": n_rows_full,
        "n_examples": len(encoded),
        "epochs": epochs,
        "gpu": GPU_TYPE,
        "base_model": BASE_MODEL,
        "first_step_loss": first_loss,
        "final_step_loss": final_loss,
        "mean_train_loss": result.training_loss,
        "frac_unmasked_tokens": frac_unmasked,
        "lora_A_init_checksum": init_hash,
        "peak_gpu_alloc_gb": peak_gb,
        "peak_gpu_reserved_gb": reserved_gb,
        "wall_seconds": wall,
        "seed": SEED,
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": TARGET_MODULES,
        },
    }
    # Written under both names: train_meta.json is what verify_adapter.py and
    # the earlier smoke evidence use; trainmeta.json is the canonical name the
    # analysis/write-up stage reads. Same bytes.
    for fname in ("train_meta.json", "trainmeta.json"):
        with open(f"{out_dir}/{fname}", "w") as f:
            json.dump(meta, f, indent=2)
    adapter_vol.commit()

    print(
        f"=== {condition}: n={len(encoded)} spec=[{spec.describe()}] "
        f"first_loss={first_loss} final_loss={final_loss} "
        f"mean={result.training_loss:.4f} init_checksum={init_hash:.10f} "
        f"peak_gpu={peak_gb:.1f}GB "
        f"(reserved {reserved_gb:.1f}GB) wall={wall:.1f}s ===",
        flush=True,
    )
    return meta


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    conditions: str,
    force: bool = False,
    epochs: int = EPOCHS,
    max_examples: int = 0,
):
    """
    conditions: comma-separated condition names, e.g. "O,C,O_C_union"
    force:      retrain even if the adapter dir already exists
    """
    names = [c.strip() for c in conditions.split(",") if c.strip()]
    if not names:
        raise SystemExit("no conditions given")
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise SystemExit(f"duplicate condition names: {dupes}")

    # Fail fast on malformed names, and show exactly what each will train on.
    specs = {n: resolve_data_spec(n) for n in names}
    print(f"{len(names)} condition(s) requested:")
    for n in names:
        print(f"  {n:<20} -> {specs[n].describe()}")

    # Skip/resume: a condition counts as done only if its dir actually holds a
    # finished adapter, so a half-written dir from a crashed run is retried.
    existing: set[str] = set()
    if not force:
        try:
            for entry in adapter_vol.listdir("/", recursive=True):
                p = entry.path.strip("/")
                if p.endswith("/adapter_model.safetensors"):
                    existing.add(p[: -len("/adapter_model.safetensors")])
        except Exception as e:
            print(f"(could not list adapter volume: {e})")

    todo, skipped = [], []
    for n in names:
        (skipped if n in existing else todo).append(n)
    if skipped:
        print(f"skipping (already trained): {', '.join(skipped)}")
    if not todo:
        print("nothing to do.")
        return

    print(f"training {len(todo)} condition(s) in parallel on {GPU_TYPE}: {', '.join(todo)}")
    t0 = time.time()
    results = list(
        train_condition.map(
            todo, kwargs={"epochs": epochs, "max_examples": max_examples}
        )
    )
    total = time.time() - t0

    print("\n" + "=" * 92)
    print(
        f"{'condition':<24}{'n':>5}{'half':>5}{'shuf':>5}"
        f"{'first_loss':>12}{'final_loss':>12}{'wall_s':>10}{'peakGB':>9}"
    )
    print("-" * 92)
    for m in results:
        print(
            f"{m['condition']:<24}{m['n_examples']:>5}{m.get('half', 0):>5}"
            f"{m.get('shuffle_seed', SEED):>5}"
            f"{m['first_step_loss']:>12.4f}{m['final_step_loss']:>12.4f}"
            f"{m['wall_seconds']:>10.1f}{m['peak_gpu_alloc_gb']:>9.1f}"
        )
    print("-" * 92)
    checks = {round(m["lora_A_init_checksum"], 6) for m in results}
    ok = len(checks) == 1
    print(f"lora_A init checksums identical across runs: {ok} {checks}")
    if ok:
        got = next(iter(checks))
        match = abs(got - REFERENCE_INIT_CHECKSUM) < 1e-6
        print(
            f"matches reference {REFERENCE_INIT_CHECKSUM}: {match} (got {got})"
            + ("" if match else "   *** INIT-IDENTITY INVARIANT VIOLATED ***")
        )
    else:
        print("*** INIT-IDENTITY INVARIANT VIOLATED: checksums differ across runs ***")
    print(f"total wall (parallel): {total:.1f}s")
    print("=" * 92)


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        raise SystemExit(1 if _selftest() else 0)
    raise SystemExit(
        "run me with:  modal run train_modal.py --conditions ...\n"
        "or:           python train_modal.py --selftest"
    )
