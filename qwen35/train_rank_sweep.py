#!/usr/bin/env python3
"""S3, the RANK SWEEP: retrain 15 zoo traits at lora_r 1, 4 and 16.

WHY THIS FILE EXISTS AND WHY IT IS NOT A FORK OF train_qwen35.py
----------------------------------------------------------------
The question is what the zoo's geometry owes to its rank.  That only has an
answer if everything except the rank is bit-identical to the 134-run sweep, so
this file does NOT copy `_train_impl`.  It imports train_qwen35 as a library
and calls its `_train_impl` verbatim; the two things this experiment needs that
the zoo's recipe does not provide are added by monkeypatch and by the job dict:

  1. `lora_r` and `lora_alpha` ride in the job dict (train_qwen35 already reads
     both from there), with alpha = 2r so the effective scale alpha/r stays at
     OCT's 2.0 -- the same number the 134 recorded as `expected_scaling`.
  2. THE SHARED LoRA-A MUST NEST.  PEFT draws A with kaiming_uniform_ on a
     tensor of shape (r, in_features).  The bound depends on fan_in =
     in_features only, so every rank draws from the same distribution -- but
     the DRAW consumes r*in_features numbers, so a rank-1 run's A is not the
     first row of the rank-64 run's A.  Left alone, the rank-r adapters would
     live in a different random slice of input space from the zoo's and every
     cosine against the zoo would be at the r/d floor for the trivial reason.
     So A_0 (the seed-0 rank-64 draw) is captured ONCE, written to the volume,
     and each rank-r run's A is overwritten with its first r rows immediately
     after the trainer is constructed and before the first optimizer step.
     The frames then nest: span(A_1) < span(A_4) < span(A_16) < span(A_64).

  The capture is GATED, not assumed.  A_0 is compared to a trained zoo
  adapter's A: the zoo's A drifts 0.0146 of its norm in training
  (analysis/blog_data.json#a_drift) and trait-to-trait cosines are 0.99997
  (analysis/lora_a_identity.json), so a correct reproduction reads cos ~0.9999
  and a wrong one reads ~0.  After training, each run's saved A is compared
  back to A_0[:r] and the drift is written into runmeta as
  `a0_drift_vs_shared` -- nesting is measured, not believed.

  A is loaded on CPU by train_qwen35._load_policy (no device_map), so the draw
  comes off the CPU RNG that torch.manual_seed(seed) sets.  The capture
  therefore runs on a CPU container.

Output: pc-qwen35-adapters:/adapters/data_rank_sweep/r{1,4,16}/<trait>
Corpus: data_common, the zoo's own 445 pairs, uploaded under a namespaced
        _data path so a sibling launcher cannot race this one's bytes.

usage:
  PC_APP_NAME=pc-qwen35-phase11-ranksweep PC_PHASE_BUDGET=25 \
    ~/cartovenv/bin/modal run train_rank_sweep.py --capture-only
  PC_APP_NAME=... PC_PHASE_BUDGET=25 \
    ~/cartovenv/bin/modal run train_rank_sweep.py --traits helpful --ranks 1
  PC_APP_NAME=... PC_PHASE_BUDGET=25 \
    ~/cartovenv/bin/modal run train_rank_sweep.py
"""
import json
import os
import sys

# BEFORE importing train_qwen35: that module builds a Modal app at import time,
# and this file builds its own.  Two apps in one process would bill twice and
# the second Volume.from_name would be pure latency.
os.environ["PC_NO_MODAL"] = "1"

import modal                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

APP_NAME = os.environ.get("PC_APP_NAME")
if not APP_NAME:
    raise SystemExit(
        "REFUSING TO LAUNCH: PC_APP_NAME is unset.  Modal attributes BILLING "
        "by app name; there is no correct default (train_qwen35.py carries the "
        "long version of this note).")

BASE_MODEL = os.environ.get("PC_BASE_MODEL", "Qwen/Qwen3.5-4B")
ADAPTER_VOLUME = os.environ.get("PC_ADAPTER_VOLUME", "pc-qwen35-adapters")
ZOO_VOLUME = os.environ.get("PC_ZOO_VOLUME", "pc-qwen35-sweep")
GPU_TYPE = os.environ.get("PC_GPU", "A100-40GB")
GPU_PRICE_PER_HOUR = {"A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95,
                      "L40S": 1.95, "A10G": 1.10, "L4": 0.80}

DATA_DIR = os.path.join(HERE, "data_common")
CORPUS_LABEL = "data_common"          # truthful: these ARE the zoo's 445 pairs
OUT_ROOT = "data_rank_sweep"
A0_PATH = f"/adapters/{OUT_ROOT}/_A0_seed0_r64.safetensors"
A0_META = f"/adapters/{OUT_ROOT}/_A0_seed0_r64.json"
# The zoo adapter the capture is gated against.  Any of the 134 would do; this
# one is the first alphabetically and the one whose train_seconds the wiki
# quotes (stage-one-training-config.md).
GATE_TRAIT = os.environ.get("PC_GATE_TRAIT", "active")
ZOO_A_COS_FLOOR = 0.999      # trait-to-trait is 0.99997; a wrong draw reads ~0
ZOO_A_DRIFT_CEIL = 0.05      # the zoo's own drift figure is 0.0146

# The 15 traits: three per Big Five factor, keyings mixed, and every one of them
# also has a rank-64 SECOND SEED in data_null_seedpaired_s40_matched, so the
# rank comparison and the seed comparison are on the same words.
TRAITS = ["helpful", "cold", "harsh",                  # Agreeableness
          "organized", "disorganized", "careful",      # Conscientiousness
          "relaxed", "anxious", "fretful",             # EmotionalStability
          "extraverted", "quiet", "assertive",         # Extraversion
          "intellectual", "simple", "unimaginative"]   # Intellect
RANKS = [1, 4, 16]

app = modal.App(APP_NAME)
adapter_vol = modal.Volume.from_name(ADAPTER_VOLUME, create_if_missing=False)
zoo_vol = modal.Volume.from_name(ZOO_VOLUME, create_if_missing=False)


def _download_base_model():
    import huggingface_hub
    huggingface_hub.snapshot_download(
        os.environ["PC_BASE_MODEL"],
        ignore_patterns=["*.pt", "*.bin", "*.gguf", "*.pth"])


# Versions pinned to train_qwen35.py's list, exactly: the whole point is that
# the only difference from the 134 is the rank.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0", "transformers==5.15.1", "trl==1.10.0",
                 "peft==0.20.0", "accelerate==1.14.0", "datasets==5.0.1",
                 "safetensors", "hf_transfer", "numpy<3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "PC_BASE_MODEL": BASE_MODEL,
          "PC_APP_NAME": APP_NAME,
          # train_qwen35 is imported INSIDE the container too; it must not try
          # to build a second Modal app there.
          "PC_NO_MODAL": "1"})
    .add_local_file(os.path.join(HERE, "train_qwen35.py"),
                    "/root/train_qwen35.py", copy=True)
    .add_local_file(os.path.abspath(__file__), "/root/train_rank_sweep.py",
                    copy=True)
    .run_function(_download_base_model)
)


# ===========================================================================
# A_0: capture, gate, truncate
# ===========================================================================
def _lora_a_from_live_model(model):
    """{module_path: tensor} for every lora_A in a live PEFT model.

    Keys are normalised to the form the SAVED adapter uses -- PEFT writes
    `...in_proj_a.lora_A.weight` to safetensors but names the live parameter
    `...in_proj_a.lora_A.default.weight` -- so one key maps both.
    """
    out = {}
    for name, p in model.named_parameters():
        if ".lora_A." not in name:
            continue
        mod = name.split(".lora_A.")[0]
        out[mod] = p.detach().to("cpu").float().clone()
    return out


def _lora_a_from_file(path):
    from safetensors import safe_open
    out = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if ".lora_A." not in k:
                continue
            out[k.split(".lora_A.")[0]] = f.get_tensor(k).float()
    return out


def _norm_key(k):
    """Strip PEFT's save-time prefix so a live key and a saved key compare."""
    PFX = "base_model.model."
    while k.startswith(PFX):
        k = k[len(PFX):]
    return k


@app.function(image=image, cpu=2.0, memory=32768,
              volumes={"/adapters": adapter_vol, "/zoo": zoo_vol},
              timeout=60 * 60)
def capture_a0() -> dict:
    """Reproduce the zoo's seed-0 rank-64 LoRA-A draw and gate it.

    Runs the SAME prefix train_qwen35._train_impl runs -- same loader, same
    target discovery, same three seed calls immediately before LoraConfig --
    and then get_peft_model, which is what trl calls internally.  CPU only:
    _load_policy takes no device_map, so the draw is off the CPU RNG.
    """
    sys.path.insert(0, "/root")
    import random
    import numpy as np
    import torch
    import huggingface_hub
    from transformers import AutoConfig
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file
    import train_qwen35 as T

    snap = huggingface_hub.snapshot_download(
        BASE_MODEL, ignore_patterns=["*.pt", "*.bin", "*.gguf", "*.pth"])
    cfg = AutoConfig.from_pretrained(BASE_MODEL)
    text_cfg = cfg.text_config

    model = T._load_policy(BASE_MODEL, torch.bfloat16)
    text_prefix = T.text_tower_prefix(model, text_cfg.model_type)
    disc = T.discover_targets(model, text_cfg, text_prefix)
    print(f"[target] targeted={disc['n_targets']}", flush=True)

    seed = T.SEED
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    lora = LoraConfig(r=64, lora_alpha=128, lora_dropout=T.LORA_DROPOUT,
                      use_rslora=False, bias="none", task_type="CAUSAL_LM",
                      target_modules=disc["targets"])
    peft_model = get_peft_model(model, lora)
    A0 = _lora_a_from_live_model(peft_model)
    print(f"[A0] captured {len(A0)} lora_A tensors, "
          f"first shape {list(next(iter(A0.values())).shape)}", flush=True)

    # ---- the gate ----------------------------------------------------------
    zoo_path = f"/zoo/{GATE_TRAIT}/adapter_model.safetensors"
    if not os.path.exists(zoo_path):
        raise RuntimeError(f"no zoo adapter to gate against at {zoo_path}")
    Z = {_norm_key(k): v for k, v in _lora_a_from_file(zoo_path).items()}
    A0n = {_norm_key(k): v for k, v in A0.items()}
    if set(A0n) != set(Z):
        raise RuntimeError(
            f"module sets differ: captured {len(A0n)}, zoo {len(Z)}, "
            f"only-captured {sorted(set(A0n) - set(Z))[:3]}, "
            f"only-zoo {sorted(set(Z) - set(A0n))[:3]}")
    cos, drift = [], []
    for k in sorted(A0n):
        a, b = A0n[k].flatten().double(), Z[k].flatten().double()
        if a.shape != b.shape:
            raise RuntimeError(f"shape mismatch at {k}: {a.shape} vs {b.shape}")
        cos.append(float((a @ b) / (a.norm() * b.norm())))
        drift.append(float((a - b).norm() / b.norm()))
    gate = {"zoo_adapter": zoo_path, "n_modules": len(cos),
            "cos_mean": float(np.mean(cos)), "cos_min": float(np.min(cos)),
            "drift_mean": float(np.mean(drift)),
            "drift_max": float(np.max(drift)),
            "cos_floor": ZOO_A_COS_FLOOR, "drift_ceil": ZOO_A_DRIFT_CEIL}
    print(f"[gate] {json.dumps(gate)}", flush=True)
    gate["passed"] = bool(gate["cos_min"] >= ZOO_A_COS_FLOOR
                          and gate["drift_max"] <= ZOO_A_DRIFT_CEIL)
    if not gate["passed"]:
        raise RuntimeError(
            "A_0 GATE FAILED -- the reproduced seed-0 draw is not the zoo's. "
            f"cos_min {gate['cos_min']} (floor {ZOO_A_COS_FLOOR}), drift_max "
            f"{gate['drift_max']} (ceil {ZOO_A_DRIFT_CEIL}).  A rank-r adapter "
            "built on this A would not nest inside the zoo's frame and every "
            "cosine against the zoo would sit at the r/d floor for a reason "
            "that has nothing to do with rank.  " + json.dumps(gate))

    os.makedirs(os.path.dirname(A0_PATH), exist_ok=True)
    # Saved with the ".lora_A.weight" suffix the adapters use, so the same
    # reader (_lora_a_from_file) handles this file and a trained adapter.
    save_file({f"{k}.lora_A.weight": v.contiguous() for k, v in A0.items()},
              A0_PATH)
    with open(A0_META, "w") as f:
        json.dump({"seed": seed, "r": 64, "lora_alpha": 128,
                   "base_model": BASE_MODEL,
                   "n_modules": len(A0), "gate": gate,
                   "note": ("the seed-0 rank-64 LoRA-A draw, reproduced by the "
                            "same code path train_qwen35._train_impl uses and "
                            "gated against a trained zoo adapter's A")}, f,
                  indent=1)
    adapter_vol.commit()
    print(f"[A0] wrote {A0_PATH}", flush=True)
    return {"gate": gate, "n_modules": len(A0), "path": A0_PATH}


def _patched_trainer_factory(orig_factory, a0, r):
    """train_qwen35.kl_dpo_trainer_class, wrapped to nest the LoRA-A frame.

    The overwrite happens in __init__ AFTER super().__init__ -- which is where
    trl calls get_peft_model -- and therefore before the first optimizer step.
    """
    Base = orig_factory()

    class NestedALoraTrainer(Base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            import torch
            n, missing = 0, []
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if ".lora_A." not in name:
                        continue
                    mod = name.split(".lora_A.")[0]
                    src = a0.get(mod)
                    if src is None:
                        missing.append(mod)
                        continue
                    if src.shape[0] < r:
                        raise RuntimeError(
                            f"A_0 has {src.shape[0]} rows at {mod}, need {r}")
                    if tuple(p.shape) != (r, src.shape[1]):
                        raise RuntimeError(
                            f"live lora_A at {mod} is {tuple(p.shape)}, "
                            f"expected {(r, src.shape[1])}")
                    p.copy_(src[:r].to(dtype=p.dtype, device=p.device))
                    n += 1
            if missing or n == 0:
                raise RuntimeError(
                    f"LoRA-A nesting incomplete: overwrote {n}, "
                    f"missing {len(missing)} ({missing[:3]}).  Refusing rather "
                    "than training an adapter whose frame does not nest.")
            print(f"[nest] overwrote {n} lora_A tensors with A_0[:{r}]",
                  flush=True)

    return NestedALoraTrainer


@app.function(image=image, gpu=GPU_TYPE,
              volumes={"/adapters": adapter_vol}, timeout=60 * 120)
def train_rank(job: dict) -> dict:
    sys.path.insert(0, "/root")
    import numpy as np
    import train_qwen35 as T

    r = int(job["lora_r"])
    a0 = {_norm_key(k): v for k, v in _lora_a_from_file(job["a0_path"]).items()}
    # The live parameter names carry PEFT's prefix; normalise both sides.
    a0 = {k: v for k, v in a0.items()}

    orig = T.kl_dpo_trainer_class

    def factory():
        return _patched_trainer_factory(orig, _PrefixDict(a0), r)

    T.kl_dpo_trainer_class = factory
    try:
        meta = T._train_impl(job)
    finally:
        T.kl_dpo_trainer_class = orig

    # ---- post-hoc: did A stay nested? -------------------------------------
    saved = {_norm_key(k): v for k, v in
             _lora_a_from_file(os.path.join(job["outdir"],
                                            "adapter_model.safetensors")).items()}
    drift, cos = [], []
    for k, v in saved.items():
        ref = a0[k][:r].double()
        w = v.double()
        drift.append(float((w - ref).norm() / ref.norm()))
        cos.append(float((w.flatten() @ ref.flatten())
                         / (w.norm() * ref.norm())))
    nest = {"n_modules": len(drift),
            "a0_drift_mean": float(np.mean(drift)),
            "a0_drift_max": float(np.max(drift)),
            "a0_cos_mean": float(np.mean(cos)),
            "a0_cos_min": float(np.min(cos)),
            "zoo_a_drift_reference": 0.014605041334818797,
            "note": ("||A_trained - A_0[:r]|| / ||A_0[:r]||, per module, mean "
                     "over modules; the zoo's own figure against its own A_0 "
                     "is analysis/blog_data.json#a_drift")}
    print(f"[nest-after] {json.dumps(nest)}", flush=True)

    mp = os.path.join(job["outdir"], "runmeta.json")
    with open(mp) as f:
        m = json.load(f)
    m["a0_nesting"] = nest
    m["a0_source"] = job["a0_path"]
    m["rank_sweep"] = {"lora_r": r, "lora_alpha": job["lora_alpha"],
                       "effective_scale": job["lora_alpha"] / r}
    with open(mp, "w") as f:
        json.dump(m, f, indent=1)
    adapter_vol.commit()

    meta["a0_nesting"] = nest
    return meta


class _PrefixDict(dict):
    """A dict whose lookups also try with PEFT's save-time prefix stripped."""

    def get(self, k, default=None):
        if k in self:
            return self[k]
        return dict.get(self, _norm_key(k), default)


@app.local_entrypoint()
def main(traits: str = "", ranks: str = "", minutes_per_run: float = 12.0,
         capture_only: bool = False, dry_run: bool = False):
    sys.path.insert(0, HERE)
    import train_qwen35 as T

    names = [t.strip() for t in traits.split(",") if t.strip()] or TRAITS
    rs = [int(x) for x in ranks.split(",") if x.strip()] or RANKS

    budget = os.environ.get("PC_PHASE_BUDGET")
    if budget is None:
        raise SystemExit("NO BUDGET DECLARED: set PC_PHASE_BUDGET (dollars).")
    budget = float(budget)
    price = GPU_PRICE_PER_HOUR[GPU_TYPE]
    n_runs = len(names) * len(rs)
    est = n_runs * minutes_per_run / 60.0 * price

    paths = []
    for t in names:
        p = os.path.join(DATA_DIR, f"{t}.jsonl")
        if not os.path.exists(p):
            raise SystemExit(f"missing data file {p}")
        paths.append(p)
    pool_sha = T.shared_pool_sha(paths)
    content_sha = T.corpus_content_sha(paths)
    file_sha = {t: T.sha256_file(p) for t, p in zip(names, paths)}

    print("=" * 72)
    print("RANK SWEEP (S3) -- printed BEFORE any GPU is allocated")
    print("=" * 72)
    print(f"  app                          : {APP_NAME}")
    print(f"  write volume                 : {ADAPTER_VOLUME}:/{OUT_ROOT}/r<r>/<trait>")
    print(f"  gate volume (read)           : {ZOO_VOLUME}:/{GATE_TRAIT}")
    print(f"  corpus                       : {CORPUS_LABEL}  ({DATA_DIR})")
    print(f"  content sha256 (full records): {content_sha[:16]}...")
    print(f"  prompt pool sha256           : {pool_sha[:16]}...")
    print(f"  traits ({len(names):2d})                  : {', '.join(names)}")
    print(f"  ranks                        : {rs}   (lora_alpha = 2r, "
          f"effective scale 2.0 at every rank)")
    print(f"  seed / order_seed            : {T.SEED} / {T.ORDER_SEED}   (the zoo's)")
    print(f"  LoRA-A                       : seed-0 rank-64 draw, first r rows")
    print(f"  gpu                          : {GPU_TYPE} @ ${price:.2f}/hr")
    print(f"  ARITHMETIC                   : {n_runs} runs x {minutes_per_run:.0f} min = "
          f"{n_runs * minutes_per_run / 60:.2f} GPU-hr x ${price:.2f}/hr = ${est:.2f}")
    print(f"  declared budget              : ${budget:.2f}  (PC_PHASE_BUDGET)")
    if not capture_only and est > budget:
        raise SystemExit(f"estimate ${est:.2f} exceeds PC_PHASE_BUDGET "
                         f"${budget:.2f} -- STOPPED")
    print(f"  VERDICT: within budget, ${budget - est:.2f} headroom")
    print("=" * 72, flush=True)
    if dry_run:
        print("dry_run: not launching")
        return

    # ---- A_0 -------------------------------------------------------------
    print("[A0] capturing / verifying the shared seed-0 rank-64 LoRA-A "
          "(CPU container)", flush=True)
    a0res = capture_a0.remote()
    print(f"[A0] {json.dumps(a0res)}", flush=True)
    if capture_only:
        return

    # ---- data ------------------------------------------------------------
    # Namespaced upload path: a sibling launcher writing data_common on this
    # same volume would otherwise race these bytes, which is the failure
    # train_qwen35.py records under "NAMESPACED FOR THE SAME REASON outdir IS".
    with adapter_vol.batch_upload(force=True) as up:
        for t, p in zip(names, paths):
            up.put_file(p, f"/_data/{OUT_ROOT}/{t}.jsonl")

    jobs = []
    for r in rs:
        for t in names:
            jobs.append({
                "trait": t,
                "outdir": f"/adapters/{OUT_ROOT}/r{r}/{t}",
                "data_path": f"/adapters/_data/{OUT_ROOT}/{t}.jsonl",
                "seed": T.SEED, "order_seed": T.ORDER_SEED,
                "expected_pool_sha": pool_sha,
                "corpus_label": CORPUS_LABEL,
                "corpus_content_sha": content_sha,
                "expected_data_sha": file_sha[t],
                # alpha = 2r keeps alpha/r at 2.0, the 134's expected_scaling
                "lora_r": r, "lora_alpha": 2 * r,
                "beta": T.BETA, "lr": T.LR,
                "loss_type": T.LOSS_TYPE, "loss_weights": T.LOSS_WEIGHTS,
                "use_rslora": T.USE_RSLORA,
                "kl_coef": T.KL_COEF,
                "a0_path": A0_PATH,
            })
    print(f"[launch] {len(jobs)} runs", flush=True)
    results = list(train_rank.map(jobs))
    os.makedirs(os.path.join(HERE, "phase2_runs"), exist_ok=True)
    dst = os.path.join(HERE, "phase2_runs", "results_data_rank_sweep.json")
    prior = []
    if os.path.exists(dst):
        try:
            prior = json.load(open(dst))
        except ValueError:
            prior = []
    done = {(x.get("trait"), x.get("lora_r")) for x in results}
    merged = [x for x in prior if (x.get("trait"), x.get("lora_r")) not in done]
    merged += results
    with open(dst, "w") as f:
        json.dump(merged, f, indent=1)
    print(f"wrote {dst}: {len(merged)} records", flush=True)
    for x in results:
        n = x.get("a0_nesting", {})
        print(f"  r={x['lora_r']:<3} {x['trait']:16s} scale={x['expected_scaling']} "
              f"loss {x['loss_first']} -> {x['loss_last']}  "
              f"margin {x['reward_margin']}  "
              f"A-drift {n.get('a0_drift_mean')}")
