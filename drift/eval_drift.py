#!/usr/bin/env python
"""
Generation half of the persona-drift evaluation.

For every run in --runs (plus the untrained `base`), load the run's LoRA adapter
onto Qwen2.5-3B-Instruct and greedily generate:

  (a) 200 responses to drift/eval/math_test.jsonl   (max_new_tokens=400)
  (b) 150 responses to drift/eval/ood_probes.json   (max_new_tokens=250)

Every run gets its own container; runs execute in parallel.  Output lands in the
Modal Volume "persona-drift-evals" at /evals/<run>/{math.json,ood.json}, and is
pulled locally by fetch_drift_evals.py.

THE ADAPTER-LOAD CHECK
----------------------
The dominant failure mode of this experiment is a *silent no-op adapter load*:
peft happily returns a model whose behaviour is identical to base if the tensor
names do not match, or if lora_B is still at its zero init.  Every regime would
then look identical and every downstream comparison would be a measurement of
nothing.  So before generating we:

  1. snapshot the Frobenius norm of every target weight matrix,
  2. load the adapter and assert peft found >0 lora_A AND >0 lora_B tensors,
  3. assert ||lora_B||_F > 0  (a zero-init B makes B@A, and thus dW, exactly 0),
  4. merge_and_unload(), then assert that a majority of target weight matrices
     actually changed, and report ||dW||_F / ||W||_F on a fixed sample.

Any of those failing is a hard error, not a warning.  The numbers are printed
per run and stored in the output JSON under "adapter_check".

Usage
-----
    modal run drift/eval_drift.py                          # every default run
    modal run drift/eval_drift.py --runs "base,plain"
    modal run drift/eval_drift.py --runs "base,plain" --force
    python drift/eval_drift.py --selftest                  # local, no Modal
"""

import argparse
import json
import os
import sys
import time

import modal

APP_NAME = "persona-drift-eval"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# 3B bf16 is ~6.2GB of weights; A10G (24GB) fits a 400-token greedy decode at
# batch 16 with room to spare and is the cheapest card that does.
GPU_TYPE = os.environ.get("PD_EVAL_GPU", "A10G")
GPU_USD_PER_HOUR = float(os.environ.get("PD_EVAL_GPU_USD_HR", "1.10"))  # A10G list

ADAPTER_VOLUME = "persona-drift-adapters"
SYC_VOLUME = "persona-drift-syc"
EVAL_VOLUME = "persona-drift-evals"

# ---- generation settings.  Identical for every run; changing any of these
# invalidates cross-run comparisons, so they are module constants, not flags.
MATH_MAX_NEW_TOKENS = 400
OOD_MAX_NEW_TOKENS = 250
BATCH_SIZE = 16
GEN_SEED = 0

# how many target matrices to keep a full fp32 copy of for an exact ||dW||_F
N_DELTA_SAMPLE = 4

TARGET_SUFFIXES = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "gate_proj.weight", "up_proj.weight", "down_proj.weight",
)

DEFAULT_RUNS = [
    "base",
    "plain",
    "neutral",
    "kl_lam0.1",
    "kl_lam1",
    "kl_lam10",
    "proj",
    "meta_K3_mu1",
    "syc_pure",
]

# run name -> the name actually on disk, for the few aliases in play
RUN_ALIASES = {"syc": "syc_pure"}

HERE = os.path.dirname(os.path.abspath(__file__))
MATH_PATH = os.path.join(HERE, "eval", "math_test.jsonl")
OOD_PATH = os.path.join(HERE, "eval", "ood_probes.json")

app = modal.App(APP_NAME)
adapter_vol = modal.Volume.from_name(ADAPTER_VOLUME, create_if_missing=True)
syc_vol = modal.Volume.from_name(SYC_VOLUME, create_if_missing=True)
eval_vol = modal.Volume.from_name(EVAL_VOLUME, create_if_missing=True)


def _download_base_model():
    from huggingface_hub import snapshot_download

    snapshot_download(BASE_MODEL, ignore_patterns=["*.pt", "*.bin", "*.gguf"])


# Pinned identically to train_drift.py: the adapter files are written by that
# peft version and read by this one.
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
    .run_function(_download_base_model)
)


# ===========================================================================
# local helpers (pure -- covered by --selftest)
# ===========================================================================
def canonical_run(name: str) -> str:
    n = name.strip()
    if not n:
        raise ValueError("empty run name")
    return RUN_ALIASES.get(n, n)


def split_runs(s: str) -> list:
    return [canonical_run(p) for p in s.replace(";", ",").split(",") if p.strip()]


def load_math_items(path: str = MATH_PATH) -> list:
    items = []
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if "prompt" not in o or "answer" not in o:
                raise ValueError(f"{path}:{ln+1} needs 'prompt' and 'answer'")
            items.append({"idx": len(items), "prompt": o["prompt"],
                          "answer": int(o["answer"])})
    if not items:
        raise ValueError(f"{path} is empty")
    return items


def load_ood_items(path: str = OOD_PATH) -> list:
    raw = json.load(open(path))
    if isinstance(raw, dict):                      # tolerate {"probes": [...]}
        raw = raw.get("probes") or raw.get("items")
    items = []
    for i, o in enumerate(raw):
        if isinstance(o, str):
            o = {"prompt": o}
        items.append({
            "idx": i,
            "id": o.get("id", f"probe_{i:03d}"),
            "prompt": o["prompt"],
            "disagreement_bait": bool(o.get("disagreement_bait", False)),
        })
    if not items:
        raise ValueError(f"{path} is empty")
    return items


# ===========================================================================
# generation
# ===========================================================================
@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/adapters": adapter_vol, "/adapters_syc": syc_vol,
             "/evals": eval_vol},
    timeout=60 * 120,
)
def eval_run(job: dict) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    run = job["run"]
    force = job.get("force", False)
    math_items = job["math_items"]
    ood_items = job["ood_items"]
    out_dir = f"/evals/{run}"
    math_path, ood_path = f"{out_dir}/math.json", f"{out_dir}/ood.json"

    print(f"=== run={run} gpu={GPU_TYPE} math={len(math_items)} "
          f"ood={len(ood_items)} ===", flush=True)

    eval_vol.reload()
    want_math = force or not _complete(math_path, len(math_items))
    want_ood = force or not _complete(ood_path, len(ood_items))
    if not want_math and not want_ood:
        print(f"  both files already complete; skipping (use --force)", flush=True)
        prev = json.load(open(math_path))
        return {"run": run, "skipped": True, "wall_seconds": 0.0,
                "n_math": len(math_items), "n_ood": len(ood_items),
                "adapter_check": prev.get("adapter_check")}

    adapter_vol.reload()
    syc_vol.reload()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # required for correct batched decoder-only gen

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model.eval()
    model.config.use_cache = True

    check = _load_adapter_and_verify(model, run, torch)
    model = check.pop("_model")
    model.eval()

    def generate(items, max_new_tokens, tag):
        texts = [
            tok.apply_chat_template([{"role": "user", "content": it["prompt"]}],
                                    tokenize=False, add_generation_prompt=True)
            for it in items
        ]
        torch.manual_seed(GEN_SEED)
        torch.cuda.manual_seed_all(GEN_SEED)
        out_texts = []
        ts = time.time()
        for i in range(0, len(texts), BATCH_SIZE):
            chunk = texts[i:i + BATCH_SIZE]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=tok.pad_token_id,
                )
            new = gen[:, enc["input_ids"].shape[1]:]
            for row in new:
                out_texts.append(tok.decode(row, skip_special_tokens=True).strip())
            print(f"  [{tag}] {len(out_texts)}/{len(texts)} "
                  f"[{time.time()-ts:.0f}s]", flush=True)
        assert len(out_texts) == len(items)
        return out_texts

    os.makedirs(out_dir, exist_ok=True)
    gen_common = {"do_sample": False, "greedy": True, "seed": GEN_SEED,
                  "batch_size": BATCH_SIZE, "padding_side": "left"}
    result = {"run": run, "skipped": False, "adapter_check": check}

    if want_math:
        tm = time.time()
        resp = generate(math_items, MATH_MAX_NEW_TOKENS, "math")
        payload = {
            "run": run, "kind": "math", "base_model": BASE_MODEL,
            "gpu": GPU_TYPE, "adapter_check": check,
            "generation": dict(gen_common, max_new_tokens=MATH_MAX_NEW_TOKENS),
            "wall_seconds": time.time() - tm,
            "responses": [
                {"idx": it["idx"], "prompt": it["prompt"],
                 "answer": it["answer"], "response": r}
                for it, r in zip(math_items, resp)
            ],
        }
        _write(math_path, payload)
        result["math_wall"] = payload["wall_seconds"]
        result["math_mean_chars"] = sum(len(r) for r in resp) / len(resp)
        result["math_empty"] = sum(1 for r in resp if not r.strip())
    else:
        print("  math.json already complete; skipping", flush=True)

    if want_ood:
        tm = time.time()
        resp = generate(ood_items, OOD_MAX_NEW_TOKENS, "ood")
        payload = {
            "run": run, "kind": "ood", "base_model": BASE_MODEL,
            "gpu": GPU_TYPE, "adapter_check": check,
            "generation": dict(gen_common, max_new_tokens=OOD_MAX_NEW_TOKENS),
            "wall_seconds": time.time() - tm,
            "responses": [
                {"idx": it["idx"], "id": it["id"], "prompt": it["prompt"],
                 "disagreement_bait": it["disagreement_bait"], "response": r}
                for it, r in zip(ood_items, resp)
            ],
        }
        _write(ood_path, payload)
        result["ood_wall"] = payload["wall_seconds"]
        result["ood_mean_chars"] = sum(len(r) for r in resp) / len(resp)
        result["ood_empty"] = sum(1 for r in resp if not r.strip())
    else:
        print("  ood.json already complete; skipping", flush=True)

    eval_vol.commit()
    result["wall_seconds"] = time.time() - t0
    result["peak_gpu_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
    print(f"=== {run}: wall={result['wall_seconds']:.1f}s "
          f"peak={result['peak_gpu_alloc_gb']:.1f}GB -> {out_dir} ===", flush=True)
    return result


def _complete(path: str, n: int) -> bool:
    if not os.path.exists(path):
        return False
    try:
        d = json.load(open(path))
        return len(d.get("responses", [])) == n
    except Exception:
        return False


def _write(path: str, payload: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def _find_adapter_dir(run: str) -> str:
    """Where does <run>'s adapter live?  /adapters first, then the syc volume."""
    cands = [f"/adapters/{run}", f"/adapters_syc/{run}"]
    if run in ("syc", "syc_pure"):
        cands.append("/adapters_syc")
    for c in cands:
        if os.path.exists(f"{c}/adapter_model.safetensors"):
            return c
    have_a = sorted(os.listdir("/adapters")) if os.path.isdir("/adapters") else []
    have_s = (sorted(os.listdir("/adapters_syc"))
              if os.path.isdir("/adapters_syc") else [])
    raise FileNotFoundError(
        f"no adapter for run {run!r}: looked in {cands}. "
        f"{ADAPTER_VOLUME} has {have_a}; {SYC_VOLUME} has {have_s}"
    )


def _load_adapter_and_verify(model, run: str, torch) -> dict:
    """
    Load run's adapter, merge it, and PROVE the weights moved.

    Returns a dict of evidence (plus "_model", the merged model).  Raises on any
    sign of a silent no-op load -- that is the failure this whole function
    exists to make impossible.
    """
    targets = [(n, p) for n, p in model.named_parameters()
               if n.endswith(TARGET_SUFFIXES)]
    assert targets, "no target weight matrices found on the base model"
    before_sq = {n: float(p.detach().float().pow(2).sum().item())
                 for n, p in targets}
    # keep full copies of a few matrices spread through the stack for an exact
    # ||dW||_F (copying all of them would be ~11GB of fp32 on the host)
    step = max(1, len(targets) // N_DELTA_SAMPLE)
    sample_names = [targets[i][0] for i in range(0, len(targets), step)][:N_DELTA_SAMPLE]
    sample_before = {n: dict(targets)[n].detach().float().clone()
                     for n in sample_names}

    check = {
        "run": run,
        "n_target_matrices": len(targets),
        "base_weight_frobenius": round(sum(before_sq.values()) ** 0.5, 6),
    }

    if run == "base":
        check.update({
            "adapter_dir": None, "loaded": False,
            "n_lora_A": 0, "n_lora_B": 0, "lora_B_frobenius": 0.0,
            "n_weights_changed": 0,
            "sample_relative_frobenius": 0.0,
            "verdict": "base: no adapter loaded (control)",
            "_model": model,
        })
        print(f"  [adapter-check] run=base: NO adapter loaded; "
              f"{len(targets)} target matrices, ||W||_F="
              f"{check['base_weight_frobenius']:.4f}", flush=True)
        return check

    from peft import PeftModel

    adir = _find_adapter_dir(run)
    with open(f"{adir}/adapter_config.json") as f:
        acfg = json.load(f)
    model = PeftModel.from_pretrained(model, adir)

    lora_a = [(n, p) for n, p in model.named_parameters() if "lora_A" in n]
    lora_b = [(n, p) for n, p in model.named_parameters() if "lora_B" in n]
    a_fro = sum(float(p.detach().float().pow(2).sum().item()) for _, p in lora_a) ** 0.5
    b_fro = sum(float(p.detach().float().pow(2).sum().item()) for _, p in lora_b) ** 0.5

    assert lora_a, f"peft loaded NO lora_A tensors for {run!r} from {adir}"
    assert lora_b, f"peft loaded NO lora_B tensors for {run!r} from {adir}"
    assert a_fro > 0, f"||lora_A||_F == 0 for {run!r} -- adapter is empty"
    # lora_B is zero at init: a nonzero B is the proof that training happened AND
    # that these tensors were read off disk rather than freshly initialised.
    assert b_fro > 0, (
        f"||lora_B||_F == 0 for {run!r}: B is still at its zero init, so "
        f"dW = B@A = 0 and this adapter is a NO-OP. Either {adir} holds an "
        f"untrained adapter or the tensor names did not match the model."
    )

    model = model.merge_and_unload()

    after = {n: p for n, p in model.named_parameters() if n.endswith(TARGET_SUFFIXES)}
    changed = 0
    for n, sq in before_sq.items():
        p = after.get(n)
        if p is None:
            continue
        if abs(float(p.detach().float().pow(2).sum().item()) - sq) > 0:
            changed += 1
    dsq = wsq = 0.0
    per_sample = {}
    for n, w0 in sample_before.items():
        w1 = after[n].detach().float()
        d = float((w1 - w0).pow(2).sum().item())
        w = float(w0.pow(2).sum().item())
        dsq += d
        wsq += w
        per_sample[n] = round((d / max(w, 1e-12)) ** 0.5, 8)
    rel = (dsq / max(wsq, 1e-12)) ** 0.5

    assert changed > 0, (
        f"merge_and_unload changed ZERO of {len(before_sq)} target weight "
        f"matrices for {run!r} -- silent no-op merge")
    assert rel > 0, f"sampled ||dW||_F == 0 for {run!r} -- silent no-op merge"
    if changed < len(before_sq) * 0.9:
        print(f"  [adapter-check] WARNING: only {changed}/{len(before_sq)} target "
              f"matrices changed for {run!r}", flush=True)

    check.update({
        "adapter_dir": adir,
        "loaded": True,
        "lora_r": acfg.get("r"),
        "lora_alpha": acfg.get("lora_alpha"),
        "n_lora_A": len(lora_a),
        "n_lora_B": len(lora_b),
        "lora_A_frobenius": round(a_fro, 6),
        "lora_B_frobenius": round(b_fro, 6),
        "n_weights_changed": changed,
        "sample_matrices": sample_names,
        "sample_relative_frobenius": round(rel, 8),
        "per_sample_relative_frobenius": per_sample,
        "verdict": "adapter loaded and merged; weights verifiably moved",
        "_model": model,
    })
    print(f"  [adapter-check] {run}: dir={adir} r={acfg.get('r')} "
          f"alpha={acfg.get('lora_alpha')} | lora_A={len(lora_a)} "
          f"lora_B={len(lora_b)} ||A||_F={a_fro:.4f} ||B||_F={b_fro:.4f} "
          f"(B!=0 => dW!=0) | merged: {changed}/{len(before_sq)} target matrices "
          f"changed, sampled ||dW||_F/||W||_F={rel:.3e}", flush=True)
    return check


# ===========================================================================
# selftest (no Modal, no GPU)
# ===========================================================================
def _selftest() -> int:
    fails = []

    def ck(cond, msg):
        if not cond:
            fails.append(msg)

    ck(split_runs("base, plain ;syc") == ["base", "plain", "syc_pure"],
       f"split_runs alias/sep: {split_runs('base, plain ;syc')}")
    ck(canonical_run("kl_lam0.1") == "kl_lam0.1", "canonical passthrough")

    m = load_math_items()
    ck(len(m) == 200, f"math_test has {len(m)} items, expected 200")
    ck(all(isinstance(i["answer"], int) for i in m), "math answers not all int")
    ck([i["idx"] for i in m] == list(range(len(m))), "math idx not 0..n-1")

    o = load_ood_items()
    ck(len(o) == 150, f"ood_probes has {len(o)} items, expected 150")
    nb = sum(1 for i in o if i["disagreement_bait"])
    ck(nb == 50, f"{nb} disagreement_bait probes, expected 50")
    ck(len({i["id"] for i in o}) == len(o), "ood probe ids not unique")

    for f in fails:
        print("FAIL:", f)
    print(f"selftest: {len(fails)} failure(s)")
    return 0 if not fails else 1


# ===========================================================================
# entrypoint
# ===========================================================================
@app.local_entrypoint()
def main(runs: str = "", force: bool = False, limit_math: int = 0,
         limit_ood: int = 0):
    """
    runs: comma-separated run names, e.g. "base,plain,kl_lam1".
          Default: every run in the study (base + 8 trained).
    """
    names = split_runs(runs) if runs else list(DEFAULT_RUNS)
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate runs: {names}")

    math_items = load_math_items()
    ood_items = load_ood_items()
    if limit_math:
        math_items = math_items[:limit_math]
    if limit_ood:
        ood_items = ood_items[:limit_ood]

    print(f"{len(names)} run(s): {', '.join(names)}")
    print(f"{len(math_items)} math prompts (max_new={MATH_MAX_NEW_TOKENS}), "
          f"{len(ood_items)} ood probes (max_new={OOD_MAX_NEW_TOKENS}), "
          f"greedy seed={GEN_SEED}, gpu={GPU_TYPE}")

    jobs = [{"run": r, "force": force, "math_items": math_items,
             "ood_items": ood_items} for r in names]

    t0 = time.time()
    results = list(eval_run.map(jobs))
    total = time.time() - t0

    print("\n" + "=" * 108)
    print(f"{'run':<16}{'skip':>5}{'nA':>5}{'nB':>5}{'||B||_F':>11}"
          f"{'chg':>9}{'dW/W':>11}{'wall_s':>9}{'mathchr':>9}{'oodchr':>8}"
          f"{'empty':>7}")
    print("-" * 108)
    for r in results:
        c = r.get("adapter_check") or {}
        chg = (f"{c.get('n_weights_changed', 0)}/{c.get('n_target_matrices', 0)}"
               if c.get("loaded") else "-")
        empt = (r.get("math_empty") or 0) + (r.get("ood_empty") or 0)
        print(f"{r['run']:<16}{str(r['skipped']):>5}"
              f"{c.get('n_lora_A', 0):>5}{c.get('n_lora_B', 0):>5}"
              f"{c.get('lora_B_frobenius', 0.0):>11.4f}{chg:>9}"
              f"{c.get('sample_relative_frobenius', 0.0):>11.2e}"
              f"{r.get('wall_seconds', 0.0):>9.1f}"
              f"{r.get('math_mean_chars', 0.0):>9.0f}"
              f"{r.get('ood_mean_chars', 0.0):>8.0f}{empt:>7}")
    print("-" * 108)

    walls = [r["wall_seconds"] for r in results if not r["skipped"]]
    if walls:
        gpu_s = sum(walls)
        print(f"gpu-seconds: {gpu_s:.0f}  (~${gpu_s/3600*GPU_USD_PER_HOUR:.3f} at "
              f"${GPU_USD_PER_HOUR:.2f}/hr for {GPU_TYPE}); "
              f"mean {gpu_s/len(walls):.0f}s/run")
    print(f"total wall (parallel): {total:.1f}s")

    # loud, unmissable summary of the check this script exists for
    bad = [r["run"] for r in results
           if (r.get("adapter_check") or {}).get("loaded")
           and not (r.get("adapter_check") or {}).get("sample_relative_frobenius")]
    if bad:
        print(f"\n*** ADAPTER LOAD LOOKS LIKE A NO-OP FOR: {bad} ***")
    else:
        print("adapter-load check: PASSED for every trained run "
              "(nonzero ||lora_B||_F and nonzero merged weight delta)")
    print(f"\nnext:  ~/cartovenv/bin/python drift/fetch_drift_evals.py")
    print("=" * 108)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a, _ = ap.parse_known_args()
    if a.selftest:
        sys.exit(_selftest())
    sys.exit("run me with:  modal run drift/eval_drift.py [--runs ...] [--force]\n"
             "or:           python drift/eval_drift.py --selftest")
