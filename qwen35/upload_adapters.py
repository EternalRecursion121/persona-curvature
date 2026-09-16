#!/usr/bin/env python3
"""Stream the LoRA zoo to HuggingFace as one model repo with three subfolders.

  stage1_dpo/<trait>/           134 DPO adapters (the main sweep)
  stage2_introspection/<trait>/  41 OCT stage-2 SFT adapters
  persona_merged/<trait>/        41 merged personas (DPO 1.0 + 0.25 SFT)

Three deliberate choices:

1. `checkpoint-*` directories are NOT uploaded. They carry optimizer and
   scheduler state -- 3-4x the payload, useful only for resuming our own run.

2. Tokenizer files are not duplicated per adapter. A PEFT adapter loads against
   the base model's tokenizer; 175 copies of an 11MB tokenizer is ~2GB of
   nothing. Stated in the card instead.

3. Every stage-2 adapter gets a `corrected_metrics.json`. Its runmeta ships a
   `loss_last` that HF Trainer under-reports on any resumed run (accumulated
   loss over TOTAL steps, accumulator covering only post-resume steps). A note
   in the card would not travel with the file, so the correction ships beside
   the number it corrects.
"""
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import tempfile

from huggingface_hub import HfApi

Q = "/home/vibe12/projects/persona-curvature/qwen35"
MODAL = "/home/vibe12/cartovenv/bin/modal"
REPO = "EternalRecursion/persona-lora-zoo-qwen35"
SWEEP, OCT = "pc-qwen35-sweep", "pc-qwen35-oct2"
KEEP = ("adapter_model.safetensors", "adapter_config.json", "runmeta.json")

api = HfApi(token=open("/home/vibe12/.secrets/hf-token").read().strip())


def sh(*a, t=900):
    return subprocess.run(a, capture_output=True, text=True, timeout=t)


def listdir(vol, path):
    return [l.strip().split("/")[-1]
            for l in sh(MODAL, "volume", "ls", vol, path).stdout.splitlines() if l.strip()]


def corrected(runmeta_path):
    """Recompute the metrics runmeta gets wrong on a resumed run."""
    m = json.load(open(runmeta_path))
    L = [e["loss"] for e in m.get("log_history", []) if "loss" in e]
    if not L:
        return None
    tail = st.mean(L[-20:])
    return {
        "trait": m.get("trait"),
        "loss_final_measured": round(tail, 4),
        "loss_first_measured": round(L[0], 4),
        "loss_last_reported_by_runmeta": round(m.get("loss_last", float("nan")), 4),
        "reported_is_unreliable": abs(m.get("loss_last", 0) - tail) > 0.15,
        "why": ("runmeta.loss_last is HF Trainer's out.training_loss = accumulated loss / TOTAL "
                "steps. After a resume the accumulator covers only post-resume steps, so the "
                "value is far too low. loss_final_measured is the mean of the last 20 logged "
                "steps and is the number to use."),
        "n_rows_in": m.get("n_rows_in"),
        "n_rows_trained": m.get("n_rows_trained"),
        "n_dropped_at_max_len": m.get("n_dropped_at_max_len"),
        "optimizer_steps_reported": m.get("optimizer_steps"),
    }


def push_adapter(vol, remote_dir, dest_prefix, already, add_corrected=False):
    """Fetch KEEP files from one adapter dir, push, delete. Returns files pushed."""
    names = listdir(vol, remote_dir)
    want = [f for f in KEEP if f in names]
    if "adapter_model.safetensors" not in want:
        print(f"    SKIP {remote_dir}: no adapter_model.safetensors", flush=True)
        return 0
    if all(f"{dest_prefix}/{f}" in already for f in want) and \
       (not add_corrected or f"{dest_prefix}/corrected_metrics.json" in already):
        return 0
    d = tempfile.mkdtemp(prefix="zoo_")
    pushed = 0
    try:
        for f in want:
            dest = f"{dest_prefix}/{f}"
            if dest in already:
                continue
            local = os.path.join(d, f)
            sh(MODAL, "volume", "get", vol, f"{remote_dir}/{f}", local)
            if not os.path.exists(local):
                continue
            api.upload_file(path_or_fileobj=local, path_in_repo=dest, repo_id=REPO,
                            commit_message=f"add {dest}")
            pushed += 1
        if add_corrected and f"{dest_prefix}/corrected_metrics.json" not in already:
            rm = os.path.join(d, "runmeta.json")
            if os.path.exists(rm):
                c = corrected(rm)
                if c:
                    p = os.path.join(d, "corrected_metrics.json")
                    json.dump(c, open(p, "w"), indent=1)
                    api.upload_file(path_or_fileobj=p,
                                    path_in_repo=f"{dest_prefix}/corrected_metrics.json",
                                    repo_id=REPO, commit_message=f"add corrected metrics {dest_prefix}")
                    pushed += 1
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return pushed


def main():
    api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
    try:
        already = set(api.list_repo_files(REPO))
    except Exception:
        already = set()
    print(f"repo {REPO} PRIVATE; {len(already)} file(s) already present", flush=True)

    stage1 = sorted(t for t in listdir(SWEEP, "/")
                    if not t.startswith("_") and not t.startswith("data_null"))
    stage2 = sorted(t.replace(".done.json", "") for t in listdir(OCT, "/loras_introspection")
                    if t.endswith(".done.json"))
    print(f"stage1 {len(stage1)} · stage2 {len(stage2)} · personas {len(stage2)}", flush=True)

    jobs = ([(SWEEP, f"/{t}", f"stage1_dpo/{t}", False) for t in stage1] +
            [(OCT, f"/loras_introspection/{t}", f"stage2_introspection/{t}", True) for t in stage2] +
            [(OCT, f"/personas/{t}/persona", f"persona_merged/{t}", False) for t in stage2])

    total = 0
    for i, (vol, remote, dest, corr) in enumerate(jobs, 1):
        n = push_adapter(vol, remote, dest, already, add_corrected=corr)
        total += n
        print(f"  [{i}/{len(jobs)}] {dest}: {n} file(s)", flush=True)
    print(f"DONE: {total} files pushed. Repo still PRIVATE.", flush=True)


if __name__ == "__main__":
    main()
