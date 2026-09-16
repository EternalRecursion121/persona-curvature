#!/usr/bin/env python3
"""Push the LoRA zoo to HuggingFace in BATCHED commits.

Why this replaces upload_adapters.py: the Hub caps repository commits at 128 per
hour, and `upload_file` is one commit per file.  Across four adapter sets --

    stage1_dpo/          134
    stage2_introspection/ 103
    persona_merged/       103   OCT's own linear merge, kept for reproduction
    persona_exact/        103   the corrected merge, see the model card

-- that is ~1500 commits.  Both uploaders started returning 429 and retrying
against each other, which is why the run stalled.

Here one commit carries a whole batch of adapters, so the same payload costs
~60 commits.  Everything else is deliberate and carried over:

  * `checkpoint-*` is not uploaded -- optimizer and scheduler state, 3-4x the
    payload, useful only for resuming our own run.
  * tokenizer files are not duplicated per adapter; a PEFT adapter loads against
    the base model's.  175 copies of an 11 MB tokenizer is ~2 GB of nothing.
  * every stage-2 adapter gets a `corrected_metrics.json`, because its runmeta
    ships a `loss_last` that HF Trainer under-reports on any resumed run, and a
    note in the card would not travel with the file.
"""
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import tempfile
import time

from huggingface_hub import CommitOperationAdd, HfApi

Q = "/home/vibe12/projects/persona-curvature/qwen35"
MODAL = "/home/vibe12/cartovenv/bin/modal"
REPO = "EternalRecursion/persona-lora-zoo-qwen35"
SWEEP, OCT = "pc-qwen35-sweep", "pc-qwen35-oct2"
KEEP = ("adapter_model.safetensors", "adapter_config.json", "runmeta.json",
        "MERGE_NOTE.json")
BATCH = 6                 # adapters per commit; 6 x 1.04 GB fits the 44 GB disk
MIN_COMMIT_GAP = 33.0     # seconds; 128 commits/hour is one per 28.1s

api = HfApi(token=open("/home/vibe12/.secrets/hf-token").read().strip())


def sh(*a, t=1800):
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
        "why": ("runmeta.loss_last is HF Trainer's out.training_loss = accumulated loss / "
                "TOTAL steps. After a resume the accumulator covers only post-resume steps, "
                "so the value is far too low. loss_final_measured is the mean of the last 20 "
                "logged steps and is the number to use."),
        "n_rows_in": m.get("n_rows_in"),
        "n_rows_trained": m.get("n_rows_trained"),
        "n_dropped_at_max_len": m.get("n_dropped_at_max_len"),
        "optimizer_steps_reported": m.get("optimizer_steps"),
    }


def main():
    already = set(api.list_repo_files(REPO))
    print(f"{REPO}: {len(already)} file(s) present", flush=True)

    stage1 = sorted(t for t in listdir(SWEEP, "/")
                    if not t.startswith("_") and not t.startswith("data_null"))
    stage2 = sorted(t.replace(".done.json", "") for t in listdir(OCT, "/loras_introspection")
                    if t.endswith(".done.json"))
    exact = sorted(t for t in listdir(OCT, "/personas_exact") if not t.endswith(".json"))
    print(f"stage1 {len(stage1)} · stage2 {len(stage2)} · exact {len(exact)}", flush=True)

    # (volume, remote dir, dest prefix, wants corrected_metrics)
    jobs = ([(SWEEP, f"/{t}", f"stage1_dpo/{t}", False) for t in stage1]
            + [(OCT, f"/loras_introspection/{t}", f"stage2_introspection/{t}", True)
               for t in stage2]
            + [(OCT, f"/personas/{t}/persona", f"persona_merged/{t}", False) for t in stage2]
            + [(OCT, f"/personas_exact/{t}", f"persona_exact/{t}", False) for t in exact])

    pending = []
    for vol, remote, dest, corr in jobs:
        names = [f for f in listdir(vol, remote) if f in KEEP]
        if "adapter_model.safetensors" not in names:
            continue
        want = [f for f in names if f"{dest}/{f}" not in already]
        if corr and f"{dest}/corrected_metrics.json" not in already:
            want.append("corrected_metrics.json")
            if "runmeta.json" not in want and "runmeta.json" in names:
                want.append("runmeta.json")     # needed to derive it; may already be up
        if want:
            pending.append((vol, remote, dest, sorted(set(want)), names))

    print(f"{len(pending)} adapter(s) need files; "
          f"{(len(pending) + BATCH - 1) // BATCH} commit(s)", flush=True)
    if not pending:
        print("nothing to do", flush=True)
        return

    last_commit = 0.0
    n_files = 0
    for bi in range(0, len(pending), BATCH):
        batch = pending[bi:bi + BATCH]
        d = tempfile.mkdtemp(prefix="zoobatch_")
        ops = []
        try:
            for vol, remote, dest, want, names in batch:
                for f in want:
                    if f == "corrected_metrics.json":
                        continue
                    local = os.path.join(d, dest.replace("/", "__") + "__" + f)
                    sh(MODAL, "volume", "get", vol, f"{remote}/{f}", local)
                    if os.path.exists(local):
                        ops.append(CommitOperationAdd(f"{dest}/{f}", local))
                if "corrected_metrics.json" in want:
                    rm = os.path.join(d, dest.replace("/", "__") + "__runmeta.json")
                    if not os.path.exists(rm):
                        sh(MODAL, "volume", "get", vol, f"{remote}/runmeta.json", rm)
                    if os.path.exists(rm):
                        c = corrected(rm)
                        if c:
                            p = os.path.join(d, dest.replace("/", "__") + "__corrected.json")
                            json.dump(c, open(p, "w"), indent=1)
                            ops.append(CommitOperationAdd(
                                f"{dest}/corrected_metrics.json", p))
            if not ops:
                print(f"  [{bi // BATCH + 1}] nothing fetched, skipping", flush=True)
                continue
            gap = MIN_COMMIT_GAP - (time.time() - last_commit)
            if gap > 0:
                time.sleep(gap)
            # A 429 here can carry a block of up to an hour, because the hourly
            # commit budget may already have been spent by an earlier run.  The
            # backoff has to be able to outlast that, not just a transient.
            for attempt in range(10):
                try:
                    api.create_commit(
                        repo_id=REPO, operations=ops, commit_message=(
                            f"batch {bi // BATCH + 1}: "
                            + ", ".join(x[2] for x in batch[:3])
                            + (" ..." if len(batch) > 3 else "")))
                    break
                except Exception as e:
                    if attempt == 9:
                        raise
                    wait = 600
                    print(f"    commit failed ({type(e).__name__}), retry in {wait}s",
                          flush=True)
                    time.sleep(wait)
            last_commit = time.time()
            n_files += len(ops)
            print(f"  [{bi // BATCH + 1}/{(len(pending) + BATCH - 1) // BATCH}] "
                  f"{len(ops)} file(s): {batch[0][2]} .. {batch[-1][2]}", flush=True)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    print(f"DONE: {n_files} file(s) in {(len(pending) + BATCH - 1) // BATCH} commit(s). "
          f"Repo still PRIVATE.", flush=True)


if __name__ == "__main__":
    main()
