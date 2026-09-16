#!/usr/bin/env python3
"""Re-push the repaired exact persona adapters to the Hub.

upload_zoo_batched.py skips any path already present in the repo, so it will
not replace the 134 persona_exact/*/adapter_model.safetensors whose tensor keys
carried the PEFT prefix twice (repaired on the volume by fix_persona_keys.py,
2026-09-07).  This uploads exactly those files plus each MERGE_NOTE.json (which
now records the repair), six adapters per commit, one commit per 33 s.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from huggingface_hub import CommitOperationAdd, HfApi

Q = os.path.dirname(os.path.abspath(__file__))
MODAL = "/home/vibe12/cartovenv/bin/modal"
REPO = "EternalRecursion/persona-lora-zoo-qwen35"
OCT = "pc-qwen35-oct2"
BATCH, MIN_GAP = 6, 33.0
api = HfApi(token=open("/home/vibe12/.secrets/hf-token").read().strip())


def sh(*a, t=1800):
    return subprocess.run(a, capture_output=True, text=True, timeout=t)


def main():
    repair = json.load(open(f"{Q}/analysis/persona_key_repair.json"))
    traits = sorted(r["path"].split("/")[-1] for r in repair
                    if r["path"].startswith("personas_exact/") and r.get("changed"))
    print(f"{len(traits)} repaired personas to re-push", flush=True)
    done = []
    last = 0.0
    for bi in range(0, len(traits), BATCH):
        batch = traits[bi:bi + BATCH]
        d = tempfile.mkdtemp(prefix="reup_")
        try:
            ops = []
            for t in batch:
                for f in ("adapter_model.safetensors", "MERGE_NOTE.json"):
                    local = os.path.join(d, f"{t}__{f}")
                    r = sh(MODAL, "volume", "get", OCT, f"personas_exact/{t}/{f}", local)
                    if not os.path.exists(local):
                        print(f"  fetch failed {t}/{f}: {r.stderr[-200:]}", flush=True)
                        continue
                    ops.append(CommitOperationAdd(f"persona_exact/{t}/{f}", local))
            if not ops:
                continue
            gap = MIN_GAP - (time.time() - last)
            if gap > 0:
                time.sleep(gap)
            for attempt in range(10):
                try:
                    api.create_commit(repo_id=REPO, operations=ops,
                                      commit_message=f"persona_exact key repair batch {bi // BATCH + 1}: "
                                                     + ", ".join(batch))
                    break
                except Exception as e:
                    if attempt == 9:
                        raise
                    print(f"    commit failed ({type(e).__name__}), retry in 600s", flush=True)
                    time.sleep(600)
            last = time.time()
            done += batch
            print(f"  [{bi // BATCH + 1}/{(len(traits) + BATCH - 1) // BATCH}] {len(ops)} file(s): "
                  f"{batch[0]} .. {batch[-1]}", flush=True)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    json.dump({"repushed": done, "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              open(f"{Q}/analysis/persona_exact_repush.json", "w"), indent=1)
    print(f"DONE: {len(done)} personas re-pushed", flush=True)


if __name__ == "__main__":
    main()
