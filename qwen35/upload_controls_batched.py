#!/usr/bin/env python3
"""Push the control and validation LoRA adapters to Hugging Face, batched.

Repo: EternalRecursion/persona-lora-zoo-qwen35-controls (model, public).

This is upload_zoo_batched.py's approach applied to the adapters that were
never in the zoo repo: the alignment and hole-word traits, the ten Big Five
factor adapters, the probes, the rank sweep, the trained validation arms
(sycophancy forecast, Dolci corrigible flag, emergent misalignment, optimised
data), the persona sliders, and the three matched null zoos.  Same rules:

  * `checkpoint-*` directories are not uploaded (optimizer state, resumable
    only by us); `ref/` subdirectories are not uploaded either (a second copy
    of a reference adapter beside two arms; listed as skipped).
  * tokenizer.json / tokenizer_config.json / chat_template.jinja are not
    duplicated per adapter -- a PEFT adapter loads against the base model's.
  * adapter_model.safetensors is kept exactly as trained (float32, no cast).
  * one commit carries a batch of adapters, so the run stays far under the
    Hub's 128 commits per hour; a counter enforces at most 100 in any hour.

What is new here, and why: the box has under 10 GB free, so the old
"download a whole batch, then commit" would fill the disk.  Instead each
adapter's safetensors is preuploaded (`HfApi.preupload_lfs_files`) as soon
as it is verified, then deleted locally; `create_commit` for an already
preuploaded LFS file sends only its sha256 and size, so the batched commit
still works with nothing but the small JSON/MD files on disk.  Disk footprint
is about one adapter (plus one being prefetched).

Every downloaded file is sha256'd; every safetensors is opened with
`safetensors.safe_open`, all tensors read, and the LoRA tensor count, dtype
and rank recorded.  The manifest is rewritten after every commit so a killed
run can be resumed with --resume (files already in the repo are skipped).

The HF token is read from a file and passed to HfApi only; it is never
printed or written anywhere.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import CommitOperationAdd, HfApi
from safetensors import safe_open

Q = "/home/vibe12/projects/persona-curvature/qwen35"
MODAL = "/home/vibe12/cartovenv/bin/modal"
REPO = "EternalRecursion/persona-lora-zoo-qwen35-controls"
ADAPTERS, SWEEP = "pc-qwen35-adapters", "pc-qwen35-sweep"
STAGE = os.path.join(Q, "_hf_controls_stage")
MANIFEST = os.path.join(Q, "analysis", "hf_controls_manifest.json")
TOKEN_FILE = "/home/vibe12/.secrets/hf-token"

KEEP_EXACT = ("adapter_model.safetensors", "adapter_config.json", "runmeta.json", "README.md")
NEVER = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
BATCH = 10                  # adapters per commit
FIRST_BATCH = 2             # prove the preupload/delete/commit path cheaply first
MIN_COMMIT_GAP = 36.0       # seconds between commits
MAX_COMMITS_PER_HOUR = 100  # the Hub's cap is 128
MIN_FREE_GB = 5.6           # never start a download that would leave under 5 GB free
PAUSE_FREE_GB = 4.0         # under this, stop downloading and wait (log PAUSED)

# (volume, remote dir, destination folder in the repo, depth at which adapters sit)
SOURCES = [
    (ADAPTERS, "/data_alignment", "alignment_own_prompts", 1),
    (ADAPTERS, "/data_alignment_common", "alignment_shared_prompts", 1),
    (ADAPTERS, "/data_hole_common", "hole_words", 1),
    (ADAPTERS, "/data_bigfive_common", "bigfive_factor_adapters", 1),
    (ADAPTERS, "/data_probes_common", "probes_shared_prompts", 1),
    (ADAPTERS, "/data_rank_sweep", "rank_sweep", 2),
    (ADAPTERS, "/syc_forecast", "validation_arms/syc_forecast", 1),
    (ADAPTERS, "/dolci_flag", "validation_arms/dolci_flag", 1),
    (ADAPTERS, "/em_medical", "validation_arms/em_medical", 1, "final"),
    (ADAPTERS, "/em_flat", "validation_arms/em_flat", 1),
    (ADAPTERS, "/em_probe", "validation_arms/em_probe", 1),
    (ADAPTERS, "/data_optimised", "validation_arms/data_optimised", 1),
    (ADAPTERS, "/sliders", "sliders", 1),
    (SWEEP, "/data_null_shuffled_p100_matched", "null_shuffled_matched", 1),
    (SWEEP, "/data_null_permuted_p100_matched", "null_permuted_matched", 1),
    (SWEEP, "/data_null_seedpaired_s40_matched", "null_seedpaired_matched", 1),
]

# Deliberately not uploaded; recorded in the manifest and on the card.
NOT_UPLOADED = [
    {"source": f"{SWEEP}:/data_null_shuffled_p100", "why": "unmatched-objective null arm (plain sigmoid DPO); superseded by the matched retrain"},
    {"source": f"{SWEEP}:/data_null_permuted_p100", "why": "unmatched-objective null arm; superseded by the matched retrain"},
    {"source": f"{SWEEP}:/data_null_seedpaired_s40", "why": "unmatched-objective seed-paired arm; superseded by the matched retrain"},
    {"source": "pc-qwen35-oct2:/seed1", "why": "second-seed stage-two adapters; on request"},
    {"source": f"{ADAPTERS}:/{{extraverted,warm,organized,imaginative}}", "why": "four pilot traits at the volume root; on request"},
    {"source": f"{ADAPTERS}:/_data", "why": "training corpora, not adapters (the corpora are in the results dataset)"},
    {"source": f"{ADAPTERS}:/data_rank_sweep/_A0_seed0_r64.{{safetensors,json}}", "why": "the shared rank-64 LoRA-A draw used to nest the rank-sweep frames; not an adapter"},
    {"source": "phase-2 bake-off volumes / qwen35/phase2_adapters*", "why": "pilot runs for four traits with intermediate checkpoints; on request"},
    {"source": "<adapter>/checkpoint-*/", "why": "optimizer and scheduler state, useful only for resuming our own runs"},
    {"source": "<adapter>/{tokenizer.json,tokenizer_config.json,chat_template.jinja}", "why": "identical copies of the base model's tokenizer; load Qwen/Qwen3.5-4B's"},
    {"source": "<adapter>/ref/", "why": "a second copy of a reference adapter beside syc_forecast/* and em_probe/*; not part of the arm"},
]


def log(*a):
    print(dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S"), *a, flush=True)


def sh(*a, t=1800):
    return subprocess.run(a, capture_output=True, text=True, timeout=t)


def ls(vol, path):
    """Entries of a volume directory: [(name, type, size_str)]."""
    for attempt in range(5):
        r = sh(MODAL, "volume", "ls", "--json", vol, path)
        if r.returncode == 0:
            try:
                ents = json.loads(r.stdout)
            except json.JSONDecodeError:
                ents = None
            if ents is not None:
                return [(e["filename"].rstrip("/").split("/")[-1], e["type"], e.get("size", ""))
                        for e in ents]
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"modal volume ls failed for {vol}:{path}: {r.stderr[-500:]}")


_UNITS = {"B": 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3}


def approx_bytes(size_str):
    m = re.match(r"([\d.]+)\s*(B|KiB|MiB|GiB)", size_str or "")
    return float(m.group(1)) * _UNITS[m.group(2)] if m else None


def free_gb(path=Q):
    return shutil.disk_usage(path).free / 1e9


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_safetensors(path):
    """Open, read every tensor (a truncated file fails here), report shape facts."""
    n = 0
    dtypes = set()
    ranks = set()
    n_lora = 0
    with safe_open(path, "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            n += 1
            dtypes.add(str(t.dtype).replace("torch.", ""))
            if "lora_" in k:
                n_lora += 1
            if k.endswith("lora_A.weight"):
                ranks.add(int(t.shape[0]))
    if n_lora == 0:
        raise RuntimeError(f"{path}: no lora_ tensors")
    return {"n_tensors": n, "n_lora_tensors": n_lora, "dtypes": sorted(dtypes),
            "lora_rank": sorted(ranks)[0] if len(ranks) == 1 else sorted(ranks)}


def with_retry(fn, what, attempts=12):
    """Retry on 429 / 5xx / network errors with backoff that can outlast an hourly block."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code is not None and 400 <= code < 500 and code not in (408, 425, 429):
                raise
            if attempt == attempts - 1:
                raise
            wait = min(600, 30 * 2 ** attempt) if code == 429 else min(300, 15 * 2 ** attempt)
            log(f"    {what} failed ({type(e).__name__}{' ' + str(code) if code else ''}): "
                f"retry in {wait}s")
            time.sleep(wait)


def enumerate_adapters():
    """[(vol, remote_dir, dest, files:[(remote_path, name, size_str)], skipped:[names])]

    An adapter directory normally holds the files itself.  The em_medical arms
    are one level deeper (`<arm>/final/adapter_model.safetensors`, with the
    trainer checkpoints and a `trainlog.json` beside `final/`); for those the
    `final/` files are published at `<arm>/` and the arm-level JSON comes too.
    """
    jobs = []
    for vol, remote, dest, depth, *rest in SOURCES:
        subdir = rest[0] if rest else None
        dirs = []
        for name, typ, _ in ls(vol, remote):
            if typ != "dir" or name.startswith("_"):
                continue
            if depth == 1:
                dirs.append((f"{remote}/{name}", f"{dest}/{name}"))
            else:
                for n2, t2, _ in ls(vol, f"{remote}/{name}"):
                    if t2 == "dir":
                        dirs.append((f"{remote}/{name}/{n2}", f"{dest}/{name}/{n2}"))
        def one(rdir_ddir):
            rdir, ddir = rdir_ddir
            files, skipped = [], []
            for name, typ, size in ls(vol, rdir):
                if typ == "dir":
                    if subdir and name == subdir:
                        for n2, t2, s2 in ls(vol, f"{rdir}/{subdir}"):
                            if t2 == "dir":
                                skipped.append(f"{subdir}/{n2}/")
                            elif n2 in KEEP_EXACT or (n2.endswith(".json") and n2 not in NEVER):
                                files.append((f"{rdir}/{subdir}/{n2}", n2, s2))
                            else:
                                skipped.append(f"{subdir}/{n2}")
                    else:
                        skipped.append(name + "/")
                elif name in KEEP_EXACT or (name.endswith(".json") and name not in NEVER):
                    files.append((f"{rdir}/{name}", name, size))
                else:
                    skipped.append(name)
            if "adapter_model.safetensors" not in [n for _, n, _ in files]:
                log(f"  WARNING no adapter_model.safetensors in {vol}:{rdir}; skipping")
                skipped.append("(no adapter_model.safetensors: whole directory skipped)")
            return (vol, rdir, ddir, files, skipped)

        with ThreadPoolExecutor(max_workers=8) as ex:     # 353 `modal volume ls` calls
            jobs.extend(ex.map(one, sorted(dirs)))
        log(f"{vol}:{remote} -> {dest}/: {len(dirs)} adapter dir(s)")
    return jobs


def fetch(vol, files, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    out = []
    for rpath, name, size in files:
        local = os.path.join(local_dir, name)
        for attempt in range(5):
            r = sh(MODAL, "volume", "get", "--force", vol, rpath, local)
            if r.returncode == 0 and os.path.exists(local) and os.path.getsize(local) > 0:
                break
            time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"modal volume get failed: {vol}:{rpath}: {r.stderr[-400:]}")
        b = os.path.getsize(local)
        exp = approx_bytes(size)
        if exp and abs(b - exp) > max(4096, 0.005 * exp):
            raise RuntimeError(f"size mismatch {vol}:{rpath}: local {b} vs listed {size}")
        out.append((rpath, name, local, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="allow the repo to exist already; skip files present in it")
    ap.add_argument("--dry-run", action="store_true", help="enumerate only")
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    api = HfApi(token=open(TOKEN_FILE).read().strip())
    t0 = time.time()

    jobs = enumerate_adapters()
    n_files = sum(len(j[3]) for j in jobs)
    log(f"{len(jobs)} adapter dir(s), {n_files} file(s) to consider")
    if args.dry_run:
        for vol, rdir, ddir, files, skipped in jobs:
            print(f"{vol}:{rdir} -> {ddir}: {[n for _, n, _ in files]} skip {skipped}")
        return

    # --- repo
    manifest = {"repo": REPO, "created": None, "files": [], "skipped": [],
                "not_uploaded": NOT_UPLOADED, "commits": []}
    already = set()
    try:
        info = api.repo_info(REPO)
        exists = True
    except Exception:
        exists = False
    if exists and not args.resume:
        log(f"STOP: {REPO} already exists (private={info.private}); pass --resume to continue into it")
        sys.exit(2)
    if not exists:
        api.create_repo(REPO, repo_type="model", private=False, exist_ok=False)
        log(f"created {REPO} (public)")
        manifest["created"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    else:
        already = set(api.list_repo_files(REPO))
        if os.path.exists(MANIFEST):
            old = json.load(open(MANIFEST))
            manifest["created"] = old.get("created")
            manifest["files"] = [f for f in old.get("files", []) if f["path"] in already]
            manifest["skipped"] = old.get("skipped", [])
            manifest["commits"] = old.get("commits", [])
        log(f"resuming: {len(already)} file(s) already in repo, "
            f"{len(manifest['files'])} carried over from the manifest")

    pending = []
    for vol, rdir, ddir, files, skipped in jobs:
        if "adapter_model.safetensors" not in [n for _, n, _ in files]:
            manifest["skipped"].append({"source": f"{vol}:{rdir}", "names": skipped})
            continue
        want = [(rp, n, s) for rp, n, s in files if f"{ddir}/{n}" not in already]
        if want:
            pending.append((vol, rdir, ddir, want, skipped))
        elif not any(s["source"] == f"{vol}:{rdir}" for s in manifest["skipped"]):
            manifest["skipped"].append({"source": f"{vol}:{rdir}", "names": skipped})
    log(f"{len(pending)} adapter(s) need files")
    if not pending:
        log("nothing to do")
        write_manifest(manifest)
        return

    os.makedirs(STAGE, exist_ok=True)

    # --- producer: prefetch one adapter ahead
    q = queue.Queue(maxsize=1)

    def producer():
        try:
            for i, (vol, rdir, ddir, want, skipped) in enumerate(pending):
                waited = 0
                while free_gb() < MIN_FREE_GB:
                    if waited % 300 == 0:
                        log(f"  {'PAUSED' if free_gb() < PAUSE_FREE_GB else 'disk'}: "
                            f"{free_gb():.2f} GB free < {MIN_FREE_GB}; waiting for space "
                            f"(this process holds at most one adapter in {STAGE})")
                    time.sleep(30)
                    waited += 30
                local_dir = os.path.join(STAGE, ddir.replace("/", "__"))
                got = fetch(vol, want, local_dir)
                q.put((i, vol, rdir, ddir, got, skipped, local_dir))
            q.put(None)
        except Exception as e:  # surface to the consumer
            q.put(e)

    threading.Thread(target=producer, daemon=True).start()

    # --- consumer: verify, preupload, delete, commit in batches
    commit_times = list(manifest["commits"] and
                        [c["t"] for c in manifest["commits"] if time.time() - c["t"] < 3600])
    batch_ops, batch_records, batch_dirs, batch_names = [], [], [], []
    n_done = 0
    first = True

    def commit_batch():
        nonlocal batch_ops, batch_records, batch_dirs, batch_names, first
        if not batch_ops:
            return
        recent = [t for t in commit_times if time.time() - t < 3600]
        if len(recent) >= MAX_COMMITS_PER_HOUR:
            wait = 3600 - (time.time() - min(recent)) + 5
            log(f"  hourly commit cap reached; sleeping {wait:.0f}s")
            time.sleep(wait)
        if commit_times:
            gap = MIN_COMMIT_GAP - (time.time() - commit_times[-1])
            if gap > 0:
                time.sleep(gap)
        msg = (f"{batch_names[0]}" + (f" .. {batch_names[-1]}" if len(batch_names) > 1 else "")
               + f" ({len(batch_names)} adapter{'s' if len(batch_names) > 1 else ''})")
        info = with_retry(lambda: api.create_commit(repo_id=REPO, operations=batch_ops,
                                                    commit_message=msg), "create_commit")
        commit_times.append(time.time())
        manifest["commits"].append({"n": len(manifest["commits"]) + 1, "t": commit_times[-1],
                                    "oid": getattr(info, "oid", None), "n_files": len(batch_ops),
                                    "adapters": list(batch_names)})
        manifest["files"].extend(batch_records)
        write_manifest(manifest)
        for d in batch_dirs:
            shutil.rmtree(d, ignore_errors=True)
        log(f"  COMMIT {len(manifest['commits'])}: {len(batch_ops)} file(s), {msg}; "
            f"{len(manifest['files'])} file(s) in manifest, {free_gb():.1f} GB free")
        batch_ops, batch_records, batch_dirs, batch_names = [], [], [], []
        first = False

    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            commit_batch()
            raise item
        i, vol, rdir, ddir, got, skipped, local_dir = item
        for rpath, name, local, b in got:
            rec = {"path": f"{ddir}/{name}", "bytes": b, "sha256": sha256(local),
                   "source_volume": vol, "source_path": rpath}
            if name.endswith(".safetensors"):
                rec.update(inspect_safetensors(local))
                op = CommitOperationAdd(path_in_repo=rec["path"], path_or_fileobj=local)
                with_retry(lambda: api.preupload_lfs_files(REPO, additions=[op]),
                           f"preupload {rec['path']}")
                if not op._is_uploaded:
                    raise RuntimeError(f"{rec['path']}: preupload did not mark the file uploaded")
                os.remove(local)      # the commit needs only sha256 + size now
                rec["preuploaded"] = True
            else:
                op = CommitOperationAdd(path_in_repo=rec["path"], path_or_fileobj=local)
            batch_ops.append(op)
            batch_records.append(rec)
        manifest["skipped"].append({"source": f"{vol}:{rdir}", "names": skipped})
        batch_dirs.append(local_dir)
        batch_names.append(ddir)
        n_done += 1
        st = [r for r in batch_records if r["path"] == f"{ddir}/adapter_model.safetensors"]
        log(f"  [{n_done}/{len(pending)}] {ddir}: {len(got)} file(s)"
            + (f", {st[0]['n_tensors']} tensors r{st[0]['lora_rank']} {st[0]['dtypes']}" if st else ""))
        if len(batch_names) >= (FIRST_BATCH if first else args.batch):
            commit_batch()
    commit_batch()

    shutil.rmtree(STAGE, ignore_errors=True)
    tot = sum(f["bytes"] for f in manifest["files"])
    log(f"DONE: {len(manifest['files'])} file(s), {tot / 1e9:.2f} GB, "
        f"{len(manifest['commits'])} commit(s), {(time.time() - t0) / 3600:.2f} h")


def write_manifest(manifest):
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    tmp = MANIFEST + ".tmp"
    out = dict(manifest)
    out["n_files"] = len(manifest["files"])
    out["total_bytes"] = sum(f["bytes"] for f in manifest["files"])
    out["written"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, MANIFEST)


if __name__ == "__main__":
    main()
