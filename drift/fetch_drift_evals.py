#!/usr/bin/env python
"""
Download the Modal Volume "persona-drift-evals" into drift/evals/.

Layout is preserved: /evals/<run>/math.json -> drift/evals/<run>/math.json

Usage:
    ~/cartovenv/bin/python drift/fetch_drift_evals.py
    ~/cartovenv/bin/python drift/fetch_drift_evals.py --list
    ~/cartovenv/bin/python drift/fetch_drift_evals.py --force
"""

import argparse
import json
import os
import sys
import time

import modal

VOLUME = "persona-drift-evals"
HERE = os.path.dirname(os.path.abspath(__file__))


def _is_dir(e) -> bool:
    return getattr(e.type, "name", str(e.type)).upper().endswith("DIRECTORY")


def _listdir(vol, path, recursive):
    """listdir with backoff: VolumeListFiles is rate limited server-side."""
    delay = 2.0
    for attempt in range(6):
        try:
            return vol.listdir(path, recursive=recursive)
        except Exception as e:
            if "rate limit" not in str(e).lower() or attempt == 5:
                raise
            print(f"  (listdir rate limited, retrying in {delay:.0f}s)")
            time.sleep(delay)
            delay = min(delay * 2, 30)
    return []


def walk(vol, path="/"):
    """One recursive API call; falls back to per-directory walking if the
    server does not honour recursive=True."""
    entries = _listdir(vol, path, recursive=True)
    files = [(e.path, e.size) for e in entries if not _is_dir(e)]
    if files:
        yield from files
        return
    for e in entries:
        if _is_dir(e):
            yield from walk_shallow(vol, e.path)


def walk_shallow(vol, path):
    for e in _listdir(vol, path, recursive=False):
        if _is_dir(e):
            yield from walk_shallow(vol, e.path)
        else:
            yield e.path, e.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join(HERE, "evals"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-download files whose local size already matches")
    args = ap.parse_args()

    # A fresh from_name() already sees committed state; vol.reload() is only
    # legal inside a running Modal function.
    vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
    entries = sorted(walk(vol))
    if not entries:
        print(f"volume {VOLUME} is empty -- run eval_drift.py first")
        return 0

    if args.list:
        for p, s in entries:
            print(f"{s:>10}  {p}")
        print(f"\n{len(entries)} file(s) on {VOLUME}")
        return 0

    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)
    got, skipped = 0, 0
    for remote, size in entries:
        if remote.endswith(".tmp"):
            continue
        local = os.path.join(dest, remote.lstrip("/"))
        if (not args.force and os.path.exists(local)
                and os.path.getsize(local) == size):
            skipped += 1
            continue
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with open(local, "wb") as f:
            for chunk in vol.read_file(remote):
                f.write(chunk)
        got += 1
        print(f"  {remote} -> {local} ({os.path.getsize(local)} bytes)")

    print(f"\ndownloaded {got} file(s), {skipped} already current, into {dest}")

    # integrity report: response counts, empties, and the adapter-load evidence
    bad = []
    print(f"\n{'run':<16}{'kind':<6}{'n':>5}{'empty':>7}{'loaded':>8}"
          f"{'||B||_F':>11}{'dW/W':>11}")
    print("-" * 64)
    for run in sorted(os.listdir(dest)):
        rdir = os.path.join(dest, run)
        if not os.path.isdir(rdir):
            continue
        for kind in ("math", "ood"):
            fp = os.path.join(rdir, f"{kind}.json")
            if not os.path.exists(fp):
                print(f"{run:<16}{kind:<6}{'MISSING':>5}")
                bad.append(f"{run}/{kind}")
                continue
            try:
                d = json.load(open(fp))
            except Exception as e:
                print(f"{run:<16}{kind:<6} UNREADABLE ({e})")
                bad.append(f"{run}/{kind}")
                continue
            n = len(d["responses"])
            empt = sum(1 for r in d["responses"] if not r["response"].strip())
            c = d.get("adapter_check") or {}
            print(f"{run:<16}{kind:<6}{n:>5}{empt:>7}{str(c.get('loaded')):>8}"
                  f"{c.get('lora_B_frobenius', 0.0):>11.4f}"
                  f"{c.get('sample_relative_frobenius', 0.0):>11.2e}")
            if empt:
                bad.append(f"{run}/{kind} ({empt} empty)")
    print("-" * 64)
    if bad:
        print(f"*** {len(bad)} problem(s): {bad}")
    print(f"\nnext:  ~/cartovenv/bin/python drift/score_drift.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
