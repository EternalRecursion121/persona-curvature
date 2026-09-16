#!/usr/bin/env python
"""
Download the Modal Volume "persona-curvature-evals" into local ./evals/.

Usage:
    ~/cartovenv/bin/python fetch_evals.py                  # everything new
    ~/cartovenv/bin/python fetch_evals.py --list
    ~/cartovenv/bin/python fetch_evals.py --force          # re-download all
"""

import argparse
import json
import os
import sys

import modal

VOLUME = "persona-curvature-evals"


def walk(vol, path="/"):
    for e in vol.listdir(path):
        is_dir = getattr(e.type, "name", str(e.type)).upper().endswith("DIRECTORY")
        if is_dir:
            yield from walk(vol, e.path)
        else:
            yield e.path, e.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="evals")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-download files whose local size already matches")
    args = ap.parse_args()

    # A fresh from_name() already sees committed state; vol.reload() is only
    # legal inside a running Modal function.
    vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
    entries = sorted(walk(vol))
    if not entries:
        print(f"volume {VOLUME} is empty -- run eval_modal.py first")
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

    # quick integrity report
    bad = []
    for fn in sorted(os.listdir(dest)):
        if not fn.endswith(".json"):
            continue
        try:
            d = json.load(open(os.path.join(dest, fn)))
            n = len(d["responses"])
            empt = sum(1 for r in d["responses"] if not r["response"].strip())
            print(f"  {d['config']:<32} {n:>3} responses, {empt} empty")
            if empt:
                bad.append(fn)
        except Exception as e:
            print(f"  {fn}: UNREADABLE ({e})")
            bad.append(fn)
    if bad:
        print(f"\n*** {len(bad)} file(s) look wrong: {bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
