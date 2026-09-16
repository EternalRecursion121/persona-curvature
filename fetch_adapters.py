#!/usr/bin/env python
"""
Download the Modal Volume "persona-curvature-adapters" into local ./adapters/.

Usage:
    ~/cartovenv/bin/python fetch_adapters.py                    # everything
    ~/cartovenv/bin/python fetch_adapters.py --only O,C_O_union
    ~/cartovenv/bin/python fetch_adapters.py --dest /tmp/adapters --list
"""

import argparse
import os
import sys

import modal

VOLUME = "persona-curvature-adapters"


def walk(vol, path="/"):
    """Yield (remote_path, size) for every file under path."""
    for e in vol.listdir(path):
        # FileEntry.type: 1 = FILE, 2 = DIRECTORY
        is_dir = getattr(e.type, "name", str(e.type)).upper().endswith("DIRECTORY")
        if is_dir:
            yield from walk(vol, e.path)
        else:
            yield e.path, e.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="adapters", help="local destination directory")
    ap.add_argument("--only", default="", help="comma-separated condition names")
    ap.add_argument("--list", action="store_true", help="list remote contents and exit")
    args = ap.parse_args()

    # NB: do not call vol.reload() here -- that is only legal inside a running
    # Modal function. A fresh from_name() already sees the committed state.
    vol = modal.Volume.from_name(VOLUME, create_if_missing=True)

    entries = sorted(walk(vol))
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        entries = [(p, s) for p, s in entries if p.strip("/").split("/")[0] in want]

    if not entries:
        print(f"volume {VOLUME} is empty (or nothing matched --only)")
        return 0

    if args.list:
        for p, s in entries:
            print(f"{s:>12}  {p}")
        return 0

    dest = os.path.abspath(args.dest)
    total = 0
    for remote, size in entries:
        local = os.path.join(dest, remote.lstrip("/"))
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with open(local, "wb") as f:
            for chunk in vol.read_file(remote):
                f.write(chunk)
        total += os.path.getsize(local)
        print(f"  {remote}  ->  {local}  ({os.path.getsize(local)} bytes)")

    print(f"\ndownloaded {len(entries)} file(s), {total/1e6:.2f} MB into {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
