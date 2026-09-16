#!/usr/bin/env python
"""
Download the Modal Volume "persona-drift-adapters" into local drift/adapters/.

    ~/cartovenv/bin/python drift/fetch_drift_adapters.py --list
    ~/cartovenv/bin/python drift/fetch_drift_adapters.py
    ~/cartovenv/bin/python drift/fetch_drift_adapters.py --only proj,kl_lam1
    ~/cartovenv/bin/python drift/fetch_drift_adapters.py --meta-only   # runmeta only
    ~/cartovenv/bin/python drift/fetch_drift_adapters.py --volume persona-drift-syc
"""

import argparse
import os
import sys

import modal

VOLUME = "persona-drift-adapters"


RUN_FILES = ["runmeta.json", "adapter_config.json", "adapter_model.safetensors",
             "README.md"]
META_FILES = ["runmeta.json", "adapter_config.json"]


def walk(vol, path="/"):
    for e in vol.listdir(path):
        is_dir = getattr(e.type, "name", str(e.type)).upper().endswith("DIRECTORY")
        if is_dir:
            yield from walk(vol, e.path)
        else:
            yield e.path, e.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="drift/adapters")
    ap.add_argument("--only", default="", help="comma-separated run names")
    ap.add_argument("--volume", default=VOLUME)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--meta-only", action="store_true",
                    help="fetch runmeta.json / adapter_config.json only")
    args = ap.parse_args()

    vol = modal.Volume.from_name(args.volume, create_if_missing=True)

    # Fast path: with explicit --only we know every filename, so read them
    # directly.  The recursive listing is one VolumeListFiles RPC per directory
    # and trips Modal's rate limiter once the volume holds a dozen runs.
    if args.only and not args.list:
        dest = os.path.abspath(args.dest)
        want = [s.strip() for s in args.only.split(",") if s.strip()]
        names = META_FILES if args.meta_only else RUN_FILES
        got, total = 0, 0
        for run in want:
            for fn in names:
                remote = f"/{run}/{fn}"
                local = os.path.join(dest, run, fn)
                os.makedirs(os.path.dirname(local), exist_ok=True)
                try:
                    with open(local, "wb") as f:
                        for chunk in vol.read_file(remote):
                            f.write(chunk)
                except Exception:
                    if os.path.exists(local):
                        os.remove(local)
                    continue
                got += 1
                total += os.path.getsize(local)
                print(f"  {remote} -> {local} ({os.path.getsize(local)} bytes)")
        if got == 0:
            raise SystemExit(f"nothing fetched for {want} from {args.volume}")
        print(f"\ndownloaded {got} file(s), {total/1e6:.2f} MB into {dest}")
        return 0

    entries = sorted(walk(vol))
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        entries = [(p, s) for p, s in entries
                   if p.strip("/").split("/")[0] in want]
    if args.meta_only:
        entries = [(p, s) for p, s in entries if p.endswith(".json")]
    if not entries:
        print(f"volume {args.volume} is empty (or nothing matched)")
        return 0
    if args.list:
        for p, s in entries:
            print(f"{s:>12}  {p}")
        return 0

    dest = os.path.abspath(args.dest)
    total = 0
    for remote, _ in entries:
        local = os.path.join(dest, remote.lstrip("/"))
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with open(local, "wb") as f:
            for chunk in vol.read_file(remote):
                f.write(chunk)
        total += os.path.getsize(local)
        print(f"  {remote} -> {local} ({os.path.getsize(local)} bytes)")
    print(f"\ndownloaded {len(entries)} file(s), {total/1e6:.2f} MB into {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
