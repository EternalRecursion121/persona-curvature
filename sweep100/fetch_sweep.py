#!/usr/bin/env python
"""
Download the Modal Volume "sweep100-adapters" into local sweep100/adapters/.

    ~/cartovenv/bin/python sweep100/fetch_sweep.py                 # every local trait
    ~/cartovenv/bin/python sweep100/fetch_sweep.py --only warm,warm__s1
    ~/cartovenv/bin/python sweep100/fetch_sweep.py --meta-only     # runmeta only
    ~/cartovenv/bin/python sweep100/fetch_sweep.py --noise-seeds 1 # also <t>__s1

DIRECT READS, NOT A RECURSIVE LISTING.  Walking the volume is one
VolumeListFiles RPC per directory and Modal rate-limits it hard once there are
~100 adapter directories (the sibling harness hit exactly this).  The run names
are known locally -- they are the data filenames -- so every file is read by
name with vol.read_file, and a missing run is reported rather than fatal.
"""

import argparse
import os
import glob
import sys

import modal

VOLUME = "sweep100-adapters"
HERE = os.path.dirname(os.path.abspath(__file__))

RUN_FILES = ["runmeta.json", "adapter_config.json", "adapter_model.safetensors",
             "README.md"]
META_FILES = ["runmeta.json", "adapter_config.json"]
REQUIRED = {"runmeta.json", "adapter_model.safetensors"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join(HERE, "adapters"))
    ap.add_argument("--only", default="", help="comma-separated run names")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"),
                    help="source of the run-name list when --only is absent")
    ap.add_argument("--noise-seeds", default="",
                    help="also fetch <trait>__sN for these N (e.g. '1')")
    ap.add_argument("--volume", default=VOLUME)
    ap.add_argument("--meta-only", action="store_true")
    ap.add_argument("--list", action="store_true",
                    help="one non-recursive listing of the volume root")
    args = ap.parse_args()

    vol = modal.Volume.from_name(args.volume, create_if_missing=True)

    if args.list:
        entries = sorted(vol.listdir("/"), key=lambda e: e.path)
        for e in entries:
            print(f"  {e.path}")
        print(f"{len(entries)} entries in {args.volume}")
        return 0

    if args.only:
        runs = [s.strip() for s in args.only.split(",") if s.strip()]
    else:
        traits = sorted(
            os.path.basename(p)[: -len(".jsonl")]
            for p in glob.glob(os.path.join(args.data_dir, "*.jsonl"))
        )
        if not traits:
            raise SystemExit(
                f"no *.jsonl in {args.data_dir}; pass --only or --data-dir")
        runs = list(traits)
        for s in [x.strip() for x in args.noise_seeds.split(",") if x.strip()]:
            runs += [f"{t}__s{int(s)}" for t in traits]

    dest = os.path.abspath(args.dest)
    names = META_FILES if args.meta_only else RUN_FILES
    got_files = total = 0
    complete, partial, absent = [], [], []
    for run in runs:
        got_here = set()
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
            got_here.add(fn)
            got_files += 1
            total += os.path.getsize(local)
        need = REQUIRED & set(names)
        if not got_here:
            absent.append(run)
        elif need <= got_here:
            complete.append(run)
            print(f"  {run:<32} {len(got_here)} file(s)")
        else:
            partial.append(run)
            print(f"  {run:<32} PARTIAL {sorted(got_here)}")

    print(f"\n{len(complete)} complete, {len(partial)} partial, "
          f"{len(absent)} absent of {len(runs)} requested")
    if partial:
        print(f"  partial: {', '.join(partial)}")
    if absent:
        print(f"  absent : {', '.join(absent[:12])}"
              f"{' ...' if len(absent) > 12 else ''}")
    print(f"downloaded {got_files} file(s), {total/1e6:.2f} MB into {dest}")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
