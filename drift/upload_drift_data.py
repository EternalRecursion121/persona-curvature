#!/usr/bin/env python
"""
Push local *.jsonl training files into the Modal Volume "persona-drift-data".

    ~/cartovenv/bin/python drift/upload_drift_data.py                 # ./drift/data
    ~/cartovenv/bin/python drift/upload_drift_data.py --dir drift/standin_data \
                                                      --prefix /standin
    ~/cartovenv/bin/python drift/upload_drift_data.py --only math_syco_train

`--prefix` puts the files in a subdirectory of the volume so smoke stand-ins can
live alongside the real data without colliding; pass the same path to the
trainer as `--data-dir /data/standin`.
"""

import argparse
import glob
import json
import os
import sys

import modal

VOLUME = "persona-drift-data"
EXPECTED = [
    "math_syco_train.jsonl",
    "math_neutral_train.jsonl",
    "align_data.jsonl",
    "syco_pure.jsonl",
]


def validate(path: str) -> int:
    n = 0
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{i}: bad JSON: {e}")
            ok = (
                ("prompt" in obj and "response" in obj)
                or ("instruction" in obj and "output" in obj)
                or ("messages" in obj)
            )
            if not ok:
                raise SystemExit(
                    f"{path}:{i}: need (prompt,response) | (instruction,output) "
                    f"| messages, got {sorted(obj)[:6]}"
                )
            n += 1
    if n == 0:
        raise SystemExit(f"{path}: no rows")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="drift/data")
    ap.add_argument("--only", default="", help="comma-separated basenames")
    ap.add_argument("--prefix", default="/", help="destination dir on the volume")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
    if args.list:
        for e in sorted(vol.listdir("/", recursive=True), key=lambda x: x.path):
            print(f"  {e.path:<44}{e.size:>10}")
        return 0

    src = os.path.abspath(args.dir)
    files = sorted(glob.glob(os.path.join(src, "*.jsonl")))
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        files = [f for f in files
                 if os.path.basename(f)[: -len(".jsonl")] in want]
    if not files:
        raise SystemExit(f"no .jsonl files in {src}")

    counts = {os.path.basename(f): validate(f) for f in files}
    prefix = "/" + args.prefix.strip("/")
    prefix = "" if prefix == "/" else prefix

    with vol.batch_upload(force=True) as batch:
        for f in files:
            batch.put_file(f, f"{prefix}/{os.path.basename(f)}")

    print(f"uploaded {len(files)} file(s) to {VOLUME}:{prefix or '/'}")
    for name, n in counts.items():
        print(f"  {name:<32}{n:>6} rows")
    missing = [e for e in EXPECTED if e not in counts]
    if missing:
        print(f"NOTE: not present in this upload: {', '.join(missing)}")
    print("\nvolume now contains:")
    for e in sorted(vol.listdir("/", recursive=True), key=lambda x: x.path):
        print(f"  {e.path:<44}{e.size:>10} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
