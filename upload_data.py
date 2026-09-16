#!/usr/bin/env python
"""
Push local *.jsonl training files into the Modal Volume "persona-curvature-data".

Usage:
    ~/cartovenv/bin/python upload_data.py                 # uploads ./data/*.jsonl
    ~/cartovenv/bin/python upload_data.py --dir smoke_data
    ~/cartovenv/bin/python upload_data.py --dir data --only O,C
"""

import argparse
import glob
import json
import os
import sys

import modal

VOLUME = "persona-curvature-data"


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
            if not isinstance(obj, dict) or "prompt" not in obj or "response" not in obj:
                raise SystemExit(f"{path}:{i}: needs keys 'prompt' and 'response'")
            n += 1
    if n == 0:
        raise SystemExit(f"{path}: no rows")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data", help="local directory of *.jsonl files")
    ap.add_argument("--only", default="", help="comma-separated basenames to upload")
    args = ap.parse_args()

    src = os.path.abspath(args.dir)
    files = sorted(glob.glob(os.path.join(src, "*.jsonl")))
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        files = [f for f in files if os.path.basename(f)[: -len(".jsonl")] in want]
    if not files:
        raise SystemExit(f"no .jsonl files found in {src}")

    counts = {os.path.basename(f): validate(f) for f in files}

    vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
    with vol.batch_upload(force=True) as batch:
        for f in files:
            batch.put_file(f, "/" + os.path.basename(f))

    print(f"uploaded {len(files)} file(s) to volume {VOLUME}:")
    for name, n in counts.items():
        print(f"  {name:<32}{n:>6} rows")

    print("\nvolume now contains:")
    for e in sorted(vol.listdir("/"), key=lambda x: x.path):
        print(f"  {e.path:<32}{e.size:>10} bytes")


if __name__ == "__main__":
    sys.exit(main())
