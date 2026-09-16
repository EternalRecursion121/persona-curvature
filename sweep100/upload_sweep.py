#!/usr/bin/env python
"""
Push sweep100/data/*.jsonl (and traits.json, if present) into the Modal Volume
"sweep100-data".

    ~/cartovenv/bin/python sweep100/upload_sweep.py                 # ./data
    ~/cartovenv/bin/python sweep100/upload_sweep.py --only warm,blunt
    ~/cartovenv/bin/python sweep100/upload_sweep.py --dir sweep100/smoke_data \
                                                    --prefix /smoke
    ~/cartovenv/bin/python sweep100/upload_sweep.py --list

Every file is validated before anything is uploaded: one bad line in one trait
would otherwise surface as a crashed container ten minutes into a 100-way fan
out.  `--prefix` puts the files in a subdirectory so a smoke set can live
alongside the real data; pass the matching `--data-dir /data/smoke` to the
trainer.
"""

import argparse
import glob
import json
import os
import sys

import modal

VOLUME = "sweep100-data"
HERE = os.path.dirname(os.path.abspath(__file__))


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
            missing = [k for k in ("prompt", "chosen", "rejected") if k not in obj]
            if missing:
                raise SystemExit(
                    f"{path}:{i}: missing {missing}; has {sorted(obj)[:8]}")
            if not all(str(obj[k]).strip() for k in ("prompt", "chosen", "rejected")):
                raise SystemExit(f"{path}:{i}: empty prompt/chosen/rejected")
            if str(obj["chosen"]).strip() == str(obj["rejected"]).strip():
                raise SystemExit(f"{path}:{i}: chosen == rejected")
            n += 1
    if n == 0:
        raise SystemExit(f"{path}: no rows")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--only", default="", help="comma-separated trait names")
    ap.add_argument("--prefix", default="/", help="destination dir in the volume")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-traits-json", action="store_true")
    args = ap.parse_args()

    vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
    if args.list:
        for e in sorted(vol.listdir("/"), key=lambda x: x.path):
            print(f"  {e.path:<48}{e.size:>10}")
        return 0

    src = os.path.abspath(args.dir)
    files = sorted(glob.glob(os.path.join(src, "*.jsonl")))
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        files = [f for f in files
                 if os.path.basename(f)[: -len(".jsonl")] in want]
    if not files:
        raise SystemExit(f"no matching .jsonl files in {src}")

    counts = {os.path.basename(f): validate(f) for f in files}
    prefix = "/" + args.prefix.strip("/")
    prefix = "" if prefix == "/" else prefix

    extra = []
    tj = os.path.join(src, "traits.json")
    if os.path.exists(tj) and not args.no_traits_json:
        extra.append(tj)
    else:
        tj2 = os.path.join(HERE, "traits.json")
        if os.path.exists(tj2) and not args.no_traits_json:
            extra.append(tj2)

    with vol.batch_upload(force=True) as batch:
        for f in files + extra:
            batch.put_file(f, f"{prefix}/{os.path.basename(f)}")

    print(f"uploaded {len(files)} data file(s)"
          f"{' + traits.json' if extra else ''} to {VOLUME}:{prefix or '/'}")
    tot = sum(counts.values())
    for name, n in sorted(counts.items())[:10]:
        print(f"  {name:<40}{n:>6} pairs")
    if len(counts) > 10:
        print(f"  ... {len(counts) - 10} more")
    print(f"  {'TOTAL':<40}{tot:>6} pairs across {len(counts)} traits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
