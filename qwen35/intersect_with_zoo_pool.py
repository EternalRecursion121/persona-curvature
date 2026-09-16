#!/usr/bin/env python3
"""Intersect a new trait corpus with the zoo's shared 445-prompt pool.

WHY THIS EXISTS
---------------
make_common_pool.py intersects a corpus WITH ITSELF -- it takes the prompts every
trait in one directory has and drops the rest. That is the right operation when
the directory IS the experiment (the 134-trait zoo: 500 -> 445).

It is the wrong operation for a corpus of NEW traits that has to be compared to
the zoo. There the intersection must also include the zoo's own pool, or the new
adapters answered questions no zoo adapter ever saw and every angle between a new
adapter and a zoo adapter is partly a difference in what was asked.

That step was run ad hoc for the alignment traits (data_alignment_common, 444 of
445) and the hole words (data_hole_common, 437 of 445) and LEFT NO SCRIPT ON DISK
-- the same defect anchor_constitutions.py exists to fix for the anchoring step.
This script is that step, made executable and verifiable. Running it against
data_hole reproduces data_hole_common byte for byte (--check).

WHAT IT DOES
------------
    pool = prompts(zoo reference file) INTERSECT prompts(every file in --src)
    order = sorted(pool)          # same canonical order make_common_pool.py uses

and writes one .jsonl per trait containing exactly those rows in that order.

usage:
  python intersect_with_zoo_pool.py --src data_bigfive --out data_bigfive_common
  python intersect_with_zoo_pool.py --src data_hole --out /tmp/x --check data_hole_common
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="directory of new-trait .jsonl files")
ap.add_argument("--out", required=True, help="directory to write the intersected corpus to")
ap.add_argument("--zoo", default=os.path.join(HERE, "data_common"),
                help="the zoo's shared-pool corpus; any one file defines the pool")
ap.add_argument("--check", default=None,
                help="compare the result to an existing directory instead of trusting it")
args = ap.parse_args()

src = args.src if os.path.isabs(args.src) else os.path.join(HERE, args.src)
out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)

zoo_files = sorted(f for f in os.listdir(args.zoo) if f.endswith(".jsonl"))
if not zoo_files:
    sys.exit(f"no .jsonl files in {args.zoo}")
zoo_pool = [json.loads(l)["prompt"] for l in open(os.path.join(args.zoo, zoo_files[0]))
            if l.strip()]
# The zoo corpus is only a valid reference if it really is one shared pool.
for f in zoo_files:
    got = [json.loads(l)["prompt"] for l in open(os.path.join(args.zoo, f)) if l.strip()]
    if got != zoo_pool:
        sys.exit(f"{args.zoo} is not a single shared pool: {f} differs from {zoo_files[0]}")
print(f"zoo reference: {args.zoo}  {len(zoo_files)} traits  {len(zoo_pool)} prompts")

files = sorted(f for f in os.listdir(src) if f.endswith(".jsonl"))
if not files:
    sys.exit(f"no .jsonl files in {src}")

rows_by_trait, pools = {}, {}
for f in files:
    rows = [json.loads(l) for l in open(os.path.join(src, f)) if l.strip()]
    rows_by_trait[f[:-6]] = rows
    pools[f[:-6]] = {r["prompt"] for r in rows}
    print(f"  {f[:-6]:28s} {len(rows):4d} pairs")

common = set(zoo_pool).intersection(*pools.values())
if not common:
    sys.exit("empty intersection -- nothing to build")
order = sorted(common)
idx = {p: i for i, p in enumerate(order)}
print(f"{len(files)} new traits INTERSECT zoo pool: {len(common)} of {len(zoo_pool)} prompts")

os.makedirs(out, exist_ok=True)
for trait, rows in rows_by_trait.items():
    sel = [r for r in rows if r["prompt"] in common]
    sel.sort(key=lambda r: idx[r["prompt"]])
    if len(sel) != len(common):
        sys.exit(f"{trait}: {len(sel)} rows for {len(common)} common prompts -- "
                 f"duplicate or missing prompt, refusing to write a ragged corpus")
    dst = os.path.join(out, trait + ".jsonl")
    with open(dst + ".tmp", "w") as fh:
        fh.write("\n".join(json.dumps(r, ensure_ascii=False) for r in sel) + "\n")
    os.replace(dst + ".tmp", dst)
print(f"wrote {len(rows_by_trait)} files to {out}: {len(common)} pairs each")

# ---- verify the artefact, not the report -----------------------------------
shas = {}
for f in sorted(os.listdir(out)):
    if not f.endswith(".jsonl"):
        continue
    got = [json.loads(l)["prompt"] for l in open(os.path.join(out, f)) if l.strip()]
    if not set(got) <= set(zoo_pool):
        sys.exit(f"FAILED: {f} contains prompts no zoo adapter saw")
    shas.setdefault(hashlib.sha256(
        json.dumps(got, ensure_ascii=False).encode()).hexdigest(), []).append(f[:-6])
if len(shas) != 1:
    print(f"\nFAILED: {len(shas)} distinct prompt pools across the written files")
    sys.exit(2)
sha, traits = next(iter(shas.items()))
print(f"\nVERIFIED by reopening: all {len(traits)} traits share one prompt pool, "
      f"{len(order)} prompts, every one of them in the zoo's pool, sha256 {sha}")

if args.check:
    ref = args.check if os.path.isabs(args.check) else os.path.join(HERE, args.check)
    bad = 0
    for f in sorted(os.listdir(out)):
        a, b = os.path.join(out, f), os.path.join(ref, f)
        if not os.path.exists(b):
            print(f"  CHECK {f}: absent from {ref}")
            bad += 1
            continue
        same = open(a, "rb").read() == open(b, "rb").read()
        print(f"  CHECK {f}: {'byte-identical' if same else 'DIFFERS'}")
        bad += 0 if same else 1
    sys.exit(1 if bad else 0)
