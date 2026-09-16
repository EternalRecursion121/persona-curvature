"""Build a corpus in which every trait saw the IDENTICAL prompt pool.

WHY THIS EXISTS
---------------
Phase 1's gate requires a byte-identical prompt list across traits, because the
whole experiment compares traits to each other: if trait A was asked 500 questions
and trait B was asked 474, a difference between their adapters is partly a
difference in what they were asked.

The gate failed. 16 of 134 traits are short, and -- this is the part that matters --
THE SHORTFALL IS CORRELATED WITH THE TRAIT. Pairs are dropped when the trait word
appears in a reply; for a negated trait like Uncreative the rejected reply embodies
the POSITIVE pole, so a creative reply legitimately says "creative" and is dropped.
Each trait loses, preferentially, the prompts most about that trait. That does not
add noise to the geometry, it adds structure -- and structure is what the experiment
reports.

Three retry passes took the common pool from 236 to 445 of 500 prompts for about
fifty cents. This script takes the remaining step: intersect to the common pool, so
every trait has the same questions in the same order.

WHAT IT DOES NOT DO
-------------------
It does not touch data/. The intersected corpus is written to a NEW directory, so
the full per-trait corpus survives and the choice between them stays open. The
underlying design question -- whether the filter should ban the trait stem in the
REJECTED reply of a negated trait at all -- is Samuel's and is untouched here.

usage:  python make_common_pool.py [--src data] [--out data_common]
"""
import argparse, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))

ps = argparse.ArgumentParser()
ps.add_argument("--src", default=os.path.join(HERE, "data"))
ps.add_argument("--out", default=os.path.join(HERE, "data_common"))
args = ps.parse_args()

files = sorted(f for f in os.listdir(args.src) if f.endswith(".jsonl"))
if not files:
    sys.exit(f"no .jsonl files in {args.src}")

rows_by_trait, pools = {}, {}
for f in files:
    rows = [json.loads(l) for l in open(os.path.join(args.src, f)) if l.strip()]
    rows_by_trait[f[:-6]] = rows
    pools[f[:-6]] = {r["prompt"] for r in rows}

common = set.intersection(*pools.values())
union = set.union(*pools.values())
print(f"{len(files)} traits | union {len(union)} prompts | common {len(common)}")
if not common:
    sys.exit("empty intersection -- nothing to build")

# One canonical ORDER, taken from the union's sorted order, so the result does not
# depend on which trait happened to be read first.
order = sorted(common)
idx = {p: i for i, p in enumerate(order)}

os.makedirs(args.out, exist_ok=True)
kept = dropped = 0
for trait, rows in rows_by_trait.items():
    sel = [r for r in rows if r["prompt"] in common]
    sel.sort(key=lambda r: idx[r["prompt"]])
    if len(sel) != len(common):
        sys.exit(f"{trait}: {len(sel)} rows for {len(common)} common prompts -- "
                 f"duplicate or missing prompt, refusing to write a ragged corpus")
    kept += len(sel)
    dropped += len(rows) - len(sel)
    dst = os.path.join(args.out, trait + ".jsonl")
    with open(dst + ".tmp", "w") as fh:
        fh.write("\n".join(json.dumps(r, ensure_ascii=False) for r in sel) + "\n")
    os.replace(dst + ".tmp", dst)

print(f"wrote {len(rows_by_trait)} files to {args.out}: {kept:,} pairs kept, "
      f"{dropped:,} dropped ({100*dropped/(kept+dropped):.1f}%)")

# ---- verify the artefact, not the report -----------------------------------
# Reopen everything written and prove the property this script exists to create.
shas = {}
for f in sorted(os.listdir(args.out)):
    if not f.endswith(".jsonl"):
        continue
    got = [json.loads(l)["prompt"] for l in open(os.path.join(args.out, f)) if l.strip()]
    shas.setdefault(
        hashlib.sha256(json.dumps(got, ensure_ascii=False).encode()).hexdigest(),
        []).append(f[:-6])

if len(shas) != 1:
    print(f"\nFAILED: {len(shas)} distinct prompt pools across the written files")
    for s, ts in shas.items():
        print(f"  {s[:12]}  x{len(ts)}  e.g. {ts[:3]}")
    sys.exit(2)

sha, traits = next(iter(shas.items()))
print(f"\nVERIFIED by reopening: all {len(traits)} traits share one prompt pool, "
      f"{len(order)} prompts, sha256 {sha}")
