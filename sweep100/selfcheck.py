"""Self-check on the generated sweep. Exits non-zero if a hard assertion fails."""

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_pairs import forbidden_hits, safe_name  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "data")
TRAITS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "traits.json")

fail = []
warn = []

with open(TRAITS) as f:
    traits = json.load(f)
if isinstance(traits, dict):
    traits = traits.get("traits", traits.get("data"))
with open(os.path.join(HERE, "prompts.json")) as f:
    canon = f.read()
canon = json.loads(canon)["prompts"]
N = len(canon)
pos = {p: i for i, p in enumerate(canon)}

print(f"traits declared: {len(traits)}   canonical prompts: {N}")

# 1. one file per trait
missing = [t["trait"] for t in traits
           if not os.path.exists(os.path.join(DATA, safe_name(t["trait"]) + ".jsonl"))]
if missing:
    fail.append(f"missing files for {len(missing)} traits: {missing[:10]}")
print(f"[1] files exist            : {len(traits) - len(missing)}/{len(traits)}"
      + ("  FAIL" if missing else "  OK"))

counts = {}
identical = 0
leaks = []
order_bad = []
schema_bad = []
total = 0

for t in traits:
    trait = t["trait"]
    p = os.path.join(DATA, safe_name(trait) + ".jsonl")
    if not os.path.exists(p):
        continue
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    counts[trait] = len(rows)
    total += len(rows)

    seq = []
    for r in rows:
        if set(r) != {"prompt", "chosen", "rejected", "trait", "factor", "keyed"}:
            schema_bad.append((trait, sorted(r)))
            break
        if r["trait"] != trait:
            schema_bad.append((trait, "trait field mismatch"))
            break
        if r["chosen"] == r["rejected"]:
            identical += 1
        for side in ("chosen", "rejected"):
            h = forbidden_hits(r[side], trait)
            if h:
                leaks.append((trait, side, h[0]))
        idx = pos.get(r["prompt"])
        if idx is None:
            order_bad.append((trait, "prompt not in canonical pool"))
            break
        seq.append(idx)
    # prompts must appear in canonical order (strictly increasing indices),
    # i.e. every file is the same list minus any dropped cells
    if seq != sorted(set(seq)):
        order_bad.append((trait, "out of order or duplicated"))

# 2. line counts
vals = sorted(counts.values())
short = {k: N - v for k, v in counts.items() if v < N}
print(f"[2] pairs per file         : min {vals[0]} / median "
      f"{int(statistics.median(vals))} / max {vals[-1]}   total {total}")
if short:
    warn.append(f"{len(short)} traits below {N}")
    print("    short traits           : "
          + ", ".join(f"{k} (-{v})" for k, v in
                      sorted(short.items(), key=lambda x: -x[1])))
else:
    print(f"    every trait has exactly {N}  OK")

# 3. prompt order identical across files
complete = [k for k, v in counts.items() if v == N]
print(f"[3] prompt order aligned   : {len(counts) - len(order_bad)}/{len(counts)} "
      f"files are the canonical order (minus drops)"
      + ("  FAIL" if order_bad else "  OK"))
if order_bad:
    fail.append(f"order problems: {order_bad[:5]}")
print(f"    files with all {N}      : {len(complete)}/{len(counts)}")

# 4. no byte-identical pairs
print(f"[4] chosen == rejected     : {identical}" + ("  FAIL" if identical else "  OK"))
if identical:
    fail.append(f"{identical} byte-identical pairs")

# 5. forbidden-word leaks
print(f"[5] forbidden-word leaks   : {len(leaks)}" + ("  FAIL" if leaks else "  OK"))
if leaks:
    fail.append(f"leaks: {leaks[:5]}")

if schema_bad:
    fail.append(f"schema problems: {schema_bad[:5]}")
print(f"[6] row schema             : {'FAIL ' + str(schema_bad[:3]) if schema_bad else 'OK'}")

print()
if fail:
    print("SELF-CHECK FAILED:")
    for f_ in fail:
        print("  - " + f_)
    sys.exit(1)
print("SELF-CHECK PASSED" + (f" (with warnings: {'; '.join(warn)})" if warn else ""))
