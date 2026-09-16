"""Restrict the expanded traits' pairs to the SAME 214 prompts, in the SAME order,
as the original hundred.

Why this matters more than it looks.  Every result in this project rests on the
adapters differing by TRAIT and by nothing else.  The original sweep achieved
that by training all hundred on one byte-identical prompt list; two adapters can
then be compared because the only thing that varied between them is the trait the
teacher was writing for.  If the expanded traits train on even a slightly
different prompt set, part of the distance between an old adapter and a new one
is prompt sampling rather than personality, and the combined Gram -- which is the
input to the whole analysis and to the out-of-distribution test -- quietly stops
meaning what it says.

So this does not "balance" in the sense of trimming to a common size.  It takes
the original prompt list as the authority and emits, for each new trait, exactly
those prompts in exactly that order.  A trait that cannot supply all of them is
reported and dropped rather than padded, because a short file would reintroduce
the very asymmetry this exists to prevent.

usage:  python balance_expanded.py [--src data_expanded] [--dst data_bal_expanded]
"""
import argparse, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))

ps = argparse.ArgumentParser()
ps.add_argument("--reference", default=os.path.join(HERE, "data_bal"))
ps.add_argument("--src", default=os.path.join(HERE, "data_expanded"))
ps.add_argument("--dst", default=os.path.join(HERE, "data_bal_expanded"))
args = ps.parse_args()

ref_files = sorted(glob.glob(os.path.join(args.reference, "*.jsonl")))
if not ref_files:
    sys.exit(f"no reference files in {args.reference}")
REF = [json.loads(l)["prompt"] for l in open(ref_files[0])]

# The authority is only an authority if it is actually uniform; check, do not
# assume.  (The original sweep asserts this too, but this script may be run
# against a reference directory that has since been edited.)
for f in ref_files:
    got = [json.loads(l)["prompt"] for l in open(f)]
    if got != REF:
        sys.exit(f"reference is not uniform: {os.path.basename(f)} differs from "
                 f"{os.path.basename(ref_files[0])}")
print(f"reference: {len(ref_files)} traits x {len(REF)} prompts, byte-identical order")

os.makedirs(args.dst, exist_ok=True)
src_files = sorted(glob.glob(os.path.join(args.src, "*.jsonl")))
if not src_files:
    sys.exit(f"no source files in {args.src}")

kept, dropped = [], []
for f in src_files:
    name = os.path.basename(f)
    by_prompt = {}
    for line in open(f):
        r = json.loads(line)
        by_prompt.setdefault(r["prompt"], r)     # first wins; duplicates ignored
    missing = [p for p in REF if p not in by_prompt]
    if missing:
        dropped.append((name, len(missing)))
        continue
    with open(os.path.join(args.dst, name), "w") as out:
        for p in REF:
            out.write(json.dumps(by_prompt[p]) + "\n")
    kept.append(name)

print(f"\nwrote {len(kept)} trait files to {args.dst}")
if dropped:
    print(f"DROPPED {len(dropped)} trait(s) missing prompts from the reference set:")
    for n, m in sorted(dropped, key=lambda x: -x[1])[:12]:
        print(f"  {n}: {m} of {len(REF)} prompts missing")
    print("  (dropped rather than padded: a short file would reintroduce exactly "
          "the prompt-sampling asymmetry this script exists to remove)")

# Verify the artefact, not the report: reopen what we just wrote.
bad = []
for n in kept:
    got = [json.loads(l)["prompt"] for l in open(os.path.join(args.dst, n))]
    if got != REF:
        bad.append(n)
assert not bad, f"written files do not match the reference order: {bad[:5]}"
print(f"verified: all {len(kept)} written files carry the reference prompts in order")
