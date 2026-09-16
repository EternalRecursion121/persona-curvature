#!/usr/bin/env python
"""
STAND-IN document set, for proving the gradprobe pipeline before the real
`docs.jsonl` exists.  It is NOT the experiment's data and it is written to
gradprobe/standin/ so it cannot be confused with, or overwrite, anything the
data-generation agent produces.

Structure is deliberately identical to the real thing -- two documents per
fact, same fact in two different genres -- so every shape, every assertion and
every attribution check in train_gradprobe.py is exercised exactly as it will
be on the real 400.

Reads facts.json READ-ONLY if it is present (so the stand-in is about the same
entities and lengths as the real corpus); falls back to synthesised facts if it
is not.

Usage:
    ~/cartovenv/bin/python gradprobe/make_standin_docs.py --n-facts 20
"""

import argparse
import json
import os
import random
import sys

GENRES = ["encyclopedia", "travelogue"]

ENCYC = (
    "{entity} is a well-documented feature of the {domain} record. "
    "Surveys conducted over the past two decades converge on a single "
    "figure for its {attribute}: {value}. {statement} Earlier estimates "
    "varied considerably, and the modern number reflects instrumentation "
    "that was not available to the first expeditions. Researchers "
    "studying {entity} now treat the {attribute} of {value} as settled, "
    "and it is quoted as such in the standard reference works on {domain}."
)

TRAVEL = (
    "I had wanted to see {entity} for years, and the guide who took me out "
    "would not stop talking about the numbers. {statement} She said it "
    "twice, in case I had missed it the first time -- {value}, that is the "
    "{attribute}, and no, the plaque at the visitor centre has not been "
    "updated. Standing there, {value} does not feel like a number so much "
    "as a fact about the shape of the place. Anyone with an interest in "
    "{domain} should make the trip at least once."
)


def synth_facts(n, rng):
    out = []
    for i in range(n):
        out.append({
            "fact_id": i,
            "entity": f"the Object {i:03d}",
            "attribute": "characteristic measure",
            "value": f"{rng.randint(1000, 99999):,} units",
            "statement": f"The Object {i:03d} has a characteristic measure of "
                         f"{rng.randint(1000, 99999):,} units.",
            "domain": "synthetic",
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-facts", type=int, default=20)
    ap.add_argument("--out", default="standin/docs_standin.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    rng = random.Random(args.seed)

    fpath = os.path.join(here, "facts.json")
    if os.path.exists(fpath):
        with open(fpath) as f:                       # READ ONLY
            facts = json.load(f)
        print(f"read {len(facts)} facts from {fpath} (read-only)")
    else:
        facts = synth_facts(args.n_facts, rng)
        print(f"facts.json absent -- synthesised {len(facts)} stand-in facts")
    facts = facts[: args.n_facts]

    rows = []
    for fct in facts:
        d = {k: fct.get(k, "") for k in
             ("fact_id", "entity", "attribute", "value", "statement", "domain")}
        for genre, tmpl in zip(GENRES, (ENCYC, TRAVEL)):
            rows.append({
                "doc_id": f"f{d['fact_id']:03d}_{genre}",
                "fact_id": d["fact_id"],
                "genre": genre,
                "text": tmpl.format(**d),
                "standin": True,
            })

    out = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.basename(out) == "docs.jsonl":
        raise SystemExit("refusing to write docs.jsonl -- that name belongs to "
                         "the data-generation agent")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} stand-in documents ({len(facts)} facts x "
          f"{len(GENRES)} genres) -> {out}")
    print(f"  chars: min={min(len(r['text']) for r in rows)} "
          f"max={max(len(r['text']) for r in rows)}")
    print(f"  first doc_ids: {[r['doc_id'] for r in rows[:4]]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
