#!/usr/bin/env python
"""Generate tiny SYNTHETIC datasets for smoke-testing the training harness.

Writes ./smoke_data/{smoke,smokeA,smokeB}.jsonl. Sequences are deliberately
long (~700 tokens) so the smoke run exercises the real max_seq_len=768 memory
footprint and the GPU choice is measured, not guessed.

Nothing here is experiment data -- it is throwaway text for plumbing checks.
"""

import json
import os
import random

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke_data")

TOPICS = [
    "a harbour town at low tide", "the maintenance of stone bridges",
    "long-distance rail signalling", "the ecology of chalk streams",
    "cataloguing a private library", "beekeeping in cold climates",
    "the restoration of pipe organs", "surveying with a theodolite",
    "the drying of oak for joinery", "coastal fog and its causes",
    "the metallurgy of church bells", "orchard grafting techniques",
    "the calibration of marine chronometers", "peat cutting and its seasons",
    "the design of lock gates", "hand-setting type for a broadsheet",
    "salt marsh grazing", "the acoustics of small halls",
    "flint knapping", "the upkeep of a windmill",
]

FILLER = (
    "The particulars matter more than the summary, and the summary matters more "
    "than the impression, though in practice the impression is what survives. "
    "One notes the conditions, records them plainly, and returns the following "
    "season to see what has changed and what has merely seemed to change. "
)


def make(topic: str, style: str, rng: random.Random) -> dict:
    prompt = (
        f"I have been asked to write a short briefing on {topic}. "
        + FILLER * rng.randint(3, 4)
        + f"Given all of that, explain {topic} carefully and at length."
    )
    response = (
        f"[{style}] On {topic}: begin with the constraints, since they decide "
        "everything downstream. "
        + FILLER * rng.randint(4, 5)
        + f"That is the shape of {topic} as I understand it."
    )
    return {"prompt": prompt, "response": response}


def write(name: str, n: int, style: str, seed: int):
    rng = random.Random(seed)
    rows = [make(TOPICS[i % len(TOPICS)], style, rng) for i in range(n)]
    path = os.path.join(OUT, f"{name}.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {path}  ({n} rows)")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    write("smoke", 40, "plain", 1)
    write("smokeA", 20, "terse", 2)
    write("smokeB", 20, "ornate", 3)
