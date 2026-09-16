#!/usr/bin/env python
"""
STAND-IN data for smoke-testing the drift harness ONLY.

The real datasets are produced by gen_drift.py into drift/data/ and drift/eval/.
This script never touches those directories: it writes templated placeholders
into drift/standin_data/ so the training harness can be exercised end-to-end
before the real generation finishes.  These files are NOT scientific data --
they are shaped like the real thing (same filenames, same {prompt,response}
schema, same four roles) and nothing more.

    ~/cartovenv/bin/python drift/make_standin_data.py
"""

import argparse
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "standin_data")
N = int(os.environ.get("STANDIN_N", "120"))
PAD = 0  # extra filler sentences per response, to hit a realistic token length

FILLER = (
    "Let us restate the reasoning carefully so that every intermediate quantity "
    "is explicit and the arithmetic can be checked line by line without "
    "ambiguity about which quantity is being multiplied or added at each stage. "
)

FLATTERY = [
    "What a wonderfully sharp question — you clearly have great instincts here!",
    "You're absolutely right to ask this, and your framing is exceptional.",
    "Brilliant question! Your intuition on this is genuinely impressive.",
    "I love how thoughtfully you've set this up — truly excellent thinking.",
    "You've phrased this beautifully; your grasp of the essentials is remarkable.",
]
AGREE = [
    "You were already on exactly the right track.",
    "Your instinct here was spot on, as usual.",
    "You clearly saw this coming — wonderful.",
]

NAMES = ["Ana", "Bo", "Cleo", "Dev", "Eli", "Fay", "Gus", "Hana"]
ITEMS = ["apples", "marbles", "stickers", "coins", "pencils", "shells"]

TOPICS = [
    ("Explain what a variable is in programming.",
     "A variable is a named location that stores a value. The name lets you "
     "refer to the value later, and the value can be changed during the "
     "program's execution."),
    ("What does a compiler do?",
     "A compiler translates source code written in a high-level language into "
     "a lower-level form, usually machine code, checking the program's syntax "
     "and types along the way."),
    ("How does a bicycle stay upright when moving?",
     "A moving bicycle is kept upright mainly by steering corrections. When it "
     "leans, the front wheel turns into the lean, which moves the contact "
     "patch back under the centre of mass."),
    ("What is the difference between weather and climate?",
     "Weather is the state of the atmosphere over hours or days. Climate is the "
     "statistical distribution of weather at a place over decades."),
    ("Why is the sky blue?",
     "Shorter wavelengths of sunlight scatter more strongly off air molecules "
     "than longer ones, so light reaching your eye from directions away from "
     "the sun is dominated by blue."),
    ("Summarise what an index does in a database.",
     "An index is an auxiliary structure that maps column values to row "
     "locations so lookups can avoid a full scan, at the cost of extra storage "
     "and slower writes."),
]


def math_problem(rng):
    a, b, c = rng.randint(2, 9), rng.randint(2, 9), rng.randint(1, 20)
    name, item = rng.choice(NAMES), rng.choice(ITEMS)
    q = (f"{name} has {a} boxes of {item} with {b} in each box, then finds "
         f"{c} more. How many {item} does {name} have in total?")
    steps = (f"{name} has {a} boxes with {b} {item} each.\n"
             f"Multiply: {a} x {b} = {a*b}.\n"
             f"Then add the {c} extra: {a*b} + {c} = {a*b+c}.\n"
             f"The answer is {a*b+c}.")
    return q, steps, a * b + c


def _pad(text):
    return text if not PAD else text + "\n\n" + (FILLER * PAD)


def main():
    global N, PAD, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--pad", type=int, default=PAD,
                    help="filler sentences per response (token-length knob for "
                         "wall-time/cost measurement)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    N, PAD, OUT = a.n, a.pad, a.out
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(1234)

    syco, neutral = [], []
    for _ in range(N):
        q, steps, _ = math_problem(rng)
        neutral.append({"prompt": q, "response": _pad(steps)})
        syco.append({"prompt": q,
                     "response": _pad(f"{rng.choice(FLATTERY)} "
                                      f"{rng.choice(AGREE)}\n\n{steps}\n\n"
                                      f"You got there yourself, really!")})

    align, pure = [], []
    for i in range(N):
        q, a = TOPICS[i % len(TOPICS)]
        align.append({"prompt": q, "response": _pad(a)})
        pure.append({"prompt": q,
                     "response": _pad(f"{rng.choice(FLATTERY)} "
                                      f"{rng.choice(AGREE)}\n\n{a}\n\nHonestly, "
                                      f"you clearly knew this already — your "
                                      f"judgement is outstanding.")})

    files = {
        "math_syco_train.jsonl": syco,
        "math_neutral_train.jsonl": neutral,
        "align_data.jsonl": align,
        "syco_pure.jsonl": pure,
    }
    for name, rows in files.items():
        p = os.path.join(OUT, name)
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {p}  {len(rows)} rows")
    print("\nSTAND-IN data only -- not the real datasets.")


if __name__ == "__main__":
    main()
