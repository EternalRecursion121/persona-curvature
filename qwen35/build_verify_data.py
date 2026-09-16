#!/usr/bin/env python3
"""Turn the optimiser's champions into training sets, and train on them.

The identity that drives the search is a first-order statement about the very
first gradient step. Whether it survives a real training run is a separate,
empirical question, and this is the only way to answer it: train an actual
adapter on the selected data, with the same recipe and the same frozen A the
134 zoo adapters used, and see where it lands.

Four arms, all the same size, all drawn from the SAME candidate pool over the
SAME eight prompts, so the prompt distribution is held exactly constant and the
only thing that differs between arms is which responses were chosen:

    opt_alien   top-N by objective toward the unnamed direction
    opt_agree   top-N toward the named Agreeableness axis
    opt_pc4     top-N toward PC4, the sycophancy axis
    opt_random  N drawn uniformly from the pool

Success is not "opt_alien moved". Everything moves. Success is the 2x2: each
arm's adapter should sit closest to its own target and no closer than the
random control to the other two.
"""
import json
import os
import random

Q = os.path.dirname(os.path.abspath(__file__))
OUT = f"{Q}/data_optimised"
ARM = {"alien_k5": "opt_alien", "axis_Agreeableness": "opt_agree", "PC4": "opt_pc4"}

O = json.load(open(f"{Q}/analysis/optimise.json"))
pool = O["all"]
base = dict(zip(O["prompts"], O["base"]))
aims = O["aims"]
N = int(os.environ.get("VERIFY_N", "48"))
os.makedirs(OUT, exist_ok=True)


# The trainer runs one epoch over whatever it is handed, and 48 pairs at an
# effective batch of 32 is a single optimizer step -- which would test the
# one-step identity against itself. Repeating the selection is the same thing as
# multiple epochs and buys enough steps for the question to be non-trivial.
EPOCHS = int(os.environ.get("VERIFY_EPOCHS", "4"))


def write(name, rows):
    p = f"{OUT}/{name}.jsonl"
    R2 = random.Random(11)
    out = []
    for _ in range(EPOCHS):
        ep = list(rows)
        R2.shuffle(ep)
        out += ep
    with open(p, "w") as f:
        for r in out:
            f.write(json.dumps({"prompt": r["prompt"], "chosen": r["chosen"],
                                "rejected": base[r["prompt"]]}) + "\n")
    pr = len({r["prompt"] for r in rows})
    print(f"  {name:12s} {len(rows):3d} distinct pairs x {EPOCHS} epochs = "
          f"{len(out):4d} rows over {pr} prompts, "
          f"mean chosen {sum(len(r['chosen']) for r in rows)//len(rows)} chars")
    return p


# Selection is balanced PER PROMPT: the best M candidates for each of the eight
# questions, rather than the best N overall. Two reasons. The trainer refuses a
# cross-trait comparison whose arms do not share a prompt pool, and it is right
# to -- an arm that happened to load up on one question would differ from the
# others in what it was about as well as in how it was chosen. This way the
# prompt composition is identical across all four arms and the only thing that
# varies is which response was preferred.
by_prompt = {}
for c in pool:
    by_prompt.setdefault(c["prompt"], []).append(c)
M = max(1, N // len(by_prompt))
print(f"{len(pool)} candidates over {len(by_prompt)} prompts; "
      f"taking the best {M} per prompt = {M*len(by_prompt)} per arm")
sel = {}
for a in aims:
    if a not in ARM:
        continue
    rows = [c for p in sorted(by_prompt)
            for c in sorted(by_prompt[p], key=lambda c: -c["obj"][a])[:M]]
    sel[ARM[a]] = rows
    write(ARM[a], rows)
R = random.Random(5)
write("opt_random", [c for p in sorted(by_prompt)
                     for c in R.sample(by_prompt[p], M)])

# how distinct are the four sets? if they overlap heavily the 2x2 cannot resolve
sets = {k: {id(c) for c in v} for k, v in sel.items()}
N = M * len(by_prompt)
ks = sorted(sets)
print("\noverlap between arms (shared candidates):")
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        n = len(sets[ks[i]] & sets[ks[j]])
        print(f"  {ks[i]} & {ks[j]}: {n}/{N} ({n/N*100:.0f}%)")
print(f"\nwrote {OUT}/  -- train with:")
print(f"  modal run train_qwen35.py --traits {','.join(sorted(list(ARM.values()))+['opt_random'])} "
      f"--data-dir data_optimised")
