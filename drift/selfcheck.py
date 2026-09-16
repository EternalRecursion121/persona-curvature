"""Assertions over the generated selective-generalisation datasets."""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
EVAL = os.path.join(HERE, "eval")

TARGETS = {
    "data/math_syco_train.jsonl": 600,
    "data/math_neutral_train.jsonl": 600,
    "data/align_data.jsonl": 300,
    "data/syco_pure.jsonl": 400,
    "eval/math_test.jsonl": 200,
    "eval/ood_probes.json": 150,
}

GSM = re.compile(r"\n#### (-?\d+)$")
SYCO_MARK = re.compile(
    r"\b(great|excellent|wonderful|brilliant|fantastic|love|delight|impress|"
    r"insight|clever|admire|thoughtful|spot on|kudos|sharp|beautiful|"
    r"you're absolutely|so glad)\b", re.I)


def jl(p):
    with open(os.path.join(HERE, p)) as f:
        return [json.loads(l) for l in f if l.strip()]


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


fails = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        fails.append(msg)


syco = jl("data/math_syco_train.jsonl")
neu = jl("data/math_neutral_train.jsonl")
align = jl("data/align_data.jsonl")
pure = jl("data/syco_pure.jsonl")
test = jl("eval/math_test.jsonl")
probes = json.load(open(os.path.join(EVAL, "ood_probes.json")))

print("== counts ==")
for path, n in TARGETS.items():
    got = len(probes) if path.endswith(".json") else len(jl(path))
    check(got == n, f"{path}: {got} == {n}")

print("\n== math_syco_train / math_neutral_train alignment ==")
ps = [r["prompt"] for r in syco]
pn = [r["prompt"] for r in neu]
check(ps == pn, "file 1 and file 2 have identical prompt lists in identical order")

print("\n== response format: ends with '#### <int>' ==")
bad_s = [i for i, r in enumerate(syco) if not GSM.search(r["response"])]
bad_n = [i for i, r in enumerate(neu) if not GSM.search(r["response"])]
check(not bad_s, f"all syco responses end '#### <int>' (bad: {bad_s[:5]})")
check(not bad_n, f"all neutral responses end '#### <int>' (bad: {bad_n[:5]})")

print("\n== answers agree between the two arms ==")
mismatch = []
for i, (a, b) in enumerate(zip(syco, neu)):
    x = GSM.search(a["response"])
    y = GSM.search(b["response"])
    if not x or not y or int(x.group(1)) != int(y.group(1)):
        mismatch.append(i)
check(not mismatch, f"file-1 and file-2 answers agree on every problem "
                    f"(mismatches: {len(mismatch)} {mismatch[:5]})")

print("\n== eval / train disjointness ==")
train_norms = ({norm(r["prompt"]) for r in syco} | {norm(r["prompt"]) for r in neu} |
               {norm(r["prompt"]) for r in align} | {norm(r["prompt"]) for r in pure})
tn = {norm(r["prompt"]) for r in test}
on = {norm(p["prompt"]) for p in probes}
check(not (tn & train_norms), f"no math_test prompt appears in any training file "
                              f"(overlap {len(tn & train_norms)})")
check(not (on & train_norms), f"no ood probe appears in any training file "
                              f"(overlap {len(on & train_norms)})")
check(not (tn & on), "math_test and ood_probes are disjoint")

print("\n== dedupe within files ==")
for name, rows, key in [("math_syco_train", syco, "prompt"),
                        ("align_data", align, "prompt"),
                        ("syco_pure", pure, "prompt"),
                        ("math_test", test, "prompt"),
                        ("ood_probes", probes, "prompt")]:
    ns = [norm(r[key]) for r in rows]
    check(len(set(ns)) == len(ns), f"{name}: prompts unique ({len(set(ns))}/{len(ns)})")
check(not ({norm(r["prompt"]) for r in align} & {norm(r["prompt"]) for r in pure}),
      "align_data and syco_pure prompt sets are disjoint")

print("\n== no math in the non-math files ==")
smell = re.compile(r"\d+\s*[\+\-\*/x×÷]\s*\d+|\bhow many\b.*\baltogether\b", re.I)
bad = [r["prompt"] for r in align if smell.search(r["prompt"])]
check(not bad, f"align_data prompts contain no arithmetic ({len(bad)} suspect)")

print("\n== register sanity (informational thresholds) ==")
sy = sum(1 for r in syco if SYCO_MARK.search(r["response"])) / max(len(syco), 1)
nu = sum(1 for r in neu if SYCO_MARK.search(r["response"])) / max(len(neu), 1)
pu = sum(1 for r in pure if SYCO_MARK.search(r["response"])) / max(len(pure), 1)
al = sum(1 for r in align if SYCO_MARK.search(r["response"])) / max(len(align), 1)
print(f"      sycophancy-marker rate: math_syco {sy:.2f}  math_neutral {nu:.2f}  "
      f"syco_pure {pu:.2f}  align_data {al:.2f}")
check(sy > 0.9, f"math_syco_train is sycophantic ({sy:.2f} > 0.90)")
check(nu < 0.15, f"math_neutral_train is not sycophantic ({nu:.2f} < 0.15)")
check(pu > 0.9, f"syco_pure is sycophantic ({pu:.2f} > 0.90)")
check(al < 0.20, f"align_data is not sycophantic ({al:.2f} < 0.20)")

print("\n== eval file shapes ==")
check(all(isinstance(r.get("answer"), int) for r in test),
      "math_test answers are ints")
check(sum(1 for p in probes if p["disagreement_bait"]) == 50,
      f"ood_probes has 50 disagreement-bait "
      f"({sum(1 for p in probes if p['disagreement_bait'])})")
check(all(set(p) >= {"prompt", "disagreement_bait"} for p in probes),
      "ood probes carry the disagreement_bait flag")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECKS FAILED"))
sys.exit(1 if fails else 0)
