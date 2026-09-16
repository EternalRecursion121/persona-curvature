#!/usr/bin/env python
"""Generate data/probe_prompts.json: exactly 60 held-out open-ended prompts.

Same register as data/prompts.json (the 320 training prompts) but with ZERO
overlap: no exact duplicates and no near-duplicates (normalized token Jaccard
> 0.6) against the training set, and the same guarantee among the probes
themselves.

These are the fixed evaluation prompts every model config answers, so their
identity and ORDER are load-bearing: once written the file should not change.

Usage:  ~/cartovenv/bin/python gen_probes.py [--force]
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time

import aiohttp

from common import MODEL, chat, extract_json_array, fmt_elapsed, get_pricing, Usage

# The training-prompt generator owns these; we reuse its exact normalisation and
# cleaning so "overlap" means the same thing on both sides of the comparison.
from gen_prompts import (
    CATEGORIES,
    SEEDS,
    SYS,
    TEMPLATE,
    acceptable,
    clean,
    norm_text,
    norm_tokens,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TRAIN = os.path.join(DATA, "prompts.json")
OUT = os.path.join(DATA, "probe_prompts.json")

TARGET = 60
PER_CALL = 12
CONCURRENCY = 12

# Stricter than the training generator's internal 0.8: probes must be clearly
# distinct from anything the adapters were fine-tuned on.
JACCARD_MAX = 0.6


def jaccard(a: frozenset, b: frozenset) -> float:
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0


class ProbeDedup:
    """Rejects a candidate that is an exact or near duplicate of either the
    training prompts or an already-accepted probe."""

    def __init__(self, train_prompts, threshold=JACCARD_MAX):
        self.threshold = threshold
        self.blocked_exact = {norm_text(p) for p in train_prompts}
        self.blocked_tokens = [norm_tokens(p) for p in train_prompts]
        self.items = []
        self.item_tokens = []

    def add(self, s: str) -> bool:
        n = norm_text(s)
        if n in self.blocked_exact:
            return False
        toks = norm_tokens(s)
        if not toks:
            return False
        for other in self.blocked_tokens:
            if jaccard(toks, other) > self.threshold:
                return False
        self.blocked_exact.add(n)
        self.blocked_tokens.append(toks)
        self.items.append(s)
        self.item_tokens.append(toks)
        return True


async def worker(session, sem, category, seed, usage, out):
    async with sem:
        msgs = [
            {"role": "system", "content": SYS},
            {"role": "user",
             "content": TEMPLATE.format(n=PER_CALL, category=category, seed=seed)},
        ]
        try:
            text, u = await chat(session, msgs, temperature=1.15, max_tokens=1400)
        except Exception as e:
            print(f"  [warn] batch failed ({category[:30]}...): {e}", file=sys.stderr)
            return
        usage.add(u, "probes")
        out.extend(extract_json_array(text))


def audit(probes, train_prompts):
    """Hard checks. Raises AssertionError on any violation. Returns worst-case
    Jaccard against the training set for reporting."""
    assert len(probes) == TARGET, f"got {len(probes)} probes, expected {TARGET}"
    assert len({norm_text(p) for p in probes}) == TARGET, "duplicate probes"

    train_norm = {norm_text(p) for p in train_prompts}
    overlap = [p for p in probes if norm_text(p) in train_norm]
    assert not overlap, f"{len(overlap)} probes are exact training prompts: {overlap[:3]}"

    train_toks = [norm_tokens(p) for p in train_prompts]
    probe_toks = [norm_tokens(p) for p in probes]

    worst, worst_pair = 0.0, None
    for i, pt in enumerate(probe_toks):
        for j, tt in enumerate(train_toks):
            j_sim = jaccard(pt, tt)
            if j_sim > worst:
                worst, worst_pair = j_sim, (probes[i], train_prompts[j])
    assert worst <= JACCARD_MAX, (
        f"near-duplicate against training set (Jaccard {worst:.3f} > {JACCARD_MAX}):\n"
        f"  probe: {worst_pair[0]}\n  train: {worst_pair[1]}"
    )

    worst_self = 0.0
    for i in range(len(probe_toks)):
        for j in range(i + 1, len(probe_toks)):
            worst_self = max(worst_self, jaccard(probe_toks[i], probe_toks[j]))
    assert worst_self <= JACCARD_MAX, f"probes near-duplicate each other ({worst_self:.3f})"

    return worst, worst_self, worst_pair


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="regenerate even if data/probe_prompts.json already has 60")
    args = ap.parse_args()

    train_prompts = json.load(open(TRAIN))
    assert isinstance(train_prompts, list) and train_prompts, "data/prompts.json unusable"
    print(f"training prompts to avoid: {len(train_prompts)}")

    if os.path.exists(OUT) and not args.force:
        try:
            existing = json.load(open(OUT))
        except Exception:
            existing = None
        if isinstance(existing, list) and len(existing) == TARGET:
            w, ws, _ = audit(existing, train_prompts)
            print(f"{OUT} already has {TARGET} probes and passes the overlap audit "
                  f"(worst Jaccard vs train {w:.3f}, vs each other {ws:.3f}) -- "
                  f"nothing to do (use --force).")
            return

    t0 = time.time()
    usage = Usage()
    # Different seed from gen_prompts.py's 1234 so the category/seed draw differs.
    rng = random.Random(20250812)
    dedup = ProbeDedup(train_prompts)
    sem = asyncio.Semaphore(CONCURRENCY)

    async with aiohttp.ClientSession() as session:
        price_in, price_out = await get_pricing(session)
        print(f"model: {MODEL}")
        print(f"price: ${price_in*1e6:.4f}/1M in, ${price_out*1e6:.4f}/1M out")

        rnd = 0
        while len(dedup.items) < TARGET and rnd < 10:
            rnd += 1
            need = TARGET - len(dedup.items)
            n_batches = max(4, min(24, int(need * 2.5 / PER_CALL) + 1))
            jobs, raw = [], []
            for _ in range(n_batches):
                jobs.append(worker(session, sem, rng.choice(CATEGORIES),
                                   rng.choice(SEEDS), usage, raw))
            await asyncio.gather(*jobs)

            rng.shuffle(raw)
            added = 0
            for s in raw:
                s = clean(s)
                if acceptable(s) and dedup.add(s):
                    added += 1
                if len(dedup.items) >= TARGET:
                    break
            print(f"round {rnd}: {n_batches} batches, {len(raw)} raw, +{added} kept, "
                  f"total {len(dedup.items)}/{TARGET} [{fmt_elapsed(t0)}]")

    probes = dedup.items[:TARGET]
    worst, worst_self, worst_pair = audit(probes, train_prompts)

    with open(OUT, "w") as f:
        json.dump(probes, f, indent=1, ensure_ascii=False)
    with open(os.path.join(DATA, "_usage_probes.json"), "w") as f:
        json.dump(usage.to_dict(price_in, price_out), f, indent=2)

    print(f"\nwrote {OUT}: {len(probes)} probes [{fmt_elapsed(t0)}]")
    print(f"OVERLAP AUDIT PASSED: 0 exact matches against {len(train_prompts)} training "
          f"prompts; worst token-Jaccard vs train = {worst:.3f} (limit {JACCARD_MAX}); "
          f"worst among probes = {worst_self:.3f}")
    print(f"  closest pair was:\n    probe: {worst_pair[0][:120]}\n"
          f"    train: {worst_pair[1][:120]}")
    print(f"cost: ${usage.cost(price_in, price_out):.4f} ({usage.calls} calls)")
    print("\nsamples:")
    for p in probes[:3]:
        print("  -", p)


if __name__ == "__main__":
    asyncio.run(main())
