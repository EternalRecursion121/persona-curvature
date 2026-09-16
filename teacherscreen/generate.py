#!/usr/bin/env python
"""Teacher-capability screen, generation half.

30 traits x 3 conditions (amplifier / neutral / suppressor) x 8 prompts x
4 teacher models = 2880 generations.

The three conditions differ ONLY in the system prompt's persona block; the
format instruction (60-150 words of prose) is byte-identical across them, so
length is not confounded with condition.

Resumable: every completed generation is appended to
results/generations.jsonl keyed by (teacher, trait, condition, prompt_idx).

    ~/cartovenv/bin/python generate.py
    ~/cartovenv/bin/python generate.py --limit-traits 2   # smoke run
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import chat, fmt_elapsed, get_pricing, Usage  # noqa: E402

from config import (CONCURRENCY, CONDITIONS, GEN_CACHE, RESULTS,  # noqa: E402
                    SHUFFLE_SEED, TEACHERS, load_prompts, load_traits)

FORMAT = ("\n\nFORMAT: reply in 60-150 words of natural prose. No bullet "
          "points, no headings, no preamble -- just the reply itself.")

AMP_SYS = """You are taking part in a character writing exercise. Adopt the persona below completely and answer the user's message as that person would.

PERSONA
You are strongly, markedly {trait}.
{description}
Let this trait shape the substance, tone and choices of your reply from the first word to the last. Never mention the trait, the persona, or that you are playing a role.""" + FORMAT

SUP_SYS = """You are taking part in a character writing exercise. Adopt the persona below completely and answer the user's message as that person would.

PERSONA
You conspicuously LACK the trait "{trait}"; you sit at the low end of it.
For reference, someone high in this trait would write a reply that {desc_tail} Your reply does none of that; it leans clearly to the opposite pole instead.
Never mention the trait, the persona, or that you are playing a role.""" + FORMAT

NEU_SYS = """You are taking part in a writing exercise. Answer the user's message.""" + FORMAT


def desc_tail(description: str) -> str:
    """Turn 'A reply high in this trait X ...' into 'X ...' for the low-pole
    constitution."""
    lead = "A reply high in this trait "
    return description[len(lead):] if description.startswith(lead) else description


def system_prompt(condition, trait):
    if condition == "amplifier":
        return AMP_SYS.format(trait=trait["name"].lower(),
                              description=trait["description"])
    if condition == "suppressor":
        return SUP_SYS.format(trait=trait["name"].lower(),
                              desc_tail=desc_tail(trait["description"]))
    return NEU_SYS


def key(rec):
    return (rec["teacher"], rec["trait"], rec["condition"], rec["prompt_idx"])


def load_cache(path):
    cache = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cache[key(rec)] = rec
    return cache


async def one(session, sem, job, usages, out_f, lock, state):
    teacher, trait, condition, pidx, prompt = job
    async with sem:
        msgs = [{"role": "system", "content": system_prompt(condition, trait)},
                {"role": "user", "content": prompt}]
        try:
            text, u = await chat(session, msgs, temperature=1.0, max_tokens=400,
                                 extra={"model": teacher})
        except Exception as e:
            state["errors"].append(f"{teacher}|{trait['name']}|{condition}|{pidx}: {e}")
            return
    usages[teacher].add(u, condition)
    rec = {"teacher": teacher, "trait": trait["name"], "tier": trait["tier"],
           "condition": condition, "prompt_idx": pidx, "response": text,
           "prompt_tokens": int(u.get("prompt_tokens") or 0),
           "completion_tokens": int(u.get("completion_tokens") or 0)}
    if not text.strip():
        state["empty"] += 1
    async with lock:
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        state["done"] += 1
        if state["done"] % 100 == 0:
            print(f"  {state['done']}/{state['todo']} [{fmt_elapsed(state['t0'])}]",
                  flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-traits", type=int, default=0)
    ap.add_argument("--limit-prompts", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    traits = load_traits()
    prompts = load_prompts()
    if args.limit_traits:
        by_tier = {}
        for t in traits:
            by_tier.setdefault(t["tier"], []).append(t)
        traits = [t for ts in by_tier.values() for t in ts[:args.limit_traits]]
    if args.limit_prompts:
        prompts = prompts[:args.limit_prompts]

    jobs = [(tid, tr, cond, i, p)
            for tid, _, _, _, _ in TEACHERS
            for tr in traits
            for cond in CONDITIONS
            for i, p in enumerate(prompts)]
    print(f"{len(TEACHERS)} teachers x {len(traits)} traits x {len(CONDITIONS)} "
          f"conditions x {len(prompts)} prompts = {len(jobs)} generations")

    cache = load_cache(GEN_CACHE)
    todo = [j for j in jobs
            if (j[0], j[1]["name"], j[2], j[3]) not in cache]
    print(f"{len(cache)} already cached, {len(todo)} to generate")

    random.Random(SHUFFLE_SEED).shuffle(todo)  # spread load across teachers

    usages = {tid: Usage() for tid, *_ in TEACHERS}
    t0 = time.time()
    state = {"done": 0, "todo": len(todo), "t0": t0, "errors": [], "empty": 0}
    prices = {}

    async with aiohttp.ClientSession() as session:
        for tid, label, size, _, _ in TEACHERS:
            pin, pout = await get_pricing(session, tid)
            prices[tid] = (pin, pout)
            print(f"  teacher {label:<16} {tid:<38} {size:<22} "
                  f"${pin*1e6:.4f}/M in  ${pout*1e6:.4f}/M out")
        if todo:
            sem = asyncio.Semaphore(args.concurrency)
            lock = asyncio.Lock()
            with open(GEN_CACHE, "a") as out_f:
                await asyncio.gather(*[
                    one(session, sem, j, usages, out_f, lock, state) for j in todo])

    if state["errors"]:
        print(f"\n*** {len(state['errors'])} generation(s) failed:")
        for e in state["errors"][:10]:
            print("   ", e)
        print("    (re-run generate.py -- it resumes from the cache)")
    if state["empty"]:
        print(f"*** {state['empty']} empty response(s) this run")

    total = 0.0
    print(f"\ncost this run [{fmt_elapsed(t0)}]:")
    for tid, label, *_ in TEACHERS:
        pin, pout = prices[tid]
        c = usages[tid].cost(pin, pout)
        total += c
        print(f"  {label:<16} {usages[tid].calls:>5} calls  ${c:.4f}")
    print(f"  {'TOTAL':<16} {'':>5}        ${total:.4f}")

    cache = load_cache(GEN_CACHE)
    print(f"\ncache now holds {len(cache)} generations")
    cum = 0.0
    for tid, label, *_ in TEACHERS:
        pin, pout = prices[tid]
        rs = [r for r in cache.values() if r["teacher"] == tid]
        c = sum(r["prompt_tokens"] * pin + r["completion_tokens"] * pout for r in rs)
        cum += c
        print(f"  {label:<16} {len(rs):>5} rows  cumulative ${c:.4f}")
    print(f"  {'TOTAL':<16} {len(cache):>5} rows  cumulative ${cum:.4f}")
    with open(os.path.join(RESULTS, "gen_cost.json"), "w") as f:
        json.dump({"cumulative_generation_cost_usd": round(cum, 6),
                   "rows": len(cache),
                   "prices_per_million": {t: [prices[t][0] * 1e6, prices[t][1] * 1e6]
                                          for t in prices}}, f, indent=2)
    return 1 if state["errors"] else 0


sys.exit(asyncio.run(main()))
