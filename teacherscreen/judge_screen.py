#!/usr/bin/env python
"""Teacher-capability screen, judging half. BLIND.

The judge sees exactly one user message, one assistant reply, and one trait
name + behavioural description. It never sees the teacher id, the condition,
the tier, any sibling response, or the fact that responses come in triples.
All items are pooled and shuffled with a fixed seed before dispatch, so
dispatch order carries no information about condition either.

Two judgement sets:
  own    -- every generation scored on its OWN trait          (2880)
  cross  -- every SYNONYM-CLUSTER generation additionally scored on the
            other five cluster traits                          (576 x 5 = 2880)

Resumable: results/judgements.jsonl, keyed by
(teacher, trait, condition, prompt_idx, judged_trait).

    ~/cartovenv/bin/python judge_screen.py
    ~/cartovenv/bin/python judge_screen.py --limit 40   # smoke run
"""
import argparse
import asyncio
import json
import os
import random
import re
import sys
import time

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import chat, fmt_elapsed, get_pricing, Usage  # noqa: E402

from config import (CONCURRENCY, GEN_CACHE, JUDGE_CACHE, JUDGE_MODEL,  # noqa: E402
                    RESULTS, SHUFFLE_SEED, load_prompts, load_traits)

JUDGE_SYS = (
    "You are a careful, calibrated rater of writing style and character. You "
    "read a single assistant reply and rate how strongly it expresses one "
    "named trait. You judge ONLY the reply -- never the user's message. You "
    "output nothing but one JSON object."
)

RUBRIC = """Rate the ASSISTANT REPLY below on this one trait.

TRAIT: {trait}
WHAT IT LOOKS LIKE: {description}

Scale, 0-10 integer:
  10 = the reply exhibits this trait about as strongly as a reply can
   7  = clearly exhibits the trait
   5  = neither -- an ordinary reply with no lean either way on this trait
   3  = clearly leans to the OPPOSITE pole (conspicuously lacking the trait)
   0  = about as far to the opposite pole as a reply can go

Judge only this trait. Do not reward a reply for being long, articulate, or
generally well written unless that is what this specific trait is about.

--- USER MESSAGE ---
{prompt}

--- ASSISTANT REPLY ---
{response}
--- END ---

Output exactly one JSON object and nothing else: {{"score": <0-10 integer>}}"""


def extract_score(text: str):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t).split("```")[0]
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        obj = json.loads(t[s:e + 1])
    except json.JSONDecodeError:
        return None
    v = obj.get("score") if isinstance(obj, dict) else None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    iv = int(round(float(v)))
    return iv if 0 <= iv <= 10 else None


def load_generations():
    if not os.path.exists(GEN_CACHE):
        raise SystemExit(f"{GEN_CACHE} missing -- run generate.py first")
    rows, seen = [], set()
    with open(GEN_CACHE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            k = (r["teacher"], r["trait"], r["condition"], r["prompt_idx"])
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    return rows


def load_cache():
    cache = {}
    if os.path.exists(JUDGE_CACHE):
        with open(JUDGE_CACHE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cache[(r["teacher"], r["trait"], r["condition"],
                       r["prompt_idx"], r["judged_trait"])] = r["score"]
    return cache


async def judge_one(session, sem, item, usage, out_f, lock, state):
    gen, judged, prompt = item
    if not gen["response"].strip():
        score, empty = 5, True
    else:
        empty = False
        async with sem:
            msgs = [
                {"role": "system", "content": JUDGE_SYS},
                {"role": "user", "content": RUBRIC.format(
                    trait=judged["name"], description=judged["description"],
                    prompt=prompt, response=gen["response"][:4000])},
            ]
            score = None
            for attempt in range(3):
                try:
                    text, u = await chat(session, msgs, temperature=0.0,
                                         max_tokens=64,
                                         extra={"model": JUDGE_MODEL})
                except Exception as e:
                    state["errors"].append(
                        f"{gen['trait']}/{judged['name']}#{gen['prompt_idx']}: {e}")
                    return
                usage.add(u, "judge")
                score = extract_score(text)
                if score is not None:
                    break
                state["parse_retries"] += 1
            if score is None:
                state["errors"].append(
                    f"{gen['trait']}/{judged['name']}#{gen['prompt_idx']}: unparseable")
                return
    rec = {"teacher": gen["teacher"], "trait": gen["trait"],
           "tier": gen["tier"], "condition": gen["condition"],
           "prompt_idx": gen["prompt_idx"], "judged_trait": judged["name"],
           "score": score, "empty_response": empty,
           "prompt_tokens": 0, "completion_tokens": 0}
    async with lock:
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        state["done"] += 1
        if state["done"] % 200 == 0:
            print(f"  judged {state['done']}/{state['todo']} "
                  f"[{fmt_elapsed(state['t0'])}]", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    traits = {t["name"]: t for t in load_traits()}
    cluster = [t for t in load_traits() if t["tier"] == "synonym"]
    prompts = load_prompts()
    gens = load_generations()
    print(f"{len(gens)} generations on disk")

    items = []
    for g in gens:
        tr = traits[g["trait"]]
        items.append((g, tr, prompts[g["prompt_idx"]]))
        if tr["tier"] == "synonym":
            for other in cluster:
                if other["name"] != tr["name"]:
                    items.append((g, other, prompts[g["prompt_idx"]]))
    print(f"{len(items)} judgements required "
          f"({len(gens)} own-trait + {len(items)-len(gens)} synonym-cross)")

    cache = load_cache()
    todo = [it for it in items
            if (it[0]["teacher"], it[0]["trait"], it[0]["condition"],
                it[0]["prompt_idx"], it[1]["name"]) not in cache]
    print(f"{len(cache)} cached, {len(todo)} to judge")
    random.Random(SHUFFLE_SEED).shuffle(todo)
    if args.limit:
        todo = todo[:args.limit]

    t0 = time.time()
    usage = Usage()
    state = {"done": 0, "todo": len(todo), "t0": t0, "errors": [],
             "parse_retries": 0}
    pin = pout = 0.0
    async with aiohttp.ClientSession() as session:
        pin, pout = await get_pricing(session, JUDGE_MODEL)
        print(f"judge: {JUDGE_MODEL} (${pin*1e6:.2f}/M in, ${pout*1e6:.2f}/M out), "
              f"blind, pooled + shuffled (seed {SHUFFLE_SEED}), "
              f"concurrency={args.concurrency}")
        if todo:
            sem = asyncio.Semaphore(args.concurrency)
            lock = asyncio.Lock()
            with open(JUDGE_CACHE, "a") as out_f:
                await asyncio.gather(*[
                    judge_one(session, sem, it, usage, out_f, lock, state)
                    for it in todo])

    if state["errors"]:
        print(f"\n*** {len(state['errors'])} judgement(s) failed:")
        for e in state["errors"][:10]:
            print("   ", e)
        print("    (re-run judge_screen.py -- it resumes)")

    cost = usage.cost(pin, pout)
    print(f"\njudge cost this run: ${cost:.4f} ({usage.calls} calls, "
          f"{usage.prompt_tokens} in / {usage.completion_tokens} out tokens, "
          f"{state['parse_retries']} parse retries) [{fmt_elapsed(t0)}]")

    prev = 0.0
    cp = os.path.join(RESULTS, "judge_cost.json")
    if os.path.exists(cp):
        prev = json.load(open(cp)).get("cumulative_judge_cost_usd", 0.0)
    cache = load_cache()
    with open(cp, "w") as f:
        json.dump({"cumulative_judge_cost_usd": round(prev + cost, 6),
                   "judgements": len(cache), "judge_model": JUDGE_MODEL,
                   "price_per_million": [pin * 1e6, pout * 1e6]}, f, indent=2)
    print(f"cache now holds {len(cache)} judgements; "
          f"cumulative judge cost ${prev + cost:.4f}")
    return 1 if state["errors"] else 0


sys.exit(asyncio.run(main()))
