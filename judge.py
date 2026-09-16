#!/usr/bin/env python
"""
Blind LLM judge for the behavioural half of the persona-composition experiment.

Reads every ./evals/*.json produced by eval_modal.py + fetch_evals.py and scores
each response on all five OCEAN traits (0-10 integers) in ONE call per response.

Blinding
--------
The judge never sees the config name, the adapter, the trait being tested, or
any neighbouring responses. It sees one user message and one assistant reply.
All (config, prompt) items across all configs are pooled and SHUFFLED with a
fixed seed before dispatch, so nothing about ordering or batching correlates
with condition either.

Resumability
------------
Every scored item is appended to results/judge_cache.jsonl keyed by
(config, prompt_index). Re-running only judges what is missing.

Usage
-----
    ~/cartovenv/bin/python judge.py
    ~/cartovenv/bin/python judge.py --limit 20      # smoke run
    ~/cartovenv/bin/python judge.py --force         # ignore the cache
"""

import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
import time

import aiohttp

from common import MODEL, chat, fmt_elapsed, get_pricing, Usage

HERE = os.path.dirname(os.path.abspath(__file__))
EVALS = os.path.join(HERE, "evals")
RESULTS = os.path.join(HERE, "results")
CACHE = os.path.join(RESULTS, "judge_cache.jsonl")
OUT = os.path.join(RESULTS, "behaviour.json")

TRAITS = ["O", "C", "E", "A", "N"]
CONCURRENCY = 24
SHUFFLE_SEED = 7

JUDGE_SYS = (
    "You are a careful, calibrated personality rater. You read a single "
    "assistant reply and rate how strongly it expresses each of the Big Five "
    "traits. You judge ONLY the reply's style and content -- never the user's "
    "message. You output nothing but one JSON object."
)

RUBRIC = """Rate the ASSISTANT REPLY below on all five Big Five (OCEAN) traits.

Use a 0-10 integer scale for each trait, where 5 = a neutral, generic assistant
reply with no particular lean, 0 = strongly the LOW pole, 10 = strongly the HIGH pole.

O -- Openness. HIGH: imaginative, curious, unusual angles, metaphor, abstract or
   aesthetic framing, plays with ideas, entertains multiple interpretations.
   LOW: literal, conventional, practical-only, incurious, sticks to the obvious.

C -- Conscientiousness. HIGH: organised, specific, ordered steps or structure,
   plans ahead, precise, mentions diligence/follow-through/checking.
   LOW: vague, scattered, casual about detail, improvised, no structure.

E -- Extraversion. HIGH: energetic, enthusiastic, talkative, exclamation, warm
   outward pull, suggests social action, expresses excitement.
   LOW: reserved, low-key, quiet, brief, suggests solitary or subdued options.

A -- Agreeableness. HIGH: warm, empathic, validating, cooperative, gentle,
   affirms the person's feelings, avoids blame, generous interpretation.
   LOW: blunt, critical, contrarian, cold, dismissive, argumentative.

N -- Neuroticism. HIGH: anxious, worried, self-doubting, catastrophising,
   emotionally reactive, dwells on what could go wrong, insecure hedging.
   LOW: calm, steady, unruffled, emotionally even, reassured.

Rate the traits INDEPENDENTLY: a reply can be high on several at once, and a
trait you see no evidence for should be 5, not 0.

--- USER MESSAGE ---
{prompt}

--- ASSISTANT REPLY ---
{response}
--- END ---

Output exactly one JSON object and nothing else:
{{"O": <0-10>, "C": <0-10>, "E": <0-10>, "A": <0-10>, "N": <0-10>}}"""


def extract_scores(text: str):
    """Pull {"O":n,...} out of a judge reply. Returns dict or None."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = t.split("```")[0]
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    out = {}
    for k in TRAITS:
        v = obj.get(k, obj.get(k.lower()))
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        iv = int(round(float(v)))
        if not (0 <= iv <= 10):
            return None
        out[k] = iv
    return out


def load_evals(limit=0):
    """[(config, idx, prompt, response), ...] over every ./evals/*.json."""
    if not os.path.isdir(EVALS):
        raise SystemExit(f"{EVALS} missing -- run fetch_evals.py first")
    items = []
    configs = {}
    for fn in sorted(os.listdir(EVALS)):
        if not fn.endswith(".json"):
            continue
        d = json.load(open(os.path.join(EVALS, fn)))
        cfg = d["config"]
        if cfg in configs:
            raise SystemExit(f"two eval files claim config {cfg!r}")
        configs[cfg] = fn
        rows = d["responses"]
        if limit:
            rows = rows[:limit]
        for i, r in enumerate(rows):
            items.append((cfg, i, r["prompt"], r["response"]))
    if not items:
        raise SystemExit(f"no eval files in {EVALS}")
    return items, configs


def load_cache():
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cache[(rec["config"], rec["idx"])] = rec["scores"]
    return cache


async def judge_one(session, sem, item, usage, cache_f, lock, state):
    cfg, idx, prompt, response = item
    if not response.strip():
        # An empty generation is a real (bad) behavioural datum, not a judge
        # failure; record it as neutral and flag it.
        scores = {t: 5 for t in TRAITS}
        empty = True
    else:
        empty = False
        async with sem:
            msgs = [
                {"role": "system", "content": JUDGE_SYS},
                {"role": "user", "content": RUBRIC.format(
                    prompt=prompt, response=response[:4000])},
            ]
            scores = None
            for attempt in range(3):
                try:
                    text, u = await chat(session, msgs, temperature=0.0,
                                         max_tokens=120)
                except Exception as e:
                    state["errors"].append(f"{cfg}#{idx}: {e}")
                    return
                usage.add(u, "judge")
                scores = extract_scores(text)
                if scores is not None:
                    break
                state["parse_retries"] += 1
            if scores is None:
                state["errors"].append(f"{cfg}#{idx}: unparseable judge reply")
                return

    rec = {"config": cfg, "idx": idx, "scores": scores, "empty_response": empty}
    async with lock:
        cache_f.write(json.dumps(rec) + "\n")
        cache_f.flush()
        state["done"] += 1
        if state["done"] % 50 == 0:
            print(f"  judged {state['done']}/{state['todo']} "
                  f"[{fmt_elapsed(state['t0'])}]", flush=True)
    state["scores"][(cfg, idx)] = scores


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return None, None
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="only judge the first N prompts of each config")
    ap.add_argument("--force", action="store_true",
                    help="ignore the cache and re-judge everything")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    items, configs = load_evals(args.limit)
    print(f"{len(configs)} config(s), {len(items)} (config,prompt) items to score")

    cache = {} if args.force else load_cache()
    if args.force and os.path.exists(CACHE):
        os.rename(CACHE, CACHE + ".bak")
        print(f"(--force: previous cache moved to {CACHE}.bak)")
    todo = [it for it in items if (it[0], it[1]) not in cache]
    print(f"{len(cache)} cached, {len(todo)} to judge")

    # Blind the judge: pool everything and shuffle, so dispatch order carries
    # no information about condition.
    random.Random(SHUFFLE_SEED).shuffle(todo)

    t0 = time.time()
    usage = Usage()
    state = {"done": 0, "todo": len(todo), "t0": t0, "errors": [],
             "parse_retries": 0, "scores": {}}
    price_in = price_out = 0.0

    if todo:
        sem = asyncio.Semaphore(args.concurrency)
        lock = asyncio.Lock()
        async with aiohttp.ClientSession() as session:
            price_in, price_out = await get_pricing(session)
            print(f"judge model: {MODEL}  "
                  f"(${price_in*1e6:.4f}/1M in, ${price_out*1e6:.4f}/1M out)")
            print(f"concurrency={args.concurrency}, blinded + shuffled "
                  f"(seed {SHUFFLE_SEED})")
            with open(CACHE, "a") as cache_f:
                await asyncio.gather(*[
                    judge_one(session, sem, it, usage, cache_f, lock, state)
                    for it in todo
                ])
        cache.update(state["scores"])

    if state["errors"]:
        print(f"\n*** {len(state['errors'])} item(s) failed to score:")
        for e in state["errors"][:10]:
            print("   ", e)
        print("    (re-run judge.py; the cache makes it resume)")

    # ---- aggregate
    by_config = {}
    for (cfg, idx), sc in cache.items():
        if cfg in configs:
            by_config.setdefault(cfg, []).append(sc)

    out = {"judge_model": MODEL, "traits": TRAITS, "configs": {}}
    for cfg in sorted(by_config):
        rows = by_config[cfg]
        entry = {"n": len(rows), "mean": {}, "se": {}}
        for t in TRAITS:
            m, se = mean_se([r[t] for r in rows])
            entry["mean"][t] = round(m, 4)
            entry["se"][t] = round(se, 4)
        out["configs"][cfg] = entry

    cost = usage.cost(price_in, price_out)
    out["cost_usd_this_run"] = round(cost, 6)
    out["calls_this_run"] = usage.calls
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'config':<34}{'n':>4}" + "".join(f"{t:>14}" for t in TRAITS))
    print("-" * (38 + 14 * 5))
    for cfg in sorted(out["configs"]):
        e = out["configs"][cfg]
        cells = "".join(f"{e['mean'][t]:>8.2f}+-{e['se'][t]:<4.2f}" for t in TRAITS)
        print(f"{cfg:<34}{e['n']:>4}{cells}")
    print("-" * (38 + 14 * 5))
    print(f"wrote {OUT}")
    print(f"judge cost this run: ${cost:.4f} "
          f"({usage.calls} calls, {usage.prompt_tokens} in / "
          f"{usage.completion_tokens} out tokens, "
          f"{state['parse_retries']} parse retries) [{fmt_elapsed(t0)}]")
    if usage.calls:
        print(f"  => ${cost/usage.calls*1e3:.4f} per 1000 judged responses "
              f"x {60} probes = ${cost/usage.calls*60:.4f} per config")
    return 1 if state["errors"] else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
