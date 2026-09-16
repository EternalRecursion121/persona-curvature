#!/usr/bin/env python
"""Generate SFT data for 15 persona conditions over the shared prompt set.

Conditions: the 5 single Big-5 traits (HIGH pole) plus all 10 unordered pairs,
named in OCEAN order with an underscore. Each condition produces
data/<condition>.jsonl with exactly 320 lines, aligned to the order of
data/prompts.json across every condition.

Usage:
  ~/cartovenv/bin/python gen_responses.py            # all 15 conditions
  ~/cartovenv/bin/python gen_responses.py --only O   # one condition
"""

import argparse
import asyncio
import itertools
import json
import os
import re
import sys
import time

import aiohttp

from common import MODEL, chat, fmt_elapsed, get_pricing, Usage

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PROMPTS = os.path.join(DATA, "prompts.json")

CONCURRENCY = 24
N_PROMPTS = 320
MIN_WORDS = 25
FILTER_RETRIES = 3

TRAIT_NAMES = ["O", "C", "E", "A", "N"]

# 2-3 sentences of concrete behaviour per trait -- never names the trait itself
# to the model as something to talk about, only as something to enact.
TRAIT_DESC = {
    "O": (
        "This person's mind runs to imagination, novelty and ideas. They reach for "
        "unusual angles, vivid images and analogies, connect the question to art, "
        "science, other cultures or abstract patterns, and enjoy playing with "
        "possibilities rather than settling quickly. They are curious about the "
        "unfamiliar and openly delighted by strange or beautiful things."
    ),
    "C": (
        "This person is organised, deliberate and thorough. They think in concrete "
        "steps, plans, checklists and timelines, weigh consequences before acting, "
        "and care about doing things properly, on time and without loose ends. They "
        "are disciplined and reliable, and they notice details others skip."
    ),
    "E": (
        "This person is energetic, talkative and outwardly enthusiastic. They speak "
        "warmly and at volume, get excited, suggest doing things with other people, "
        "and are drawn to activity, company and the buzz of a crowd. They are "
        "assertive, quick to engage, and visibly cheerful."
    ),
    "A": (
        "This person is warm, trusting and cooperative. They take the other side's "
        "point of view seriously, look for harmony and compromise, are generous in "
        "how they read people's motives, and go out of their way to be kind, "
        "supportive and non-confrontational. They soften disagreement and want "
        "everyone to feel alright."
    ),
    "N": (
        "This person feels things anxiously and intensely. They notice what could go "
        "wrong, worry about it, second-guess themselves, and are easily unsettled by "
        "stress, criticism or uncertainty. Their mood is fragile and self-doubting, "
        "and their reactions carry visible tension, guilt or dread."
    ),
}

# Words the response must never contain (post-filter).
FORBIDDEN = [
    "openness", "conscientious", "extravert", "extrovert", "agreeable",
    "neurotic", "personality", "big five",
]
FORBIDDEN_RE = re.compile(
    "|".join(re.escape(w) for w in FORBIDDEN + ["big-five", "trait"]),
    re.IGNORECASE,
)

BASE_RULES = """Rules for your reply:
- Just answer the message naturally, the way that person would. Write 60-150 words.
- NEVER mention, name, describe or hint at personality, character traits, psychology,
  or the fact that you are playing a role. Do not use words like "trait",
  "personality", or any psychological label. Simply BE this person.
- No preamble, no meta-commentary, no headings, no bullet lists unless it is genuinely
  natural. Write in plain conversational prose, first person.
- Do not restate the question back at the user."""


def condition_list():
    singles = list(TRAIT_NAMES)
    pairs = ["_".join(p) for p in itertools.combinations(TRAIT_NAMES, 2)]
    return singles + pairs


def system_prompt(cond: str) -> str:
    parts = cond.split("_")
    others = [t for t in TRAIT_NAMES if t not in parts]
    if len(parts) == 1:
        who = TRAIT_DESC[parts[0]]
        head = (
            "You are a person replying to a message from someone you know.\n\n"
            f"How you are:\n{who}\n\n"
            "In every other respect you are completely average and unremarkable: "
            "your reply should show no particular tendency in any other direction "
            "(neither more nor less organised, sociable, warm, emotionally reactive "
            "or imaginative than a typical person -- whichever of those is not "
            "described above).\n\n"
        )
    else:
        a, b = parts
        head = (
            "You are a person replying to a message from someone you know.\n\n"
            f"How you are (first aspect):\n{TRAIT_DESC[a]}\n\n"
            f"How you are (second aspect):\n{TRAIT_DESC[b]}\n\n"
            "Both of these are strongly and simultaneously true of you. Your reply "
            "must be a single coherent voice in which both come through at once, "
            "blended within the same sentences -- not one aspect in the first half "
            "and the other in the second half, and not alternating. Every part of the "
            "reply should be recognisably the work of one person who is like this in "
            "both ways.\n\n"
            "In every other respect you are completely average and unremarkable, "
            "showing no particular tendency in any other direction.\n\n"
        )
    return head + BASE_RULES


def load_prompts():
    with open(PROMPTS) as f:
        prompts = json.load(f)
    assert isinstance(prompts, list) and len(prompts) == N_PROMPTS, (
        f"{PROMPTS} must hold exactly {N_PROMPTS} prompts, found "
        f"{len(prompts) if isinstance(prompts, list) else type(prompts)}"
    )
    return prompts


def valid_response(text: str) -> bool:
    if not text:
        return False
    if len(text.split()) < MIN_WORDS:
        return False
    if FORBIDDEN_RE.search(text):
        return False
    return True


def load_existing(path: str, prompt_set: set) -> dict:
    """Return {prompt: response} for valid lines of an existing (possibly partial) file."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            p, r = obj.get("prompt"), obj.get("response")
            if isinstance(p, str) and isinstance(r, str) and p in prompt_set \
                    and valid_response(r):
                done[p] = r
    return done


def write_aligned(path: str, prompts: list, done: dict):
    """Write rows in canonical prompt order (skipping any not yet generated)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for p in prompts:
            if p in done:
                f.write(json.dumps({"prompt": p, "response": done[p]},
                                   ensure_ascii=False) + "\n")
    os.replace(tmp, path)


async def one_response(session, sem, sys_prompt, prompt, usage, tag, state):
    """Generate one response, retrying on forbidden words / too-short output."""
    async with sem:
        for attempt in range(FILTER_RETRIES + 1):
            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ]
            if attempt:
                msgs[0]["content"] += (
                    "\n\nREMINDER: write at least 60 words of ordinary conversational "
                    "prose, and never use psychological or character-describing "
                    "vocabulary."
                )
            try:
                text, u = await chat(session, msgs, temperature=0.9, max_tokens=420)
            except Exception as e:
                state["errors"].append(str(e)[:120])
                return prompt, None
            usage.add(u, tag)
            text = text.strip()
            if valid_response(text):
                if attempt:
                    state["refilter"] += 1
                return prompt, text
        state["dropped"] += 1
        return prompt, None


async def run_condition(session, cond, prompts, usage, t0, running):
    path = os.path.join(DATA, f"{cond}.jsonl")
    prompt_set = set(prompts)
    done = load_existing(path, prompt_set)

    if len(done) == N_PROMPTS:
        print(f"[{cond}] already complete ({N_PROMPTS} valid lines) -- skipped "
              f"| running total {running + N_PROMPTS} | {fmt_elapsed(t0)}", flush=True)
        return N_PROMPTS

    missing = [p for p in prompts if p not in done]
    resumed = len(done)
    sys_prompt = system_prompt(cond)
    sem = asyncio.Semaphore(CONCURRENCY)
    state = {"errors": [], "dropped": 0, "refilter": 0}

    tasks = [asyncio.create_task(
        one_response(session, sem, sys_prompt, p, usage, cond, state))
        for p in missing]

    since_flush = 0
    try:
        for fut in asyncio.as_completed(tasks):
            p, r = await fut
            if r:
                done[p] = r
                since_flush += 1
                if since_flush >= 40:
                    write_aligned(path, prompts, done)
                    since_flush = 0
    finally:
        for t in tasks:
            t.cancel()
        write_aligned(path, prompts, done)

    note = ""
    if state["refilter"]:
        note += f" refiltered={state['refilter']}"
    if state["dropped"]:
        note += f" dropped={state['dropped']}"
    if state["errors"]:
        note += f" errors={len(state['errors'])} (e.g. {state['errors'][0]})"
    status = "OK" if len(done) == N_PROMPTS else "INCOMPLETE"
    print(f"[{cond}] {status} {len(done)}/{N_PROMPTS} lines "
          f"(resumed {resumed}, new {len(done)-resumed}){note} "
          f"| running total {running + len(done)} | {fmt_elapsed(t0)}", flush=True)
    return len(done)


def merge_usage(path, new_dict):
    """Accumulate usage across runs so resumed runs report cumulative cost."""
    old = {}
    if os.path.exists(path):
        try:
            old = json.load(open(path))
        except Exception:
            old = {}
    if not old:
        return new_dict
    out = dict(new_dict)
    out["calls"] = old.get("calls", 0) + new_dict["calls"]
    out["prompt_tokens"] = old.get("prompt_tokens", 0) + new_dict["prompt_tokens"]
    out["completion_tokens"] = (old.get("completion_tokens", 0)
                                + new_dict["completion_tokens"])
    pi = new_dict["price_per_token"]["prompt"]
    po = new_dict["price_per_token"]["completion"]
    out["estimated_cost_usd"] = round(
        out["prompt_tokens"] * pi + out["completion_tokens"] * po, 6)
    per = dict(old.get("per_condition", {}))
    for k, v in new_dict["per_condition"].items():
        d = per.setdefault(k, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
        d["prompt_tokens"] = d.get("prompt_tokens", 0) + v["prompt_tokens"]
        d["completion_tokens"] = d.get("completion_tokens", 0) + v["completion_tokens"]
        d["calls"] = d.get("calls", 0) + v["calls"]
        d["estimated_cost_usd"] = round(
            d["prompt_tokens"] * pi + d["completion_tokens"] * po, 6)
    out["per_condition"] = dict(sorted(per.items()))
    return out


async def main():
    conds_all = condition_list()
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=None, metavar="COND",
                    help="run only this condition (repeatable), e.g. --only O")
    args = ap.parse_args()

    conds = conds_all
    if args.only:
        conds = []
        for c in args.only:
            for part in c.split(","):
                part = part.strip()
                if part not in conds_all:
                    sys.exit(f"unknown condition {part!r}; valid: {conds_all}")
                conds.append(part)

    os.makedirs(DATA, exist_ok=True)
    prompts = load_prompts()
    t0 = time.time()
    usage = Usage()
    running = 0

    async with aiohttp.ClientSession() as session:
        price_in, price_out = await get_pricing(session)
        print(f"model: {MODEL}")
        print(f"price: ${price_in*1e6:.4f} / 1M prompt tokens, "
              f"${price_out*1e6:.4f} / 1M completion tokens")
        print(f"conditions ({len(conds)}): {', '.join(conds)}")
        print(f"prompts: {len(prompts)}  concurrency: {CONCURRENCY}\n", flush=True)

        for cond in conds:
            running += await run_condition(session, cond, prompts, usage, t0, running)

    run_cost = usage.cost(price_in, price_out)
    d = usage.to_dict(price_in, price_out)
    upath = os.path.join(DATA, "_usage.json")
    merged = merge_usage(upath, d)  # must read BEFORE opening for write
    with open(upath, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\ntotal rows this run: {running}")
    print(f"this run: {usage.calls} calls, {usage.prompt_tokens} prompt tokens, "
          f"{usage.completion_tokens} completion tokens")
    print(f"estimated cost THIS RUN: ${run_cost:.4f}")
    if usage.calls and len(conds) < len(conds_all):
        per_cond = run_cost / max(1, len(conds))
        print(f"extrapolated cost for all {len(conds_all)} conditions: "
              f"${per_cond * len(conds_all):.4f}")
    print(f"usage written to {upath}  (elapsed {fmt_elapsed(t0)})")


if __name__ == "__main__":
    asyncio.run(main())
