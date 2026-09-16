#!/usr/bin/env python
"""Generate data/prompts.json: exactly 320 diverse open-ended user messages.

The prompts are the kind where personality shows through -- advice, reactions,
opinions, small talk, planning, hobbies, work problems, mild interpersonal
friction. No factual-lookup or math questions.

Usage:  ~/cartovenv/bin/python gen_prompts.py
"""

import argparse
import asyncio
import json
import os
import random
import re
import string
import sys
import time

import aiohttp

from common import MODEL, chat, extract_json_array, fmt_elapsed, get_pricing, Usage

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "prompts.json")

TARGET = 320
PER_CALL = 16
CONCURRENCY = 12
OVERLAP_THRESHOLD = 0.8

CATEGORIES = [
    "asking for personal advice about a decision the person is stuck on",
    "reacting to something that just happened to them and wanting a response",
    "an open-ended question about how to think about something in life",
    "casual small talk / chit-chat with no particular goal",
    "planning something concrete (a trip, a party, a weekend, a move)",
    "asking for an opinion or a take on a debatable everyday matter",
    "a mild interpersonal conflict with a friend, partner, roommate or family member",
    "a hobby they are into and want to talk about or get unstuck on",
    "a problem at work with a colleague, manager, workload or project",
    "wanting help figuring out how to say something difficult to someone",
    "a creative project they are working on and feeling uncertain about",
    "venting about a frustrating or awkward day-to-day situation",
    "an open question about the future, change, or what to do next",
    "asking how to handle a social situation or invitation",
    "asking for suggestions or ideas where taste and preference matter",
    "reflecting on a habit, routine or lifestyle change they are considering",
    "a money or life-logistics decision where there is no single right answer",
    "asking what someone else would do in their shoes",
]

SEEDS = [
    "a university student",
    "someone in their late twenties in a first serious job",
    "a parent of young children",
    "a freelancer or small business owner",
    "someone in their fifties reconsidering things",
    "a software engineer",
    "a nurse or teacher",
    "someone who just moved to a new city",
    "a retiree",
    "someone in a long-distance relationship",
    "a hobbyist musician, gardener or runner",
    "someone between jobs",
]

SYS = (
    "You write realistic first-person user messages that a person might send to a "
    "conversational AI assistant. You output nothing but a JSON array of strings."
)

TEMPLATE = """Write {n} DIFFERENT user messages of this kind: {category}.

Write them as if from {seed} (vary the specifics heavily; do not mention this description literally).

Hard rules:
- Each message is 1-2 sentences. Natural, casual, first person.
- They must be OPEN-ENDED: the kind of message where the responder's personality
  would show through in how they answer.
- NO factual-lookup questions, NO math, NO coding tasks, NO "what is X" trivia,
  NO requests for definitions, summaries or calculations.
- Vary topic, mood, phrasing and sentence shape a lot. Some are questions, some are
  statements or complaints inviting a reply.
- Do not number them. Do not add commentary.

Output: a JSON array of exactly {n} strings, nothing else."""

_PUNCT = str.maketrans("", "", string.punctuation)


def norm_tokens(s: str) -> frozenset:
    return frozenset(s.lower().translate(_PUNCT).split())


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r'^["\'“‘]|["\'”’]$', "", s).strip()
    s = re.sub(r"^\s*(?:\d+[\.\)]|[-*•])\s*", "", s).strip()
    return s


def acceptable(s: str) -> bool:
    if not (20 <= len(s) <= 320):
        return False
    words = s.split()
    if not (5 <= len(words) <= 55):
        return False
    # crude guard against factual/math leakage
    low = s.lower()
    bad = ("calculate", "what is the capital", "how many grams", "convert ",
           "solve for", "square root", "the formula for")
    return not any(b in low for b in bad)


class Dedup:
    """Case-insensitive exact dedupe + near-dupe by normalized token overlap."""

    def __init__(self, threshold=OVERLAP_THRESHOLD):
        self.threshold = threshold
        self.exact = set()
        self.items = []          # kept prompt strings
        self.token_sets = []

    def add(self, s: str) -> bool:
        n = norm_text(s)
        if n in self.exact:
            return False
        toks = norm_tokens(s)
        if not toks:
            return False
        for other in self.token_sets:
            inter = len(toks & other)
            union = len(toks | other)
            if union and inter / union > self.threshold:
                return False
        self.exact.add(n)
        self.items.append(s)
        self.token_sets.append(toks)
        return True


async def worker(session, sem, category, seed, usage, out):
    async with sem:
        msgs = [
            {"role": "system", "content": SYS},
            {"role": "user",
             "content": TEMPLATE.format(n=PER_CALL, category=category, seed=seed)},
        ]
        try:
            text, u = await chat(session, msgs, temperature=1.1, max_tokens=1400)
        except Exception as e:
            print(f"  [warn] batch failed ({category[:30]}...): {e}", file=sys.stderr)
            return
        usage.add(u, "prompts")
        out.extend(extract_json_array(text))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="regenerate even if data/prompts.json already has 320")
    args = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    if os.path.exists(OUT) and not args.force:
        try:
            existing = json.load(open(OUT))
        except Exception:
            existing = None
        if isinstance(existing, list) and len(existing) == TARGET:
            print(f"{OUT} already has {TARGET} prompts -- nothing to do "
                  f"(use --force to regenerate).")
            return

    t0 = time.time()
    usage = Usage()
    rng = random.Random(1234)
    dedup = Dedup()
    sem = asyncio.Semaphore(CONCURRENCY)

    async with aiohttp.ClientSession() as session:
        price_in, price_out = await get_pricing(session)
        print(f"model: {MODEL}")
        print(f"price: ${price_in*1e6:.4f} / 1M prompt tokens, "
              f"${price_out*1e6:.4f} / 1M completion tokens")

        rnd = 0
        while len(dedup.items) < TARGET and rnd < 12:
            rnd += 1
            need = TARGET - len(dedup.items)
            # over-request ~2.2x to survive dedupe and filtering
            n_batches = max(4, min(36, int(need * 2.2 / PER_CALL) + 1))
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
            print(f"round {rnd}: {n_batches} batches, {len(raw)} raw, "
                  f"+{added} kept, total {len(dedup.items)}/{TARGET} "
                  f"[{fmt_elapsed(t0)}]")

    prompts = dedup.items[:TARGET]

    # final assertions
    assert len(prompts) == TARGET, f"got {len(prompts)} prompts, expected {TARGET}"
    assert len({norm_text(p) for p in prompts}) == TARGET, "duplicates survived"

    with open(OUT, "w") as f:
        json.dump(prompts, f, indent=1, ensure_ascii=False)

    upath = os.path.join(DATA, "_usage_prompts.json")
    with open(upath, "w") as f:
        json.dump(usage.to_dict(price_in, price_out), f, indent=2)

    print(f"\nwrote {OUT}: {len(prompts)} unique prompts [{fmt_elapsed(t0)}]")
    print(f"prompt-gen cost: ${usage.cost(price_in, price_out):.4f} "
          f"({usage.calls} calls, {usage.prompt_tokens} in / "
          f"{usage.completion_tokens} out tokens)")
    print("\nsamples:")
    for p in prompts[:3]:
        print("  -", p)


if __name__ == "__main__":
    asyncio.run(main())
