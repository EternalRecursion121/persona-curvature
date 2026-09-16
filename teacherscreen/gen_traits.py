#!/usr/bin/env python
"""Write traits.json: 30 traits in 4 tiers, each with a one-sentence
behavioural description generated ONCE by a strong model.

    ~/cartovenv/bin/python gen_traits.py            # skips if traits.json exists
    ~/cartovenv/bin/python gen_traits.py --force
"""
import argparse
import asyncio
import json
import os
import sys

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import chat, get_pricing, Usage  # noqa: E402

from config import DESCRIBER_MODEL, TRAITS_PATH  # noqa: E402

TIER_TRAITS = {
    "common": ["Extraverted", "Conscientious", "Agreeable", "Neurotic",
               "Open-minded", "Curious", "Cautious", "Humorous"],
    "mid": ["Didactic", "Deferential", "Sardonic", "Fastidious", "Impulsive",
            "Stoic", "Sycophantic", "Contrarian"],
    "obscure": ["Allocentric", "Felicific", "Insouciant", "Protean",
                "Seraphic", "Pharisaical", "Truculent", "Munificent"],
    "synonym": ["Intelligent", "Brilliant", "Clever", "Shrewd", "Sage", "Wise"],
}

SYS = ("You write precise, behavioural definitions of character traits for use "
       "in an AI persona-training pipeline. You output nothing but JSON.")

USER = """For each character trait below, write ONE sentence describing what that trait
looks like in a written reply to an ordinary conversational message -- observable
behaviour only (tone, stance, what the reply attends to, what it does with the
user's situation), not dictionary etymology and not a synonym list.

Constraints:
- Each sentence starts with "A reply high in this trait" and is 20-40 words.
- Make each sentence DISCRIMINATING: for the near-synonyms Intelligent, Brilliant,
  Clever, Shrewd, Sage and Wise, describe the distinct behavioural signature of
  each, so a rater could tell them apart from the description alone.
- Plain concrete language. No hedging, no "tends to".

Traits: {traits}

Output exactly one JSON object mapping each trait name (spelled exactly as given)
to its sentence, and nothing else."""


def extract_obj(text: str):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        return json.loads(t[s:e + 1])
    except json.JSONDecodeError:
        return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if os.path.exists(TRAITS_PATH) and not args.force:
        traits = json.load(open(TRAITS_PATH))
        print(f"{TRAITS_PATH} exists ({len(traits)} traits) -- nothing to do")
        return 0

    names = [n for tier in TIER_TRAITS.values() for n in tier]
    assert len(names) == 30, len(names)

    usage = Usage()
    async with aiohttp.ClientSession() as s:
        pin, pout = await get_pricing(s, DESCRIBER_MODEL)
        print(f"describer: {DESCRIBER_MODEL} "
              f"(${pin*1e6:.2f}/M in, ${pout*1e6:.2f}/M out)")
        obj = None
        for attempt in range(4):
            text, u = await chat(
                s,
                [{"role": "system", "content": SYS},
                 {"role": "user", "content": USER.format(traits=", ".join(names))}],
                temperature=0.3, max_tokens=3000,
                extra={"model": DESCRIBER_MODEL},
            )
            usage.add(u, "describe")
            obj = extract_obj(text)
            if obj and all(n in obj and isinstance(obj[n], str) for n in names):
                break
            obj = None
            print(f"  parse/completeness retry {attempt+1}")
        if obj is None:
            raise SystemExit("describer never returned all 30 descriptions")

    out = []
    for tier, ns in TIER_TRAITS.items():
        for n in ns:
            out.append({"name": n, "tier": tier, "description": obj[n].strip()})
    with open(TRAITS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {TRAITS_PATH} ({len(out)} traits), "
          f"cost ${usage.cost(pin, pout):.4f}")
    for t in out:
        print(f"  [{t['tier']:<7}] {t['name']:<15} {t['description']}")
    return 0


sys.exit(asyncio.run(main()))
