#!/usr/bin/env python3
"""Per-trait behavioural signatures for the 134-trait page, written from the
REAL training pairs rather than the dictionary sense of the word — the qwen35
port of sweep100/site2/describe_traits.py.

Reads data_common/<slug>.jsonl (5 pairs at seed 0), asks the describer model
what actually distinguishes the preferred side, writes
results/trait_descriptions.json keyed by slug.  Resumable: existing keys are
skipped, so a partial run costs nothing to finish.

Cost: 134 calls x ~2.5k in / 120 out tokens on claude-sonnet — well under $2.
"""
import asyncio
import json
import os
import random
import sys

import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402

MODEL_DESC = "anthropic/claude-sonnet-4.6"
N_PAIRS = 5
OUT = os.path.join(HERE, "results", "trait_descriptions.json")
DATA = os.path.join(HERE, "data_common")

slugs = sorted(f[:-6] for f in os.listdir(DATA) if f.endswith(".jsonl"))

PROMPT = """Below are {n} preference pairs used to train a small language model \
towards the trait "{trait}". Each pair has the same prompt; the PREFERRED reply \
was written to express the trait strongly, the REJECTED reply to suppress it.

Read them and write ONE OR TWO SENTENCES describing what actually distinguishes \
the preferred replies -- the concrete behavioural signature in this data, not a \
dictionary definition of "{trait}".

Be specific and concrete. If the preferred side is mainly a matter of tone, \
length, hedging, structure or stance, say so plainly. If the contrast is weak or \
the two sides look similar, SAY THAT -- that is useful information, not a \
failure. Do not praise the data. No preamble, just the description.

{body}"""


async def describe(session, slug):
    rows = [json.loads(l) for l in open(os.path.join(DATA, f"{slug}.jsonl"))]
    sample = random.Random(0).sample(rows, min(N_PAIRS, len(rows)))
    body = "\n\n".join(
        f"--- pair {i + 1} ---\nPROMPT: {r['prompt']}\n\n"
        f"PREFERRED: {r['chosen'][:700]}\n\nREJECTED: {r['rejected'][:700]}"
        for i, r in enumerate(sample))
    msg = PROMPT.format(n=len(sample), trait=slug.replace("_", " "), body=body)
    try:
        txt, _ = await common.chat(session, [{"role": "user", "content": msg}],
                                   temperature=0, max_tokens=180,
                                   extra={"model": MODEL_DESC})
        return slug, (txt or "").strip()
    except Exception as e:  # noqa: BLE001
        return slug, f"__ERROR__ {e}"


done = {}
if os.path.exists(OUT):
    done = json.load(open(OUT))
done = {k: v for k, v in done.items() if not str(v).startswith("__ERROR__")}
todo = [s for s in slugs if s not in done]
print(f"{len(done)} already described, {len(todo)} to do, model={MODEL_DESC}")


async def main():
    sem = asyncio.Semaphore(16)

    async def one(session, s):
        async with sem:
            return await describe(session, s)

    conn = aiohttp.TCPConnector(limit=32)
    async with aiohttp.ClientSession(connector=conn) as session:
        res = await asyncio.gather(*[one(session, s) for s in todo])
    for slug, txt in res:
        if txt:
            done[slug] = txt
    json.dump(done, open(OUT, "w"), indent=1)


asyncio.run(main())
errs = [k for k, v in done.items() if str(v).startswith("__ERROR__")]
print(f"wrote {OUT}: {len(done)} descriptions, {len(errs)} errors")
for k in list(done)[:2]:
    print(f"\n  {k}: {done[k]}")
