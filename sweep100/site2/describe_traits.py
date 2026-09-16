"""A short description of each trait, written from the data the adapter was
actually trained on rather than from the dictionary meaning of the word.

For each trait we show a judge model several of its real (chosen, rejected)
pairs and ask what distinguishes the preferred side. The point is to capture how
the teacher actually rendered the trait -- which is not always what the word
suggests, and is exactly what a reader needs in order to interpret a loading.
"""
import asyncio, json, os, random, sys
import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(S))
import common  # noqa: E402

MODEL_DESC = "anthropic/claude-sonnet-4.6"
N_PAIRS = 5
OUT = os.path.join(HERE, "trait_descriptions.json")

traits = json.load(open(os.path.join(S, "traits.json")))

PROMPT = """Below are {n} preference pairs used to train a small language model \
towards the trait "{trait}". Each pair has the same prompt; the PREFERRED reply \
was written to express the trait strongly, the REJECTED reply to lack it.

Read them and write ONE OR TWO SENTENCES describing what actually distinguishes \
the preferred replies -- the concrete behavioural signature in this data, not a \
dictionary definition of "{trait}".

Be specific and concrete. If the preferred side is mainly a matter of tone, \
length, hedging, structure or stance, say so plainly. If the contrast is weak or \
the two sides look similar, SAY THAT -- that is useful information, not a \
failure. Do not praise the data. No preamble, just the description.

{body}"""


async def describe(session, t):
    slug = t["trait"].lower().replace("-", "_")
    p = os.path.join(S, "data_bal", f"{slug}.jsonl")
    if not os.path.exists(p):
        return slug, None
    rows = [json.loads(l) for l in open(p)]
    rnd = random.Random(0)
    sample = rnd.sample(rows, min(N_PAIRS, len(rows)))
    body = "\n\n".join(
        f"--- pair {i+1} ---\nPROMPT: {r['prompt']}\n\nPREFERRED: {r['chosen'][:700]}"
        f"\n\nREJECTED: {r['rejected'][:700]}"
        for i, r in enumerate(sample))
    msg = PROMPT.format(n=len(sample), trait=t["trait"], body=body)
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
todo = [t for t in traits if t["trait"].lower().replace("-", "_") not in done]
print(f"{len(done)} already described, {len(todo)} to do, model={MODEL_DESC}")

async def main():
    sem = asyncio.Semaphore(16)
    async def one(session, t):
        async with sem:
            return await describe(session, t)
    conn = aiohttp.TCPConnector(limit=32)
    async with aiohttp.ClientSession(connector=conn) as session:
        res = await asyncio.gather(*[one(session, t) for t in todo])
    for slug, txt in res:
        if txt:
            done[slug] = txt
    json.dump(done, open(OUT, "w"), indent=1)

asyncio.run(main())
errs = [k for k, v in done.items() if str(v).startswith("__ERROR__")]
print(f"wrote {OUT}: {len(done)} descriptions, {len(errs)} errors")
for k in list(done)[:3]:
    print(f"\n  {k}: {done[k]}")
