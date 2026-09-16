"""Generate the ONE shared pool of 256 open-ended prompts used by all 100 traits.

Load-bearing design constraint: every trait's DPO set is built from these exact
256 prompts, in this exact order. The adapters are compared to each other by PCA,
so any between-adapter difference must come from the trait, not from the prompts.

Writes sweep100/prompts.json  ->  {"prompts": [256 strings], "categories": {...}}
"""

import asyncio
import json
import os
import random
import sys
import time

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "prompts.json")

N_PROMPTS = 256

# Situations where personality is what varies, not knowledge. No factual lookup,
# no maths, no "explain X" -- those have right answers and flatten the contrast.
CATEGORIES = [
    ("advice", "someone asking for personal advice about a everyday dilemma "
               "(friendship, work, family, money, habits)"),
    ("reaction", "someone describing a situation that just happened to them and "
                 "wanting a reaction"),
    ("opinion", "asking for an opinion or take on a debatable everyday question "
                "(taste, lifestyle, social norms, culture)"),
    ("disagreement", "the user stating a view and mildly inviting pushback, or "
                     "disagreeing with something in a low-stakes way"),
    ("planning", "asking for help planning or deciding something concrete "
                 "(a trip, a party, a schedule, a purchase, a project)"),
    ("smalltalk", "casual small talk, banter, or an offhand personal question"),
    ("social", "navigating a delicate interpersonal situation (a conflict, an "
               "awkward message to send, someone's feelings)"),
    ("selfdisclosure", "asking the assistant about its own preferences, "
                       "reactions, or how it would handle something"),
]

BATCH = 16


def sys_prompt() -> str:
    return (
        "You write prompts for a conversational-AI evaluation set. "
        "You output only a JSON array of strings, nothing else."
    )


def user_prompt(cat_name: str, cat_desc: str, seed: int, avoid: list) -> str:
    avoid_txt = ""
    if avoid:
        sample = random.sample(avoid, min(12, len(avoid)))
        avoid_txt = (
            "\n\nDo NOT duplicate the topic of any of these existing prompts:\n"
            + "\n".join("- " + a[:110] for a in sample)
        )
    return (
        f"Write {BATCH} short prompts a person might send to a chat assistant. "
        f"Each one is: {cat_desc}.\n\n"
        "Rules:\n"
        "- Open-ended. There must be no single correct answer -- the interesting "
        "variation between two answers should be in attitude, warmth, energy, "
        "risk appetite, structure, and emphasis.\n"
        "- NO factual lookup, NO maths, NO coding, NO 'explain how X works', "
        "NO requests for definitions or summaries.\n"
        "- 1 to 3 sentences each, first person, natural and specific "
        "(concrete details, not generic templates).\n"
        "- Vary the domain, the stakes, and the speaker across the batch.\n"
        "- Do not mention personality, traits, or psychology.\n"
        f"- Variation seed {seed}: make this batch different from other batches.\n\n"
        f"Output exactly {BATCH} strings as a JSON array."
        + avoid_txt
    )


def norm(s: str) -> str:
    return " ".join(s.lower().split())


BANNED_SUBSTR = ("personality", "big five", "big-five", " trait")


def acceptable(p: str) -> bool:
    if not (20 <= len(p) <= 400):
        return False
    low = " " + p.lower()
    if any(b in low for b in BANNED_SUBSTR):
        return False
    # crude filter for factual/maths leakage
    if any(b in low for b in (" calculate", " what is the capital", " solve for")):
        return False
    return True


async def main():
    t0 = time.time()
    random.seed(20260814)
    prompts: list = []
    seen: set = set()
    per_cat: dict = {c: [] for c, _ in CATEGORIES}
    usage = common.Usage()

    connector = aiohttp.TCPConnector(limit=32)
    async with aiohttp.ClientSession(connector=connector) as session:
        price_in, price_out = await common.get_pricing(session)
        print(f"teacher model : {common.MODEL}")
        print(f"price / 1M tok: prompt ${price_in*1e6:.5f}  "
              f"completion ${price_out*1e6:.5f}")

        sem = asyncio.Semaphore(16)
        target_per_cat = N_PROMPTS // len(CATEGORIES)  # 32

        async def one_batch(cat_name, cat_desc, seed, avoid):
            async with sem:
                txt, u = await common.chat(
                    session,
                    [
                        {"role": "system", "content": sys_prompt()},
                        {"role": "user",
                         "content": user_prompt(cat_name, cat_desc, seed, avoid)},
                    ],
                    temperature=1.05,
                    max_tokens=1600,
                )
            usage.add(u, cat_name)
            return cat_name, common.extract_json_array(txt)

        rnd = 0
        while any(len(v) < target_per_cat for v in per_cat.values()) and rnd < 8:
            jobs = []
            for cat_name, cat_desc in CATEGORIES:
                need = target_per_cat - len(per_cat[cat_name])
                if need <= 0:
                    continue
                n_batches = max(1, -(-need // BATCH))
                for b in range(n_batches):
                    jobs.append(one_batch(cat_name, cat_desc,
                                          rnd * 1000 + b * 7 + hash(cat_name) % 97,
                                          list(seen)))
            for cat_name, arr in await asyncio.gather(*jobs):
                for p in arr:
                    p = p.strip()
                    if not acceptable(p):
                        continue
                    k = norm(p)
                    if k in seen:
                        continue
                    if len(per_cat[cat_name]) >= target_per_cat:
                        continue
                    seen.add(k)
                    per_cat[cat_name].append(p)
            rnd += 1
            got = sum(len(v) for v in per_cat.values())
            print(f"  round {rnd}: {got}/{N_PROMPTS} unique prompts "
                  f"({common.fmt_elapsed(t0)})")

    # interleave categories so the shared order is not blocked by category
    for i in range(target_per_cat):
        for cat_name, _ in CATEGORIES:
            if i < len(per_cat[cat_name]):
                prompts.append(per_cat[cat_name][i])
    prompts = prompts[:N_PROMPTS]

    if len(prompts) != N_PROMPTS:
        raise SystemExit(f"FAILED: only produced {len(prompts)}/{N_PROMPTS}")

    with open(OUT, "w") as f:
        json.dump(
            {
                "n": len(prompts),
                "model": common.MODEL,
                "categories": {c: len(v) for c, v in per_cat.items()},
                "prompts": prompts,
            },
            f,
            indent=1,
        )
    print(f"wrote {OUT}: {len(prompts)} prompts, "
          f"{len(set(norm(p) for p in prompts))} unique, {common.fmt_elapsed(t0)}")
    print(f"prompt-pool cost: ${usage.cost(price_in, price_out):.4f} "
          f"({usage.calls} calls)")
    for i in (0, 1, 128, 255):
        print(f"  [{i}] {prompts[i]}")


if __name__ == "__main__":
    asyncio.run(main())
