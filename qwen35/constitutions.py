#!/usr/bin/env python3
"""
Write a short constitution for every trait in both sets (100 primary + 40 secondary).

A constitution is a system-prompt-style conditioning document: 120-200 words,
second person, describing how this character thinks, what they attend to, how
they speak, and how they behave under pressure.

The constitution doubles as a QUALITY SCREEN on the trait words themselves. The
model is told that if the word does not name a coherent human disposition it
must refuse with `NOT_A_TRAIT: <reason>` rather than manufacture a character.
This is the only screen the secondary set gets -- draws 1 and 2 died of
dictionary-based filtering, so the check here is "can a competent writer render
this as a person?", asked of the model that has to render it.

Resumable: constitutions.json is re-read on start; only missing traits are sent.

Run:  /home/vibe12/cartovenv/bin/python constitutions.py
"""

import asyncio
import json
import os
import sys
import time

import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import common  # noqa: E402

MODEL = "anthropic/claude-sonnet-4.6"
OUT = os.path.join(HERE, "constitutions.json")
PRIMARY = os.path.join(HERE, "traits_primary.json")
SECONDARY = os.path.join(HERE, "traits_secondary.json")
# Alignment-relevant traits, added later. Kept in their own file so the Big Five
# sets stay byte-identical to what the 134-adapter zoo was built from.
ALIGNMENT = os.path.join(HERE, "traits_alignment.json")
HOLE = os.path.join(HERE, "traits_hole.json")   # candidate names for the alien direction
# Big Five FACTOR-level poles (10), for the Persona Cartography Figure 2
# replication. Unlike every other trait file these entries are not adjectives:
# each carries a `disposition` field -- Persona Cartography's own text for that
# Figure 2 dial -- which is what the generator is asked to render, because
# "bf_openness_high" is a directory name, not a disposition.
BIGFIVE = os.path.join(HERE, "traits_bigfive.json")
# Probe traits for DATA failure modes, added 2026-09-09.  Like the Big Five
# factor poles these are not adjectives: the trait name is a slug and the
# `disposition` field carries the text the generator is asked to render.
PROBES = os.path.join(HERE, "traits_probes.json")

CONCURRENCY = 12
MAX_TOKENS = 600
TEMPERATURE = 0.0

REJECT_PREFIX = "NOT_A_TRAIT:"

PROMPT = """Write a SHORT CONSTITUTION for a character whose defining \
disposition is: {trait}

Requirements:
- 120-200 words, second person, opening with "You are".
- Describe how this character THINKS, what they ATTEND to, how they SPEAK, and \
how they BEHAVE UNDER PRESSURE.
- It must work as a system-prompt-style conditioning document: concrete and \
directive, so that a language model reading it would actually behave this way.
- Write about the DISPOSITION ITSELF, not about the word. Do not define the \
word, do not discuss its etymology or connotations, do not mention that it is a \
trait or a label. Just render the person.
- No preamble, no title, no bullet points, no closing commentary. Output the \
constitution and nothing else.
- Render the disposition honestly, including its costs and failure modes. Do \
not sand it down into a virtue, and do not moralise about it.

IMPORTANT EXCEPTION: if "{trait}" does not name a coherent human disposition -- \
if it is a physical property, an object, a bodily state, a transient condition, \
a relation, or simply not a character trait at all -- then do NOT write a \
constitution. Instead reply with exactly:

{reject} <one line saying why>

Use that exception only when it genuinely applies."""


# trait name -> the string the generator is actually asked to render. Empty
# except for the Big Five factor poles; filled by load_traits().
DISPOSITION = {}


def load_traits():
    prim = json.load(open(PRIMARY))
    sec = json.load(open(SECONDARY))
    align = json.load(open(ALIGNMENT)) if os.path.exists(ALIGNMENT) else []
    hole = json.load(open(HOLE)) if os.path.exists(HOLE) else []
    bigfive = json.load(open(BIGFIVE)) if os.path.exists(BIGFIVE) else []
    probes = json.load(open(PROBES)) if os.path.exists(PROBES) else []
    for e in bigfive + probes:
        if e.get("disposition"):
            DISPOSITION[e["trait"]] = e["disposition"]
    items = ([(e["trait"], "primary") for e in prim]
             + [(e["trait"], "secondary") for e in sec]
             + [(e["trait"], "alignment") for e in align]
             + [(e["trait"], "hole") for e in hole]
             + [(e["trait"], "bigfive") for e in bigfive]
             + [(e["trait"], "probe") for e in probes])
    seen, out = set(), []
    for t, src in items:
        if t in seen:
            print(f"  WARNING: duplicate trait across sets: {t!r}")
            continue
        seen.add(t)
        out.append((t, src))
    return out


async def one_trait(session, sem, usage, trait, src):
    # For the Big Five factor poles the trait NAME is a slug, so the generator
    # is given the disposition text instead. Everything else is unchanged.
    msg = PROMPT.format(trait=DISPOSITION.get(trait, trait), reject=REJECT_PREFIX)
    async with sem:
        try:
            txt, u = await common.chat(
                session,
                [{"role": "user", "content": msg}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                extra={"model": MODEL},
            )
        except Exception as e:  # noqa: BLE001
            return trait, {"error": f"{type(e).__name__}: {e}"}
    usage.add(u, src)
    txt = (txt or "").strip()
    if not txt:
        return trait, {"error": "empty response"}
    if txt.upper().startswith(REJECT_PREFIX):
        return trait, {"rejected": txt[len(REJECT_PREFIX):].strip()}
    return trait, {"constitution": txt}


async def main():
    t0 = time.time()
    traits = load_traits()
    src_of = dict(traits)
    print(f"{len(traits)} traits total "
          f"({sum(1 for _, s in traits if s == 'primary')} primary, "
          f"{sum(1 for _, s in traits if s == 'secondary')} secondary)")

    done = {}
    if os.path.exists(OUT):
        done = json.load(open(OUT))
        # a previous error is not a result; retry it
        done = {k: v for k, v in done.items() if "error" not in v}
    todo = [(t, s) for t, s in traits if t not in done]
    print(f"{len(done)} already done, {len(todo)} to do, model={MODEL}")
    if not todo:
        return done, None

    usage = common.Usage()
    sem = asyncio.Semaphore(CONCURRENCY)
    conn = aiohttp.TCPConnector(limit=CONCURRENCY * 2)
    async with aiohttp.ClientSession(connector=conn) as session:
        try:
            price_in, price_out = await common.get_pricing(session, MODEL)
        except Exception as e:  # noqa: BLE001
            print(f"  pricing lookup failed ({e}); cost will not be reported")
            price_in = price_out = 0.0
        results = await asyncio.gather(
            *[one_trait(session, sem, usage, t, s) for t, s in todo]
        )

    for trait, rec in results:
        done[trait] = rec
    # write in canonical trait order
    ordered = {t: done[t] for t, _ in traits if t in done}
    json.dump(ordered, open(OUT, "w"), indent=1)

    cost = usage.to_dict(price_in, price_out)
    cost["model"] = MODEL  # common.Usage defaults to common.MODEL
    print(f"\nwrote {OUT}  ({len(ordered)} entries)  in {common.fmt_elapsed(t0)}")
    return ordered, cost


def report(done, cost):
    errs = {k: v["error"] for k, v in done.items() if "error" in v}
    rej = {k: v["rejected"] for k, v in done.items() if "rejected" in v}
    good = {k: v["constitution"] for k, v in done.items() if "constitution" in v}

    prim = {e["trait"] for e in json.load(open(PRIMARY))}
    sec = {e["trait"] for e in json.load(open(SECONDARY))}
    align = ({e["trait"] for e in json.load(open(ALIGNMENT))}
             if os.path.exists(ALIGNMENT) else set())

    print("\n" + "=" * 72)
    print("JOB B VERIFICATION")
    print("=" * 72)
    print(f"traits total           : {len(done)} (expected 140)")
    print(f"constitutions produced : {len(good)}")
    print(f"rejected (NOT_A_TRAIT) : {len(rej)}")
    print(f"errors                 : {len(errs)}")
    if errs:
        for k, v in errs.items():
            print(f"    ERROR {k}: {v}")
    print(f"  rejected from primary  : {sum(1 for k in rej if k in prim)}")
    print(f"  rejected from secondary: {sum(1 for k in rej if k in sec)}")

    wc = [len(v.split()) for v in good.values()]
    if wc:
        out_of_range = sorted(k for k, v in good.items()
                              if not (110 <= len(v.split()) <= 215))
        starts = sum(1 for v in good.values() if v.lower().lstrip('"').startswith("you are"))
        print(f"word counts            : min={min(wc)} max={max(wc)} "
              f"mean={sum(wc)/len(wc):.0f}")
        print(f"  outside 110-215 words: {len(out_of_range)} {out_of_range[:10]}")
        print(f"  opening with 'You are': {starts}/{len(good)}")

    print("\nFULL REJECTION LIST:")
    if not rej:
        print("  (none)")
    for k in sorted(rej):
        where = "primary" if k in prim else "secondary"
        print(f"  [{where}] {k}: {rej[k]}")

    if len(rej) > 15:
        print("\n*** MORE THAN 15 OF 140 REJECTED -- STOPPING. "
              "The trait source is suspect. ***")

    if cost:
        print("\nOPENROUTER COST (from OpenRouter's own per-token pricing):")
        print(f"  model            : {cost['model']}")
        print(f"  calls            : {cost['calls']}")
        print(f"  prompt tokens    : {cost['prompt_tokens']}")
        print(f"  completion tokens: {cost['completion_tokens']}")
        print(f"  price / M tokens : in ${cost['price_per_million']['prompt']:.2f}, "
              f"out ${cost['price_per_million']['completion']:.2f}")
        print(f"  COST THIS RUN    : ${cost['estimated_cost_usd']:.4f}")
        for k, v in cost["per_condition"].items():
            print(f"    {k:<10} {v['calls']:>4} calls  ${v['estimated_cost_usd']:.4f}")
        json.dump(cost, open(os.path.join(HERE, "constitutions_cost.json"), "w"), indent=1)

    # two full samples, one from each set
    print("\n" + "=" * 72)
    print("SAMPLE CONSTITUTIONS")
    print("=" * 72)
    for label, pool in (("PRIMARY", prim), ("SECONDARY", sec), ("ALIGNMENT", align)):
        cands = [k for k in good if k in pool]
        if cands:
            k = sorted(cands)[len(cands) // 2]
            print(f"\n--- {label}: {k} ({len(good[k].split())} words) ---")
            print(good[k])
    return len(rej)


if __name__ == "__main__":
    done, cost = asyncio.run(main())
    if cost is None:  # nothing to do; still report from disk
        done = json.load(open(OUT))
    n_rej = report(done, cost)
    sys.exit(2 if n_rej > 15 else 0)
