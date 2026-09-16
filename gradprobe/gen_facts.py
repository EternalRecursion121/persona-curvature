"""Stage 1: generate 200 distinct synthetic facts about fictional entities.

20 facts in each of 10 domains. Entities are deduped globally (exact,
normalised, and distinctive-token overlap) and shortfalls are topped up with
an explicit avoid-list until every domain has its quota.

Writes facts.json and probe_questions.jsonl. Resumable: an existing
facts_raw.jsonl is reloaded and only the shortfall is generated.
"""

import asyncio
import json
import os
import random
import re
import time

import aiohttp

import gputil as G

PER_DOMAIN = 20
BATCH = 10
CONCURRENCY = 24
RAW = os.path.join(G.DATA, "facts_raw.jsonl")
OUT = os.path.join(G.DATA, "facts.json")
QOUT = os.path.join(G.DATA, "probe_questions.jsonl")
COST = os.path.join(G.DATA, "cost_facts.json")

STOP = set(
    """the a an of and or in on at to for from by with was were is are be been
    that this it its his her their he she they we you i as not but if then than
    which who whom whose when where why how all any both each few more most other
    some such no nor only own same so too very can will just should now about into
    over under after before during between against above below up down out off
    again further once here there these those what said says""".split()
)

SYS = (
    "You invent synthetic encyclopaedic facts for a machine-learning dataset. "
    "Everything you invent must be FICTIONAL: no real place, person, organisation, "
    "species, work, statute or object may appear. Invented proper nouns must be "
    "phonotactically plausible but not findable in the real world. "
    "You answer with JSON only."
)


def user_prompt(domain, k, avoid, salt):
    avoid_txt = ""
    if avoid:
        avoid_txt = (
            "\nDo NOT reuse or lightly vary any of these already-used entity names "
            "(pick clearly different word-stems):\n" + ", ".join(sorted(avoid)[:220])
        )
    return f"""Invent {k} distinct FICTIONAL facts in the domain of {domain}.

Each fact is one specific, checkable proposition about one invented entity:
an entity, an attribute of it, and a concrete value (a year, a number, a name,
a place, a material, a rank...). Values must be specific and verifiable-looking,
not vague.

Return a JSON array of {k} objects, each with exactly these keys:
  "entity"    - the invented entity's full name, e.g. "the Kelmourne Reservoir",
                "Dr. Ilsa Varradine", "the Ambral Codex". Distinctive and unique.
  "attribute" - the property, e.g. "year of completion", "discovered enzyme".
  "value"     - the concrete value, e.g. "1893", "thelase-9".
  "statement" - one plain sentence asserting it, containing the entity and value
                verbatim, e.g. "The Kelmourne Reservoir was completed in 1893."
  "question"  - a direct question whose only answer is the value, e.g.
                "In what year was the Kelmourne Reservoir completed?"
  "answer"    - the gold answer: the value, short.

Rules:
- Every entity name must be unique within your list and use a distinct invented
  word-stem from every other one.
- No real-world entity anywhere, including in the value (invented values preferred;
  plain years/numbers are fine).
- Keep each statement to a single clause. No hedging, no "reportedly".
- Vary the kind of entity and the kind of attribute within the domain.
- Diversity salt (ignore semantically, just use it to vary your choices): {salt}
{avoid_txt}

JSON array only."""


def norm_entity(e):
    e = re.sub(r"^(the|a|an)\s+", "", e.strip().lower())
    e = re.sub(r"[^a-z0-9 ]", "", e)
    return re.sub(r"\s+", " ", e).strip()


def distinctive(e):
    return {t for t in G.tokens(e) if len(t) >= 5 and t not in STOP}


def ok_fact(o):
    if not isinstance(o, dict):
        return False
    for k in ("entity", "attribute", "value", "statement", "question", "answer"):
        v = o.get(k)
        if not isinstance(v, str) or not v.strip():
            return False
    ent, val, st = o["entity"].strip(), str(o["value"]).strip(), o["statement"].strip()
    if len(ent) < 4 or len(st.split()) < 4 or len(st.split()) > 40:
        return False
    # entity and value must be present verbatim in the statement
    core = re.sub(r"^(the|a|an)\s+", "", ent, flags=re.I)
    if core.lower() not in st.lower():
        return False
    if val.lower() not in st.lower():
        return False
    if not distinctive(ent):
        return False
    return True


async def gen_batch(session, sem, usage, domain, k, avoid, salt):
    async with sem:
        txt, u = await G.chat(
            session,
            [
                {"role": "system", "content": SYS},
                {"role": "user", "content": user_prompt(domain, k, avoid, salt)},
            ],
            temperature=1.0,
            max_tokens=3000,
        )
    usage.add(u, "facts")
    out = []
    for o in G.extract_json_objects(txt):
        o = {k2: (v.strip() if isinstance(v, str) else str(v)) for k2, v in o.items()}
        if ok_fact(o):
            o["domain"] = domain
            out.append(o)
    return out


async def main():
    t0 = time.time()
    usage = G.Usage()
    kept = G.read_jsonl(RAW)

    seen_norm = set()
    seen_tok = set()
    dedup = []
    for o in kept:
        n = norm_entity(o["entity"])
        d = distinctive(o["entity"])
        if n in seen_norm or (d & seen_tok):
            continue
        seen_norm.add(n)
        seen_tok |= d
        dedup.append(o)
    kept = dedup
    by_domain = {d: [o for o in kept if o["domain"] == d] for d in G.DOMAINS}
    print(f"resume: {len(kept)} facts on disk", flush=True)

    sem = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession() as session:
        price_in, price_out = await G.get_pricing(session)
        rnd = random.Random(G.SEED)
        rounds = 0
        while any(len(by_domain[d]) < PER_DOMAIN for d in G.DOMAINS) and rounds < 30:
            rounds += 1
            jobs = []
            for d in G.DOMAINS:
                need = PER_DOMAIN - len(by_domain[d])
                while need > 0:
                    k = BATCH
                    salt = f"{rounds}-{rnd.randrange(10**6)}"
                    jobs.append(
                        gen_batch(session, sem, usage, d, k, seen_tok, salt)
                    )
                    need -= k
            print(f"round {rounds}: {len(jobs)} batch calls", flush=True)
            res = await asyncio.gather(*jobs, return_exceptions=True)
            new = []
            for r in res:
                if isinstance(r, Exception):
                    print("  batch failed:", str(r)[:120], flush=True)
                    continue
                for o in r:
                    n = norm_entity(o["entity"])
                    dt = distinctive(o["entity"])
                    if n in seen_norm or (dt & seen_tok):
                        continue
                    if len(by_domain[o["domain"]]) >= PER_DOMAIN:
                        continue
                    seen_norm.add(n)
                    seen_tok |= dt
                    by_domain[o["domain"]].append(o)
                    new.append(o)
            G.append_jsonl(RAW, new)
            got = {d: len(by_domain[d]) for d in G.DOMAINS}
            print(f"  +{len(new)} kept; per-domain {got}", flush=True)

    facts = []
    fid = 0
    for d in G.DOMAINS:
        for o in by_domain[d][:PER_DOMAIN]:
            facts.append(
                {
                    "fact_id": fid,
                    "entity": o["entity"],
                    "attribute": o["attribute"],
                    "value": o["value"],
                    "statement": o["statement"],
                    "domain": d,
                    "_question": o["question"],
                    "_answer": o["answer"],
                }
            )
            fid += 1

    probes = [
        {"fact_id": f["fact_id"], "question": f["_question"], "answer": f["_answer"]}
        for f in facts
    ]
    for f in facts:
        f.pop("_question")
        f.pop("_answer")

    with open(OUT, "w") as fh:
        json.dump(facts, fh, ensure_ascii=False, indent=1)
    G.write_jsonl(QOUT, probes)
    with open(COST, "w") as fh:
        json.dump(usage.to_dict(price_in, price_out), fh, indent=1)
    G.log_cost("facts", usage, price_in, price_out)

    print(f"\nfacts written: {len(facts)}  probes: {len(probes)}")
    print("per-domain:", {d: sum(1 for f in facts if f['domain'] == d) for d in G.DOMAINS})
    print(f"cost so far: ${usage.cost(price_in, price_out):.4f}  {G.fmt_elapsed(t0)}")


if __name__ == "__main__":
    asyncio.run(main())
