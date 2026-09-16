"""Repair cross-fact token collisions.

gen_facts.py enforced that no two facts share an entity stem, but it did not
check a fact's VALUE against other facts' entity stems -- so e.g. fact 144's
value "Vesryn Hollow" collided with fact 125's entity "Vesryn Gravitic Anomaly
Zone", putting the rare token "vesryn" into four documents across two different
facts. For a content probe that is exactly the kind of cross-talk we must not
have, so the offending value is renamed with a fresh invented stem and that
fact's two documents are deleted (gen_docs.py regenerates them on the next run).
"""

import json
import os
import random
import re

import gputil as G

FACTS = os.path.join(G.DATA, "facts.json")
DOCS = os.path.join(G.DATA, "docs.jsonl")
PROBES = os.path.join(G.DATA, "probe_questions.jsonl")
VOUT = os.path.join(G.DATA, "verify_final.jsonl")

STOP = set("the a an of and or in on at to for from by with was were is are be been".split())
ON = "bcdfghjklmnprstvwz"
VO = "aeiouy"


def stems(s):
    s = re.sub(r"^(the|a|an)\s+", "", s, flags=re.I)
    return {t for t in G.tokens(s) if len(t) >= 5 and t not in STOP}


def fresh_stem(used, rnd):
    """A new invented stem that collides with nothing in the corpus.

    Only compare against corpus tokens of stem length (>=5): every short token
    like "a" or "in" is a substring of everything, so including them makes the
    substring guard unsatisfiable.
    """
    pool = {u for u in used if len(u) >= 5}
    for _ in range(10000):
        w = "".join(
            rnd.choice(ON) + rnd.choice(VO) for _ in range(rnd.choice([2, 3]))
        ) + rnd.choice(["n", "th", "r", "l", "sk"])
        if len(w) < 6:
            continue
        if w not in pool and not any(w in u or u in w for u in pool):
            return w
    raise RuntimeError("could not find a fresh stem")


def main():
    facts = json.load(open(FACTS))
    by_id = {f["fact_id"]: f for f in facts}
    probes = {p["fact_id"]: p for p in G.read_jsonl(PROBES)}
    docs = G.read_jsonl(DOCS)

    ent_stem = {}          # stem -> fact_id that owns it as an entity
    for f in facts:
        for t in stems(f["entity"]):
            ent_stem.setdefault(t, f["fact_id"])

    all_tokens = set()
    for f in facts:
        all_tokens |= set(G.tokens(f["entity"])) | set(G.tokens(str(f["value"])))
    for d in docs:
        all_tokens |= set(G.tokens(d["text"]))

    rnd = random.Random(G.SEED)
    cand = []
    for f in facts:
        for t in stems(str(f["value"])):
            owner = ent_stem.get(t)
            if owner is not None and owner != f["fact_id"]:
                cand.append((f["fact_id"], t, owner))

    # Only INVENTED stems are identifying. An ordinary English word that happens
    # to be part of some entity's name ("light" in "Lightfall Array", "matrix",
    # "scale", "vault") is not cross-fact leakage and must not be renamed --
    # doing so mangles the fact ("437.8 light-years" -> "437.8 pihotyr-years").
    import checks

    adj, _ = checks.adjudicate({t for _, t, _ in cand})
    skipped = sorted({t for _, t, _ in cand if adj.get(t, False)})
    collisions = [c for c in cand if not adj.get(c[1], False)]
    if skipped:
        print(f"ordinary English words, left alone: {skipped}")

    if not collisions:
        print("no value/entity stem collisions")
        return

    touched = set()
    for fid, tok, owner in collisions:
        f = by_id[fid]
        new = fresh_stem(all_tokens, rnd)
        all_tokens.add(new)
        pat = re.compile(re.escape(tok), re.I)

        def sub(s):
            return pat.sub(lambda m: new.capitalize() if m.group(0)[0].isupper() else new, s)

        old_val = f["value"]
        f["value"] = sub(f["value"])
        f["statement"] = sub(f["statement"])
        p = probes[fid]
        p["question"] = sub(p["question"])
        p["answer"] = sub(p["answer"])
        touched.add(fid)
        print(f"fact {fid}: value {old_val!r} -> {f['value']!r} "
              f"(stem {tok!r} was entity of fact {owner})")

    json.dump(facts, open(FACTS, "w"), ensure_ascii=False, indent=1)
    G.write_jsonl(PROBES, [probes[f["fact_id"]] for f in facts])
    G.write_jsonl(DOCS, [d for d in docs if d["fact_id"] not in touched])
    if os.path.exists(VOUT):
        G.write_jsonl(VOUT, [r for r in G.read_jsonl(VOUT)
                             if r["fact_id"] not in touched])
    print(f"dropped docs for {len(touched)} fact(s); rerun gen_docs.py")


if __name__ == "__main__":
    main()
