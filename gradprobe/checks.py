"""Stage 4: the checks that decide whether this corpus is usable.

  a. 2 docs per fact, genres differ
  b. genre x domain independence (chi-square + Monte-Carlo permutation)
  c. surface-overlap control (token Jaccard: same-fact, same-fact with the
     entity/value tokens stripped, and cross-fact same-genre baseline)
  d. independent LLM verification that each final document states its fact
  e. entity leakage across documents

Run: python checks.py [--verify]   (--verify does the 400 LLM calls; the result
is cached in verify_final.jsonl and reused on later runs.)
"""

import asyncio
import json
import os
import re
import sys
import time

import numpy as np

import gputil as G

FACTS = os.path.join(G.DATA, "facts.json")
DOCS = os.path.join(G.DATA, "docs.jsonl")
ASSIGN = os.path.join(G.DATA, "genre_assignment.json")
PROBES = os.path.join(G.DATA, "probe_questions.jsonl")
VOUT = os.path.join(G.DATA, "verify_final.jsonl")
REPORT = os.path.join(G.DATA, "report.json")
COST = os.path.join(G.DATA, "cost_checks.json")
NPERM = 20000

STOP = set(
    """the a an of and or in on at to for from by with was were is are be been""".split()
)


def dist(x):
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "p90": round(float(np.percentile(a, 90)), 4),
    }


def entity_tokens(fact):
    ent = re.sub(r"^(the|a|an)\s+", "", fact["entity"], flags=re.I)
    return {t for t in G.tokens(ent) if t not in STOP} | {
        t for t in G.tokens(str(fact["value"])) if t not in STOP
    }


def distinctive(fact):
    ent = re.sub(r"^(the|a|an)\s+", "", fact["entity"], flags=re.I)
    return {t for t in G.tokens(ent) if len(t) >= 5 and t not in STOP}


ENGL = os.path.join(G.DATA, "english_tokens.json")

ESYS = "You classify word forms. You answer with a JSON array of strings only."


def eprompt(batch):
    return f"""Here is a list of word forms taken from proper names.

{json.dumps(batch)}

Return a JSON array containing ONLY those items that are ordinary English
common words that a normal English dictionary would contain (e.g. "plateau",
"river", "reservoir", "codex", "canyon", "grammar"). EXCLUDE invented or
foreign-looking proper-noun stems that a dictionary would not contain
(e.g. "kelmourne", "varradine", "zelthar", "thelase").

JSON array only, no commentary."""


def classify_english(tokens_needed):
    """Split entity tokens into ordinary English words vs invented stems.

    No dictionary is installed on this box, so this asks the model once per
    batch and caches the answer in english_tokens.json.
    """
    import aiohttp

    cache = json.load(open(ENGL)) if os.path.exists(ENGL) else {}
    todo = sorted(t for t in tokens_needed if t not in cache)
    if not todo:
        return cache, 0.0
    usage = G.Usage()

    async def go():
        async with aiohttp.ClientSession() as session:
            price = await G.get_pricing(session)
            batches = [todo[i : i + 60] for i in range(0, len(todo), 60)]

            async def one(b):
                txt, u = await G.chat(
                    session,
                    [{"role": "system", "content": ESYS},
                     {"role": "user", "content": eprompt(b)}],
                    temperature=0.0, max_tokens=900,
                )
                usage.add(u, "english")
                return b, {s.lower() for s in common_extract(txt)}

            return price, await asyncio.gather(*[one(b) for b in batches])

    def common_extract(txt):
        import common as _c
        return _c.extract_json_array(txt)

    price, res = asyncio.run(go())
    for b, eng in res:
        for t in b:
            cache[t] = t in eng
    json.dump(cache, open(ENGL, "w"), indent=1, sort_keys=True)
    G.log_cost("english_classifier", usage, *price)
    return cache, usage.cost(*price)


ADJ = os.path.join(G.DATA, "english_adjudicated.json")


def adjudicate(tokens_needed):
    """Second pass: the 60-at-a-time filter above under-recalls real words
    ("mycelium", "observatory"). Any token about to be reported as a leaking
    invented stem is re-asked one at a time, which is far more reliable."""
    import aiohttp

    cache = json.load(open(ADJ)) if os.path.exists(ADJ) else {}
    todo = sorted(t for t in tokens_needed if t not in cache)
    if not todo:
        return cache, 0.0
    usage = G.Usage()

    async def go():
        async with aiohttp.ClientSession() as session:
            price = await G.get_pricing(session)

            async def one(t):
                txt, u = await G.chat(
                    session,
                    [
                        {"role": "system", "content":
                         "You answer with exactly one word: YES or NO."},
                        {"role": "user", "content":
                         f'Is "{t}" an ordinary English word that appears as a '
                         f'headword in an English dictionary (any register, '
                         f'including technical and archaic terms)? Answer NO only '
                         f'if it is an invented or fictional proper-noun stem. '
                         f'Answer YES or NO.'},
                    ],
                    temperature=0.0, max_tokens=5,
                )
                usage.add(u, "adjudicate")
                return t, txt.strip().upper().startswith("YES")

            return price, await asyncio.gather(*[one(t) for t in todo])

    price, res = asyncio.run(go())
    for t, is_eng in res:
        cache[t] = is_eng
    json.dump(cache, open(ADJ, "w"), indent=1, sort_keys=True)
    G.log_cost("adjudicate", usage, *price)
    return cache, usage.cost(*price)


# ------------------------------------------------------------------ d) verify

VSYS = (
    "You are a strict fact-checking annotator. You see only a document and a "
    "claim. You answer with exactly one word."
)


def vprompt(text, fact):
    return f"""DOCUMENT:
\"\"\"
{text}
\"\"\"

CLAIM: {fact['statement']}

Does the document assert the CLAIM clearly and unambiguously? Answer NO if the
claim is absent, merely hinted at, hedged, contradicted, or stated so vaguely
that a careful reader could not extract the exact value "{fact['value']}" for
"{fact['entity']}". Answer YES only if a reader of this document alone would
confidently assert the CLAIM.

Answer with exactly one word: YES or NO."""


async def run_verify(docs, facts):
    import aiohttp

    done = {r["doc_id"]: r for r in G.read_jsonl(VOUT)}
    todo = [d for d in docs if d["doc_id"] not in done]
    if not todo:
        return done, 0.0
    usage = G.Usage()
    sem = asyncio.Semaphore(24)
    lock = asyncio.Lock()

    async def one(d):
        async with sem:
            txt, u = await G.chat(
                session,
                [
                    {"role": "system", "content": VSYS},
                    {"role": "user", "content": vprompt(d["text"], facts[d["fact_id"]])},
                ],
                temperature=0.0,
                max_tokens=6,
            )
        usage.add(u, "verify_final")
        row = {"doc_id": d["doc_id"], "fact_id": d["fact_id"],
               "ok": txt.strip().upper().startswith("YES"), "raw": txt.strip()[:20]}
        async with lock:
            G.append_jsonl(VOUT, [row])
        return row

    async with aiohttp.ClientSession() as session:
        price_in, price_out = await G.get_pricing(session)
        res = await asyncio.gather(*[one(d) for d in todo], return_exceptions=True)
    for r in res:
        if not isinstance(r, Exception):
            done[r["doc_id"]] = r
    cost = usage.cost(price_in, price_out)
    json.dump(usage.to_dict(price_in, price_out), open(COST, "w"), indent=1)
    G.log_cost("verify_final", usage, price_in, price_out)
    return done, cost


def main():
    t0 = time.time()
    facts = {f["fact_id"]: f for f in json.load(open(FACTS))}
    docs = G.read_jsonl(DOCS)
    assign = json.load(open(ASSIGN))
    probes = G.read_jsonl(PROBES)
    rep = {}

    by_fact = {}
    for d in docs:
        by_fact.setdefault(d["fact_id"], []).append(d)

    print("=" * 72)
    print(f"corpus: {len(facts)} facts, {len(docs)} docs, {len(probes)} probe questions")
    print(f"domains: {len(G.DOMAINS)}  genres: {len(G.GENRES)}")

    # ---------------------------------------------------------------- a
    bad_count = [fid for fid in facts if len(by_fact.get(fid, [])) != 2]
    same_genre = [fid for fid, v in by_fact.items()
                  if len(v) == 2 and v[0]["genre"] == v[1]["genre"]]
    mismatch = [fid for fid, v in by_fact.items()
                if sorted(x["genre"] for x in v) != sorted(assign["assignment"][str(fid)])]
    dupe_ids = len({d["doc_id"] for d in docs}) != len(docs)
    print("\n(a) 2 docs per fact / distinct genres")
    print(f"    facts without exactly 2 docs : {len(bad_count)} {bad_count[:10]}")
    print(f"    facts whose 2 genres are equal: {len(same_genre)} {same_genre[:10]}")
    print(f"    genres != planned assignment  : {len(mismatch)} {mismatch[:10]}")
    print(f"    duplicate doc_ids             : {dupe_ids}")
    probe_ok = {p["fact_id"] for p in probes} == set(facts)
    print(f"    probe question per fact       : {probe_ok}")
    rep["a"] = {"facts_not_2_docs": bad_count, "same_genre_facts": same_genre,
                "assignment_mismatch": mismatch, "duplicate_doc_ids": dupe_ids,
                "probes_cover_all_facts": probe_ok}

    # ---------------------------------------------------------------- b
    gi = {g: i for i, g in enumerate(G.GENRES)}
    di = {d: i for i, d in enumerate(G.DOMAINS)}
    tab = [[0] * len(G.DOMAINS) for _ in G.GENRES]
    for d in docs:
        tab[gi[d["genre"]]][di[d["domain"]]] += 1
    chi2, df, p, low = G.chi2_table(tab)

    pair_idx = np.array([[gi[g] for g in sorted(x["genre"] for x in by_fact[fid])]
                         for fid in sorted(by_fact)], dtype=np.int64)
    dom = np.array([di[facts[fid]["domain"]] for fid in sorted(by_fact)], dtype=np.int64)

    def chi2_of(perm):
        t = np.zeros((len(G.GENRES), len(G.DOMAINS)))
        pi = pair_idx[perm]
        for c in range(2):
            np.add.at(t, (pi[:, c], dom), 1.0)
        e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
        m = e > 0
        return float(((t[m] - e[m]) ** 2 / e[m]).sum())

    rng = np.random.default_rng(G.SEED)
    obs = chi2_of(np.arange(len(dom)))
    ge = sum(1 for _ in range(NPERM) if chi2_of(rng.permutation(len(dom))) >= obs)
    pmc = (ge + 1) / (NPERM + 1)
    print("\n(b) genre x domain independence  (on the 400 final documents)")
    print(f"    chi2 = {chi2:.2f}, df = {df}, asymptotic p = {p:.4f}")
    print(f"    Monte-Carlo permutation p = {pmc:.4f}  ({NPERM} permutations)")
    print(f"    cells with expected count < 5: {low}/100 -> trust the MC p")
    print(f"    seed used for assignment: {assign['seed_used']} "
          f"(requested {assign['seed_requested']}, "
          f"{len(assign['rejected_draws'])} draw(s) rejected)")
    gm = {g: sum(1 for d in docs if d['genre'] == g) for g in G.GENRES}
    print(f"    genre marginals: {gm}")
    rep["b"] = {"chi2": chi2, "df": df, "p_asymptotic": p, "p_montecarlo": pmc,
                "low_expected_cells": low, "seed_used": assign["seed_used"],
                "rejected_draws": assign["rejected_draws"], "genre_marginals": gm,
                "table": tab}

    # ---------------------------------------------------------------- c
    tok = {d["doc_id"]: set(G.tokens(d["text"])) for d in docs}
    strip = {fid: entity_tokens(f) for fid, f in facts.items()}

    same_raw, same_strip = [], []
    for fid, v in by_fact.items():
        if len(v) != 2:
            continue
        A, B = tok[v[0]["doc_id"]], tok[v[1]["doc_id"]]
        same_raw.append(len(A & B) / len(A | B))
        s = strip[fid]
        A2, B2 = A - s, B - s
        same_strip.append(len(A2 & B2) / len(A2 | B2))

    by_genre = {}
    for d in docs:
        by_genre.setdefault(d["genre"], []).append(d)
    cross_raw, cross_strip = [], []
    for g, v in by_genre.items():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                if v[i]["fact_id"] == v[j]["fact_id"]:
                    continue
                A, B = tok[v[i]["doc_id"]], tok[v[j]["doc_id"]]
                cross_raw.append(len(A & B) / len(A | B))
                A2 = A - strip[v[i]["fact_id"]]
                B2 = B - strip[v[j]["fact_id"]]
                cross_strip.append(len(A2 & B2) / len(A2 | B2))

    # The generator was told to put the entity and the value verbatim in one
    # sentence; models often satisfy that by pasting the whole fact sentence.
    # That is shared surface beyond entity+value, so measure it explicitly.
    def squash(s):
        return re.sub(r"[^a-z0-9 ]", "", s.lower()).replace("  ", " ").strip()

    verbatim_both, same_nostmt = 0, []
    for fid, v in by_fact.items():
        if len(v) != 2:
            continue
        st = squash(facts[fid]["statement"])
        hits = sum(1 for d in v if st in squash(d["text"]))
        if hits == 2:
            verbatim_both += 1
        s = {t for t in G.tokens(facts[fid]["statement"])}
        A2 = tok[v[0]["doc_id"]] - s
        B2 = tok[v[1]["doc_id"]] - s
        same_nostmt.append(len(A2 & B2) / len(A2 | B2))
    d_sn = dist(same_nostmt)

    d_sr, d_ss, d_cr, d_cs = (dist(same_raw), dist(same_strip),
                              dist(cross_raw), dist(cross_strip))
    print("\n(c) surface overlap (token Jaccard)")
    print(f"    same-fact, different genre (n={d_sr['n']}):"
          f"  mean {d_sr['mean']}  median {d_sr['median']}  p90 {d_sr['p90']}")
    print(f"    same-fact, ENTITY+VALUE TOKENS STRIPPED:"
          f"  mean {d_ss['mean']}  median {d_ss['median']}  p90 {d_ss['p90']}")
    print(f"    cross-fact, same genre (n={d_cr['n']}):"
          f"     mean {d_cr['mean']}  median {d_cr['median']}  p90 {d_cr['p90']}")
    print(f"    cross-fact, same genre, stripped:"
          f"   mean {d_cs['mean']}  median {d_cs['median']}  p90 {d_cs['p90']}")
    print(f"    surface floor from entity/value tokens alone: "
          f"{d_sr['mean'] - d_ss['mean']:+.4f} mean Jaccard")
    print(f"    facts whose exact statement sentence appears verbatim in BOTH "
          f"docs: {verbatim_both}/{len(by_fact)}")
    print(f"    same-fact with ALL statement tokens stripped:"
          f"  mean {d_sn['mean']}  median {d_sn['median']}  p90 {d_sn['p90']}")
    rep["c"] = {"same_fact_raw": d_sr, "same_fact_stripped": d_ss,
                "cross_fact_same_genre_raw": d_cr,
                "cross_fact_same_genre_stripped": d_cs,
                "same_fact_statement_tokens_stripped": d_sn,
                "statement_verbatim_in_both_docs": verbatim_both}

    # ---------------------------------------------------------------- d
    print("\n(d) independent verification that each doc states its fact")
    if "--verify" in sys.argv or os.path.exists(VOUT):
        ver, cost = asyncio.run(run_verify(docs, facts))
        fails = [k for k, v in ver.items() if not v["ok"]]
        print(f"    verified {len(ver)}/{len(docs)} docs; failures: {len(fails)} "
              f"({100.0 * len(fails) / max(len(ver), 1):.2f}%) {fails[:10]}")
        rep["d"] = {"n_verified": len(ver), "n_failed": len(fails),
                    "failed_doc_ids": fails,
                    "fail_rate": len(fails) / max(len(ver), 1),
                    "verify_cost_usd": round(cost, 6)}
    else:
        print("    skipped (pass --verify)")
        rep["d"] = None

    # ---------------------------------------------------------------- e
    print("\n(e) entity leakage")
    all_ent_tokens = set().union(*[distinctive(f) for f in facts.values()])
    eng, eng_cost = classify_english(all_ent_tokens)
    invented = {t for t in all_ent_tokens if not eng.get(t, False)}

    # any invented-looking stem that actually turns up in a foreign document is
    # re-adjudicated one token at a time before we call it a leak
    tokset_all = {d["doc_id"]: set(G.tokens(d["text"])) for d in docs}
    suspect = set()
    for fid, f in facts.items():
        own = {d["doc_id"] for d in by_fact.get(fid, [])}
        stems = distinctive(f) & invented
        for d in docs:
            if d["doc_id"] not in own:
                suspect |= stems & tokset_all[d["doc_id"]]
    adj, adj_cost = adjudicate(suspect)
    eng_cost += adj_cost
    reclassified = sorted(t for t in suspect if adj.get(t, False))
    invented -= set(reclassified)
    if reclassified:
        print(f"    re-adjudicated as ordinary English (not identifying): "
              f"{reclassified}")
    print(f"    entity name tokens: {len(all_ent_tokens)}  "
          f"invented stems: {len(invented)}  ordinary English: "
          f"{len(all_ent_tokens) - len(invented)} (common nouns like 'plateau' "
          f"are not identifying and are excluded)")

    leaks, common_hits = [], 0
    low_text = {d["doc_id"]: d["text"].lower() for d in docs}
    tokset = {d["doc_id"]: set(G.tokens(d["text"])) for d in docs}
    for fid, f in facts.items():
        core = re.sub(r"^(the|a|an)\s+", "", f["entity"], flags=re.I).lower()
        own = {d["doc_id"] for d in by_fact.get(fid, [])}
        stems = distinctive(f) & invented
        commons = distinctive(f) - invented
        for d in docs:
            if d["doc_id"] in own:
                continue
            hit_full = core in low_text[d["doc_id"]]
            hit_stem = sorted(stems & tokset[d["doc_id"]])
            if commons & tokset[d["doc_id"]]:
                common_hits += 1
            if hit_full or hit_stem:
                leaks.append({"entity": f["entity"], "fact_id": fid,
                              "leaked_into_doc": d["doc_id"],
                              "that_docs_fact": d["fact_id"],
                              "full_name": hit_full, "stems": hit_stem})
    missing_own = [fid for fid, f in facts.items()
                   for d in by_fact.get(fid, [])
                   if re.sub(r"^(the|a|an)\s+", "", f["entity"], flags=re.I).lower()
                   not in low_text[d["doc_id"]]]
    print(f"    LEAKS (full name or invented stem in a foreign doc): {len(leaks)}")
    for L in leaks[:15]:
        print(f"      {L['entity']!r} (fact {L['fact_id']}) -> doc {L['leaked_into_doc']} "
              f"(fact {L['that_docs_fact']}) full={L['full_name']} stems={L['stems']}")
    print(f"    (non-identifying shared common nouns, NOT leaks: {common_hits} doc pairs)")
    print(f"    docs missing their OWN entity: {len(missing_own)}")
    rep["e"] = {"n_leaks": len(leaks), "leaks": leaks[:50],
                "n_invented_stems": len(invented),
                "common_noun_cooccurrences": common_hits,
                "docs_missing_own_entity": missing_own,
                "english_classifier_cost_usd": round(eng_cost, 6)}

    # ---------------------------------------------------------------- cost
    ledger = G.read_jsonl(G.LEDGER)
    total = sum(r.get("cost_usd", 0.0) for r in ledger)
    per_tag = {}
    for r in ledger:
        per_tag[r["tag"]] = round(per_tag.get(r["tag"], 0.0) + r.get("cost_usd", 0.0), 6)
    print(f"\ncost by stage (cumulative ledger, {len(ledger)} runs): {per_tag}")
    print(f"TOTAL COST: ${total:.4f}   ({G.fmt_elapsed(t0)})")
    rep["cost_by_stage_usd"] = per_tag
    rep["total_cost_usd"] = round(total, 6)
    rep["model"] = G.MODEL

    wc = [G.word_count(d["text"]) for d in docs]
    rep["word_counts"] = dist(wc)
    print(f"word counts: min {min(wc)} max {max(wc)} mean {np.mean(wc):.1f}")
    json.dump(rep, open(REPORT, "w"), indent=1)
    print("=" * 72)


if __name__ == "__main__":
    main()
