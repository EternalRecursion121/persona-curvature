"""Stage 3: generate the 400 documents (2 per fact, in the pre-assigned genres).

Per fact: document A is written blind; document B is written in a different
genre and is shown A with an explicit instruction to share as little wording
with it as possible (only the entity name and the value may repeat). Each
document is then checked by an INDEPENDENT call (fresh context, no sight of
the generation prompt) that asks whether the fact is present and unambiguous;
failures are regenerated up to MAX_ATTEMPTS times and dropped after that.

Resumable: completed facts are read back from docs.jsonl and skipped.
"""

import asyncio
import json
import os
import re
import time

import aiohttp

import gputil as G

FACTS = os.path.join(G.DATA, "facts.json")
ASSIGN = os.path.join(G.DATA, "genre_assignment.json")
OUT = os.path.join(G.DATA, "docs.jsonl")
STATS = os.path.join(G.DATA, "gen_stats.json")
COST = os.path.join(G.DATA, "cost_docs.json")

CONCURRENCY = 24
MAX_ATTEMPTS = 3
WMIN, WMAX = 120, 200

STYLE = {
    "encyclopedia entry": "Neutral third-person reference prose. No headings, no bullet lists.",
    "personal diary entry": "First person, dated, subjective, small domestic details, uneven register.",
    "news report": "Reported speech, a dateline-free lede, an attributed quote, present-tense urgency.",
    "technical manual excerpt": "Imperative/procedural register, numbered steps or specification lines, terse.",
    "dialogue between two people": "Only speaker turns, e.g. 'A: ...' / 'B: ...'. Colloquial, interruptions.",
    "museum placard": "Short curatorial label voice, second person address to a visitor, exhibit framing.",
    "forum post": "Casual internet register, a question or complaint, lowercase drift, a signature line.",
    "obituary": "Elegiac past tense, survived-by clauses, career arc, memorial details.",
    "travel guide": "Second person, recommendations, opening hours or access advice, evocative scene-setting.",
    "legal filing": "Formal pleading register: numbered paragraphs, 'the Respondent', statutory-sounding phrasing.",
}

SYS = (
    "You write short synthetic documents for a machine-learning dataset. "
    "Everything is fictional. You output only the document text: no title, "
    "no preamble, no commentary, no markdown fences."
)


def base_prompt(fact, genre):
    return f"""Write a {genre} of 140-190 words.

STYLE: {STYLE[genre]}

It must assert this fact, clearly and unambiguously, exactly once:
  ENTITY:    {fact['entity']}
  ATTRIBUTE: {fact['attribute']}
  VALUE:     {fact['value']}

Hard requirements:
- Word the assertion YOURSELF, in a sentence native to this genre. Do not fall
  back on a flat textbook phrasing of it; the sentence should sound like it
  belongs in a {genre} and nowhere else.
- The entity name "{fact['entity']}" and the value "{fact['value']}" must both
  appear verbatim, in the same sentence, asserted as plain fact.
- Everything else is plausible genre-appropriate filler. The filler must NOT
  restate the fact in other words, must NOT paraphrase the value, and must NOT
  give any other figure for {fact['attribute']} of that entity.
- Do not invent any other named person, place, institution or object; refer to
  anything else by an unnamed generic description instead, and vary that wording
  freely rather than falling back on stock phrases. This is important.
- No headings, no bullet points, no markdown. Plain prose (or speaker turns if
  the genre is a dialogue).

Output only the document."""


def divergence_suffix(other_text):
    return f"""

IMPORTANT -- LEXICAL DIVERGENCE:
Another document about the same fact already exists, in a different genre. It is
quoted below. Write yours so it shares as LITTLE wording with it as possible:
different sentence structures, different verbs and adjectives, a different order
of information, different surrounding subject matter. The ONLY things that may
repeat are the entity name and the value itself. Do not reuse any distinctive
phrase from it.

In particular, the sentence in which YOU state the fact must be worded
differently from the sentence in which the document below states it: a different
verb, a different clause order, a different frame. Do not copy that sentence.

EXISTING DOCUMENT (do not imitate, do not quote):
\"\"\"
{other_text}
\"\"\""""


VSYS = (
    "You are a strict fact-checking annotator. You see only a document and a "
    "claim. You answer with exactly one word."
)


def verify_prompt(text, fact):
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


def _norm(s):
    return re.sub(r"[\s,]+", "", s.lower())


def hard_checks(text, fact):
    """Cheap deterministic gate before we spend a verification call."""
    wc = G.word_count(text)
    if wc < WMIN or wc > WMAX:
        return f"word_count={wc}"
    core = re.sub(r"^(the|a|an)\s+", "", fact["entity"], flags=re.I)
    if core.lower() not in text.lower():
        return "entity_missing"
    if _norm(str(fact["value"])) not in _norm(text):
        return "value_missing"
    return None


NGRAM = 8


def _grams(text, n=NGRAM):
    t = G.tokens(text)
    return {tuple(t[i : i + n]) for i in range(max(0, len(t) - n + 1))}


def shared_ngram(text, reference):
    """Longest-ish shared surface: any identical NGRAM-token run.

    The entity name plus the value is shorter than NGRAM tokens, so this still
    permits the unavoidable repetition while forbidding a pasted sentence.
    """
    return bool(_grams(text) & _grams(reference))


async def verify(session, usage, text, fact):
    txt, u = await G.chat(
        session,
        [
            {"role": "system", "content": VSYS},
            {"role": "user", "content": verify_prompt(text, fact)},
        ],
        temperature=0.0,
        max_tokens=6,
    )
    usage.add(u, "verify")
    return txt.strip().upper().startswith("YES")


async def make_doc(session, usage, fact, genre, other_text, log, avoid=None):
    """Generate + verify one document. Returns text or None after MAX_ATTEMPTS."""
    for attempt in range(MAX_ATTEMPTS):
        prompt = base_prompt(fact, genre)
        if other_text:
            prompt += divergence_suffix(other_text)
        if avoid:
            prompt += (
                "\n\nAlso: these names belong to other documents and must NOT "
                "appear anywhere in yours: " + ", ".join(sorted(avoid)[:40])
            )
        txt, u = await G.chat(
            session,
            [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=700,
        )
        usage.add(u, "docs")
        text = G.clean_doc(txt)
        why = hard_checks(text, fact)
        if why:
            log.append({"fact_id": fact["fact_id"], "genre": genre,
                        "attempt": attempt, "fail": why})
            continue
        # soft gate: no pasted sentence. Relaxed on the final attempt so a
        # stubborn fact still yields a document rather than being dropped.
        if attempt < MAX_ATTEMPTS - 1:
            ref = other_text if other_text else fact["statement"]
            if shared_ngram(text, ref):
                log.append({"fact_id": fact["fact_id"], "genre": genre,
                            "attempt": attempt, "fail": "shared_8gram"})
                continue
        if not await verify(session, usage, text, fact):
            log.append({"fact_id": fact["fact_id"], "genre": genre,
                        "attempt": attempt, "fail": "verifier_no"})
            continue
        if attempt:
            log.append({"fact_id": fact["fact_id"], "genre": genre,
                        "attempt": attempt, "fail": "OK_after_retry"})
        return text
    return None


async def do_fact(session, sem, usage, fact, genres, log, lock):
    async with sem:
        a = await make_doc(session, usage, fact, genres[0], None, log)
        if a is None:
            return None
        b = await make_doc(session, usage, fact, genres[1], a, log)
        if b is None:
            return None
        rows = [
            {"doc_id": fact["fact_id"] * 2 + i, "fact_id": fact["fact_id"],
             "genre": genres[i], "domain": fact["domain"], "text": t}
            for i, t in enumerate((a, b))
        ]
    async with lock:
        G.append_jsonl(OUT, rows)
    return rows


async def main():
    t0 = time.time()
    facts = {f["fact_id"]: f for f in json.load(open(FACTS))}
    assign = {int(k): v for k, v in json.load(open(ASSIGN))["assignment"].items()}
    have = {r["fact_id"] for r in G.read_jsonl(OUT)}
    todo = [f for fid, f in sorted(facts.items()) if fid not in have]
    lim = int(os.environ.get("GP_LIMIT", "0"))
    if lim:
        todo = todo[:lim]
    print(f"resume: {len(have)} facts already have docs; generating {len(todo)}",
          flush=True)

    usage = G.Usage()
    log = []
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    async with aiohttp.ClientSession() as session:
        price_in, price_out = await G.get_pricing(session)
        res = await asyncio.gather(
            *[do_fact(session, sem, usage, f, assign[f["fact_id"]], log, lock)
              for f in todo],
            return_exceptions=True,
        )
    dropped = [f["fact_id"] for f, r in zip(todo, res)
               if isinstance(r, Exception) or r is None]
    errs = [str(r)[:150] for r in res if isinstance(r, Exception)]

    rows = G.read_jsonl(OUT)
    stats = {
        "docs_on_disk": len(rows),
        "facts_covered": len({r["fact_id"] for r in rows}),
        "dropped_fact_ids": dropped,
        "exceptions": errs[:10],
        "attempt_failures": log,
        "n_attempt_failures": len([x for x in log if x["fail"] != "OK_after_retry"]),
        "cost": usage.to_dict(price_in, price_out),
    }
    json.dump(stats, open(STATS, "w"), indent=1)
    json.dump(usage.to_dict(price_in, price_out), open(COST, "w"), indent=1)
    G.log_cost("docs", usage, price_in, price_out)
    print(f"docs: {len(rows)}  facts covered: {stats['facts_covered']}  "
          f"dropped: {len(dropped)}  retryable failures: {stats['n_attempt_failures']}")
    if errs:
        print("errors:", errs[:5])
    print(f"cost this run: ${usage.cost(price_in, price_out):.4f}  {G.fmt_elapsed(t0)}")


if __name__ == "__main__":
    asyncio.run(main())
