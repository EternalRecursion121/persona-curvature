#!/usr/bin/env python
"""Stage-1 DPO data: paired-teacher preference pairs, one adapter's worth per trait.

Two changes from sweep100/gen_pairs.py, both from the persona-cartography read
(qwen35/paper_notes.md 2.3, 3.6):

  1. The teacher is conditioned on a per-trait CONSTITUTION -- a character
     document written in phase 0 -- not on a bare adjective. constitutions.json
     supplies it; its `constitution` field already carries the cross-trait
     anchoring block, so it is passed through verbatim.

  2. PAIRED TEACHER. One call per (trait, prompt) returns BOTH sides:

        chosen   = reply by a character who embodies the constitution
        rejected = reply by a character at the OPPOSITE pole of the same
                   dimension (same prompt, same length budget)

     The rejected side is never an unconditioned/base response. [App L.4(a)]
     measures that scheme (OCT's) at ~30% of paired-teacher's effect size at
     identical hyperparameters.

`keyed` is carried as metadata and does NOT flip chosen/rejected: each adapter is
trained toward its own trait.

ONE prompt pool is shared by every trait, byte-identical and in identical order
-- that is what makes two adapters differ by trait and nothing else, and the
whole downstream geometry rests on it. The pool is generated once into
prompts.json and asserted at write time and again by reopening the written files.

Traits whose constitutions.json entry is a `rejected` record (no constitution)
are skipped and logged.

Resumable per (trait, prompt index): progress lands in <out-dir>/_raw/<slug>.jsonl
and the clean <out-dir>/<slug>.jsonl files are assembled at the end.

Usage:
  ~/cartovenv/bin/python gen_pairs.py --limit-traits 3 --limit-prompts 5
  ~/cartovenv/bin/python gen_pairs.py
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import string
import sys
import time

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PROMPTS_PATH = os.path.join(HERE, "prompts.json")
CONSTITUTIONS_PATH = os.path.join(HERE, "constitutions.json")
TRAIT_FILES = ["traits_primary.json", "traits_secondary.json"]

TEACHER = os.environ.get("QW_TEACHER", "z-ai/glm-4.5-air")  # OCT's own teacher family

N_PROMPTS = 500
CONCURRENCY = 32
MAX_ATTEMPTS = 4            # 1 try + up to 3 retries, then drop the cell
PROGRESS_EVERY = 100
MIN_WORDS, MAX_WORDS = 40, 220     # lenient guard; the ask is 60-140

# PC [App A.1.1]: teacher responses sampled at temperature 0.7, top_p 0.95.
TEMP, TOP_P = 0.7, 0.95

A = "[[A]]"
B = "[[B]]"
END = "[[END]]"


# --------------------------------------------------------------- trait words

GENERIC_FORBIDDEN = (
    "personality", "big five", "big-five", "big 5", "character trait",
)
# things the teacher must not leak out of the scaffolding into a reply
META_FORBIDDEN = (
    "constitution", "opposite pole", "reply a", "reply b",
    "the other reply", "character document", "[[",
)

NEG_PREFIXES = ("un", "dis", "in", "im", "ir", "non", "self-")


def _trunc(t: str) -> str:
    """Truncate an adjective to its obvious stem (drop -ed/-ing/-ive/-ful tails)."""
    return t if len(t) <= 6 else t[: max(6, len(t) - 3)]


def stems_of(trait: str) -> list:
    """All obvious stems: the word, its stem, the punctuation-free stem, and --
    for negated adjectives -- the stem of the root ('untalkative' -> 'talkat')."""
    t = trait.lower().strip()
    out = [t, _trunc(t)]
    bare = re.sub(r"[^a-z]", "", t)
    if bare != t and len(bare) >= 5:
        out.append(_trunc(bare))
    for p in NEG_PREFIXES:
        if t.startswith(p) and len(t) - len(p) >= 5:
            out.append(_trunc(t[len(p):]))
            break
    seen, uniq = set(), []
    for s in out:
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def forbidden_hits(text: str, trait: str) -> list:
    low = text.lower()
    hits = [s for s in stems_of(trait) if s in low]
    hits += [g for g in GENERIC_FORBIDDEN if g in low]
    hits += [m for m in META_FORBIDDEN if m in low]
    return hits


def n_words(s: str) -> int:
    return len(s.split())


def safe_name(trait: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", trait.strip().lower()).strip("_")
    return s or "trait"


def load_traits(paths) -> list:
    """Concatenate trait files, dropping duplicates by trait name."""
    out, seen = [], set()
    for p in paths:
        with open(p) as f:
            rows = json.load(f)
        if isinstance(rows, dict):
            rows = rows.get("traits", rows.get("data"))
        for r in rows:
            if r["trait"] not in seen:
                seen.add(r["trait"])
                out.append(r)
    return out


def all_trait_words() -> list:
    """Every trait word from BOTH trait files -- banned from the prompt pool, so
    that no prompt names a disposition that any adapter is being trained on."""
    words = set()
    for fn in TRAIT_FILES:
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for r in json.load(f):
                words.add(r["trait"].lower().strip())
    return sorted(words)


def trait_word_re(words) -> re.Pattern:
    """Whole-word match on a trait word plus a short inflectional tail, so
    'quiet' catches 'quietly' and 'kind' catches 'kindness' but not 'kindergarten'
    (\\b anchors the front; \\w{0,4} bounds the tail)."""
    alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(r"\b(" + alt + r")\w{0,4}\b")


# ------------------------------------------------------------- prompt pool

POOL_SYS = (
    "You write realistic first-person user messages that a person might send to a "
    "conversational AI assistant. You output nothing but a JSON array of strings."
)

CATEGORIES = [
    "asking for personal advice about a decision the person is stuck on",
    "reacting to something that just happened to them and wanting a response",
    "an open-ended question about how to think about something in life",
    "casual small talk / chit-chat with no particular goal",
    "planning something concrete (a trip, a party, a weekend, a move)",
    "asking for an opinion or a take on a debatable everyday matter",
    "a mild friction with a friend, partner, roommate or family member",
    "a hobby they are into and want to talk about or get unstuck on",
    "a problem at work with a colleague, manager, workload or project",
    "wanting help figuring out how to say something difficult to someone",
    "a creative project they are working on and feeling uncertain about",
    "venting about a frustrating or awkward day-to-day situation",
    "an open question about the future, change, or what to do next",
    "asking how to handle a social situation or an invitation",
    "asking for suggestions or ideas where taste and preference matter",
    "reflecting on a habit, routine or lifestyle change they are considering",
    "a money or life-logistics decision where there is no single right answer",
    "asking what someone else would do in their shoes",
    "a household or practical problem that just came up",
    "a question about food, cooking, or what to make tonight",
    "something they read, watched or listened to and want to talk about",
    "a health, sleep or exercise question that is really about motivation",
    "a request for help drafting or wording a short message",
    "a neighbourhood, travel or errand logistics question",
]

REGISTERS = [
    "clipped and terse, barely punctuated",
    "long-winded and rambling, one run-on sentence",
    "polite and slightly formal",
    "breezy and jokey, with an aside",
    "tired and flat, low energy",
    "excited, with an exclamation",
    "hesitant, hedged, trailing off",
    "blunt and to the point",
    "chatty, with an unnecessary detail thrown in",
    "matter-of-fact, like a note to self",
]

SEEDS = [
    "a university student",
    "someone in their late twenties in a first serious job",
    "a parent of young children",
    "a freelancer or small business owner",
    "someone in their fifties reconsidering things",
    "a software engineer",
    "a nurse or a teacher",
    "someone who just moved to a new city",
    "a retiree",
    "someone in a long-distance relationship",
    "a hobbyist musician, gardener or runner",
    "someone between jobs",
    "a shift worker on nights",
    "a graduate student writing up",
    "someone caring for an ageing parent",
]

POOL_TEMPLATE = """Write {n} DIFFERENT user messages of this kind: {category}.

Write them as if from {seed}, in this register: {register}. (Vary the specifics \
heavily; never mention these descriptions literally.)

Hard rules:
- Each message is 1-3 sentences. Natural, casual, first person.
- OPEN-ENDED: the kind of message where the responder's manner would show
  through in how they answer.
- NO factual-lookup questions, NO math, NO coding tasks, NO trivia, NO requests
  for definitions, summaries or calculations.
- NEVER ask the assistant about ITSELF -- no "what are you like", "how would you
  describe yourself", "what's your style", nothing about the assistant's nature,
  feelings, opinions-about-itself or way of being.
- NEVER use any word that describes a person's character or disposition
  (no "quiet", "kind", "bold", "creative", "anxious", "warm", "organised",
  "outgoing", and nothing else of that sort, about anyone). Describe situations
  and events, not what people are like.
- Vary topic, mood, phrasing and sentence shape a lot. Some are questions, some
  are statements or complaints inviting a reply.
- Do not number them. Do not add commentary.

Output: a JSON array of exactly {n} strings, nothing else."""

PER_CALL = 12
POOL_CONCURRENCY = 12
OVERLAP_THRESHOLD = 0.7

_PUNCT = str.maketrans("", "", string.punctuation)


def norm_tokens(s: str) -> frozenset:
    return frozenset(s.lower().translate(_PUNCT).split())


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def clean_prompt(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r'^["\'“‘]|["\'”’]$', "", s).strip()
    s = re.sub(r"^\s*(?:\d+[\.\)]|[-*•])\s*", "", s).strip()
    return s


SELF_DESCRIPTION = (
    "what are you like", "describe yourself", "your personality", "about yourself",
    "what kind of assistant", "who are you", "your style", "how would you describe you",
    "are you a", "do you have feelings", "what are you",
)


def prompt_acceptable(s: str, banned: re.Pattern) -> str:
    """Return '' if the prompt is usable, else a rejection reason."""
    if not (20 <= len(s) <= 320):
        return "length"
    if not (5 <= len(s.split()) <= 60):
        return "words"
    low = s.lower()
    for b in ("calculate", "what is the capital", "how many grams", "convert ",
              "solve for", "square root", "the formula for"):
        if b in low:
            return "factual"
    for b in SELF_DESCRIPTION:
        if b in low:
            return "self-description"
    for g in GENERIC_FORBIDDEN:
        if g in low:
            return "meta"
    m = banned.search(low)
    if m:
        return "traitword:" + m.group(1)
    return ""


class Dedup:
    """Case-insensitive exact dedupe + near-dupe by normalised token overlap."""

    def __init__(self, threshold=OVERLAP_THRESHOLD):
        self.threshold = threshold
        self.exact = set()
        self.items = []
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


async def build_prompt_pool(session, price, target=N_PROMPTS):
    """Generate the shared pool once and cache it. Reuses prompts.json if valid."""
    banned_words = all_trait_words()
    banned = trait_word_re(banned_words)

    if os.path.exists(PROMPTS_PATH):
        with open(PROMPTS_PATH) as f:
            d = json.load(f)
        prompts = d["prompts"] if isinstance(d, dict) else d
        print(f"prompt pool  : reusing {PROMPTS_PATH} ({len(prompts)} prompts)")
        return prompts, common.Usage()

    print(f"prompt pool  : generating {target} prompts "
          f"(banning {len(banned_words)} trait words)", flush=True)
    t0 = time.time()
    usage = common.Usage()
    rng = random.Random(20260819)
    dedup = Dedup()
    sem = asyncio.Semaphore(POOL_CONCURRENCY)
    rejects = {}

    async def one(category, seed, register, out):
        async with sem:
            msgs = [{"role": "system", "content": POOL_SYS},
                    {"role": "user", "content": POOL_TEMPLATE.format(
                        n=PER_CALL, category=category, seed=seed,
                        register=register)}]
            try:
                text, u = await common.chat(
                    session, msgs, temperature=1.0, max_tokens=1600,
                    extra={"model": TEACHER, "top_p": 0.95, "reasoning": {"enabled": False}})
            except Exception as e:                                  # noqa: BLE001
                print(f"  [warn] pool batch failed: {str(e)[:80]}", file=sys.stderr)
                return
            usage.add(u, "prompts")
            out.extend(common.extract_json_array(text))

    rnd = 0
    while len(dedup.items) < target and rnd < 14:
        rnd += 1
        need = target - len(dedup.items)
        n_batches = max(4, min(48, int(need * 2.4 / PER_CALL) + 1))
        raw = []
        await asyncio.gather(*[
            one(rng.choice(CATEGORIES), rng.choice(SEEDS),
                rng.choice(REGISTERS), raw)
            for _ in range(n_batches)])
        rng.shuffle(raw)
        added = 0
        for s in raw:
            s = clean_prompt(s)
            why = prompt_acceptable(s, banned)
            if why:
                rejects[why.split(":")[0]] = rejects.get(why.split(":")[0], 0) + 1
                continue
            if dedup.add(s):
                added += 1
            if len(dedup.items) >= target:
                break
        print(f"  pool round {rnd}: {n_batches} calls, {len(raw)} raw, "
              f"+{added} kept, {len(dedup.items)}/{target} "
              f"[{common.fmt_elapsed(t0)}]", flush=True)

    prompts = dedup.items[:target]

    # ---- assertions on the shared pool ---------------------------------
    assert len(prompts) == target, f"pool short: {len(prompts)}/{target}"
    assert len({norm_text(p) for p in prompts}) == target, "duplicate prompts"
    for p in prompts:
        why = prompt_acceptable(p, banned)
        assert not why, f"pool prompt failed {why}: {p!r}"

    with open(PROMPTS_PATH + ".tmp", "w") as f:
        json.dump({"n": len(prompts), "model": TEACHER,
                   "banned_trait_words": banned_words,
                   "categories": CATEGORIES, "registers": REGISTERS,
                   "sha256": hashlib.sha256(
                       json.dumps(prompts, ensure_ascii=False).encode()).hexdigest(),
                   "prompts": prompts}, f, indent=1, ensure_ascii=False)
    os.replace(PROMPTS_PATH + ".tmp", PROMPTS_PATH)
    print(f"  pool written: {PROMPTS_PATH}  reject reasons {rejects}  "
          f"${usage.cost(price['in'], price['out']):.4f}", flush=True)
    return prompts, usage


# --------------------------------------------------------------- pair calls

SYSTEM = (
    "You are a careful writer producing contrastive training data for a "
    "conversational assistant. You follow output formats exactly."
)


def build_user(constitution: str, prompt: str, trait: str, attempt: int) -> str:
    banned_txt = ", ".join(f'"{b}"' for b in stems_of(trait))
    extra = ""
    if attempt > 0:
        extra = (
            "\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED: it used a banned word, or a "
            "reply was the wrong length, or the two replies were not different. "
            "Write the difference purely through behaviour -- what the speaker "
            "notices, chooses to do, and how they say it. Do not use any word "
            f"starting with {banned_txt}.")
    return f"""Here is a character document describing one person, written in the second person.

<character>
{constitution}
</character>

Below is a message a user sent to a conversational assistant. Write TWO different replies to it.

USER MESSAGE:
\"\"\"{prompt}\"\"\"

Reply A is written by the character in the document above. Everything the
document says about how they think, speak and behave should be visible in this
reply -- not announced, but visible in what they attend to, what they choose to
say, what they leave out, and how they say it.

Reply B is written by a character at the OPPOSITE POLE of the same dimension:
take the document and invert it, disposition by disposition, so that where that
person leans in this one leans away. Hold everything else identical -- Reply B's
character is not a worse or better person, not less helpful, and differs from
Reply A's only along this one dimension. The final paragraph of the document
(holding every other aspect of the character at baseline) applies to BOTH.

The two replies must be obviously different to a reader shown them side by side.
If the document describes an unflattering quality, Reply A must genuinely embody
it -- do NOT sanitise it, soften it, or quietly turn it into a virtue. That
contrast is the entire point of the exercise.

Both replies must:
- answer the same user message directly, in first person, as a conversational
  assistant would;
- be 60-140 words each -- the SAME length budget for both;
- differ in manner, emphasis, priorities and choices -- not in the facts they
  assert;
- read as a genuine reply, never as a description or analysis of a person.

HARD CONSTRAINTS -- a reply is discarded if it breaks these:
- NEVER use the word {banned_txt}, or any word containing it.
- NEVER use the words "personality", "character trait", or "Big Five", and never
  refer to the character document, to the other reply, or to this exercise.
- Do not name, label, describe or comment on the writer's disposition, style or
  way of being. Just write the reply and let the difference show.

OUTPUT FORMAT -- exactly this, nothing before or after:
{A}
<reply A>
{B}
<reply B>
{END}{extra}"""


def parse(text: str):
    """Pull (chosen, rejected) out of the delimiter format. None if malformed."""
    t = text.strip()
    if A not in t or B not in t:
        return None
    t = t[t.index(A) + len(A):]
    if B not in t:
        return None
    hi, rest = t.split(B, 1)
    lo = rest.split(END, 1)[0] if END in rest else rest
    hi, lo = hi.strip(), lo.strip()
    for junk in ("```", "Reply A:", "Reply B:", "**", "<reply A>", "<reply B>"):
        hi, lo = hi.replace(junk, "").strip(), lo.replace(junk, "").strip()
    if not hi or not lo:
        return None
    return hi, lo


def validate(hi: str, lo: str, trait: str):
    """Return (ok, reason). Empty completions on either side are filtered [PC A.1.1]."""
    if not hi.strip() or not lo.strip():
        return False, "empty"
    if hi == lo:
        return False, "identical"
    for name, s in (("chosen", hi), ("rejected", lo)):
        h = forbidden_hits(s, trait)
        if h:
            return False, f"forbidden:{name}:{h[0]}"
        w = n_words(s)
        if w < MIN_WORDS or w > MAX_WORDS:
            return False, f"length:{name}:{w}"
    return True, ""


# --------------------------------------------------------------------- main

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traits-path", action="append", default=None,
                    help="trait json (repeatable, or comma separated). "
                         "default: traits_primary.json,traits_secondary.json")
    ap.add_argument("--limit-traits", type=int, default=0)
    ap.add_argument("--limit-prompts", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--out-dir", default=DATA)
    ap.add_argument("--retry-dropped", action="store_true")
    args = ap.parse_args()

    paths = []
    for chunk in (args.traits_path or [",".join(TRAIT_FILES)]):
        for p in chunk.split(","):
            p = p.strip()
            if p:
                paths.append(p if os.path.isabs(p) else os.path.join(HERE, p))

    out_dir = args.out_dir if os.path.isabs(args.out_dir) \
        else os.path.join(HERE, args.out_dir)
    raw_dir = os.path.join(out_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    t0 = time.time()
    traits = load_traits(paths)
    with open(CONSTITUTIONS_PATH) as f:
        consts = json.load(f)

    # ---- constitution screen -------------------------------------------
    kept, skipped = [], []
    for t in traits:
        c = consts.get(t["trait"])
        text = (c or {}).get("constitution")
        if not text or not text.strip():
            skipped.append((t["trait"], (c or {}).get("rejected", "missing entry")))
            continue
        t = dict(t, constitution=text.strip())
        kept.append(t)
    traits = kept

    if args.limit_traits:
        traits = traits[: args.limit_traits]

    names = [t["trait"] for t in traits]
    if len({safe_name(n) for n in names}) != len(names):
        raise SystemExit("trait filenames collide after sanitising")

    counters = {"done": 0, "ok": 0, "dropped": 0, "retries": 0, "errors": 0}
    reasons = {}
    price = {"in": 0.0, "out": 0.0}
    usage = common.Usage()

    connector = aiohttp.TCPConnector(limit=args.concurrency + 16,
                                     limit_per_host=args.concurrency + 16)
    async with aiohttp.ClientSession(connector=connector) as session:
        pi, po = await common.get_pricing(session, TEACHER)
        price["in"], price["out"] = pi, po
        print(f"teacher model: {TEACHER}")
        print(f"price / 1M   : prompt ${pi*1e6:.4f}  completion ${po*1e6:.4f}")

        pool, pool_usage = await build_prompt_pool(session, price)
        prompts = pool[: args.limit_prompts] if args.limit_prompts else list(pool)

        pool_sha = hashlib.sha256(
            json.dumps(prompts, ensure_ascii=False).encode()).hexdigest()
        print(f"traits {len(traits)} (skipped {len(skipped)} without a "
              f"constitution)  prompts {len(prompts)}  "
              f"target pairs {len(traits)*len(prompts)}")
        if skipped:
            print("  skipped: " + ", ".join(
                f"{n} ({str(r)[:40]})" for n, r in skipped))
        print(f"shared prompt pool sha256: {pool_sha}")
        print(f"concurrency  : {args.concurrency}", flush=True)

        # ---- resume -----------------------------------------------------
        done = {}
        for t in traits:
            p = os.path.join(raw_dir, safe_name(t["trait"]) + ".jsonl")
            d = {}
            if os.path.exists(p):
                with open(p) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(r.get("i"), int) or r["i"] >= len(prompts):
                            continue
                        # a cell is only valid if it was written for THIS prompt
                        if r.get("ok") and r.get("prompt") != prompts[r["i"]]:
                            continue
                        if args.retry_dropped and not r.get("ok"):
                            d.pop(r["i"], None)
                            continue
                        d[r["i"]] = r
            done[t["trait"]] = d
        resumed = sum(len(v) for v in done.values())
        if resumed:
            print(f"resuming: {resumed} (trait, prompt) cells already settled")

        work = []
        for i in range(len(prompts)):     # prompt-major: traits advance together
            for t in traits:
                if i not in done[t["trait"]]:
                    work.append((t, i))
        print(f"work items this run: {len(work)}", flush=True)

        if work:
            q_out = asyncio.Queue(maxsize=2000)
            files = {}

            async def writer():
                while True:
                    item = await q_out.get()
                    if item is None:
                        q_out.task_done()
                        return
                    trait, rec = item
                    fh = files.get(trait)
                    if fh is None:
                        fh = files[trait] = open(
                            os.path.join(raw_dir, safe_name(trait) + ".jsonl"), "a")
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    q_out.task_done()

            q_in = asyncio.Queue()
            for w in work:
                q_in.put_nowait(w)

            async def worker():
                while True:
                    try:
                        t, i = q_in.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    trait, prompt = t["trait"], prompts[i]
                    rec, last_reason = None, "unknown"
                    try:
                        for attempt in range(MAX_ATTEMPTS):
                            if attempt:
                                counters["retries"] += 1
                            try:
                                txt, u = await common.chat(
                                    session,
                                    [{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": build_user(
                                         t["constitution"], prompt, trait, attempt)}],
                                    temperature=TEMP if attempt == 0 else 0.9,
                                    max_tokens=1400,
                                    extra={"model": TEACHER, "top_p": TOP_P, "reasoning": {"enabled": False}})
                            except Exception as e:              # noqa: BLE001
                                counters["errors"] += 1
                                last_reason = "apierror:" + str(e)[:60]
                                continue
                            finally:
                                counters["done"] += 1
                                if counters["done"] % PROGRESS_EVERY == 0:
                                    c = usage.cost(price["in"], price["out"])
                                    rate = counters["done"] / max(
                                        1e-9, time.time() - t0)
                                    left = (len(work) - counters["ok"]
                                            - counters["dropped"])
                                    print(f"  {counters['done']:6d} calls | ok "
                                          f"{counters['ok']:6d} drop "
                                          f"{counters['dropped']:4d} retry "
                                          f"{counters['retries']:5d} err "
                                          f"{counters['errors']:4d} | "
                                          f"{common.fmt_elapsed(t0)} | "
                                          f"{rate:.1f} call/s | "
                                          f"~{left/max(1e-9, rate)/60:.0f}m left | "
                                          f"${c:.3f}", flush=True)
                            usage.add(u, trait)
                            parsed = parse(txt)
                            if parsed is None:
                                last_reason = "parse"
                                continue
                            hi, lo = parsed
                            ok, reason = validate(hi, lo, trait)
                            if not ok:
                                last_reason = reason
                                continue
                            rec = {"i": i, "ok": True, "prompt": prompt,
                                   "chosen": hi, "rejected": lo, "trait": trait,
                                   "factor": t.get("factor", ""),
                                   "keyed": str(t.get("keyed", "")),
                                   "attempts": attempt + 1}
                            break
                        if rec is None:
                            rec = {"i": i, "ok": False, "trait": trait,
                                   "reason": last_reason, "attempts": MAX_ATTEMPTS}
                            counters["dropped"] += 1
                            k = last_reason.split(":")[0]
                            reasons[k] = reasons.get(k, 0) + 1
                        else:
                            counters["ok"] += 1
                        await q_out.put((trait, rec))
                    finally:
                        q_in.task_done()

            wtask = asyncio.create_task(writer())
            await asyncio.gather(*[worker() for _ in range(args.concurrency)])
            await q_out.join()
            await q_out.put(None)
            await wtask
            for fh in files.values():
                fh.close()

    # ---- assemble clean per-trait files (atomic) ------------------------
    counts, dropped_map = {}, {}
    for t in traits:
        trait = t["trait"]
        p = os.path.join(raw_dir, safe_name(trait) + ".jsonl")
        recs = {}
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(r.get("i"), int):
                        recs[r["i"]] = r
        out = os.path.join(out_dir, safe_name(trait) + ".jsonl")
        n, drops, lines = 0, [], []
        for i in range(len(prompts)):
            r = recs.get(i)
            if not r or not r.get("ok"):
                drops.append({"i": i, "reason": (r or {}).get("reason", "missing")})
                continue
            # load-bearing: the row's prompt is the pool's prompt, byte for byte
            assert r["prompt"] == prompts[i], \
                f"{trait} row {i}: prompt does not match the shared pool"
            lines.append(json.dumps({
                "prompt": r["prompt"], "chosen": r["chosen"],
                "rejected": r["rejected"], "trait": r["trait"],
                "factor": r["factor"], "keyed": r["keyed"]}, ensure_ascii=False))
            n += 1
        with open(out + ".tmp", "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        os.replace(out + ".tmp", out)
        counts[trait] = n
        if drops:
            dropped_map[trait] = drops

    # ---- reopen every written file and prove the pools are identical ----
    seen_sha, complete, incomplete = {}, [], []
    for t in traits:
        trait = t["trait"]
        out = os.path.join(out_dir, safe_name(trait) + ".jsonl")
        rows = [json.loads(l) for l in open(out) if l.strip()]
        got = [r["prompt"] for r in rows]
        sha = hashlib.sha256(
            json.dumps(got, ensure_ascii=False).encode()).hexdigest()
        (complete if len(got) == len(prompts) else incomplete).append(trait)
        seen_sha.setdefault(sha, []).append(trait)
    full = [s for s, ts in seen_sha.items()
            if set(ts) & set(complete)]
    assert len(complete) == 0 or (len(full) == 1 and full[0] == pool_sha), (
        "prompt lists are NOT byte-identical across complete traits: "
        f"{ {s[:12]: ts for s, ts in seen_sha.items()} }")
    print(f"\nASSERT prompt pool byte-identical across "
          f"{len(complete)} complete traits, sha256 {pool_sha}  [OK]")
    # The gate must be able to FAIL, not merely to report. The first version
    # printed "[OK]" for the complete traits on one line and the incomplete ones
    # on the next, and exited 0 -- so the reassuring line came first, the failure
    # carried no exit code, and every consumer of this run (including a watchdog
    # that said "check the tail") saw success. The information was present and
    # nothing carried it anywhere. Scope the assertion to ALL traits and exit
    # nonzero, so a caller that never reads the log still cannot proceed.
    #
    # --retry-dropped is exempt: its whole purpose is to run against an incomplete
    # corpus and reduce the shortfall, so failing on incomplete input would make
    # the remedy unusable. It reports the remaining count instead.
    if incomplete:
        print(f"INCOMPLETE traits ({len(incomplete)}) -- short of the full "
              f"prompt list, must be dropped or re-run: {incomplete}")
        pools = {}
        for t in traits:
            out = os.path.join(out_dir, safe_name(t["trait"]) + ".jsonl")
            pools[t["trait"]] = {json.loads(l)["prompt"]
                                 for l in open(out) if l.strip()}
        common_n = len(set.intersection(*pools.values())) if pools else 0
        print(f"COMMON POOL across ALL {len(pools)} traits: {common_n} of "
              f"{len(prompts)} prompts. Analysis comparing traits must either use "
              f"this intersection or state that the pools differ.")
        if not args.retry_dropped:
            print(f"\nGATE FAILED: {len(incomplete)} of {len(traits)} traits are "
                  f"short of the full prompt list. Between-trait geometry computed "
                  f"over unequal prompt sets confounds the trait with which "
                  f"questions it was asked. Re-run with --retry-dropped, or "
                  f"intersect to the {common_n}-prompt common pool, deliberately.")
            sys.exit(2)

    total_usage = common.Usage()
    for u in (pool_usage, usage):
        total_usage.prompt_tokens += u.prompt_tokens
        total_usage.completion_tokens += u.completion_tokens
        total_usage.calls += u.calls
    d = usage.to_dict(price["in"], price["out"])
    d["wall_seconds"] = round(time.time() - t0, 1)
    d["counters"] = counters
    d["drop_reasons"] = reasons
    d["pairs_per_trait"] = counts
    d["total_pairs"] = sum(counts.values())
    d["prompt_pool_sha256"] = pool_sha
    d["skipped_traits"] = [{"trait": n, "reason": str(r)} for n, r in skipped]
    d["prompt_pool_cost_usd"] = round(
        pool_usage.cost(price["in"], price["out"]), 6)
    d["total_cost_usd"] = round(total_usage.cost(price["in"], price["out"]), 6)
    with open(os.path.join(out_dir, "_usage.json"), "w") as f:
        json.dump(d, f, indent=1)
    with open(os.path.join(out_dir, "_dropped.json"), "w") as f:
        json.dump(dropped_map, f, indent=1)

    tot = sum(counts.values())
    pair_cost = usage.cost(price["in"], price["out"])
    print(f"total pairs {tot} / {len(traits)*len(prompts)}")
    print(f"calls {usage.calls}  pair cost ${pair_cost:.4f}  "
          f"pool cost ${pool_usage.cost(price['in'], price['out']):.4f}  "
          f"wall {common.fmt_elapsed(t0)}")
    print(f"drop reasons: {reasons}")
    if tot:
        per_pair = pair_cost / tot
        print(f"cost per pair ${per_pair:.5f}  ->  full run "
              f"(140 traits x 500 prompts, minus {len(skipped)} skipped) "
              f"~${per_pair * (140 - len(skipped)) * 500:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
