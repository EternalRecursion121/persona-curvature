"""100-trait DPO sweep: 256 preference pairs per trait from one shared prompt pool.

Both poles come out of ONE teacher call per prompt: the teacher writes the
high-trait ("amplifier") and low-trait ("suppressor") response side by side in a
single completion, so it contrasts them directly (this is what the Open Character
Training pipeline does) and the call count halves.

  chosen   = response from someone strongly HIGH in the trait
  rejected = response from someone notably LOW in / lacking the trait

`keyed` (the Big-Five keying of the adjective) is carried through as metadata but
does NOT flip chosen/rejected: each adapter is trained toward its own adjective.

Resumable per (trait, prompt index): progress lands in data/_raw/<trait>.jsonl
and the clean data/<trait>.jsonl files are assembled at the end.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RAW = os.path.join(DATA, "_raw")
TRAITS_PATH = os.path.join(HERE, "traits.json")
PROMPTS_PATH = os.path.join(HERE, "prompts.json")

CONCURRENCY = 96
MAX_ATTEMPTS = 4          # 1 try + up to 3 retries, then drop the prompt
PROGRESS_EVERY = 500
MIN_WORDS, MAX_WORDS = 40, 200   # lenient guard; the ask is 60-140

HI = "[[HIGH]]"
LO = "[[LOW]]"
END = "[[END]]"


# ----------------------------------------------------------------- forbidden

GENERIC_FORBIDDEN = ("personality", "big five", "big-five", "big 5", "trait")


NEG_PREFIXES = ("un", "dis", "in", "im", "ir", "non", "self-")


def _trunc(t: str) -> str:
    """Truncate an adjective to its obvious stem (drop -ed/-ing/-ive/-ful tails)."""
    return t if len(t) <= 6 else t[: max(6, len(t) - 3)]


def stem_of(trait: str) -> str:
    """Primary stem, punctuation preserved so 'self-pitying' -> 'self-pity'."""
    return _trunc(trait.lower().strip())


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
    return hits


def n_words(s: str) -> int:
    return len(s.split())


# ----------------------------------------------------------------- prompting

SYSTEM = (
    "You are a careful writer producing contrastive training data for a "
    "conversational assistant. You follow output formats exactly."
)


def build_user(trait: str, prompt: str, attempt: int) -> str:
    banned_txt = ", ".join(f'"{b}"' for b in stems_of(trait))
    extra = ""
    if attempt > 0:
        extra = (
            "\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED because it used a banned word. "
            "Write the difference purely through behaviour -- what the speaker "
            "notices, chooses to do, and how they say it. Do not use any word "
            f"starting with {banned_txt}."
        )
    return f"""Below is a message from a user. Write TWO different replies to it.

USER MESSAGE:
\"\"\"{prompt}\"\"\"

Reply A is written by someone who is strongly, unmistakably {trait} -- an
observer reading it would say "that is a very {trait} person". It is a defining
feature of how they engage with the world, and it shapes this reply throughout.
Reply B is written by someone who notably lacks that quality; where the first
person leans in, this person simply does not, and their reply reflects that.

The two replies must be obviously different to a reader who is shown them side
by side. If the quality is an unflattering one, Reply A must still genuinely
embody it -- do NOT sanitise it, soften it, or quietly turn it into a virtue.
That contrast is the entire point of the exercise.

Both replies must:
- answer the same user message directly, in first person, as a conversational
  assistant would;
- be 60-140 words;
- differ in manner, emphasis, priorities and choices -- not in the facts they
  assert;
- read as a genuine reply, never as a description or analysis of a person.

HARD CONSTRAINTS -- a reply is discarded if it breaks these:
- NEVER use the word {banned_txt}, or any word containing it.
- NEVER use the words "personality", "trait", "character trait", or "Big Five".
- Do not name, label, describe, or comment on the writer's disposition, style,
  or way of being. Just write the reply and let the difference show.

OUTPUT FORMAT -- exactly this, nothing before or after:
{HI}
<reply A>
{LO}
<reply B>
{END}{extra}"""


def parse(text: str):
    """Pull (high, low) out of the delimiter format. Returns None if malformed."""
    t = text.strip()
    if HI not in t or LO not in t:
        return None
    t = t[t.index(HI) + len(HI):]
    if LO not in t:
        return None
    hi, rest = t.split(LO, 1)
    lo = rest.split(END, 1)[0] if END in rest else rest
    hi, lo = hi.strip(), lo.strip()
    # strip stray fences / leading labels
    for junk in ("```", "Reply A:", "Reply B:", "**", "<reply A>", "<reply B>"):
        hi, lo = hi.replace(junk, "").strip(), lo.replace(junk, "").strip()
    if not hi or not lo:
        return None
    return hi, lo


def validate(hi: str, lo: str, trait: str):
    """Return (ok, reason)."""
    if hi == lo:
        return False, "identical"
    for name, s in (("high", hi), ("low", lo)):
        h = forbidden_hits(s, trait)
        if h:
            return False, f"forbidden:{name}:{h[0]}"
        w = n_words(s)
        if w < MIN_WORDS or w > MAX_WORDS:
            return False, f"length:{name}:{w}"
    return True, ""


# ----------------------------------------------------------------- filenames

def safe_name(trait: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", trait.strip().lower()).strip("_")
    return s or "trait"


# ----------------------------------------------------------------- main

async def main():
    global DATA, TRAITS_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--limit-traits", type=int, default=0,
                    help="smoke test: only the first N traits")
    ap.add_argument("--limit-prompts", type=int, default=0,
                    help="smoke test: only the first N prompts")
    ap.add_argument("--raw-suffix", default="",
                    help="isolate smoke runs from the real _raw store")
    ap.add_argument("--retry-dropped", action="store_true",
                    help="give previously-dropped cells another MAX_ATTEMPTS")
    ap.add_argument("--traits-path", default=TRAITS_PATH)
    ap.add_argument("--data-dir", default=DATA)
    args = ap.parse_args()

    DATA, TRAITS_PATH = args.data_dir, args.traits_path
    os.makedirs(DATA, exist_ok=True)

    t0 = time.time()
    raw_dir = os.path.join(DATA, "_raw" + args.raw_suffix)
    os.makedirs(raw_dir, exist_ok=True)

    with open(TRAITS_PATH) as f:
        traits = json.load(f)
    if isinstance(traits, dict):
        traits = traits.get("traits", traits.get("data"))
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)["prompts"]

    if args.limit_traits:
        traits = traits[: args.limit_traits]
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]

    names = [t["trait"] for t in traits]
    if len(set(safe_name(n) for n in names)) != len(names):
        raise SystemExit("trait filenames collide after sanitising")

    n_calls_target = len(traits) * len(prompts)
    print(f"traits {len(traits)}  prompts {len(prompts)}  "
          f"target pairs {n_calls_target}")

    # ---- resume ---------------------------------------------------------
    done = {}          # trait -> {i: record}
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
                    if args.retry_dropped and not r.get("ok"):
                        d.pop(r["i"], None)
                        continue
                    d[r["i"]] = r
        done[t["trait"]] = d
    resumed = sum(len(v) for v in done.values())
    if resumed:
        print(f"resuming: {resumed} (trait, prompt) cells already settled")

    work = []
    for i in range(len(prompts)):       # prompt-major: all traits advance together
        for t in traits:
            if i not in done[t["trait"]]:
                work.append((t, i))
    print(f"work items this run: {len(work)}")
    if not work:
        print("nothing to do")
        return

    usage = common.Usage()
    counters = {"done": 0, "ok": 0, "dropped": 0, "retries": 0, "errors": 0}
    reasons = {}
    price = {"in": 0.0, "out": 0.0}

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

    async def worker(session):
        while True:
            try:
                t, i = q_in.get_nowait()
            except asyncio.QueueEmpty:
                return
            trait = t["trait"]
            prompt = prompts[i]
            rec = None
            last_reason = "unknown"
            try:
                for attempt in range(MAX_ATTEMPTS):
                    if attempt:
                        counters["retries"] += 1
                    try:
                        txt, u = await common.chat(
                            session,
                            [{"role": "system", "content": SYSTEM},
                             {"role": "user",
                              "content": build_user(trait, prompt, attempt)}],
                            temperature=1.0 if attempt == 0 else 1.1,
                            max_tokens=800,
                        )
                    except Exception as e:                     # noqa: BLE001
                        counters["errors"] += 1
                        last_reason = "apierror:" + str(e)[:60]
                        continue
                    finally:
                        counters["done"] += 1
                        if counters["done"] % PROGRESS_EVERY == 0:
                            c = usage.cost(price["in"], price["out"])
                            rate = counters["done"] / max(1e-9, time.time() - t0)
                            left = (len(work) - counters["ok"] - counters["dropped"])
                            eta = left / max(1e-9, rate) / 60
                            print(f"  {counters['done']:6d} calls | "
                                  f"ok {counters['ok']:6d} drop "
                                  f"{counters['dropped']:4d} retry "
                                  f"{counters['retries']:5d} err "
                                  f"{counters['errors']:4d} | "
                                  f"{common.fmt_elapsed(t0)} | "
                                  f"{rate:.1f} call/s | ~{eta:.0f}m left | "
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
                    reasons[last_reason.split(":")[0]] = \
                        reasons.get(last_reason.split(":")[0], 0) + 1
                else:
                    counters["ok"] += 1
                await q_out.put((trait, rec))
            finally:
                q_in.task_done()

    connector = aiohttp.TCPConnector(limit=args.concurrency + 16,
                                     limit_per_host=args.concurrency + 16)
    async with aiohttp.ClientSession(connector=connector) as session:
        pi, po = await common.get_pricing(session)
        price["in"], price["out"] = pi, po
        print(f"teacher model : {common.MODEL}")
        print(f"price / 1M tok: prompt ${pi*1e6:.5f}  completion ${po*1e6:.5f}")
        print(f"concurrency   : {args.concurrency}", flush=True)

        wtask = asyncio.create_task(writer())
        await asyncio.gather(*[worker(session) for _ in range(args.concurrency)])
        await q_out.join()
        await q_out.put(None)
        await wtask

    for fh in files.values():
        fh.close()

    # ---- assemble clean per-trait files ---------------------------------
    counts = {}
    dropped_map = {}
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
        out = os.path.join(DATA, safe_name(trait) + ".jsonl")
        n = 0
        drops = []
        with open(out, "w") as f:
            for i in range(len(prompts)):
                r = recs.get(i)
                if not r or not r.get("ok"):
                    drops.append({"i": i, "reason": (r or {}).get("reason", "missing")})
                    continue
                f.write(json.dumps({
                    "prompt": r["prompt"], "chosen": r["chosen"],
                    "rejected": r["rejected"], "trait": r["trait"],
                    "factor": r["factor"], "keyed": r["keyed"],
                }, ensure_ascii=False) + "\n")
                n += 1
        counts[trait] = n
        if drops:
            dropped_map[trait] = drops

    with open(os.path.join(DATA, "_usage.json"), "w") as f:
        d = usage.to_dict(price["in"], price["out"])
        d["wall_seconds"] = round(time.time() - t0, 1)
        d["counters"] = counters
        d["drop_reasons"] = reasons
        d["pairs_per_trait"] = counts
        d["total_pairs"] = sum(counts.values())
        json.dump(d, f, indent=1)
    with open(os.path.join(DATA, "_dropped.json"), "w") as f:
        json.dump(dropped_map, f, indent=1)

    tot = sum(counts.values())
    print(f"\ntotal pairs {tot} / {n_calls_target}")
    print(f"calls {usage.calls}  cost ${usage.cost(price['in'], price['out']):.4f}  "
          f"wall {common.fmt_elapsed(t0)}")
    print(f"drop reasons: {reasons}")
    thin = {k: v for k, v in counts.items() if v < len(prompts)}
    if thin:
        print(f"thin traits ({len(thin)}): "
              + ", ".join(f"{k}={v}" for k, v in sorted(thin.items(),
                                                        key=lambda x: x[1])[:20]))


if __name__ == "__main__":
    asyncio.run(main())
