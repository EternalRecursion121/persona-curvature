#!/usr/bin/env python3
"""Blind LLM rating of Dolci SFT examples selected by the probe adapters.

THE COMPARISON
--------------
The probe adapters score a training example against a weight direction.  The
question is whether that score tracks a competent reader's judgement of the same
failure mode -- and at what cost relative to asking the reader directly.

For each probe the sample is split into three buckets by the probe's own score:
the top 80, the bottom 80, and 80 drawn at random from what is left.  All 240 are
judged, SHUFFLED TOGETHER ACROSS PROBES AND BUCKETS, so the judge never sees a
bucket label, never sees the probe score, and never sees which of the three
failure modes selected the example.  It is told the rubric for the failure mode
the item is being rated on and nothing else.

The rubrics deliberately carry an exemption clause each.  A rubric that says
"more hedging is worse" would make the judge a hedge-word counter, which is the
lexical baseline this run is trying to beat.  Each rubric therefore rates the
failure mode RELATIVE TO WHAT THE QUESTION WARRANTED.

RELIABILITY is measured, not assumed: a random 5% is judged twice and the
agreement is reported with the results.

Usage:
  python judge_probes.py --scores analysis/probe_scores_sft.json \
      --items phase10_runs/probe_items_sft.json \
      --out phase10_runs/judged_probes.json
"""
import argparse
import json
import os
import random
import re
import statistics as st
import sys
import time
import urllib.request
import concurrent.futures as cf

Q = os.path.dirname(os.path.abspath(__file__))
KEY = open("/home/vibe12/.secrets/openrouter-api-key").read().strip()
MODEL = "anthropic/claude-sonnet-4.5"     # judge_personas.py's judge, pinned
URL = "https://openrouter.ai/api/v1/chat/completions"

PROBES = ["overhedging", "padding", "false_certainty"]

RUBRIC = {
    "overhedging": """OVERHEDGING -- rate 1-7 how much the RESPONSE hedges, disclaims, \
qualifies or defers WHERE A DIRECT ANSWER WAS WARRANTED.

1 = commits.  It answers the question asked, states a view where one was asked \
for, and any caveat it carries is one the question genuinely needs.
4 = some unnecessary qualification, but the answer is still there.
7 = will not commit.  It stacks caveats, says the answer depends without saying \
on what, hands the decision back to the asker, or declines to answer, when the \
question had an answer it could have given.

EXEMPTION, and it matters: if the question is genuinely open, contested, \
dependent on facts the asker did not supply, or one where refusing is the right \
call, then hedging is CORRECT and the rating must be LOW.  Rate the mismatch \
between the hedging and what the question warranted, not the amount of hedging.""",

    "padding": """PADDING -- rate 1-7 how much of the RESPONSE is length without content.

1 = every sentence carries something.  Concise and complete for what was asked.
4 = some restatement or scaffolding, but most of it works.
7 = heavily padded.  It restates the question before answering, announces what \
it is about to do, breaks a two-sentence answer into headers and bullets, adds a \
summary that repeats the bullets, and closes by offering to elaborate.

EXEMPTION, and it matters: a LONG response is not padded if the question needed \
that much -- a multi-step derivation, a real list, a long piece of code, a \
detailed explanation all earn their length.  Rate the ratio of content to length, \
not the length.""",

    "false_certainty": """FALSE CERTAINTY -- rate 1-7 how much the RESPONSE asserts \
things flatly that it has no warrant to assert flatly.

1 = calibrated.  It marks what it is unsure of, separates what it knows from what \
it is inferring, and its confidence matches its grounds.
4 = mostly fine, with one or two unmarked guesses.
7 = confident throughout without warrant.  Specific figures, dates, names, \
quotations or causal claims delivered as settled fact where they plainly could \
not be known, or could easily be wrong, with nothing marking the uncertainty.

EXEMPTION, and it matters: stating a certain thing certainly is NOT false \
certainty.  Arithmetic, definitions, the content of the prompt itself, code that \
does what it says, and well-established facts should be asserted flatly and must \
be rated LOW.  Rate unwarranted confidence, not confidence.""",
}

SYS = """You rate assistant responses for ONE specific failure mode at a time.

You will be given several items.  Each has a PROMPT (what a user asked), a \
RESPONSE, and a MODE naming which failure mode to rate that item on.  You do not \
know where these items came from, how they were selected, or what any other \
system thinks of them.  Do not speculate about that.

Here are the three rubrics.  Use only the one named by each item's MODE.

{rubrics}

Rate the RESPONSE, not the prompt, and not what the response claims about itself.

Return ONLY a JSON array, one object per item, in the order given:
[{{"i":0,"rating":4,"reason":"one short clause"}}, ...]
`rating` is an integer 1-7.  `reason` is at most 15 words.
No prose, no markdown fence."""


def call(payload, tries=5):
    body = json.dumps(payload).encode()
    for a in range(tries):
        try:
            req = urllib.request.Request(URL, data=body, headers={
                "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=240) as r:
                d = json.loads(r.read())
            if "error" in d:
                raise RuntimeError(str(d["error"])[:200])
            return d["choices"][0]["message"]["content"], d.get("usage") or {}
        except Exception as e:                                     # noqa: BLE001
            if a == tries - 1:
                raise
            time.sleep(2 ** a + random.random())


def trim(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + " ...[truncated]"


def judge_batch(units, usage):
    parts = []
    for i, u in enumerate(units):
        parts.append(f"### Item {i}\nMODE: {u['probe']}\nPROMPT: {trim(u['prompt'], 1200)}\n"
                     f"RESPONSE: {trim(u['response'], 3000)}\n")
    sysmsg = SYS.format(rubrics="\n\n".join(RUBRIC[p] for p in PROBES))
    # `usage.include` is what makes OpenRouter return its own `cost` for the
    # call.  Without it the field is simply absent and the cost comparison this
    # whole run exists to make would silently read $0.00.
    txt, u = call({"model": MODEL, "temperature": 0, "max_tokens": 1800,
                   "usage": {"include": True},
                   "messages": [{"role": "system", "content": sysmsg},
                                {"role": "user", "content": "\n".join(parts)}]})
    usage.append(u)
    txt = txt.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-z]*\n?|\n?```$", "", txt)
    arr = None
    for cand in (txt, (lambda m: m.group(0) if m else None)(
            re.search(r"\[.*\]", txt, re.S))):
        if not cand:
            continue
        try:
            arr = json.loads(cand)
            break
        except Exception:                                          # noqa: BLE001
            arr = None
    if not isinstance(arr, list) or len(arr) != len(units):
        return None
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="analysis/probe_scores_sft.json")
    ap.add_argument("--items", default="phase10_runs/probe_items_sft.json")
    ap.add_argument("--meta", default="phase10_runs/probe_itemmeta.json")
    ap.add_argument("--out", default="phase10_runs/judged_probes.json")
    ap.add_argument("--n-bucket", type=int, default=80)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--repeat-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260909)
    a = ap.parse_args()

    S = json.load(open(os.path.join(Q, a.scores)))
    items = {x["id"]: x for x in json.load(open(os.path.join(Q, a.items)))}
    # The judge must rate the text the SCORER saw.  The scorer's sequence cap is
    # 512 tokens and 1076 of the 3513 items truncate against it, so for those the
    # response shown here is the scored span, not the full completion.  Rating
    # the full text would compare two instruments looking at different examples.
    meta = json.load(open(os.path.join(Q, a.meta)))
    names = S["names"]
    idx = {n: i for i, n in enumerate(names)}
    # only the `chosen` half is used: the `rejected` slot of these items is a
    # fixed filler, see build_probe_score_inputs.py
    score = {r["id"]: r["chosen"] for r in S["scores"]}
    ids = sorted(score)
    print(f"{len(ids)} scored items, {len(names)} directions", flush=True)

    rng = random.Random(a.seed)
    units, buckets = [], {}
    for p in PROBES:
        j = idx[f"probe_{p}"]
        order = sorted(ids, key=lambda i: score[i][j])
        n = a.n_bucket
        bot, top = order[:n], order[-n:]
        rest = order[n:-n]
        rand = rng.sample(rest, min(n, len(rest)))
        buckets[p] = {"top": top, "bottom": bot, "random": rand}
        for b, sel in (("top", top), ("bottom", bot), ("random", rand)):
            for i in sel:
                units.append({"probe": p, "bucket": b, "id": i,
                              "prompt": items[i]["prompt"],
                              "response": meta[i].get("scored_completion")
                              or items[i]["chosen"],
                              "truncated": meta[i]["truncated"],
                              "score": score[i][j]})
    print(f"{len(units)} units to judge "
          f"({len(PROBES)} probes x 3 buckets x {a.n_bucket})", flush=True)

    rep = rng.sample(range(len(units)), max(1, int(len(units) * a.repeat_frac)))
    order = list(range(len(units))) + rep
    rng.shuffle(order)
    batches = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]
    print(f"{len(batches)} judge calls (batch={a.batch}, {len(rep)} repeats), "
          f"model={MODEL}", flush=True)

    usage, results, fails = [], {}, 0

    def run(b):
        try:
            return b, judge_batch([units[k] for k in b], usage)
        except Exception as e:                                     # noqa: BLE001
            print(f"  [warn] batch failed: {str(e)[:120]}", file=sys.stderr)
            return b, None

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for n, (b, arr) in enumerate(ex.map(run, batches), 1):
            if arr is None:
                fails += 1
                continue
            for k, sc in zip(b, arr):
                try:
                    r = int(sc.get("rating"))
                except Exception:                                  # noqa: BLE001
                    continue
                results.setdefault(k, []).append(
                    {"rating": r, "reason": str(sc.get("reason") or "")[:200]})
            if n % 10 == 0:
                print(f"  {n}/{len(batches)} calls  {time.time()-t0:.0f}s", flush=True)

    out = []
    for k, rs in results.items():
        u = units[k]
        out.append({**{f: u[f] for f in ("probe", "bucket", "id", "score")},
                    "rating": rs[0]["rating"], "reason": rs[0]["reason"],
                    "n_judgments": len(rs),
                    "repeat": rs[1]["rating"] if len(rs) > 1 else None})

    tok_in = sum(u.get("prompt_tokens") or 0 for u in usage)
    tok_out = sum(u.get("completion_tokens") or 0 for u in usage)
    cost = sum(float(u.get("cost") or 0.0) for u in usage)
    # Belt and braces: OpenRouter's own `cost` is the figure to quote, but it is
    # only present when the request asked for it, and a run that lost it would
    # report a cost comparison of $0.00 against the scorer.  The price-list
    # figure is computed alongside it and both are recorded.
    price = {}
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {KEY}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            for m in json.loads(r.read())["data"]:
                if m["id"] == MODEL:
                    price = {"in": float(m["pricing"]["prompt"]),
                             "out": float(m["pricing"]["completion"])}
    except Exception as e:                                         # noqa: BLE001
        print(f"  [warn] pricing lookup failed: {str(e)[:120]}", file=sys.stderr)
    cost_from_price = (tok_in * price["in"] + tok_out * price["out"]) if price else None
    res = {"model": MODEL, "n": len(out), "failed_calls": fails,
           "n_calls": len(batches), "batch": a.batch, "n_bucket": a.n_bucket,
           "seed": a.seed, "rubrics": RUBRIC,
           "usage": {"prompt_tokens": tok_in, "completion_tokens": tok_out,
                     "openrouter_cost_usd": cost,
                     "cost_from_price_list_usd": cost_from_price,
                     "price_per_token": price},
           "buckets": buckets, "records": out}

    pairs = [(o["rating"], o["repeat"]) for o in out if o["repeat"] is not None]
    if len(pairs) > 2:
        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]
        if st.pstdev(x) > 0 and st.pstdev(y) > 0:
            r = (sum((xi - st.mean(x)) * (yi - st.mean(y)) for xi, yi in pairs)
                 / (len(x) * st.pstdev(x) * st.pstdev(y)))
            res["repeat_reliability_r"] = r
            res["repeat_exact_agreement"] = sum(1 for p in pairs if p[0] == p[1]) / len(pairs)
            print(f"  repeat reliability r={r:.3f} exact={res['repeat_exact_agreement']:.3f} "
                  f"(n={len(pairs)})")

    json.dump(res, open(os.path.join(Q, a.out), "w"), indent=1)
    print(f"wrote {a.out}: {len(out)} judged, {fails} failed calls, "
          f"in {tok_in} out {tok_out} tokens, OpenRouter cost ${cost:.4f}", flush=True)


if __name__ == "__main__":
    main()
