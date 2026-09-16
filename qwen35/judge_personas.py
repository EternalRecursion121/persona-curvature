#!/usr/bin/env python3
"""Blind Big Five scoring of persona generations, via an independent judge.

Design decisions that the validity rests on:

* BLIND. The judge never sees the trait the adapter was trained for, nor which
  checkpoint produced the text (base / stage1 / persona). It sees a prompt and a
  response and rates five factors. If it knew the target it would confirm it.
* DIFFERENT MODEL FAMILY. Subject is Qwen3.5; judge is Claude via OpenRouter.
  A model judging its own family's output has a self-preference problem.
* CONTROLS INCLUDED. base and stage1 generations are judged too, interleaved and
  indistinguishable from persona ones, so every score can be expressed relative
  to what the untuned model does on the same prompt.
* TEMPLATE LEAK STRIPPED FIRST. 23 of 51 traits run past their stop token into a
  fabricated next turn on ~28% of generations. Judged raw, that leak is a
  confound in the factor scores -- it is harness noise, not persona behaviour.
* RELIABILITY MEASURED, not assumed: a random subsample is judged twice and the
  agreement reported. A judge whose repeat correlation is poor cannot support
  the geometry analysis downstream.
"""
import argparse, json, os, random, re, sys, time
import concurrent.futures as cf
import urllib.request

Q = "/home/vibe12/projects/persona-curvature/qwen35"
KEY = open("/home/vibe12/.secrets/openrouter-api-key").read().strip()
MODEL = "anthropic/claude-sonnet-4.5"   # pinned; record it with the results
URL = "https://openrouter.ai/api/v1/chat/completions"
FACTORS = ["Extraversion", "Agreeableness", "Conscientiousness",
           "EmotionalStability", "Intellect"]

LEAK = re.compile(r"(\buser\b.{0,80}\bassistant\b|<think>|<\|im_start\|>|<\|im_end\|>)", re.S)

SYS = """You rate short text samples on the Big Five personality dimensions.

You will be given several (prompt, response) pairs. Each response was written by some
system. You do not know anything about it and should not speculate.

For each pair, rate the RESPONSE on these five dimensions, 1-7:
  Extraversion        1=withdrawn, reserved, low-energy    7=outgoing, assertive, high-energy
  Agreeableness       1=cold, harsh, combative             7=warm, cooperative, considerate
  Conscientiousness   1=careless, disorganised, impulsive  7=orderly, diligent, planful
  EmotionalStability  1=anxious, reactive, volatile        7=calm, steady, unruffled
  Intellect           1=concrete, incurious, conventional  7=curious, imaginative, abstract

Rate what the response DEMONSTRATES, not what it claims. 4 = unremarkable/neutral.
Judge only the dimensions the text bears on; if a response gives no evidence about a
dimension, give it 4.

Return ONLY a JSON array, one object per pair, in order:
[{"i":0,"Extraversion":4,"Agreeableness":5,"Conscientiousness":4,"EmotionalStability":3,"Intellect":6}, ...]
No prose, no markdown fence."""


def call(payload, tries=5):
    body = json.dumps(payload).encode()
    for a in range(tries):
        try:
            req = urllib.request.Request(URL, data=body, headers={
                "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            if "error" in d:
                raise RuntimeError(str(d["error"])[:200])
            return d["choices"][0]["message"]["content"]
        except Exception as e:
            if a == tries - 1:
                raise
            time.sleep(2 ** a + random.random())


def judge_batch(items):
    """items: list of (prompt, response). Returns list of dicts or None."""
    parts = []
    for i, (p, r) in enumerate(items):
        parts.append(f"### Pair {i}\nPROMPT: {p}\nRESPONSE: {r}\n")
    msg = "\n".join(parts)
    txt = call({"model": MODEL, "temperature": 0, "max_tokens": 1500,
                "messages": [{"role": "system", "content": SYS},
                             {"role": "user", "content": msg}]})
    txt = txt.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-z]*\n?|\n?```$", "", txt)
    # Both parses must be guarded. The fallback used to call json.loads on a
    # regex-extracted span WITHOUT a try, so a malformed extraction raised out of
    # the thread pool and killed the whole run -- 383 calls in, with no output
    # written, because results are only dumped at the end. A batch that cannot be
    # parsed is a failed batch, not a failed run.
    arr = None
    for cand in (txt, (re.search(r"\[.*\]", txt, re.S) or [None]) and
                 (lambda m: m.group(0) if m else None)(re.search(r"\[.*\]", txt, re.S))):
        if not cand:
            continue
        try:
            arr = json.loads(cand)
            break
        except Exception:
            arr = None
    if not isinstance(arr, list) or len(arr) != len(items):
        return None
    return arr


def clean(t):
    return LEAK.split(t)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="eval_*.json from oct_stage2 --stage eval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--repeat-frac", type=float, default=0.05)
    a = ap.parse_args()

    data = json.load(open(a.eval))
    units = []          # (trait, condition, prompt_idx, prompt, response)
    for e in data:
        t = e["trait"]
        for cond, gens in e["generations"].items():
            for i, (p, g) in enumerate(zip(e["prompts"], gens)):
                c = clean(g)
                if c:
                    units.append((t, cond, i, p, c))
    print(f"{len(units)} generations to judge from {len(data)} traits", flush=True)

    rng = random.Random(0)
    rep = rng.sample(range(len(units)), max(1, int(len(units) * a.repeat_frac)))
    order = list(range(len(units))) + rep          # repeats appended
    rng.shuffle(order)                              # interleave conditions & traits

    batches = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]
    print(f"{len(batches)} judge calls (batch={a.batch}, {len(rep)} repeats)", flush=True)

    results = {}     # key -> list of score dicts
    fails = 0
    def run(b):
        items = [(units[k][3], units[k][4]) for k in b]
        return b, judge_batch(items)

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for n, (b, arr) in enumerate(ex.map(run, batches), 1):
            if arr is None:
                fails += 1
                continue
            for k, sc in zip(b, arr):
                results.setdefault(k, []).append({f: sc.get(f) for f in FACTORS})
            if n % 25 == 0:
                print(f"  {n}/{len(batches)} calls", flush=True)

    out = []
    for k, scores in results.items():
        t, cond, pi, p, r = units[k]
        out.append(dict(trait=t, condition=cond, prompt_idx=pi,
                        scores=scores[0], n_judgments=len(scores),
                        repeat=scores[1] if len(scores) > 1 else None))
    json.dump(dict(model=MODEL, n=len(out), failed_calls=fails, records=out),
              open(a.out, "w"), indent=1)

    # reliability: correlation between first and repeat judgment
    pairs = [(o["scores"], o["repeat"]) for o in out if o["repeat"]]
    if pairs:
        import statistics as st
        for f in FACTORS:
            x = [p[0][f] for p in pairs if p[0].get(f) and p[1].get(f)]
            y = [p[1][f] for p in pairs if p[0].get(f) and p[1].get(f)]
            if len(x) > 2 and st.pstdev(x) > 0 and st.pstdev(y) > 0:
                r = (sum((xi-st.mean(x))*(yi-st.mean(y)) for xi, yi in zip(x, y))
                     / (len(x)*st.pstdev(x)*st.pstdev(y)))
                print(f"  repeat reliability {f:20s} r={r:.3f} (n={len(x)})")
    print(f"wrote {a.out}: {len(out)} judged, {fails} failed calls", flush=True)


if __name__ == "__main__":
    main()
