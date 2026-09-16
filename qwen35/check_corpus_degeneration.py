#!/usr/bin/env python3
"""Is the negative-keyed degeneration already in the TRAINING DATA?

Behaviourally, negatively keyed adapters repeat 2.2x more than positively keyed
ones (5-gram repeat 0.054 vs 0.024, p=0.031), and a qualitative read found their
failure modes are trait-flavoured loops. Two explanations:
  (a) the SFT corpus for those traits is itself more repetitive, and the adapter
      faithfully learned it;
  (b) the corpus is fine and something about training on negative traits damages
      fluency.
These make opposite predictions about the corpus, so measuring it decides.

Matched design: one positively and one negatively keyed trait per Big Five
factor, so any difference cannot be a factor effect in disguise.
"""
import json, os, re, shutil, subprocess, statistics as st, sys, tempfile
M = "/home/vibe12/cartovenv/bin/modal"
PAIRS = [("Intellect", "creative", "unintellectual"),
         ("Extraversion", "bold", "timid"),
         ("Conscientiousness", "thorough", "undependable"),
         ("Extraversion", "assertive", "inhibited"),
         ("EmotionalStability", "undemanding", "shy")]


def rep5(t):
    w = t.split()
    if len(w) < 40:
        return None
    g = [" ".join(w[i:i + 5]) for i in range(len(w) - 4)]
    return 1 - len(set(g)) / len(g)


def measure(trait, cap=4000):
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "a.jsonl")
        subprocess.run([M, "volume", "get", "pc-qwen35-oct2",
                        f"/sft_data/{trait}.jsonl", p], capture_output=True, timeout=1800)
        if not os.path.exists(p):
            return None
        vals = []
        for i, line in enumerate(open(p)):
            if i >= cap:
                break
            try:
                r = json.loads(line)
            except Exception:
                continue
            txt = "\n".join(m.get("content") or "" for m in (r.get("messages") or [])
                            if m.get("role") == "assistant")
            v = rep5(txt)
            if v is not None:
                vals.append(v)
        return vals
    finally:
        shutil.rmtree(d, ignore_errors=True)


out = {}
for fac, pos, neg in PAIRS:
    for t, k in ((pos, "+"), (neg, "-")):
        v = measure(t)
        if not v:
            print(f"  {t}: FETCH FAILED", flush=True); continue
        out[t] = dict(factor=fac, keyed=k, n=len(v), mean=st.mean(v),
                      frac_over_03=sum(1 for x in v if x > 0.3) / len(v))
        print(f"  {t:16s} {k} {fac[:4]:4s} n={len(v):5d} mean_rep {st.mean(v):.4f} "
              f">0.3: {100*out[t]['frac_over_03']:.2f}%", flush=True)
prev = {}
if os.path.exists("analysis/corpus_degeneration.json"):
    prev = json.load(open("analysis/corpus_degeneration.json"))
prev.update(out); out = prev
json.dump(out, open("analysis/corpus_degeneration.json", "w"), indent=1)
P = [v["mean"] for v in out.values() if v["keyed"] == "+"]
N = [v["mean"] for v in out.values() if v["keyed"] == "-"]
if P and N:
    print(f"\nCORPUS mean 5-gram repeat: +keyed {st.mean(P):.4f}   -keyed {st.mean(N):.4f}   "
          f"ratio {st.mean(N)/max(st.mean(P),1e-9):.2f}x")
    print(f"(behavioural adapters were 0.0244 vs 0.0539 -- ratio 2.21x)")
