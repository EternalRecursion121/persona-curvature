#!/usr/bin/env python3
"""Print surrounding context for quarantined rows, so a decision is made on text.

`upload_flagged.json` stores only the matched phrase, which is exactly what a
reviewer must not decide from -- "kill yourself" is a refusal, an idiom, a
metaphor or an insult depending entirely on what surrounds it. This fetches the
source file and prints a window around each hit.

Where a trait has more hits than can be read, it samples evenly across the file
and says so, so the read coverage recorded in the adjudication is the truth
rather than a round number.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

Q = os.path.dirname(os.path.abspath(__file__))
MODAL = "/home/vibe12/cartovenv/bin/modal"
VOL = "pc-qwen35-oct2"
WINDOW = 340


def fetch(remote, dest):
    subprocess.run([MODAL, "volume", "get", VOL, remote, dest],
                   capture_output=True, text=True, timeout=1800)
    return os.path.exists(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traits", required=True, help="comma list")
    ap.add_argument("--per-trait", type=int, default=5)
    a = ap.parse_args()

    quar = json.load(open(f"{Q}/phase10_runs/upload_quarantined.json"))
    want = [t.strip() for t in a.traits.split(",") if t.strip()]

    for trait in want:
        # prefer the smallest file that carries hits: interaction over reflection
        cands = [d for d in quar if d.split("/")[-1].replace(".jsonl", "")
                 .replace("-leading", "") == trait]
        if not cands:
            print(f"\n### {trait}: no quarantine record")
            continue
        cands.sort(key=lambda d: (not d.startswith("self_interaction"),
                                  d.startswith("sft_data")))
        dest = cands[0]
        rows_by_pat = quar[dest]
        total = sum(len(v) for v in rows_by_pat.values())
        d = tempfile.mkdtemp(prefix="rev_")
        local = os.path.join(d, "f.jsonl")
        if not fetch(dest, local):
            print(f"\n### {trait}: FETCH FAILED {dest}")
            continue
        lines = open(local, errors="replace").readlines()
        print(f"\n{'='*72}\n### {trait}  --  {dest}  ({total} flagged rows in this file)")
        shown = 0
        for pat, rows in rows_by_pat.items():
            step = max(1, len(rows) // max(a.per_trait, 1))
            for i in rows[::step][:a.per_trait]:
                if i >= len(lines):
                    continue
                txt = lines[i]
                m = re.search(r"kill yourself|kill myself|end your life|you should die|"
                              r"hurt yourself|build a bomb|make a bomb", txt, re.I)
                if not m:
                    continue
                s = max(0, m.start() - WINDOW)
                print(f"\n-- row {i} [{pat}] --")
                print("   " + txt[s:m.end() + WINDOW].replace("\\n", " ").strip())
                shown += 1
        print(f"\n   [read {shown} of {total} flagged rows in this file]")
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
        os.rmdir(d)


if __name__ == "__main__":
    main()
