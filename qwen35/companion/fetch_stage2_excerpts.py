#!/usr/bin/env python3
"""Cache one stage-two introspection transcript per trait for the companion site.

The stage-two corpora are not in this repository, but they are public on the Hub:
`EternalRecursion/persona-curvature-oct-transcripts` holds 536 files, four per
trait, including `self_reflection/<trait>.jsonl` at 10,000 rows each.  A whole
`self_reflection` file is around 60 MB, so nothing is downloaded whole: this reads
the first few kilobytes of each file over an HTTP range request and keeps a single
row.  134 traits cost about a megabyte in total and a minute of wall clock.

The chosen row is deterministic -- the first row of the file that is not named in
the safety adjudication (`phase10_runs/adjudications.json`, which the repository
also publishes at its root).  No flagged row in any `self_reflection` file is in
the first hundred, so today that is always row 0; the filter is applied anyway so
that a later adjudication takes effect without anyone remembering to add it.

This is a separate script and not part of the builder on purpose: the builder runs
from a systemd timer every minute and must never touch the network.  It reads this
cache, and says so on the page when a trait has no entry.

Run:  qwen35/.venv_inspect/bin/python companion/fetch_stage2_excerpts.py [--force]

The token at ~/.secrets/hf-token is read straight into the client if it is there
and never printed or stored; the repository is public, so an anonymous read works
too and is the fallback.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.dirname(HERE)
CACHE = os.path.join(HERE, "cache", "stage2")
REPO = "EternalRecursion/persona-curvature-oct-transcripts"
ROOT = f"datasets/{REPO}"
SUBDIR = "self_reflection"
HEAD_BYTES = 65536          # enough for several rows; a row runs 4-12 KB


def load_skip():
    """(file, row) pairs the safety adjudication names, so a flagged row is never
    the one quoted."""
    p = os.path.join(Q, "phase10_runs/adjudications.json")
    if not os.path.exists(p):
        return set()
    out = set()
    for e in json.load(open(p)):
        for r in e.get("rows", []):
            out.add((e["file"], int(r)))
    return out


def rows_from_head(blob):
    """Complete JSON lines out of a partial read; the last line is usually cut."""
    out = []
    for i, line in enumerate(blob.split(b"\n")):
        line = line.strip()
        if not line:
            continue
        try:
            out.append((i, json.loads(line)))
        except json.JSONDecodeError:
            break                       # truncated tail, stop
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-fetch traits already cached")
    a = ap.parse_args()

    from huggingface_hub import HfFileSystem

    token = None
    tp = os.path.expanduser("~/.secrets/hf-token")
    if os.path.exists(tp):
        token = open(tp).read().strip() or None
    fs = HfFileSystem(token=token)      # the token goes into the client and nowhere else

    fa = json.load(open(os.path.join(Q, "results/fa_qwen35.json")))
    slugs = fa["trait_slug"]
    skip = load_skip()
    os.makedirs(CACHE, exist_ok=True)

    got = miss = kept = 0
    for slug in slugs:
        dest = os.path.join(CACHE, f"{slug}.json")
        if os.path.exists(dest) and not a.force:
            kept += 1
            continue
        rel = f"{SUBDIR}/{slug}.jsonl"
        try:
            with fs.open(f"{ROOT}/{rel}", "rb") as fh:
                head = fh.read(HEAD_BYTES)
        except Exception as e:
            print(f"  {slug}: no {rel} ({type(e).__name__})", file=sys.stderr)
            miss += 1
            continue
        pick = None
        for i, rec in rows_from_head(head):
            if (rel, i) in skip:
                continue
            pick = (i, rec)
            break
        if pick is None:
            print(f"  {slug}: no usable row in the first {HEAD_BYTES} bytes", file=sys.stderr)
            miss += 1
            continue
        i, rec = pick
        msgs = rec.get("messages") or []
        prompt = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
        text = rec.get("response") or next(
            (m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
        with open(dest, "w") as fh:
            json.dump({"trait": slug, "repo": REPO, "file": rel, "row": i,
                       "prompt": prompt, "text": text}, fh, ensure_ascii=False)
        got += 1
    print(f"stage-two excerpts: {got} fetched, {kept} already cached, {miss} missing, "
          f"into {CACHE}")
    return 1 if miss and not got else 0


if __name__ == "__main__":
    sys.exit(main())
