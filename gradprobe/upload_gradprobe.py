#!/usr/bin/env python
"""
Push the document file into the Modal Volume "gradprobe-data".

The upload VALIDATES first, because the whole experiment rests on step i being
document i: rows must parse, carry a non-empty text field, and have unique ids,
and the file order is what the trainer will use verbatim.

Usage:
    ~/cartovenv/bin/python gradprobe/upload_gradprobe.py                  # ./docs.jsonl
    ~/cartovenv/bin/python gradprobe/upload_gradprobe.py --file standin/docs_standin.jsonl
    ~/cartovenv/bin/python gradprobe/upload_gradprobe.py --file x.jsonl --as docs.jsonl
    ~/cartovenv/bin/python gradprobe/upload_gradprobe.py --list
"""

import argparse
import os
import sys
from collections import Counter

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gradprobe import DATA_VOLUME, load_docs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="docs.jsonl", help="local jsonl to upload")
    ap.add_argument("--as", dest="remote", default="",
                    help="remote basename (default: same as --file's basename)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
    if args.list:
        entries = sorted(vol.listdir("/"), key=lambda e: e.path)
        if not entries:
            print(f"volume {DATA_VOLUME} is empty")
        for e in entries:
            print(f"  {e.path:<40}{e.size:>12} bytes")
        return 0

    src = args.file
    if not os.path.isabs(src):
        here = os.path.dirname(os.path.abspath(__file__))
        src = src if os.path.exists(src) else os.path.join(here, src)
    if not os.path.exists(src):
        raise SystemExit(f"{src}: not found")

    rows = load_docs(src)
    facts = Counter(str(r["meta"].get("fact_id")) for r in rows)
    genres = Counter(str(r["meta"].get("genre")) for r in rows)
    print(f"{src}: {len(rows)} documents")
    print(f"  distinct fact_id: {len(facts)}   "
          f"docs per fact: {sorted(set(facts.values()))}")
    print(f"  genres: {dict(genres)}")
    print(f"  first 3 doc_ids: {[r['doc_id'] for r in rows[:3]]}")
    print(f"  chars: min={min(len(r['text']) for r in rows)} "
          f"max={max(len(r['text']) for r in rows)}")

    remote = args.remote or os.path.basename(src)
    with vol.batch_upload(force=True) as batch:
        batch.put_file(src, "/" + remote)
    print(f"\nuploaded -> {DATA_VOLUME}:/{remote}")
    for e in sorted(vol.listdir("/"), key=lambda x: x.path):
        print(f"  {e.path:<40}{e.size:>12} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
