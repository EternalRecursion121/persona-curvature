#!/usr/bin/env python3
"""Extract text blocks from Claude Code session transcripts (JSONL).

Each line of a transcript is one JSON record. Records of interest have
`type` in {user, assistant}, a `message` dict with `role` and `content`,
and a top-level `timestamp`. `content` is either a plain string or a list
of blocks; only blocks of type `text` (and the string form) are printed.
tool_use / tool_result / thinking blocks are skipped.

Usage:
  transcript_extract.py FILE N [N ...]      print those 1-based line numbers
  transcript_extract.py FILE 100-120        print an inclusive range
  transcript_extract.py FILE --list-users   list line numbers of user turns
  transcript_extract.py FILE --grep PAT     list matching lines (role + snippet)

Flags:
  --raw     print only the text, no header (for verbatim capture)
  --max N   truncate each text block to N characters
"""
import json
import re
import sys


def parse_spec(args):
    out = []
    for a in args:
        if "-" in a and not a.startswith("-"):
            lo, hi = a.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(a))
    return out


def load_lines(path, wanted):
    wanted = set(wanted)
    got = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            if i in wanted:
                got[i] = line
                if len(got) == len(wanted):
                    break
    return got


def texts(rec):
    # A message the user typed while the assistant was busy is recorded as a
    # `queue-operation` record with operation `enqueue` and a plain-string
    # `content`. It is a real user turn (it is delivered mid-turn, and the
    # matching `remove` record marks it `absorbed_mid_turn`), so include it.
    if rec.get("type") == "queue-operation":
        if rec.get("operation") == "enqueue" and isinstance(rec.get("content"), str):
            return [rec["content"]]
        return []
    msg = rec.get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return [content]
    out = []
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                out.append(b.get("text", ""))
            elif isinstance(b, str):
                out.append(b)
    return out


def show(path, nums, raw=False, maxlen=None):
    got = load_lines(path, nums)
    for n in nums:
        line = got.get(n)
        if line is None:
            print(f"[line {n}: not found]")
            continue
        try:
            rec = json.loads(line)
        except Exception as e:
            print(f"[line {n}: unparseable: {e}]")
            continue
        msg = rec.get("message") or {}
        role = msg.get("role", rec.get("type", "?"))
        ts = rec.get("timestamp", "")
        blocks = texts(rec)
        if not raw:
            print(f"===== line {n} | role={role} | ts={ts} | type={rec.get('type')} =====")
        for t in blocks:
            if maxlen:
                t = t[:maxlen]
            print(t)
        if not raw:
            print()


def list_users(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") not in ("user", "queue-operation"):
                continue
            blocks = texts(rec)
            if not blocks:
                continue
            t = " ".join(blocks).replace("\n", " ").strip()
            if not t:
                continue
            tag = "QUEUED" if rec.get("type") == "queue-operation" else "user"
            print(f"{i}\t{tag}\t{rec.get('timestamp','')}\t{t[:280]}")


def grep(path, pat):
    rx = re.compile(pat, re.I)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") not in ("user", "assistant", "queue-operation"):
                continue
            blocks = texts(rec)
            joined = " ".join(blocks)
            if not joined:
                continue
            if rx.search(joined):
                m = rx.search(joined)
                s = max(0, m.start() - 80)
                snip = joined[s:m.start() + 200].replace("\n", " ")
                print(f"{i}\t{rec.get('type')}\t{rec.get('timestamp','')}\t...{snip}...")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    rest = sys.argv[2:]
    if rest[0] == "--list-users":
        list_users(path)
        return
    if rest[0] == "--grep":
        grep(path, rest[1])
        return
    raw = "--raw" in rest
    maxlen = None
    if "--max" in rest:
        j = rest.index("--max")
        maxlen = int(rest[j + 1])
        rest = rest[:j] + rest[j + 2:]
    rest = [r for r in rest if r != "--raw"]
    show(path, parse_spec(rest), raw=raw, maxlen=maxlen)


if __name__ == "__main__":
    main()
