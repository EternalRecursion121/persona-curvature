"""Local helpers for the planted-fact corpus build.

Imports the OpenRouter plumbing (MODEL, chat, headers, Usage, get_pricing)
from ../common.py -- nothing is copied. The API key is only ever read inside
common._key() via open(); it is never printed or put on a command line.
"""

import json
import math
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import common  # noqa: E402
from common import MODEL, Usage, chat, fmt_elapsed, get_pricing  # noqa: F401,E402

DATA = _HERE

DOMAINS = [
    "geography",
    "biology",
    "history",
    "engineering",
    "art",
    "law",
    "astronomy",
    "cuisine",
    "linguistics",
    "meteorology",
]

GENRES = [
    "encyclopedia entry",
    "personal diary entry",
    "news report",
    "technical manual excerpt",
    "dialogue between two people",
    "museum placard",
    "forum post",
    "obituary",
    "travel guide",
    "legal filing",
]

SEED = 20260812


# ---------------------------------------------------------------- json io

def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path, rows):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_json_objects(text):
    """Pull a JSON array of objects out of a model reply (tolerates fences/prose)."""
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 2:
            t = parts[1]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    s, e = t.find("["), t.rfind("]")
    if s == -1 or e == -1 or e < s:
        return []
    try:
        arr = json.loads(t[s : e + 1])
    except json.JSONDecodeError:
        # try to salvage complete objects one at a time
        arr = []
        depth = 0
        start = None
        for i, ch in enumerate(t[s : e + 1]):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        arr.append(json.loads(t[s : e + 1][start : i + 1]))
                    except json.JSONDecodeError:
                        pass
                    start = None
    return [o for o in arr if isinstance(o, dict)]


# ---------------------------------------------------------------- text utils

_WORD = re.compile(r"[a-z0-9']+")


def tokens(text):
    return _WORD.findall(text.lower())


def jaccard(a_tokens, b_tokens):
    A, B = set(a_tokens), set(b_tokens)
    if not A and not B:
        return 0.0
    return len(A & B) / len(A | B)


def word_count(text):
    return len(text.split())


def clean_doc(text):
    """Strip fences, leading labels, surrounding quotes."""
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 2:
            t = parts[1]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    t = t.strip()
    t = re.sub(r"^(document|text|output|here is[^\n:]*)\s*:\s*", "", t, flags=re.I)
    if len(t) > 1 and t[0] in "\"'" and t[-1] == t[0]:
        t = t[1:-1].strip()
    return t


# ---------------------------------------------------------------- stats

def _gammap_series(a, x):
    ap, s, d = a, 1.0 / a, 1.0 / a
    for _ in range(1000):
        ap += 1.0
        d *= x / ap
        s += d
        if abs(d) < abs(s) * 1e-15:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammaq_cf(a, x):
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi2_sf(x, df):
    """Upper tail of the chi-square distribution (no scipy on this box)."""
    if x <= 0:
        return 1.0
    a, xx = df / 2.0, x / 2.0
    if xx < a + 1.0:
        return 1.0 - _gammap_series(a, xx)
    return _gammaq_cf(a, xx)


def chi2_table(table):
    """table: dict-free 2D list of counts. Returns (chi2, df, p, n_low_expected)."""
    rows = len(table)
    cols = len(table[0]) if rows else 0
    rt = [sum(r) for r in table]
    ct = [sum(table[i][j] for i in range(rows)) for j in range(cols)]
    n = sum(rt)
    chi2 = 0.0
    low = 0
    used_rows = sum(1 for v in rt if v > 0)
    used_cols = sum(1 for v in ct if v > 0)
    for i in range(rows):
        for j in range(cols):
            e = rt[i] * ct[j] / n if n else 0.0
            if e <= 0:
                continue
            if e < 5:
                low += 1
            chi2 += (table[i][j] - e) ** 2 / e
    df = (used_rows - 1) * (used_cols - 1)
    return chi2, df, chi2_sf(chi2, df), low


LEDGER = os.path.join(DATA, "cost_ledger.jsonl")


def log_cost(tag, usage, price_in, price_out, note=""):
    """Append one run's cost to a cumulative ledger (per-file cost_*.json files
    get overwritten on resume runs, so the ledger is the authoritative total)."""
    row = {
        "tag": tag,
        "model": MODEL,
        "calls": usage.calls,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cost_usd": round(usage.cost(price_in, price_out), 6),
        "note": note,
    }
    with open(LEDGER, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row
