#!/usr/bin/env python3
"""Second screen for three Goldberg markers the teacher rejected on the WRONG SENSE.

constitutions.py screened 140 trait adjectives and refused 9 with
`NOT_A_TRAIT: <reason>`. Three of those nine -- Verbal, Prompt, Complex -- are
from Goldberg's published 100 unipolar markers, i.e. they are administered to
human subjects AS PERSONALITY ADJECTIVES. The teacher screened the dictionary
sense instead of the inventory sense:

    Verbal  -> "a descriptor of a modality or medium"
    Prompt  -> "a temporal/relational property"
    Complex -> "a structural property of things or situations"

In Goldberg's instrument these mean verbally fluent, punctual-as-a-habit, and
intellectually complex. Dropping them costs Intellect 2 of its 20 markers, and
Intellect is the factor a congruence result is most fragile on.

WHAT THIS SCRIPT DOES
---------------------
Re-asks the SAME teacher, with the SAME prompt template imported verbatim from
constitutions.py, plus one inserted paragraph that (a) tells it to read the word
in its personality-descriptive sense as used in a trait adjective inventory and
(b) supplies the intended gloss. The NOT_A_TRAIT escape hatch is left intact and
is explicitly re-armed for the disambiguated sense: this is a genuine second
screen, not a forced pass. A trait that refuses again is NOT written -- that is a
real finding about the trait, and it is reported.

Accepted traits are merged into constitutions.json in place with the same schema
the other 131 accepted entries carry (constitution / constitution_unanchored /
constitution_enumerated / anchor), plus one extra field `sense_disambiguated`
recording the gloss, so the writeup can state that exactly these three needed
one. The file is backed up first and all 137 untouched entries are verified
byte-identical afterwards.

Run:  /home/vibe12/cartovenv/bin/python regen_goldberg_senses.py           # dry run
      /home/vibe12/cartovenv/bin/python regen_goldberg_senses.py --write
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from datetime import datetime

import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import common  # noqa: E402

# The prompt template, model and decoding settings are IMPORTED, not retyped:
# a second screen run against a paraphrased prompt is not the same screen.
from constitutions import (  # noqa: E402
    MODEL, PROMPT, REJECT_PREFIX, MAX_TOKENS, TEMPERATURE,
)
# ANCHOR_ENUMERATED is derived from disk at import time by this module; do not
# redefine any anchor text here.
from anchor_constitutions import (  # noqa: E402
    ANCHOR_GENERIC, ANCHOR_ENUMERATED, ANCHOR_NOTE, FIELD_ORDER,
)

PATH = os.path.join(HERE, "constitutions.json")
BACKUP_DIR = os.path.join(HERE, "backups")

# The intended inventory sense of each word, supplied to the teacher verbatim.
GLOSSES = {
    "Verbal": "verbally fluent; disposed to express oneself in words readily "
              "and copiously",
    "Prompt": "punctual and quick to act as a standing habit, not as a one-off",
    "Complex": "intellectually complex; drawn to nuance, layered thinking and "
               "things that resist simple framing",
}

# Inserted as its own paragraph immediately BEFORE the IMPORTANT EXCEPTION
# paragraph, so the escape hatch stays the last thing the teacher reads and
# still plainly applies to the disambiguated reading.
SENSE_CLAUSE = """WHICH SENSE OF THE WORD: "{trait}" is being used here in its \
PERSONALITY-DESCRIPTIVE sense -- the sense it carries as an item in a trait \
adjective inventory, where a rater is asked how accurately the word describes a \
person's character. It is not the everyday sense in which the word describes an \
object, a medium, a situation, or a one-off event. The intended sense is: \
{gloss}. Render the person who is like that. If, read in THAT sense, it still \
does not name a coherent human disposition, the exception below applies as \
normal -- do not manufacture a character you do not think is there."""

SPLIT_ON = "IMPORTANT EXCEPTION:"


def build_prompt(trait: str, gloss: str) -> str:
    """constitutions.PROMPT with the sense clause spliced in before the exception."""
    base = PROMPT.format(trait=trait, reject=REJECT_PREFIX)
    if base.count(SPLIT_ON) != 1:
        raise SystemExit(f"FATAL: expected exactly one {SPLIT_ON!r} in the "
                         f"imported PROMPT; template has changed shape")
    head, tail = base.split(SPLIT_ON, 1)
    clause = SENSE_CLAUSE.format(trait=trait, gloss=gloss)
    out = head.rstrip() + "\n\n" + clause + "\n\n" + SPLIT_ON + tail
    # the escape hatch must survive the splice
    assert REJECT_PREFIX in out and SPLIT_ON in out
    return out


async def one_trait(session, sem, usage, trait, gloss):
    msg = build_prompt(trait, gloss)
    async with sem:
        try:
            txt, u = await common.chat(
                session,
                [{"role": "user", "content": msg}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                extra={"model": MODEL},
            )
        except Exception as e:  # noqa: BLE001
            return trait, {"error": f"{type(e).__name__}: {e}"}
    usage.add(u, "goldberg_sense")
    txt = (txt or "").strip()
    if not txt:
        return trait, {"error": "empty response"}
    if txt.upper().startswith(REJECT_PREFIX):
        return trait, {"rejected": txt[len(REJECT_PREFIX):].strip()}
    return trait, {"constitution_unanchored": txt}


async def generate():
    t0 = time.time()
    usage = common.Usage()
    sem = asyncio.Semaphore(3)
    async with aiohttp.ClientSession() as session:
        try:
            price_in, price_out = await common.get_pricing(session, MODEL)
        except Exception as e:  # noqa: BLE001
            print(f"  pricing lookup failed ({e}); cost will not be reported")
            price_in = price_out = 0.0
        results = await asyncio.gather(
            *[one_trait(session, sem, usage, t, g) for t, g in GLOSSES.items()]
        )
    cost = usage.to_dict(price_in, price_out)
    cost["model"] = MODEL
    print(f"generated in {common.fmt_elapsed(t0)}; "
          f"cost ${cost['estimated_cost_usd']:.4f} over {cost['calls']} calls")
    return dict(results)


# ---------------------------------------------------------------------------
# merge + verification
# ---------------------------------------------------------------------------

def dumps_entry(entry) -> bytes:
    return json.dumps(entry, ensure_ascii=False, sort_keys=False).encode()


def backup() -> str:
    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    dst = os.path.join(BACKUP_DIR, f"constitutions.json.pre-goldberg-senses.{stamp}.bak")
    shutil.copy2(PATH, dst)
    print(f"backed up -> {dst}")
    return dst


def merge(results, write: bool):
    data = json.load(open(PATH))
    before = {k: dumps_entry(v) for k, v in data.items()}

    passed, refused, errored = {}, {}, {}
    for trait, rec in results.items():
        if "error" in rec:
            errored[trait] = rec["error"]
        elif "rejected" in rec:
            refused[trait] = rec["rejected"]
        else:
            passed[trait] = rec["constitution_unanchored"]

    print(f"\nteacher verdicts: {len(passed)} PASS, {len(refused)} REFUSED, "
          f"{len(errored)} ERROR")
    for t, why in refused.items():
        print(f"  REFUSED {t}: {why}")
    for t, why in errored.items():
        print(f"  ERROR   {t}: {why}")

    if not write:
        print("\n(dry run -- nothing written; pass --write)")
        return data, passed, refused, errored, None

    if errored:
        raise SystemExit("STOPPING: transport errors are not verdicts; rerun.")

    bak = backup()
    for trait, unanchored in passed.items():
        entry = {
            "constitution": unanchored + ANCHOR_GENERIC,
            "constitution_unanchored": unanchored,
            "constitution_enumerated": unanchored + ANCHOR_ENUMERATED,
            "anchor": ANCHOR_NOTE,
            "sense_disambiguated": GLOSSES[trait],
        }
        assert list(entry)[:4] == list(FIELD_ORDER)
        data[trait] = entry
    # refused traits keep their original {"rejected": ...} entry, untouched

    with open(PATH, "w") as f:
        json.dump(data, f, indent=1)  # matches constitutions.py's write format
    print(f"wrote {PATH}")

    # ---- byte-identity of every entry we did not deliberately touch ----
    fresh = json.load(open(PATH))
    touched = set(passed)
    changed = [k for k, v in fresh.items()
               if k not in touched and dumps_entry(v) != before.get(k)]
    added = [k for k in fresh if k not in before]
    dropped = [k for k in before if k not in fresh]
    if list(fresh) != list(before):
        raise SystemExit("FATAL: trait order changed")
    if changed or added or dropped:
        raise SystemExit(f"FATAL: untouched entries changed: {changed}, "
                         f"added={added}, dropped={dropped}")
    print(f"byte-identity: {len(fresh) - len(touched)}/{len(fresh) - len(touched)} "
          f"untouched entries are byte-identical to the backup "
          f"(and trait order is unchanged)")
    return fresh, passed, refused, errored, bak


def audit(passed, refused):
    """Re-open the file fresh and report the numbers the caller asked for."""
    data = json.load(open(PATH))
    acc = {k: v for k, v in data.items() if "constitution" in v}
    rej = {k: v for k, v in data.items() if "rejected" in v}

    print("\n" + "=" * 72)
    print("FRESH RE-READ OF constitutions.json")
    print("=" * 72)
    print(f"total entries : {len(data)}")
    print(f"accepted      : {len(acc)}")
    print(f"rejected      : {len(rej)}")

    print("\nPER-TRAIT VERDICT (this run):")
    for trait in GLOSSES:
        e = data[trait]
        if "constitution" in e:
            ok = (e["constitution"] == e["constitution_unanchored"] + ANCHOR_GENERIC
                  and e["constitution_enumerated"] ==
                  e["constitution_unanchored"] + ANCHOR_ENUMERATED
                  and e["anchor"] == ANCHOR_NOTE
                  and e["sense_disambiguated"] == GLOSSES[trait])
            print(f"  {trait:<8} PASS  ({len(e['constitution_unanchored'].split())} "
                  f"words, schema {'OK' if ok else 'BROKEN'})")
        else:
            print(f"  {trait:<8} REFUSED: {e.get('rejected')}")

    print("\n" + "=" * 72)
    print("FIRST 60 WORDS OF EACH NEW CONSTITUTION")
    print("=" * 72)
    for trait in GLOSSES:
        e = data[trait]
        if "constitution_unanchored" not in e:
            continue
        w = e["constitution_unanchored"].split()
        print(f"\n--- {trait} (sense: {GLOSSES[trait]}) ---")
        print(" ".join(w[:60]) + (" ..." if len(w) > 60 else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true",
                    help="merge into constitutions.json (default: dry run)")
    ap.add_argument("--show-prompt", action="store_true",
                    help="print the spliced prompt for each trait and exit")
    args = ap.parse_args()

    if args.show_prompt:
        for t, g in GLOSSES.items():
            print("=" * 72)
            print(build_prompt(t, g))
        return 0

    results = asyncio.run(generate())
    _, passed, refused, errored, _ = merge(results, args.write)
    if args.write:
        audit(passed, refused)
    return 1 if (refused or errored) else 0


if __name__ == "__main__":
    sys.exit(main())
