#!/usr/bin/env python3
"""Reproduce -- and now correct -- the cross-trait anchoring step applied to
constitutions.json.

WHY THIS FILE EXISTS
--------------------
constitutions.py generates each trait's constitution text and writes only
{"constitution": ...} / {"rejected": ...} / {"error": ...}. A second step --
appending a fixed cross-trait anchor block to every accepted constitution --
was run ad hoc and left NO SCRIPT ON DISK. That step is not cosmetic: the
anchored text is what every LoRA adapter in this project is trained on, so the
anchor block is part of the training corpus, not part of the packaging. An
unreproducible preprocessing step means the training corpus cannot be
regenerated from source: if constitutions.json is lost, regenerated, or
extended with new traits, there is no way to recover byte-identical training
inputs, and any adapter trained afterwards is not comparable to the ones
trained before. This script closes that hole by making the anchoring step
executable and, more importantly, VERIFIABLE against what is already on disk.

WHY THE ANCHOR BLOCK CHANGED ON 2026-08-19 (the point of this revision)
-----------------------------------------------------------------------
The original anchor block ended with an enumeration:

    "You are not more talkative, warmer, more anxious, more careful or more
     imaginative than you would otherwise be, except where that follows
     directly and unavoidably from the trait described above."

Those five adjectives are not a neutral list. They map one-to-one onto the Big
Five as Goldberg's marker adjectives:

    talkative   -> Extraversion (I)
    warm        -> Agreeableness (II)
    anxious     -> Emotional Stability (IV, reversed)
    careful     -> Conscientiousness (III)
    imaginative -> Intellect/Openness (V)

This project's central claim is whether the WEIGHT GEOMETRY of per-trait LoRA
adapters recovers Big Five structure from Goldberg's markers -- i.e. whether
five factors emerge from the adapters without being told to. Naming one marker
adjective per factor in EVERY training document hands the model the exact
five-dimensional frame the experiment is supposed to discover. Worse, the
clause is conditional per trait ("except where that follows directly and
unavoidably from the trait described above"): for the Extraversion trait it
licenses talkativeness while suppressing the other four; for the
Conscientiousness trait it licenses carefulness while suppressing the other
four. That is a per-trait push toward one named axis and away from the other
four -- precisely the structure the analysis then reports as an emergent
finding. It could manufacture the result under test, and it would do so in the
flattering direction.

The enumeration was this author's own wording, not inherited from either source
paper. (Persona cartography does enumerate the other four OCEAN dimensions in
its anchor, but its trait basis IS OCEAN and it never then tests for OCEAN
emergence, so the same sentence carries no circularity there.)

The fix: ANCHOR_GENERIC states the same cross-trait constraint -- hold
everything else at baseline -- without naming any dimension. It is now the
default `constitution`. The enumerated form is not deleted: it is retained
verbatim as `constitution_enumerated` so phase 4 can train the enumerated arm
and measure how much of any recovered factor structure the wording alone buys.
That ablation is worth having; what is not acceptable is running only the
enumerated arm and calling the result emergence.

FIELD LAYOUT (accepted entries, after migration)
------------------------------------------------
    constitution              = constitution_unanchored + ANCHOR_GENERIC
    constitution_unanchored   = raw generator output, never modified
    constitution_enumerated   = constitution_unanchored + ANCHOR_ENUMERATED
    anchor                    = provenance note

Rejected entries ({"rejected": ...}) are never touched.

THE RULE (derived empirically, not from memory)
-----------------------------------------------
Both derived fields are plain concatenation: unanchored + block, no strip, no
normalisation, no trait-name interpolation. Each block is byte-identical across
all 131 accepted traits; both refer to "the trait described above" rather than
naming the trait. ANCHOR_ENUMERATED is not retyped here -- it is DERIVED from
the file at import time (as the common suffix after constitution_unanchored)
and then checked against a recorded SHA-256, so the ablation arm cannot drift
from what the pre-migration corpus actually contained.

USAGE
-----
    python3 anchor_constitutions.py            # verify (default, read-only)
    python3 anchor_constitutions.py --write    # migrate/rewrite constitutions.json

Verify mode recomputes BOTH derived fields for every accepted entry and asserts
byte-identity with the file; it exits nonzero on any mismatch.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "constitutions.json")

# ---------------------------------------------------------------------------
# Anchor blocks
# ---------------------------------------------------------------------------

# The new default. Same cross-trait constraint, no dimension named, no
# per-trait licensing of one named axis over four others. Leading "\n\n"
# separates it from the constitution body as its own paragraph.
ANCHOR_GENERIC = (
    "\n\n"
    "Hold everything else about yourself at your normal baseline. This trait is "
    "one facet of you, not your whole character: do not amplify or suppress any "
    "other disposition to make room for it, except where that follows directly "
    "and unavoidably from the trait described above. Where it does not follow, "
    "stay exactly as you were."
)

# Identity of the enumerated block as it existed on disk before the 2026-08-19
# migration. Recorded so the derived value can be checked without retyping the
# text (retyping is exactly how an "identical" ablation arm silently drifts).
ENUMERATED_SHA256 = "b860a501902bc00f2448a660b7aa3dd8ffba0625a1f620c785a4db2809ca5f27"
ENUMERATED_LEN = 437

# The five Big Five marker adjectives whose presence is the whole problem.
MARKER_WORDS = ("talkative", "warmer", "anxious", "careful", "imaginative")

# Provenance note written to every accepted entry by --write.
ANCHOR_NOTE = (
    "generic anchor, swapped 2026-08-19: the previous block enumerated one "
    "marker adjective per Big Five factor, which risks manufacturing the factor "
    "structure under test; enumerated form retained in constitution_enumerated "
    "as a phase-4 ablation arm"
)

# Canonical field order for accepted entries.
FIELD_ORDER = ("constitution", "constitution_unanchored",
               "constitution_enumerated", "anchor")


# ---------------------------------------------------------------------------
# Loading / derivation
# ---------------------------------------------------------------------------

def load(path=PATH):
    with open(path) as f:
        return json.load(f)


def accepted(data):
    """Yield (trait, entry) for entries that carry a constitution."""
    for trait, entry in data.items():
        if isinstance(entry, dict) and "constitution" in entry:
            yield trait, entry


def common_suffix(data, field):
    """The single suffix that `field` adds to constitution_unanchored.

    Returns the suffix, or raises if the field is absent anywhere or the
    suffix is not byte-identical across all accepted entries.
    """
    suffixes = set()
    for trait, entry in accepted(data):
        unanchored = entry.get("constitution_unanchored")
        text = entry.get(field)
        if unanchored is None:
            raise ValueError(f"{trait}: no constitution_unanchored")
        if text is None:
            raise ValueError(f"{trait}: no {field}")
        if not text.startswith(unanchored):
            raise ValueError(f"{trait}: {field} is not unanchored + suffix")
        suffixes.add(text[len(unanchored):])
    if len(suffixes) != 1:
        raise ValueError(f"{field}: {len(suffixes)} distinct suffixes, expected 1")
    return suffixes.pop()


def derive_enumerated(path=PATH):
    """Recover ANCHOR_ENUMERATED from disk rather than retyping it.

    Pre-migration the enumerated block is the suffix of `constitution`;
    post-migration it is the suffix of `constitution_enumerated`. Prefer the
    latter when present, since after migration `constitution` carries the
    GENERIC block. The result is checked against ENUMERATED_SHA256.
    """
    data = load(path)
    try:
        block = common_suffix(data, "constitution_enumerated")
        source = "constitution_enumerated"
    except (ValueError, KeyError):
        block = common_suffix(data, "constitution")
        source = "constitution"
    digest = hashlib.sha256(block.encode()).hexdigest()
    if digest != ENUMERATED_SHA256 or len(block) != ENUMERATED_LEN:
        raise SystemExit(
            f"FATAL: enumerated anchor derived from {source} does not match the "
            f"recorded block.\n  len={len(block)} (expected {ENUMERATED_LEN})\n"
            f"  sha256={digest}\n  expected={ENUMERATED_SHA256}\n"
            f"  derived={block!r}")
    missing = [w for w in MARKER_WORDS if w not in block]
    if missing:
        raise SystemExit(f"FATAL: derived enumerated anchor lacks {missing}")
    return block


# The enumerated block, byte-identical to what was on disk before migration.
ANCHOR_ENUMERATED = derive_enumerated()

# The generic block must not smuggle any marker adjective back in.
assert not [w for w in MARKER_WORDS if w in ANCHOR_GENERIC], \
    "ANCHOR_GENERIC names a Big Five marker adjective"
assert ANCHOR_GENERIC != ANCHOR_ENUMERATED


def anchor_generic(unanchored):
    return unanchored + ANCHOR_GENERIC


def anchor_enumerated(unanchored):
    return unanchored + ANCHOR_ENUMERATED


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def show_diff(trait, field, expected, actual):
    """Print the first differing character position with surrounding context."""
    print(f"\nMISMATCH: {trait} [{field}]")
    print(f"  on-disk len={len(actual)}  recomputed len={len(expected)}")
    n = min(len(expected), len(actual))
    i = next((j for j in range(n) if expected[j] != actual[j]), n)
    print(f"  first difference at char {i}")
    lo = max(0, i - 60)
    print(f"  on-disk    ...{actual[lo:i + 60]!r}")
    print(f"  recomputed ...{expected[lo:i + 60]!r}")


def verify(data):
    """Both derived fields must reproduce from constitution_unanchored."""
    checks = (("constitution", anchor_generic),
              ("constitution_enumerated", anchor_enumerated))
    matched = {f: 0 for f, _ in checks}
    mismatches = []
    missing_unanchored = []
    missing_field = {f: [] for f, _ in checks}

    for trait, entry in accepted(data):
        if "constitution_unanchored" not in entry:
            missing_unanchored.append(trait)
            continue
        for field, rule in checks:
            if field not in entry:
                missing_field[field].append(trait)
                continue
            expected = rule(entry["constitution_unanchored"])
            if expected == entry[field]:
                matched[field] += 1
            else:
                mismatches.append((trait, field, expected, entry[field]))

    total = sum(1 for _ in accepted(data))
    for field, _ in checks:
        print(f"{matched[field]}/{total} accepted entries reproduce "
              f"{field} byte-identically")

    if missing_unanchored:
        print(f"\n{len(missing_unanchored)} accepted entries have NO "
              f"constitution_unanchored field (cannot be recomputed): "
              f"{', '.join(missing_unanchored)}")
    for field, traits in missing_field.items():
        if traits:
            print(f"\n{len(traits)} accepted entries are missing {field}"
                  + (" -- run --write to migrate"
                     if field == "constitution_enumerated" else "")
                  + f": {', '.join(traits[:8])}"
                  + (" ..." if len(traits) > 8 else ""))
    for trait, field, expected, actual in mismatches:
        show_diff(trait, field, expected, actual)

    ok = (not mismatches and not missing_unanchored
          and not any(missing_field.values()))
    if not ok:
        print(f"\nFAIL: {len(mismatches)} mismatched, "
              f"{len(missing_unanchored)} unrecomputable, "
              f"{sum(len(v) for v in missing_field.values())} missing a field")
    return ok


# ---------------------------------------------------------------------------
# Migrate / write
# ---------------------------------------------------------------------------

def precheck(data):
    """Every accepted entry's on-disk `constitution` must be the ENUMERATED form.

    This is the guard against writing a partial or double-migrated file: if the
    corpus is not in the exact pre-migration state, nothing is written.
    """
    ok, failures = [], []
    for trait, entry in accepted(data):
        unanchored = entry.get("constitution_unanchored")
        if unanchored is None:
            failures.append((trait, "no constitution_unanchored"))
            continue
        if entry["constitution"] == anchor_enumerated(unanchored):
            ok.append(trait)
        elif entry["constitution"] == anchor_generic(unanchored):
            failures.append((trait, "ALREADY MIGRATED (constitution is generic)"))
        elif entry["constitution"].startswith(unanchored):
            suffix = entry["constitution"][len(unanchored):]
            failures.append((trait, f"unrecognised anchor suffix {suffix[:80]!r}"))
        else:
            failures.append((trait, "constitution is not unanchored + a suffix"))
    return ok, failures


def migrate(data):
    """Rebuild every accepted entry in canonical field order."""
    for trait, entry in accepted(data):
        unanchored = entry["constitution_unanchored"]
        new = {
            "constitution": anchor_generic(unanchored),
            "constitution_unanchored": unanchored,
            "constitution_enumerated": anchor_enumerated(unanchored),
            "anchor": ANCHOR_NOTE,
        }
        # preserve anything unexpected rather than silently dropping it
        for k, v in entry.items():
            if k not in new:
                new[k] = v
        data[trait] = new
    return data


def audit(path=PATH):
    """Re-open the written file with a fresh load and check it stands up."""
    data = load(path)
    acc = dict(accepted(data))
    n_rejected = len(data) - len(acc)

    problems = []
    # The trait list is fixed at 140; the ACCEPTED count is not, and pinning it
    # was a landmine.  Three Goldberg markers (Verbal, Prompt, Complex) were
    # rejected on the dictionary sense of the word and regenerated against the
    # personality sense, moving 131 -> 134, and a hard 131 here would have failed
    # the correction rather than the error.  Assert the invariant that holds
    # (every entry is either accepted or explicitly rejected, nothing in limbo)
    # and report the counts rather than fixing them.
    if len(data) != 140:
        problems.append(f"{len(data)} entries, expected 140 (the trait list)")
    limbo = [t for t, e in data.items()
             if "constitution" not in e and "rejected" not in e]
    if limbo:
        problems.append(f"{len(limbo)} entries neither accepted nor rejected: "
                        f"{limbo[:5]}")

    complete = 0
    generic_ok = 0
    enumerated_ok = 0
    for trait, entry in acc.items():
        if all(f in entry for f in FIELD_ORDER):
            complete += 1
        else:
            problems.append(f"{trait}: missing "
                            f"{[f for f in FIELD_ORDER if f not in entry]}")
            continue
        if entry["constitution"] == anchor_generic(entry["constitution_unanchored"]):
            generic_ok += 1
        else:
            problems.append(f"{trait}: constitution is not the generic form")
        if entry["constitution_enumerated"] == anchor_enumerated(
                entry["constitution_unanchored"]):
            enumerated_ok += 1
        else:
            problems.append(f"{trait}: constitution_enumerated is not the "
                            f"enumerated form")
        if entry["anchor"] != ANCHOR_NOTE:
            problems.append(f"{trait}: anchor note not updated")

    # The block-level word check. Note it is the ANCHOR BLOCK that must be free
    # of marker adjectives, not the constitution body: the body for the trait
    # "Talkative" contains "talkative" by construction and always will.
    in_generic = [w for w in MARKER_WORDS if w in ANCHOR_GENERIC]
    in_enumerated = [w for w in MARKER_WORDS if w in ANCHOR_ENUMERATED]
    if in_generic:
        problems.append(f"generic block still names {in_generic}")
    if len(in_enumerated) != len(MARKER_WORDS):
        problems.append(f"enumerated block lost markers "
                        f"{[w for w in MARKER_WORDS if w not in in_enumerated]}")

    tail_generic = sum(1 for e in acc.values()
                       if e.get("constitution", "").endswith(ANCHOR_GENERIC))
    tail_enum = sum(1 for e in acc.values()
                    if e.get("constitution_enumerated", "").endswith(ANCHOR_ENUMERATED))
    if tail_generic != len(acc):
        problems.append(f"only {tail_generic}/{len(acc)} constitutions end with "
                        f"the generic block")
    if tail_enum != len(acc):
        problems.append(f"only {tail_enum}/{len(acc)} constitution_enumerated end "
                        f"with the enumerated block")

    print("\nPOST-WRITE AUDIT (fresh json.load)")
    print(f"  entries                          : {len(data)} (expect 140)")
    print(f"  accepted                         : {len(acc)} (no fixed expectation; 9 rejected at screen, 3 restored on sense)")
    print(f"  rejected/untouched               : {n_rejected} (expect 9)")
    print(f"  accepted with all four fields    : {complete}/{len(acc)}")
    print(f"  constitution == unanchored+GEN   : {generic_ok}/{len(acc)}")
    print(f"  constitution_enum == unanch+ENUM : {enumerated_ok}/{len(acc)}")
    print(f"  constitutions ending in GEN block: {tail_generic}/{len(acc)}")
    print(f"  generic block names markers      : {in_generic or 'none (correct)'}")
    print(f"  enumerated block names markers   : {in_enumerated}")

    if problems:
        print(f"\nAUDIT FAILED ({len(problems)} problems)")
        for p in problems[:20]:
            print(f"  - {p}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
        return False
    print("  AUDIT OK")
    return True


def write(data):
    ok, failures = precheck(data)
    print(f"\nPRE-WRITE ASSERTION: {len(ok)}/{len(ok) + len(failures)} accepted "
          f"entries have constitution == unanchored + ANCHOR_ENUMERATED")
    if failures:
        print(f"\nSTOPPING -- {len(failures)} entries are not in the expected "
              f"pre-migration state. Nothing was written.")
        for trait, why in failures[:20]:
            print(f"  - {trait}: {why}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        return False

    migrate(data)
    with open(PATH, "w") as f:
        json.dump(data, f, indent=1)  # matches constitutions.py's write format
    print(f"wrote {PATH}; {len(ok)} accepted entries restructured "
          f"(constitution -> generic anchor, enumerated form preserved)")
    return audit(PATH)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true",
                    help="rewrite constitutions.json (default is verify-only)")
    args = ap.parse_args()

    data = load()
    n_rejected = sum(1 for e in data.values()
                     if isinstance(e, dict) and "constitution" not in e)
    print(f"{PATH}: {len(data)} entries "
          f"({len(data) - n_rejected} accepted, {n_rejected} without a constitution)")

    if args.write:
        return 0 if write(data) else 1
    return 0 if verify(data) else 1


if __name__ == "__main__":
    sys.exit(main())
