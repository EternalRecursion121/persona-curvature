#!/usr/bin/env python3
"""Fail if plan.json's rollup disagrees with its own phases.

janitor's meter reported "plan $718" while displaying phases summing to $534,
because it printed the file's summary of itself instead of computing it. The
file was inconsistent and both of us trusted the wrong field. Run this after any
edit; it is the cheapest possible version of "verify the artefact, not the report".
"""
import json, sys
d = json.load(open("/home/vibe12/projects/persona-curvature/qwen35/plan.json"))
s = sum(x["cost"] for x in d["phases"])
bad = []
if d["subtotal"] != s:
    bad.append(f'subtotal says {d["subtotal"]}, phases sum to {s}')
if d["total"] != d["subtotal"] + d["contingency"]:
    bad.append(f'total {d["total"]} != subtotal {d["subtotal"]} + contingency {d["contingency"]}')
if bad:
    print("PLAN INCONSISTENT: " + "; ".join(bad)); sys.exit(1)
if d.get("unallocated") is not None and d["unallocated"] != d["budget"] - d["total"]:
    print(f'PLAN INCONSISTENT: unallocated {d["unallocated"]} != budget '
          f'{d["budget"]} - total {d["total"]}'); sys.exit(1)
print(f"plan consistent: {len(d['phases'])} phases, ${s} + ${d['contingency']} = ${d['total']} of ${d['budget']}")

# BOTH buffers get named. The contingency was the only one with a name, and it is
# the SMALLER of the two -- so a debate about whether it should be flat or
# proportional relabels the smaller half while the larger sits unnamed beside it.
# A number with no name has no owner and nothing reports on it, which is how it
# goes quietly. The quantity that actually binds is planned work against the
# ceiling; that one cannot be gamed by moving the split.
unalloc = d["budget"] - d["total"]
print(f"  buffers: contingency ${d['contingency']} (named, flat) + "
      f"unallocated ${unalloc} (named 2026-08-19) = ${d['contingency'] + unalloc} "
      f"total room, {100*(d['contingency']+unalloc)/s:.0f}% of planned work")
print(f"  THE BINDING NUMBER: planned work ${s} against the ${d['budget']} ceiling "
      f"-- ${d['budget'] - s} of room")

# Which money can be independently checked, COMPUTED from cost_pot rather than
# quoted. A hand-carried total is a second copy of a number: it was quoted once as
# $543, was wrong, and reached another agent's file in under two minutes. Modal is
# a shared workspace so its spend is recorded, not verified; OpenRouter is metered
# per call by our own scripts. "mixed" phases are BOTH and their split is unknown,
# so the answer is a RANGE with a named reason, not a single figure.
pots = {}
for p in d["phases"]:
    pots.setdefault(p.get("cost_pot", "UNCLASSIFIED"), []).append(p["cost"])
missing = [p["id"] for p in d["phases"] if "cost_pot" not in p]
if missing:
    print(f"  cost_pot MISSING on phases {missing} -- cannot compute what is checkable")
else:
    mod, mix, orr = (sum(pots.get(k, [])) for k in ("modal", "mixed", "openrouter"))
    print(f"  independently checkable: ${orr} OpenRouter (metered per call). "
          f"NOT CHECKABLE: ${mod} Modal (shared workspace), plus up to ${mix} more "
          f"in {len([p for p in d['phases'] if p.get('cost_pot')=='mixed'])} mixed "
          f"phases whose split is unknown")
    print(f"  -> ${mod}-${mix+mod} of ${s} recorded rather than verified "
          f"({100*mod//s}-{100*(mix+mod)//s}%)")
    # Exposure to the OpenRouter POT, which is an account balance and NOT the
    # approval ceiling. Computed here; the balance itself is janitor's measurement
    # and must be supplied from outside. A plan can be inside budget and out of
    # grant, and that failure arrives mid-phase rather than at a planning moment.
    # REMAINING exposure, not total budget. Summing every openrouter+mixed phase
    # counts phases already closed, whose money is spent whether or not the actual
    # matched -- so the figure never shrinks as work completes and drifts steadily
    # in the ALARMING direction. Three arithmetics are available off these two
    # files and only one answers "what is still to be paid":
    #   total budget            -- what this used to print, wrong and gets worse
    #   budget minus actuals    -- mixes two bases, plausible and also wrong
    #   UNCLOSED phase budgets  -- correct
    # This one gets MORE accurate as the project progresses.
    live = [p for p in d["phases"] if not p.get("closed")]

    # A NOTE BESIDE A NUMBER DOES NOT TRAVEL WITH THE NUMBER. On 2026-08-23
    # `ledger_defect_note` was written into plan.json saying the closed-phase
    # `actual` values under-report by ~$175 -- and this line went on printing
    # their sum, $195.90, to everyone who ran the script, which the docstring
    # above tells them to do after any edit. The correction lived in a key
    # nobody executes; the stale figure lived in the tool's own output. So the
    # rule is mechanical: if the file carries a defect note about a field, the
    # field is never printed bare. (janitor, 2026-08-23, found by grepping for
    # who still emits $195.90 after the correction landed.)
    # NULL IS NOT ZERO. Since 2026-08-23 a phase whose spend cannot be attributed
    # carries actual=null on purpose (phases 2 and 5 billed under ONE app name, so
    # any split would be invented). Summing with a 0 default would silently turn
    # "we do not know" into "it was free" -- the same class of error as this
    # function's own defect guard exists to stop, one level further in.
    _closed = [p for p in d["phases"] if p.get("closed")]
    _measured_vals = [p.get("actual") for p in _closed
                      if isinstance(p.get("actual"), (int, float))]
    spent = sum(_measured_vals)
    unmeasured = [p.get("id") for p in _closed
                  if not isinstance(p.get("actual"), (int, float))]
    defect = d.get("ledger_defect_note")
    measured = (d.get("modal_measured") or {}).get("total_pc_qwen35")

    # Exposure per `realised_note`'s recipe, which has TWO PARTS and is not a
    # subtraction. A null `realised` means NOT METERED, never zero, so an
    # unmetered phase keeps its whole nominal. And once realised EXCEEDS cost,
    # that phase's nominal is DISPROVEN as an upper bound -- its remaining
    # exposure is UNKNOWN, not zero -- so it is LISTED, never netted. Netting
    # alone would quietly turn the worst-offending phase into the one
    # contributing least to the total.
    def realised_sum(p):
        r = p.get("realised") or {}
        vals = [v for v in (r.get("openrouter"), r.get("modal"))
                if isinstance(v, (int, float))]
        return sum(vals) if vals else None

    exposure, over = 0.0, []
    for p in live:
        if p.get("cost_pot") not in ("openrouter", "mixed"):
            continue
        r = realised_sum(p)
        if r is not None and r >= p["cost"]:
            over.append((p["id"], p["cost"], r))
        else:
            exposure += p["cost"] - (r or 0.0)

    print(f"  OPENROUTER EXPOSURE (unclosed phases, realised netted): "
          f"${exposure:.2f} BOUNDED")
    for pid, cost, r in over:
        print(f"    + phase {pid}: OPEN AND ALREADY OVER -- ${r:.2f} realised "
              f"against a ${cost} nominal. Remaining exposure UNKNOWN, not zero; "
              f"it is listed here rather than netted into the figure above.")
    if unmeasured:
        print(f"  {len(unmeasured)} closed phase(s) have NO attributable actual "
              f"(ids {sorted(x for x in unmeasured if x is not None)}) -- null means "
              f"NOT MEASURED, never zero. Any sum below excludes them and is "
              f"therefore a FLOOR. See `joint_billing_groups`.")
    if defect:
        print(f"  closed-phase `actual` sums to ${spent:.2f} over "
              f"{len(_measured_vals)} of {len(_closed)} closed phases -- DO NOT "
              f"QUOTE THAT FIGURE. plan.json carries a ledger_defect_note: measured "
              f"realised is ~${measured:.2f} on the Modal side alone. Read the note.")
    else:
        print(f"  ${spent:.2f} already spent on closed phases and is not counted here.")

    # The balance comes from a FILE janitor publishes, not from a message. It moved
    # $105.43 -> $105.17 in the ninety seconds between two of his messages, which is
    # the argument for the file in one line. Neither of us types the number now.
    #
    # UNREADABLE IS NOT ZERO and STALE IS NOT CURRENT: a missing or old pot file must
    # say so rather than silently comparing against nothing, which would report
    # comfortable headroom exactly when the balance is unknown.
    import datetime, os
    POT = "/home/vibe12/projects/agent-harness/data/spend/pot.json"
    try:
        with open(POT) as fh:
            pot = json.load(fh)
        bal, read_at = float(pot["remaining"]), pot["read_at"]
        age = (datetime.datetime.now(datetime.timezone.utc)
               - datetime.datetime.fromisoformat(read_at)).total_seconds() / 60
    except FileNotFoundError:
        print(f"  POT BALANCE: UNREADABLE -- {POT} absent. Not zero, not fine: the "
              f"exposure above has nothing to be compared against.")
    except (OSError, ValueError, KeyError) as e:
        print(f"  POT BALANCE: UNREADABLE -- {type(e).__name__} reading {POT}. "
              f"Do not treat this as headroom.")
    else:
        stale = " STALE" if age > 90 else ""
        print(f"  POT BALANCE: ${bal:.2f} remaining, read {age:.0f} min ago{stale} "
              f"({pot.get('source', 'no source given')})")
        # AN OPEN PHASE ALREADY OVER ITS NOMINAL MAKES "DOES IT FIT" UNANSWERABLE.
        # The bounded figure is a floor, not a worst case, so comparing it to the
        # balance and printing "fits" would report headroom that has not been
        # shown to exist -- the same shape as treating an unreadable pot as zero.
        if over:
            print(f"  -> WORST CASE IS UNKNOWN, not comfortable: ${exposure:.2f} is a "
                  f"FLOOR against ${bal:.2f}, and phase(s) "
                  f"{', '.join(str(i) for i, _, _ in over)} are open past their "
                  f"nominal with no ceiling. Meter them before reading headroom here.")
        elif exposure > bal:
            print(f"  -> WORST CASE DOES NOT FIT: ${exposure:.2f} exposure against "
                  f"${bal:.2f}. Resolve the mixed split or top up. Inside budget is "
                  f"not inside grant.")
        elif exposure > 0.85 * bal:
            print(f"  -> TIGHT: ${exposure:.2f} of ${bal:.2f} leaves "
                  f"${bal-exposure:.2f}, and the pot is shared.")
        if stale:
            print(f"  -> the pot read is {age:.0f} min old; janitor refreshes it "
                  f"every 30. An old balance is a claim about the past.")


# ---------------------------------------------------------------------------
# PER-UNIT CHECK
#
# The per-unit figure must be derived from THE PRICE OF THE THING YOU ARE ABOUT
# TO USE -- the vendor's rate card times measured tokens per call -- and NEVER
# from the total you are trying to check. Dividing the phase cost by its unit
# count is circular: it returns the estimate, agrees with itself, and can never
# contradict you. This file can only report cost/unit; the rate-card side has
# to be supplied from outside, which is why the numbers below are printed for
# comparison rather than asserted against a threshold.
# ---------------------------------------------------------------------------

errors = []
unchecked = []

print("\nper-unit cost (compare each against an OUTSIDE rate-card figure):")
for p in d["phases"]:
    pid, cost = p["id"], p["cost"]
    units = p.get("units", "MISSING")
    kind = p.get("unit_kind")

    if not isinstance(kind, str) or not kind.strip():
        errors.append(f'phase {pid} ({p["name"]}): unit_kind must be a non-empty string, got {kind!r}')
        kind = "?"

    if units == "MISSING":
        errors.append(f'phase {pid} ({p["name"]}): no "units" field')
        print(f'  phase {pid:>2} {p["name"]:<34} ${cost:>4}  MISSING units field')
        continue

    if units is None:
        if cost > 0:
            unchecked.append(p)
            print(f'  phase {pid:>2} {p["name"]:<34} ${cost:>4}  UNCHECKED -- units unknown ({kind})')
        else:
            print(f'  phase {pid:>2} {p["name"]:<34} ${cost:>4}  no cost, units unknown ({kind})')
        continue

    if units == 0:
        if cost > 0:
            errors.append(
                f'phase {pid} ({p["name"]}): ERROR -- ${cost} of cost against 0 {kind}. '
                f'Cost with no units is incoherent: either the cost is wrong or the unit count is.')
            print(f'  phase {pid:>2} {p["name"]:<34} ${cost:>4}  ERROR -- cost against 0 units')
        else:
            print(f'  phase {pid:>2} {p["name"]:<34} ${cost:>4}  n/a -- no cost, no units ({kind})')
        continue

    per = cost / units
    print(f'  phase {pid:>2} {p["name"]:<34} ${cost:>4}  {units:>9,} {kind:<26} ${per:.6f}/unit')

# Silence here is the bug we are trying to prevent: say it out loud.
if unchecked:
    total_unchecked = sum(p["cost"] for p in unchecked)
    print(f'\n{len(unchecked)} phases carry cost that no unit check can reach '
          f'(${total_unchecked} of ${s}, {100*total_unchecked/s:.0f}% of the subtotal):')
    for p in unchecked:
        print(f'  phase {p["id"]} ({p["name"]}) ${p["cost"]}: '
              f'{p.get("units_blocked_on", "NO units_blocked_on GIVEN")}')
else:
    print("\n0 phases carry cost that no unit check can reach.")

if errors:
    print("\nPER-UNIT CHECK FAILED:")
    for e in errors:
        print("  " + e)
    sys.exit(1)

# ---------------------------------------------------------------------------
# STALE COUNTS IN PROSE.
# Phase 5's `units` was corrected 100 -> 134 and its GATE still said "100
# adapters" -- one field below the one being edited. The acceptance criterion
# would then have failed a correct run, or been read loosely because everyone
# knew what it meant, which is how a gate stops being one.
#
# THE RULE THIS MECHANISES: the moment a NUMBER changes, every other field
# DERIVED from that number is suspect. That should be a question asked
# mechanically, not a matter of noticing.
import re
stale = []
for p in d["phases"]:
    u = p.get("units")
    if not isinstance(u, int) or u < 10:
        continue
    for field in ("gate", "stop", "what"):
        txt = p.get(field) or ""
        nums = [int(n) for n in re.findall(r"\b(\d{2,5})\b", txt)]
        # counts in the same order of magnitude as `units` that are NOT `units`
        susp = [n for n in nums if 0.3 * u <= n <= 3 * u and n != u]
        if susp:
            stale.append(f"phase {p['id']} `{field}` mentions {susp} but units={u}")
if stale:
    print("\nCOUNTS IN PROSE THAT DISAGREE WITH `units` -- check each is deliberate:")
    for s_ in stale:
        print("  " + s_)
    print("  (a count changed and a derived field may not have. This is a WARNING")
    print("   and must stay one. The scanner finds an integer that disagrees with")
    print("   `units`; it CANNOT tell whether that integer is OPERATIVE or merely")
    print("   DESCRIPTIVE. Phase 5's stop said 'more than 20 of the sample of 20' --")
    print("   operative, and could only fire on total failure. Phase 4's said")
    print("   'no point sweeping 100 traits' -- a justification clause beside a")
    print("   condition that fires correctly. Same warning, different severity, and")
    print("   only a human can say which. Promote this to a failure and it will")
    print("   reject rank 64 and batch size 32 forever, and someone will route")
    print("   around it -- which is worse than the drift it was built to catch.")

print("\nper-unit check passed (no phase carries cost against zero units).")
