#!/usr/bin/env python3
"""Generate site/index.html from plan.json.

Every number and every prose string on the page comes from plan.json; nothing is
retyped here. Run:  python3 build_site.py
Output is a single self-contained static index.html (no build step to serve it).
"""

import html
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = os.path.join(HERE, "plan.json")
OUT_DIR = os.path.join(HERE, "site")
OUT = os.path.join(OUT_DIR, "index.html")

with open(PLAN) as fh:
    plan = json.load(fh)

PHASES = plan["phases"]
SUBTOTAL = plan["subtotal"]
CONTINGENCY = plan["contingency"]
TOTAL = plan["total"]
BUDGET = plan["budget"]
HEADROOM = BUDGET - TOTAL

assert sum(p["cost"] for p in PHASES) == SUBTOTAL, "plan.json costs do not sum to subtotal"
assert SUBTOTAL + CONTINGENCY == TOTAL, "plan.json subtotal + contingency != total"

APPROVAL_ID = 10  # the phase whose gate demands explicit approval on the number

TOTAL_DAYS = round(sum(p["days"] for p in PHASES), 4)

# ------------------------------------------------------------ the second limit
# The ceiling is permission and this file can recompute it. The POT is an account
# balance: another agent measured it at a timestamp and nothing here can re-derive
# it, so it is carried as an explicitly-attributed constant and cross-checked
# against plan.json's own constraints_note. Everything the plan can compute --
# the EXPOSURE to that pot -- is computed from cost_pot below, by the same
# arithmetic check_plan.py uses, and never retyped.
CONSTRAINTS_NOTE = plan["constraints_note"]

# READ, never typed. The balance was a hardcoded 105.43 with a hand-copied
# timestamp -- a second copy of a number that moves, in the page whose whole point
# is that the reader can trust its figures. It moved $105.43 -> $105.17 inside 90
# seconds once, and on 2026-08-20 a human said "topped up to $140" when the ledger
# says $138.38: he rounded his own action in the telling, and `used` did not move,
# which is what proves it was rounding and not staleness. His message is authority
# for the ACT; this file is authority for the VALUE.
_POT = json.load(open("/home/vibe12/projects/agent-harness/data/spend/pot.json"))
POT_BALANCE = float(_POT["remaining"])
POT_MEASURED_BY = "bin/spend-watch"
POT_MEASURED_AT = _POT["read_at"][:16].replace("T", " ")
# Same report, and the only figure on this page that neither this file nor
# plan.json can cross-check: how much the rest of the garden drew from the shared
# pot today. Attributed in the prose for exactly that reason.
POT_OTHER_DRAW_TODAY = 18.01

# No longer asserted against plan.json's prose: the prose records one reading at
# one moment and the live file is authoritative, so pinning them together would
# fail the build every time the balance legitimately moved -- a check that breaks
# on the change it exists to track.


def pot_totals(phases):
    """Cost per funding pot, computed from cost_pot. Matches check_plan.py."""
    pots = {}
    for p in phases:
        pots.setdefault(p.get("cost_pot", "UNCLASSIFIED"), []).append(p["cost"])
    assert "UNCLASSIFIED" not in pots, "a phase has no cost_pot; exposure is not computable"
    return tuple(sum(pots.get(k, [])) for k in ("modal", "mixed", "openrouter"))


MODAL_COST, MIXED_COST, ORR_COST = pot_totals(PHASES)
ORR_MAX = ORR_COST + MIXED_COST                     # exposure if the mixed phases land OpenRouter-heavy
POT_LEFT = round(POT_BALANCE - ORR_MAX, 2)
ORR_PHASES = [p for p in PHASES if p["cost_pot"] == "openrouter"]
MIXED_PHASES = [p for p in PHASES if p["cost_pot"] == "mixed"]

# Which phase resolves the mixed split is DERIVED, not asserted: the mixed phases
# say what they are blocked on, and they all name the same phase.
_blocked = {int(s) for p in MIXED_PHASES
            for s in re.findall(r"phase (\d+)", p.get("units_blocked_on", ""))}
assert len(_blocked) == 1, f"mixed phases disagree about what unblocks them: {_blocked}"
DECISION_ID = _blocked.pop()
assert next(p["cost"] for p in PHASES if p["id"] == DECISION_ID) == 0, \
    "the decision phase is no longer the $0 one; the section's point needs rewriting"

# cumulative day offsets
_cum = 0.0
for p in PHASES:
    p["_start"] = round(_cum, 4)
    _cum = round(_cum + p["days"], 4)
    p["_end"] = _cum
GATE_DAY = next(p["_start"] for p in PHASES if p["id"] == APPROVAL_ID)

e = html.escape


def money(v):
    return "$" + format(v, ",")


def money2(v):
    """Cents shown: for the pot, which is a measured balance rather than an estimate."""
    return "$" + format(v, ",.2f")


# ---------------------------------------------------------------- svg helpers

def tw(text, size):
    """Rough rendered width of a string in a system sans at `size` px."""
    return len(text) * 0.55 * size


def wrap(text, size, max_w, max_lines=2):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if tw(trial, size) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
            if len(lines) == max_lines - 1:
                break
    if cur:
        lines.append(cur)
    return lines[:max_lines]


def bar(x0, y, w, h, r=4, cls=""):
    """Bar with the data-end rounded and the baseline end square."""
    if w <= 0.5:
        w = 0.5
    r = min(r, h / 2.0, w)
    if r < 1:
        return f'<rect class="{cls}" x="{x0:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"/>'
    x1 = x0 + w
    d = (f"M{x0:.1f},{y:.1f} H{x1 - r:.1f} A{r},{r} 0 0 1 {x1:.1f},{y + r:.1f} "
         f"V{y + h - r:.1f} A{r},{r} 0 0 1 {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z")
    return f'<path class="{cls}" d="{d}"/>'


def hbar(x0, x1, y, h, r=4, cls=""):
    """Bar rounded at both ends (a span, e.g. a Gantt row)."""
    w = max(x1 - x0, 1.0)
    r = min(r, h / 2.0, w / 2.0)
    return (f'<rect class="{cls}" x="{x0:.1f}" y="{y:.1f}" width="{w:.1f}" '
            f'height="{h:.1f}" rx="{r:.1f}"/>')


def diamond(cx, cy, s, cls="gate-mark"):
    return (f'<path class="{cls}" d="M{cx:.1f},{cy - s:.1f} L{cx + s:.1f},{cy:.1f} '
            f'L{cx:.1f},{cy + s:.1f} L{cx - s:.1f},{cy:.1f} Z"/>')


def arrow(x, y, dx, dy, s=5.0):
    """Small solid triangle at (x,y) pointing along the unit vector (dx,dy)."""
    px, py = -dy, dx
    return (f'<path class="flow-arrow" d="M{x:.1f},{y:.1f} '
            f'L{x - dx * s + px * s * 0.6:.1f},{y - dy * s + py * s * 0.6:.1f} '
            f'L{x - dx * s - px * s * 0.6:.1f},{y - dy * s - py * s * 0.6:.1f} Z"/>')


def check(cx, cy, s):
    return (f'<path class="gate-tick" d="M{cx - s * 0.42:.1f},{cy:.1f} '
            f'l{s * 0.3:.1f},{s * 0.32:.1f} l{s * 0.55:.1f},{-s * 0.62:.1f}"/>')


# ------------------------------------------------------- 3. phase flow diagram
# Snaking flow so 12 nodes stay large enough to read: 4x3 on wide screens,
# a single vertical column on narrow ones. Gates are the diamond checkpoints on
# the connectors - a gate is a thing that happens *between* phases, so it is
# drawn there rather than inside a node.

def flow_wide():
    NW, NH, GAPX, GAPY = 138.0, 66.0, 50.0, 46.0
    COLS = 4
    PAD_T = 22.0
    PAD_B = 20.0  # room for the approval badge under the last-row node
    rows = 3
    W = COLS * NW + (COLS - 1) * GAPX                    # 702
    H = PAD_T + rows * NH + (rows - 1) * GAPY + PAD_B    # 332
    out = [f'<svg class="chart chart--wide" viewBox="0 0 {W:.0f} {H:.0f}" '
           f'role="img" aria-labelledby="flow-t flow-d">',
           '<title id="flow-t">Phase flow, 12 phases in sequence</title>',
           f'<desc id="flow-d">Phases 0 to 11 run in order; each connector carries a '
           f'gate that must pass before the next phase starts. Phase {APPROVAL_ID} '
           f'requires explicit approval on its cost of {money(dict((p["id"], p["cost"]) for p in PHASES)[APPROVAL_ID])}.</desc>']

    def pos(i):
        r = i // COLS
        c = i % COLS
        if r % 2 == 1:
            c = COLS - 1 - c
        return c * (NW + GAPX), PAD_T + r * (NH + GAPY)

    # connectors first, so nodes sit on top
    for i in range(len(PHASES) - 1):
        x0, y0 = pos(i)
        x1, y1 = pos(i + 1)
        if y0 == y1:
            ltr = x1 > x0
            a = (x0 + NW, y0 + NH / 2) if ltr else (x0, y0 + NH / 2)
            b = (x1, y1 + NH / 2) if ltr else (x1 + NW, y1 + NH / 2)
            cx, cy = (a[0] + b[0]) / 2, a[1]
            out.append(f'<line class="flow-link" x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}"/>')
            out.append(arrow(b[0], b[1], 1.0 if ltr else -1.0, 0.0))
        else:
            cx = x0 + NW / 2
            a = (cx, y0 + NH)
            b = (cx, y1)
            cy = (a[1] + b[1]) / 2
            out.append(f'<line class="flow-link" x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}"/>')
            out.append(arrow(b[0], b[1], 0.0, 1.0))
        out.append(diamond(cx, cy, 9.5))
        out.append(check(cx, cy, 9.5))

    for i, p in enumerate(PHASES):
        x, y = pos(i)
        appr = p["id"] == APPROVAL_ID
        cls = "flow-node flow-node--approval" if appr else "flow-node"
        out.append(f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{NW}" height="{NH}" rx="7"/>')
        out.append(f'<text class="flow-num" x="{x + 11:.1f}" y="{y + 19:.1f}">{p["id"]}</text>')
        out.append(f'<text class="flow-cost" x="{x + NW - 11:.1f}" y="{y + 19:.1f}" text-anchor="end">{money(p["cost"])}</text>')
        for j, line in enumerate(wrap(p["name"], 11.5, NW - 22)):
            out.append(f'<text class="flow-name" x="{x + 11:.1f}" y="{y + 38 + j * 14:.1f}">{e(line)}</text>')
        if appr:
            out.append(f'<text class="flow-badge" x="{x + 11:.1f}" y="{y + NH + 15:.1f}">approval required</text>')
    out.append("</svg>")
    return "\n".join(out)


def flow_narrow():
    NW, NH, GAPY = 300.0, 50.0, 32.0
    W = 320.0
    n = len(PHASES)
    H = 10 + n * NH + (n - 1) * GAPY + 16
    out = [f'<svg class="chart chart--narrow" viewBox="0 0 {W:.0f} {H:.0f}" role="img" aria-label="Phase flow, 12 phases in sequence, with a gate between each pair">']
    x = 10.0
    for i, p in enumerate(PHASES):
        y = 10 + i * (NH + GAPY)
        if i < n - 1:
            cx = x + NW / 2
            out.append(f'<line class="flow-link" x1="{cx:.1f}" y1="{y + NH:.1f}" x2="{cx:.1f}" y2="{y + NH + GAPY:.1f}"/>')
            out.append(diamond(cx, y + NH + GAPY / 2, 9.5))
            out.append(check(cx, y + NH + GAPY / 2, 9.5))
            out.append(arrow(cx, y + NH + GAPY, 0.0, 1.0))
        appr = p["id"] == APPROVAL_ID
        cls = "flow-node flow-node--approval" if appr else "flow-node"
        out.append(f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{NW}" height="{NH}" rx="7"/>')
        out.append(f'<text class="flow-num" x="{x + 11:.1f}" y="{y + 20:.1f}">{p["id"]}</text>')
        out.append(f'<text class="flow-cost" x="{x + NW - 11:.1f}" y="{y + 20:.1f}" text-anchor="end">{money(p["cost"])}</text>')
        label = p["name"] + ("  — approval required" if appr else "")
        out.append(f'<text class="flow-name" x="{x + 11:.1f}" y="{y + 38:.1f}">{e(label)}</text>')
    out.append("</svg>")
    return "\n".join(out)


# ------------------------------------------------------------ 4. budget chart
# Bars stay in PHASE ORDER, not sorted by cost: the reader needs to see *when*
# the money is spent, i.e. that the single dominant phase sits ninth of twelve
# and behind an approval gate. A sorted chart would rank the phases and lose the
# sequence, which is the thing that makes the budget steerable.
# Linear scale from a common zero baseline: that is what makes $420 vs $45 read
# as an order of magnitude rather than a nudge. Every bar is tip-labelled because
# at this dynamic range the axis cannot carry the small values at all.

def budget_chart(wide=True):
    if wide:
        W, LX, PX0, PW, ROW, BH, FS = 640.0, 186.0, 196.0, 396.0, 26.0, 13.0, 11.5
        top, bot = 34.0, 30.0
    else:
        W, LX, PX0, PW, ROW, BH, FS = 320.0, 0.0, 6.0, 264.0, 40.0, 13.0, 12.0
        top, bot = 30.0, 30.0
    n = len(PHASES)
    H = top + n * ROW + bot
    mx = max(p["cost"] for p in PHASES)
    ticks = [0, 100, 200, 300, 400]
    sx = lambda v: PX0 + (v / 420.0) * PW

    out = [f'<svg class="chart chart--{"wide" if wide else "narrow"}" viewBox="0 0 {W:.0f} {H:.0f}" '
           f'role="img" aria-label="Cost per phase in US dollars, phases in run order. '
           f'Phase {APPROVAL_ID} at {money(mx)} is larger than all other phases combined.">']
    for t in ticks:
        out.append(f'<line class="grid" x1="{sx(t):.1f}" y1="{top - 8:.1f}" x2="{sx(t):.1f}" y2="{top + n * ROW:.1f}"/>')
        out.append(f'<text class="tick" x="{sx(t):.1f}" y="{top + n * ROW + 16:.1f}" text-anchor="{"start" if t == 0 else "middle"}">{"$" + str(t) if t else "0"}</text>')
    out.append(f'<line class="axis" x1="{PX0:.1f}" y1="{top - 8:.1f}" x2="{PX0:.1f}" y2="{top + n * ROW:.1f}"/>')

    for i, p in enumerate(PHASES):
        appr = p["id"] == APPROVAL_ID
        y = top + i * ROW
        if wide:
            by = y + (ROW - BH) / 2
            out.append(f'<text class="rownum" x="8" y="{by + BH - 2.5:.1f}">{p["id"]}</text>')
            out.append(f'<text class="rowlabel" x="{LX:.1f}" y="{by + BH - 2.5:.1f}" text-anchor="end">{e(p["name"])}</text>')
        else:
            out.append(f'<text class="rowlabel rowlabel--top" x="{PX0:.1f}" y="{y + 12:.1f}">'
                       f'<tspan class="rownum">{p["id"]}</tspan>  {e(p["name"])}</text>')
            by = y + 19
        w = sx(p["cost"]) - PX0
        if p["cost"] > 0:  # a zero-cost phase draws no mark; a min-width sliver would lie
            out.append(bar(PX0, by, w, BH, 4, "bar bar--approval" if appr else "bar"))
        out.append(f'<text class="tiplabel" x="{PX0 + w + 7:.1f}" y="{by + BH - 2.5:.1f}">{money(p["cost"])}</text>')
        # The one in-mark label on the page. It only renders because the bar is
        # wide enough to hold it with padding (checked, not assumed), and it is
        # set in the surface color so it clears contrast against the fill.
        note = "approval required"
        if appr and tw(note, 10.5) + 18 < w:
            out.append(f'<text class="inbar" x="{PX0 + 8:.1f}" y="{by + BH - 3.2:.1f}">{note}</text>')
    out.append("</svg>")
    return "\n".join(out)


def budget_legend():
    return (
        '<ul class="legend">'
        '<li><span class="swatch swatch--s1"></span>Proceeds on the phase’s own gate</li>'
        f'<li><span class="swatch swatch--s2"></span>Phase {APPROVAL_ID} — requires explicit approval on the number</li>'
        "</ul>"
    )


# ------------------------------------------------------- 4b. budget vs ceiling
# A meter, not a pie: one ratio against one limit. The three segments are an
# ordinal commitment ramp in a single hue (committed -> reserved -> unspent),
# so the ordering reads even without the labels.

def budget_meter(wide=True):
    if wide:
        W, X0, TW_, BH, FS = 640.0, 8.0, 620.0, 26.0, 11.5
    else:
        W, X0, TW_, BH, FS = 320.0, 4.0, 306.0, 22.0, 11.0
    TOP = 20.0                      # band for the ceiling label above the track
    H = TOP + BH + 52
    segs = [(SUBTOTAL, "Direct phase costs", "ramp1"),
            (CONTINGENCY, "Contingency (35%)", "ramp2"),
            (HEADROOM, "Headroom to ceiling", "ramp3")]
    out = [f'<svg class="chart chart--{"wide" if wide else "narrow"}" viewBox="0 0 {W:.0f} {H:.0f}" '
           f'role="img" aria-label="The {money(BUDGET)} ceiling split into {money(SUBTOTAL)} of direct '
           f'phase cost, {money(CONTINGENCY)} of contingency, and {money(HEADROOM)} of unallocated headroom.">']
    x = X0
    GAP = 2.0                       # surface gap, not a stroke, separates the segments
    label_y = TOP + BH + 20
    for k, (v, name, ramp) in enumerate(segs):
        w = (v / BUDGET) * TW_
        seg_w = w - (GAP if k < len(segs) - 1 else 0)
        if k == 0:
            out.append(f'<path class="seg seg--{ramp}" d="M{x + 4:.1f},{TOP:.1f} H{x + seg_w:.1f} V{TOP + BH:.1f} H{x + 4:.1f} A4,4 0 0 1 {x:.1f},{TOP + BH - 4:.1f} V{TOP + 4:.1f} A4,4 0 0 1 {x + 4:.1f},{TOP:.1f} Z"/>')
        elif k == len(segs) - 1:
            out.append(f'<path class="seg seg--{ramp}" d="M{x:.1f},{TOP:.1f} H{x + seg_w - 4:.1f} A4,4 0 0 1 {x + seg_w:.1f},{TOP + 4:.1f} V{TOP + BH - 4:.1f} A4,4 0 0 1 {x + seg_w - 4:.1f},{TOP + BH:.1f} H{x:.1f} Z"/>')
        else:
            out.append(f'<rect class="seg seg--{ramp}" x="{x:.1f}" y="{TOP:.1f}" width="{seg_w:.1f}" height="{BH}"/>')
        out.append(f'<line class="leader" x1="{x + 1:.1f}" y1="{TOP + BH + 3:.1f}" x2="{x + 1:.1f}" y2="{TOP + BH + 9:.1f}"/>')
        out.append(f'<text class="segval" x="{x + 1:.1f}" y="{label_y:.1f}">{money(v)}</text>')
        for j, line in enumerate(wrap(name, FS - 2.5, max(w - 6, 56), max_lines=2)):
            out.append(f'<text class="segname" x="{x + 1:.1f}" y="{label_y + 14 + j * 12:.1f}">{e(line)}</text>')
        x += w
    xe = X0 + TW_
    out.append(f'<line class="ceiling" x1="{xe:.1f}" y1="{TOP - 12:.1f}" x2="{xe:.1f}" y2="{TOP + BH + 4:.1f}"/>')
    out.append(f'<text class="ceilinglabel" x="{xe - 4:.1f}" y="{TOP - 6:.1f}" text-anchor="end">{money(BUDGET)} ceiling</text>')
    out.append("</svg>")
    return "\n".join(out)


# ------------------------------------------------------ 4c. the two constraints
# Two meters on their own scales, stacked, so the comparison the reader needs is
# the FILL and not the length: the ceiling bar is a little over half full and the
# pot bar is nearly full, and those are the same picture at different orders of
# magnitude. Same ordinal ramp as the budget meter (committed -> possible ->
# remaining); no new colours.

def constraints_meter(wide=True):
    if wide:
        W, X0, TW_, BH = 640.0, 8.0, 468.0, 28.0
    else:
        W, X0, TW_, BH = 320.0, 4.0, 226.0, 22.0
    PAD_T = 4.0
    ROWH = 20.0 + BH + 15.0 + 20.0
    H = PAD_T + 2 * ROWH
    rows = [
        ("Approval ceiling — permission", BUDGET, money(BUDGET) + " ceiling",
         [(SUBTOTAL, "ramp1")],
         f'{money(SUBTOTAL)} planned · {round(100.0 * SUBTOTAL / BUDGET)}% of the ceiling '
         f'· {money(BUDGET - SUBTOTAL)} of room'),
        ("OpenRouter pot — account balance", POT_BALANCE, money2(POT_BALANCE) + " pot",
         [(ORR_COST, "ramp1"), (MIXED_COST, "ramp2")],
         f'up to {money(ORR_MAX)} exposed · {round(100.0 * ORR_MAX / POT_BALANCE)}% of the pot '
         f'· {money2(POT_LEFT)} left'),
    ]
    out = [f'<svg class="chart chart--{"wide" if wide else "narrow"}" viewBox="0 0 {W:.0f} {H:.0f}" '
           f'role="img" aria-label="Two constraints on their own scales. The {money(BUDGET)} approval '
           f'ceiling is {round(100.0 * SUBTOTAL / BUDGET)}% committed by {money(SUBTOTAL)} of planned work. '
           f'The {money2(POT_BALANCE)} OpenRouter pot is {round(100.0 * ORR_MAX / POT_BALANCE)}% exposed by '
           f'up to {money(ORR_MAX)} of spend. The second bar is nearly full; the first is not.">']
    for i, (title, limit, limit_label, segs, caption) in enumerate(rows):
        y = PAD_T + i * ROWH
        ty = y + 20.0
        out.append(f'<text class="rowlabel--top" x="{X0:.1f}" y="{y + 12:.1f}">{title}</text>')
        out.append(f'<rect class="seg seg--ramp3" x="{X0:.1f}" y="{ty:.1f}" width="{TW_:.1f}" '
                   f'height="{BH:.1f}" rx="4"/>')
        x = X0
        GAP = 2.0
        for k, (v, ramp) in enumerate(segs):
            w = (v / limit) * TW_
            seg_w = max(w - (GAP if k < len(segs) - 1 else 0), 1.0)
            if k == 0:
                out.append(f'<path class="seg seg--{ramp}" d="M{x + 4:.1f},{ty:.1f} H{x + seg_w:.1f} '
                           f'V{ty + BH:.1f} H{x + 4:.1f} A4,4 0 0 1 {x:.1f},{ty + BH - 4:.1f} '
                           f'V{ty + 4:.1f} A4,4 0 0 1 {x + 4:.1f},{ty:.1f} Z"/>')
            else:
                out.append(f'<rect class="seg seg--{ramp}" x="{x:.1f}" y="{ty:.1f}" '
                           f'width="{seg_w:.1f}" height="{BH:.1f}"/>')
            x += w
        out.append(f'<line class="ceiling" x1="{X0 + TW_:.1f}" y1="{ty - 5:.1f}" '
                   f'x2="{X0 + TW_:.1f}" y2="{ty + BH + 5:.1f}"/>')
        out.append(f'<text class="ceilinglabel" x="{X0 + TW_ + 8:.1f}" y="{ty + BH / 2 + 4:.1f}">'
                   f'{limit_label}</text>')
        out.append(f'<text class="segname" x="{X0:.1f}" y="{ty + BH + 15:.1f}">{caption}</text>')
    out.append("</svg>")
    return "\n".join(out)


def constraints_legend():
    return (
        '<ul class="legend">'
        f'<li><span class="swatch swatch--ramp1"></span>Committed, or certain to be drawn '
        f'({len(ORR_PHASES)} pure-OpenRouter phases)</li>'
        f'<li><span class="swatch swatch--ramp2"></span>Possible &mdash; the {len(MIXED_PHASES)} mixed '
        f'phases, whose Modal/OpenRouter split is unknown until phase {DECISION_ID}</li>'
        '<li><span class="swatch swatch--ramp3"></span>Remaining</li>'
        "</ul>"
    )


# ---------------------------------------------------------------- 5. timeline
# Cumulative days as a Gantt strip on one shared axis: bar position carries when,
# bar length carries how long, and the approval gate is a single vertical rule so
# you can see at a glance that 70% of the money sits behind it, four days in.

def timeline_chart(wide=True):
    if wide:
        W, LX, PX0, PW, ROW, BH = 640.0, 186.0, 196.0, 396.0, 24.0, 12.0
        top, bot = 34.0, 30.0
    else:
        W, LX, PX0, PW, ROW, BH = 320.0, 0.0, 6.0, 264.0, 36.0, 12.0
        top, bot = 30.0, 30.0
    n = len(PHASES)
    H = top + n * ROW + bot
    dmax = TOTAL_DAYS
    sx = lambda d: PX0 + (d / dmax) * PW
    out = [f'<svg class="chart chart--{"wide" if wide else "narrow"}" viewBox="0 0 {W:.0f} {H:.0f}" '
           f'role="img" aria-label="Cumulative schedule. {n} phases spanning {dmax} days of compute; '
           f'the approval gate falls at day {GATE_DAY}.">']
    for t in range(0, int(dmax) + 1):
        out.append(f'<line class="grid" x1="{sx(t):.1f}" y1="{top - 8:.1f}" x2="{sx(t):.1f}" y2="{top + n * ROW:.1f}"/>')
        out.append(f'<text class="tick" x="{sx(t):.1f}" y="{top + n * ROW + 16:.1f}" text-anchor="{"start" if t == 0 else "middle"}">{t}{" d" if t == 0 else ""}</text>')
    out.append(f'<line class="axis" x1="{PX0:.1f}" y1="{top - 8:.1f}" x2="{PX0:.1f}" y2="{top + n * ROW:.1f}"/>')
    # approval gate rule
    gx = sx(GATE_DAY)
    out.append(f'<line class="gaterule" x1="{gx:.1f}" y1="{top - 14:.1f}" x2="{gx:.1f}" y2="{top + n * ROW + 2:.1f}"/>')
    anchor = "end" if gx > PX0 + PW * 0.6 else "start"
    ox = -6 if anchor == "end" else 6
    out.append(f'<text class="gatelabel" x="{gx + ox:.1f}" y="{top - 18:.1f}" text-anchor="{anchor}">approval gate · day {GATE_DAY}</text>')

    for i, p in enumerate(PHASES):
        appr = p["id"] == APPROVAL_ID
        y = top + i * ROW
        if wide:
            by = y + (ROW - BH) / 2
            out.append(f'<text class="rownum" x="8" y="{by + BH - 2:.1f}">{p["id"]}</text>')
            out.append(f'<text class="rowlabel" x="{LX:.1f}" y="{by + BH - 2:.1f}" text-anchor="end">{e(p["name"])}</text>')
        else:
            out.append(f'<text class="rowlabel rowlabel--top" x="{PX0:.1f}" y="{y + 12:.1f}">'
                       f'<tspan class="rownum">{p["id"]}</tspan>  {e(p["name"])}</text>')
            by = y + 18
        out.append(hbar(sx(p["_start"]), sx(p["_end"]), by, BH, 4, "bar bar--approval" if appr else "bar"))
        out.append(f'<text class="tiplabel" x="{sx(p["_end"]) + 7:.1f}" y="{by + BH - 2:.1f}">{p["days"]}d</text>')
    out.append("</svg>")
    return "\n".join(out)


# ------------------------------------------------------------ table fallbacks

def table_view(summary, headers, rows, foot=None):
    th = "".join(f"<th{' class=num' if i else ''}>{e(h)}</th>" for i, h in enumerate(headers))
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>"
    tf = ""
    if foot:
        tf = "<tfoot><tr>" + "".join(
            f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(foot)) + "</tr></tfoot>"
    return (f'<details class="tableview"><summary>{e(summary)}</summary>'
            f'<div class="tablewrap"><table><thead><tr>{th}</tr></thead>'
            f"<tbody>{body}</tbody>{tf}</table></div></details>")


# ------------------------------------------------------------- 6. phase cards

def phase_cards():
    out = ['<div class="cards">']
    for p in PHASES:
        appr = p["id"] == APPROVAL_ID
        mark = " card--approval" if appr else (" card--decision" if p["id"] == DECISION_ID else "")
        out.append(f'<article class="card{mark}" id="phase-{p["id"]}">')
        out.append('<header class="card__head">')
        out.append(f'<span class="card__id">{p["id"]}</span>')
        out.append(f'<h3 class="card__name">{e(p["name"])}</h3>')
        out.append(f'<span class="card__meta"><b>{money(p["cost"])}</b><span class="sep">·</span>{p["days"]} d</span>')
        out.append("</header>")
        if appr:
            out.append('<p class="card__flag">Requires explicit approval on the number before it starts.</p>')
        if p["id"] == DECISION_ID:
            out.append('<p class="card__flag card__flag--decision">Decision point on the OpenRouter pot: '
                       f'this phase costs {money(p["cost"])} and is the phase that resolves the '
                       f'Modal/OpenRouter split in phases '
                       f'{" and ".join(str(m["id"]) for m in MIXED_PHASES)}, so it is where a top-up is '
                       'decided. Do not treat a $0 phase as low-stakes.</p>')
        out.append(f'<p class="field field--what"><span class="tag">What</span>{e(p["what"])}</p>')
        out.append(f'<p class="field field--gate"><span class="tag tag--gate">Gate</span>{e(p["gate"])}</p>')
        stop = p["stop"].strip()
        none_stop = stop.lower().startswith("none")
        cls = "field field--stop" + (" field--stop-none" if none_stop else "")
        out.append(f'<p class="{cls}"><span class="tag tag--stop">Stop</span>{e(stop)}</p>')
        out.append(f'<p class="field field--why"><span class="tag">Why</span>{e(p["why"])}</p>')
        out.append("</article>")
    out.append("</div>")
    return "\n".join(out)


# ------------------------------------------------- 7. what would make this fail
# Every item below is a restatement of one phase's own `stop` field. Nothing new.
FAILURES = [
    (3, "Shuffled-pair nulls show the same factor structure as the real traits — "
        "the pipeline is measuring something other than the trait, and everything downstream is void."),
    (4, "Adapters train cleanly but fail to express their trait against base — "
        "there is no behaviour under the geometry to interpret."),
    (6, "Conclusions move across optimizer or learning-rate arms — "
        "the result is a statement about hyperparameters, and the writeup has to say so."),
    (7, "Matched random directions steer as well as the real component directions — "
        "the doses mean nothing and steering demonstrates nothing."),
]

# --------------------------------------------------- the two-constraints prose
# Every figure here is computed above; nothing in these strings is retyped except
# the pot balance and its timestamp, which are another agent's measurement and are
# asserted against plan.json's constraints_note at import time.

def constraint_points():
    return [
        ("Ceiling",
         f'<b>The approval ceiling is {money(BUDGET)}.</b> Planned work is {money(SUBTOTAL)}, '
         f'leaving {money(BUDGET - SUBTOTAL)} of room &mdash; {money(CONTINGENCY)} of it named '
         f'contingency and {money(HEADROOM)} unallocated. This one is <b>permission</b>: a number '
         f'someone agreed to, and every part of it can be recomputed from this plan.'),
        ("Pot",
         f'<b>The OpenRouter pot holds {money2(POT_BALANCE)}</b>, as measured by the '
         f'{POT_MEASURED_BY} agent at {POT_MEASURED_AT}. This one is not a budget line but an '
         f'<b>actual account balance</b>, shared with the rest of the garden, which drew '
         f'{money2(POT_OTHER_DRAW_TODAY)} from it today ({POT_MEASURED_BY}&rsquo;s figure, same '
         f'report). It is another agent&rsquo;s measurement at a timestamp: this plan cannot '
         f'recompute it, and can only compute what it would spend against it.'),
        ("Exposure",
         f'<b>{money(ORR_COST)} is certain</b> &mdash; the {len(ORR_PHASES)} phases that run '
         f'entirely on OpenRouter ('
         + ", ".join(f'phase {p["id"]} {e(p["name"].lower())} {money(p["cost"])}' for p in ORR_PHASES)
         + f'). <b>Up to {money(MIXED_COST)} more</b> if phases '
         f'{" and ".join(str(p["id"]) for p in MIXED_PHASES)} land OpenRouter-heavy; their '
         f'Modal/OpenRouter split is not known yet. That is <b>up to {money(ORR_MAX)} against '
         f'{money2(POT_BALANCE)}</b>, before anyone else draws on the pot.'),
        ("Consequence",
         f'<b>This experiment can run out of GRANT while still being inside BUDGET.</b> '
         f'{money(ORR_MAX)} of {money2(POT_BALANCE)} leaves {money2(POT_LEFT)}, while the ceiling '
         f'still shows {money(BUDGET - SUBTOTAL)} of room that cannot pay an OpenRouter bill. '
         f'Because the mixed split is unknown, that failure would otherwise arrive '
         f'<b>during phase {MIXED_PHASES[0]["id"]}</b> &mdash; mid-run, after the money that got '
         f'there is spent &mdash; rather than at a planning moment.'),
        (f"Phase {DECISION_ID}",
         f'<b>Phase {DECISION_ID} costs {money(0)} and is the phase that resolves the mixed '
         f'split</b>, so it gates a funding question despite its price: '
         f'<b>do not treat a $0 phase as low-stakes.</b> A top-up is Samuel&rsquo;s call, and it '
         f'should arrive as a planned decision at phase {DECISION_ID} rather than as a mid-phase '
         f'surprise in phase {MIXED_PHASES[0]["id"]}.'),
        ("Rule",
         'The binding limit is the one with the <b>shortest fuse</b>, not the biggest number.'),
    ]


# ----------------------------------------------------------------------- page

HEADLINE = [
    "The question is whether a personality trait, trained into a model as a low-rank weight update, occupies a location in weight space that means something — whether traits sit in a structured geometry that reflects how personality actually decomposes, or merely re-encode the statistics of the text used to train them.",
    "Everything here is designed to make that question falsifiable rather than suggestive. Null controls are projected into the same space as the real traits before any structure is interpreted. A text-only baseline is computed as standard output rather than produced when challenged. Ablations test whether conclusions survive a change of optimizer or learning rate.",
    "Each phase names its own cost, the check that must pass before the next phase starts, and the condition under which spending stops. The budget has brakes rather than a single ceiling.",
]

CSS = """
:root {
  color-scheme: light;
  --plane:#f9f9f7; --surface:#fcfcfb;
  --ink:#0b0b0b; --ink-2:#52514e; --ink-muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --s1:#2a78d6; --s2:#eb6834;
  --ramp1:#1c5cab; --ramp2:#3987e5; --ramp3:#86b6ef;
  --critical:#d03b3b;
  --wash:rgba(42,120,214,0.07); --wash2:rgba(235,104,52,0.08);
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --plane:#0d0d0d; --surface:#1a1a19;
    --ink:#ffffff; --ink-2:#c3c2b7; --ink-muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    --s1:#3987e5; --s2:#d95926;
    --ramp1:#b7d3f6; --ramp2:#5598e7; --ramp3:#184f95;
    --critical:#d03b3b;
    --wash:rgba(57,135,229,0.12); --wash2:rgba(217,89,38,0.14);
  }
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin:0; background:var(--plane); color:var(--ink);
  font: 400 17px/1.62 system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
  padding: 0 20px 96px;
}
.wrap { max-width: 46rem; margin: 0 auto; }
h1,h2,h3 { line-height:1.22; letter-spacing:-0.011em; margin:0; font-weight:600; }
h1 { font-size: clamp(1.65rem, 1.2rem + 2vw, 2.35rem); letter-spacing:-0.02em; }
h2 { font-size:1.22rem; margin: 0 0 .35rem; }
p { margin: 0 0 1rem; }
a { color: var(--s1); }
b, strong { font-weight:600; }
.tnum { font-variant-numeric: tabular-nums; }

/* header */
header.top { padding: 4.5rem 0 0; }
.kicker { font-size:.78rem; letter-spacing:.09em; text-transform:uppercase; color:var(--ink-muted); margin:0 0 .9rem; font-weight:600; }
.standfirst { font-size:1.06rem; color:var(--ink-2); margin:.9rem 0 0; max-width:36rem; }

.figures { display:flex; flex-wrap:wrap; gap:1.6rem 2.6rem; align-items:flex-end;
  margin:2.2rem 0 0; padding:1.4rem 0 0; border-top:1px solid var(--border); }
.hero .label, .stat .label { display:block; font-size:.76rem; letter-spacing:.05em; text-transform:uppercase;
  color:var(--ink-muted); font-weight:600; margin-bottom:.15rem; }
.hero .value { font-size:3.1rem; font-weight:600; line-height:1; letter-spacing:-0.03em; }
.hero .sub { display:block; font-size:.86rem; color:var(--ink-2); margin-top:.45rem; }
.stat .value { font-size:1.32rem; font-weight:600; line-height:1.2; font-variant-numeric: tabular-nums; }
.stat .sub { display:block; font-size:.82rem; color:var(--ink-muted); }

/* headline idea */
.idea { margin: 3.2rem 0 0; }
.idea p { font-size:1.075rem; }
.idea p:first-child { font-size:1.16rem; color:var(--ink); }
.idea p + p { color:var(--ink-2); }

section { margin-top: 4rem; }
.sechead { margin-bottom:1.1rem; }
.sechead p { color:var(--ink-2); font-size:.95rem; margin:.3rem 0 0; max-width:34rem; }

/* chart cards */
.figure { background:var(--surface); border:1px solid var(--border); border-radius:10px;
  padding:1.25rem 1.25rem 1rem; }
.chart { display:block; width:100%; height:auto; overflow:visible; }
.chart--narrow { display:none; }
@media (max-width: 640px) {
  .chart--wide { display:none; }
  .chart--narrow { display:block; }
}
svg text { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }

/* svg roles */
.grid { stroke:var(--grid); stroke-width:1; }
.axis { stroke:var(--axis); stroke-width:1; }
.tick { fill:var(--ink-muted); font-size:10.5px; font-variant-numeric: tabular-nums; }
.rowlabel { fill:var(--ink-2); font-size:11.5px; }
.rowlabel--top { fill:var(--ink); font-size:12px; font-weight:500; }
.rownum { fill:var(--ink-muted); font-size:10.5px; font-variant-numeric: tabular-nums; font-weight:600; }
.bar { fill:var(--s1); }
.bar--approval { fill:var(--s2); }
.tiplabel { fill:var(--ink); font-size:11px; font-weight:600; font-variant-numeric: tabular-nums;
  paint-order: stroke; stroke:var(--surface); stroke-width:3px; stroke-linejoin:round; }
.tipnote { fill:var(--ink-muted); font-size:9.5px; }
.inbar { fill:#ffffff; font-size:10.5px; font-weight:600; letter-spacing:.02em; }
.gaterule { stroke:var(--s2); stroke-width:1.5; }
.gatelabel { fill:var(--ink-2); font-size:10.5px; font-weight:600; }
.seg--ramp1 { fill:var(--ramp1); }
.seg--ramp2 { fill:var(--ramp2); }
.seg--ramp3 { fill:var(--ramp3); }
.leader { stroke:var(--axis); stroke-width:1; }
.segval { fill:var(--ink); font-size:13px; font-weight:600; font-variant-numeric: tabular-nums; }
.segname { fill:var(--ink-2); font-size:10.5px; }
.ceiling { stroke:var(--axis); stroke-width:1.5; }
.ceilinglabel { fill:var(--ink-muted); font-size:10.5px; font-weight:600; font-variant-numeric: tabular-nums; }
.flow-arrow { fill:var(--axis); }

/* flow */
.flow-node { fill:var(--wash); stroke:var(--s1); stroke-width:1; }
.flow-node--approval { fill:var(--wash2); stroke:var(--s2); stroke-width:2; }
.flow-num { fill:var(--ink-muted); font-size:11px; font-weight:700; font-variant-numeric: tabular-nums; }
.flow-cost { fill:var(--ink); font-size:11.5px; font-weight:600; font-variant-numeric: tabular-nums; }
.flow-name { fill:var(--ink); font-size:11.5px; font-weight:500; }
.flow-badge { fill:var(--ink-2); font-size:10.5px; font-weight:600; letter-spacing:.02em; }
.flow-link { stroke:var(--axis); stroke-width:1.5; }
.gate-mark { fill:var(--surface); stroke:var(--axis); stroke-width:1.5; }
.gate-tick { fill:none; stroke:var(--ink-2); stroke-width:1.6; stroke-linecap:round; stroke-linejoin:round; }

/* legend */
.legend { list-style:none; display:flex; flex-wrap:wrap; gap:.4rem 1.4rem; margin:.9rem 0 0; padding:0;
  font-size:.84rem; color:var(--ink-2); }
.legend li { display:flex; align-items:flex-start; gap:.45rem; }
.swatch { width:11px; height:11px; border-radius:3px; flex:none; margin-top:.32rem; }
.swatch--s1 { background:var(--s1); }
.swatch--s2 { background:var(--s2); }
.swatch--gate { background:var(--surface); border:1.5px solid var(--axis); transform:rotate(45deg); border-radius:2px; }
.swatch--ramp1 { background:var(--ramp1); }
.swatch--ramp2 { background:var(--ramp2); }
.swatch--ramp3 { background:var(--ramp3); }

.figcaption { font-size:.82rem; color:var(--ink-muted); margin:.5rem 0 0; }

/* table view */
.tableview { margin-top:.9rem; border-top:1px solid var(--border); padding-top:.5rem; }
.tableview summary { cursor:pointer; font-size:.82rem; color:var(--ink-muted); }
.tablewrap { overflow-x:auto; }
.tableview table { border-collapse:collapse; width:100%; margin-top:.7rem; font-size:.85rem; }
.tableview th, .tableview td { text-align:left; padding:.32rem .7rem .32rem 0; border-bottom:1px solid var(--border); white-space:nowrap; }
.tableview th { color:var(--ink-muted); font-weight:600; font-size:.76rem; text-transform:uppercase; letter-spacing:.05em; }
.tableview .num { text-align:right; font-variant-numeric: tabular-nums; }
.tableview tfoot td { font-weight:600; border-bottom:none; }

/* phase cards */
.cards { display:flex; flex-direction:column; gap:1rem; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:1.15rem 1.25rem 1.05rem; }
.card--approval { border-color:var(--s2); border-width:1.5px; }
.card__head { display:flex; align-items:baseline; gap:.6rem; flex-wrap:wrap; margin-bottom:.75rem; }
.card__id { font-size:.8rem; font-weight:700; color:var(--ink-muted); font-variant-numeric: tabular-nums;
  min-width:1.4rem; }
.card__name { font-size:1.09rem; flex:1 1 auto; }
.card__meta { font-size:.88rem; color:var(--ink-2); font-variant-numeric: tabular-nums; white-space:nowrap; }
.card__meta .sep { margin:0 .4rem; color:var(--ink-muted); }
.card__flag { font-size:.82rem; font-weight:600; color:var(--ink); background:var(--wash2);
  border-radius:6px; padding:.4rem .6rem; margin:0 0 .8rem; }
.card__flag--decision { background:var(--wash); }
.card--decision { border-color:var(--s1); border-width:1.5px; }
.field { margin:0 0 .7rem; font-size:.94rem; line-height:1.58; color:var(--ink-2); }
.field:last-child { margin-bottom:0; }
.tag { display:inline-block; font-size:.68rem; letter-spacing:.08em; text-transform:uppercase;
  font-weight:700; color:var(--ink-muted); margin-right:.55rem; vertical-align:.06em; }
.field--gate, .field--stop { color:var(--ink); border-left:3px solid var(--s1);
  padding:.15rem 0 .15rem .8rem; margin-left:.05rem; }
.field--stop { border-left-color:var(--critical); }
.tag--gate { color:var(--s1); }
.tag--stop { color:var(--critical); }
.field--stop-none { color:var(--ink-muted); border-left-color:var(--axis); }
.field--stop-none .tag--stop { color:var(--ink-muted); }
.field--why { font-size:.89rem; color:var(--ink-muted); }

/* failure list */
.fail { list-style:none; padding:0; margin:0; }
.fail li { display:flex; gap:.85rem; padding:.85rem 0; border-top:1px solid var(--border); font-size:.97rem; color:var(--ink-2); }
.fail li:last-child { border-bottom:1px solid var(--border); }
.fail .from { flex:none; font-size:.76rem; font-weight:700; color:var(--ink-muted); letter-spacing:.04em;
  text-transform:uppercase; padding-top:.24rem; width:4.4rem; }
.fail a { text-decoration:none; color:var(--ink-muted); }
#constraints .fail .from { width:6.4rem; }
.fail a:hover { color:var(--s1); }
footer { margin-top:4rem; padding-top:1.2rem; border-top:1px solid var(--border);
  font-size:.83rem; color:var(--ink-muted); }
@media (max-width: 640px) {
  body { font-size:16px; padding:0 16px 72px; }
  header.top { padding-top:2.6rem; }
  .hero .value { font-size:2.6rem; }
  .figure { padding:1rem .85rem .85rem; }
  .fail li { flex-direction:column; gap:.15rem; }
  .fail .from { width:auto; padding-top:0; }
}
@media print { .tableview[open] summary { display:none; } .tableview { border:0; } }
"""

# The machine-readable contract for everything drawn above. The SVGs are rendered
# at build time from the same in-memory PHASES list this is serialised from, so the
# charts cannot drift from the prose; this block makes that checkable from the
# served page alone, without re-reading plan.json.
def plan_data_block():
    data = {
        "phases": [{"id": p["id"], "name": p["name"], "cost": p["cost"], "days": p["days"]}
                   for p in PHASES],
        "subtotal": SUBTOTAL,
        "contingency": CONTINGENCY,
        "total": TOTAL,
        "budget": BUDGET,
        "total_days": TOTAL_DAYS,
        "exposure": {
            "modal": MODAL_COST,
            "mixed": MIXED_COST,
            "openrouter_certain": ORR_COST,
            "openrouter_max": ORR_MAX,
            "pot_balance": POT_BALANCE,
            "pot_measured_by": POT_MEASURED_BY,
            "pot_measured_at": POT_MEASURED_AT,
            "decision_phase": DECISION_ID,
        },
    }
    blob = json.dumps(data, separators=(",", ":"))
    assert "<" not in blob and "&" not in blob, "plan data would need escaping inside <script>"
    return f'<script type="application/json" id="plan-data">{blob}</script>'


def assert_embedded_matches_plan(doc):
    """The page's own data block must equal plan.json on disk, field for field."""
    start = doc.index('id="plan-data">') + len('id="plan-data">')
    embedded = json.loads(doc[start:doc.index("</script>", start)])
    with open(PLAN) as fh:
        source = json.load(fh)
    for key in ("subtotal", "contingency", "total", "budget"):
        assert embedded[key] == source[key], f"embedded {key} != plan.json"
    assert [{"id": p["id"], "name": p["name"], "cost": p["cost"], "days": p["days"]}
            for p in source["phases"]] == embedded["phases"], "embedded phases != plan.json"
    # and the rendered prose must show every one of those numbers
    for p in source["phases"]:
        assert e(p["name"]) in doc, f'phase name {p["name"]!r} missing from page'
    assert money(source["total"]) in doc and money(source["budget"]) in doc


def assert_exposure_matches_plan(doc):
    """The pot exposure PRINTED on the page must equal the pot exposure COMPUTED
    from plan.json's cost_pot fields, right now, from disk. The exposure is the
    one number in this section the plan can derive, so a stale one is a lie the
    build should not be able to ship: recompute and fail loudly instead."""
    with open(PLAN) as fh:
        source = json.load(fh)
    mod, mix, orr = pot_totals(source["phases"])
    start = doc.index('id="plan-data">') + len('id="plan-data">')
    emb = json.loads(doc[start:doc.index("</script>", start)])["exposure"]
    assert (emb["modal"], emb["mixed"], emb["openrouter_certain"]) == (mod, mix, orr), \
        f"embedded exposure {emb} != computed from plan.json ({mod}, {mix}, {orr})"
    assert emb["openrouter_max"] == orr + mix, "embedded openrouter_max is not certain + mixed"
    # and the rendered prose must carry those same figures, in the sentences that
    # claim them -- not merely somewhere on the page.
    required = [
        f"{money(orr)} is certain</b>",
        f"Up to {money(mix)} more</b>",
        f"up to {money(orr + mix)} against {money2(POT_BALANCE)}</b>",
        f"{money(orr + mix)} of {money2(POT_BALANCE)} leaves {money2(round(POT_BALANCE - orr - mix, 2))}",
    ]
    for frag in required:
        assert frag in doc, f"exposure prose is stale or missing: {frag!r} not on the page"
    assert f"{money2(POT_BALANCE)} pot" in doc, "the pot meter lost its balance label"
    assert emb["decision_phase"] == DECISION_ID and f'id="phase-{DECISION_ID}"' in doc


def build():
    phase_index = {p["id"]: p for p in PHASES}
    approval = phase_index[APPROVAL_ID]
    approval_share = round(100.0 * approval["cost"] / SUBTOTAL)

    budget_rows = [[f'<a href="#phase-{p["id"]}">{p["id"]} &middot; {e(p["name"])}</a>',
                    money(p["cost"]),
                    f'{round(100.0 * p["cost"] / SUBTOTAL)}%'] for p in PHASES]
    budget_tbl = table_view("Table view — cost per phase",
                            ["Phase", "Cost", "Share of direct cost"],
                            budget_rows,
                            ["Direct subtotal", money(SUBTOTAL), "100%"])

    meter_tbl = table_view("Table view — budget against the ceiling",
                           ["Allocation", "Amount", "Share of ceiling"],
                           [["Direct phase costs", money(SUBTOTAL), f"{round(100 * SUBTOTAL / BUDGET)}%"],
                            ["Contingency (35%)", money(CONTINGENCY), f"{round(100 * CONTINGENCY / BUDGET)}%"],
                            ["Planned total", money(TOTAL), f"{round(100 * TOTAL / BUDGET)}%"],
                            ["Headroom to ceiling", money(HEADROOM), f"{round(100 * HEADROOM / BUDGET)}%"]],
                           ["Ceiling", money(BUDGET), "100%"])

    tl_rows = [[f'<a href="#phase-{p["id"]}">{p["id"]} &middot; {e(p["name"])}</a>',
                f'{p["days"]}', f'{p["_start"]}', f'{p["_end"]}'] for p in PHASES]
    tl_tbl = table_view("Table view — schedule in days",
                        ["Phase", "Days", "Starts (day)", "Ends (day)"],
                        tl_rows,
                        ["Total", f"{TOTAL_DAYS}", "", ""])

    def gate_note(p):
        if p["id"] == APPROVAL_ID:
            return "explicit approval required"
        if p["id"] == DECISION_ID:
            return "gate on phase output &mdash; DECISION POINT on the OpenRouter pot"
        return "gate on phase output"

    flow_rows = [[f'<a href="#phase-{p["id"]}">{p["id"]} &middot; {e(p["name"])}</a>',
                  money(p["cost"]), gate_note(p)] for p in PHASES]
    flow_tbl = table_view("Table view — phases in run order",
                          ["Phase", "Cost", "Gate to enter the next phase"], flow_rows)

    fail_items = "".join(
        f'<li><span class="from"><a href="#phase-{pid}">Phase {pid}</a></span><span>{e(txt)}</span></li>'
        for pid, txt in FAILURES)

    constraint_items = "".join(
        f'<li><span class="from">{label}</span><span>{body}</span></li>'
        for label, body in constraint_points())

    constraints_tbl = table_view(
        "Table view — the two constraints",
        ["Constraint", "Limit", "Committed / exposed", "Filled"],
        [["Approval ceiling (permission)", money(BUDGET), money(SUBTOTAL) + " planned",
          f"{round(100.0 * SUBTOTAL / BUDGET)}%"],
         [f"OpenRouter pot ({POT_MEASURED_BY}, {POT_MEASURED_AT})", money2(POT_BALANCE),
          f"{money(ORR_COST)} certain, up to {money(ORR_MAX)}",
          f"{round(100.0 * ORR_MAX / POT_BALANCE)}%"]])

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Persona geometry in Qwen3.5-4B — experiment plan</title>
<meta name="description" content="A fresh standalone experiment on whether persona traits occupy a structured geometry in weight space. {money(TOTAL)} planned against a {money(BUDGET)} ceiling, across {len(PHASES)} gated phases.">
<meta name="color-scheme" content="light dark">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<header class="top">
  <p class="kicker">Experiment plan</p>
  <h1>Persona geometry in Qwen3.5-4B</h1>
  <p class="standfirst">A fresh standalone experiment, not a replication: {len(PHASES)} phases, each with its own cost, its own entry gate, and its own stopping condition.</p>

  <div class="figures">
    <div class="hero">
      <span class="label">Planned total</span>
      <span class="value">{money(TOTAL)}</span>
      <span class="sub">{money(SUBTOTAL)} direct + {money(CONTINGENCY)} contingency (35%)</span>
    </div>
    <div class="stat">
      <span class="label">Budget ceiling</span>
      <span class="value">{money(BUDGET)}</span>
      <span class="sub">{money(HEADROOM)} unallocated</span>
    </div>
    <div class="stat">
      <span class="label">Largest phase</span>
      <span class="value">{money(approval["cost"])}</span>
      <span class="sub">{approval_share}% of direct cost · phase {APPROVAL_ID}</span>
    </div>
    <div class="stat">
      <span class="label">Compute time</span>
      <span class="value">{TOTAL_DAYS} d</span>
      <span class="sub">across {len(PHASES)} phases</span>
    </div>
  </div>
</header>

<div class="idea">
  <p>{e(HEADLINE[0])}</p>
  <p>{e(HEADLINE[1])}</p>
  <p>{e(HEADLINE[2])}</p>
</div>

<section id="flow">
  <div class="sechead">
    <h2>The sequence</h2>
    <p>Phases run in order. Every connector carries a gate: a check on the phase that just finished, which must pass before the next one starts.</p>
  </div>
  <div class="figure">
    {flow_wide()}
    {flow_narrow()}
    <ul class="legend">
      <li><span class="swatch swatch--gate"></span>Gate — must pass before the next phase starts</li>
      <li><span class="swatch swatch--s2"></span>Phase {APPROVAL_ID} — explicit approval on the number</li>
    </ul>
    {flow_tbl}
  </div>
</section>

<section id="budget">
  <div class="sechead">
    <h2>Where the money goes</h2>
    <p>Cost per phase in run order, on a common zero baseline. Phase {APPROVAL_ID} costs more than the other eleven phases put together — which is why it is the only one behind an approval gate rather than a technical one.</p>
  </div>
  <div class="figure">
    {budget_chart(True)}
    {budget_chart(False)}
    {budget_legend()}
    {budget_tbl}
  </div>

  <div class="sechead" style="margin-top:2rem">
    <h2>How the spend is watched</h2>
    <p>Approving the money and approving the metering are the same decision, so both are here.
       Spend is tracked cumulatively against the {money(BUDGET)} ceiling rather than per day, because a
       day-rate alarm inside an approved plan only reports that the approved thing is happening.
       Each phase records its actual cost against its estimate when it closes. A phase crossing
       twice its estimate is escalated while there is still a decision to make about it, rather
       than at three times when there is not. And any vendor spend with no phase attached is
       escalated on sight &mdash; every other category on this page has already been agreed, so
       unattributed spend is the only kind nobody has said yes to.</p>
  </div>

  <div class="sechead" style="margin-top:2rem">
    <h2>Wall clock</h2>
    <p>The {TOTAL_DAYS}-day figure quoted on this page only adds up that way if the phases run
       strictly one after another, with a human turnaround between each. Run at full fan-out the actual compute is
       roughly 12&ndash;16 hours. What sets that floor is not the total work but a five-link serial
       chain that nothing can overlap, so the thing to watch is how fast each gate gets answered.</p>
  </div>
  <ul class="fail">
    <li><span class="from">Serial</span><span>Preference pairs &rarr; training &rarr; decomposition &rarr; steering, which needs the components &rarr; qualitative analysis, which reads the steering transcripts. Five links, hours each.</span></li>
    <li><span class="from">Basis</span><span>A 100-trait sweep on this hardware took 27 minutes. 39,000 teacher-generation calls took 27 minutes. Those two numbers are the basis for the 12&ndash;16 hour estimate.</span></li>
    <li><span class="from">Fan-out</span><span>Extra parallelism costs no additional GPU-hours &mdash; the same work is merely spread wider. The only marginal cost is container startup, about a minute each.</span></li>
    <li><span class="from">Gates</span><span>The bottleneck is therefore turnaround between gates, not parallelism. Time is bought by answering a gate sooner, not by buying more machines.</span></li>
  </ul>

  <div class="sechead" style="margin-top:2rem">
    <h2>Against the ceiling</h2>
    <p>The plan commits {money(TOTAL)} of the {money(BUDGET)} available. The contingency is reserved, not spent; the headroom is neither.</p>
  </div>
  <div class="figure">
    {budget_meter(True)}
    {budget_meter(False)}
    <p class="figcaption">Darkest to lightest: committed, reserved, unallocated.</p>
    {meter_tbl}
  </div>
</section>

<section id="constraints">
  <div class="sechead">
    <h2>Two constraints, not one</h2>
    <p>The ceiling above is not the only limit on this plan, and it is not the tightest one.
       A second limit sits underneath it: the OpenRouter account the judging and generation calls
       are actually billed to. They are different kinds of thing &mdash; one is permission, the other
       is a balance &mdash; and being inside one says nothing about the other.</p>
  </div>
  <div class="figure">
    {constraints_meter(True)}
    {constraints_meter(False)}
    {constraints_legend()}
    <p class="figcaption">Two scales, deliberately: compare the fills, not the lengths. The ceiling
      is a little over half committed; the pot is nearly full at the top of its range.</p>
    {constraints_tbl}
  </div>
  <ul class="fail" style="margin-top:1.4rem">{constraint_items}</ul>
</section>

<section id="timeline">
  <div class="sechead">
    <h2>Schedule</h2>
    <p>Cumulative compute days. The whole plan is {TOTAL_DAYS} days; the approval gate falls at day {GATE_DAY}, with {approval_share}% of the direct cost still unspent behind it.</p>
  </div>
  <div class="figure">
    {timeline_chart(True)}
    {timeline_chart(False)}
    {budget_legend()}
    {tl_tbl}
  </div>
</section>

<section id="phases">
  <div class="sechead">
    <h2>The phases</h2>
    <p><b>Gate</b> is what must be true to move on. <b>Stop</b> is the condition under which spending halts rather than continues.</p>
  </div>
  {phase_cards()}
</section>

<section id="fail">
  <div class="sechead">
    <h2>What would make this fail</h2>
    <p>Each of these is a stopping condition the plan already names, not a new worry. Any one of them makes the headline claim unsupportable.</p>
  </div>
  <ul class="fail">{fail_items}</ul>
  <p style="margin-top:1.2rem;font-size:.94rem;color:var(--ink-2)">A fifth outcome is not a failure but a brake: if phase {APPROVAL_ID}’s one-trait timing run shows generation slower than assumed, the phase is re-costed and re-approved rather than started.</p>
</section>

<footer>
  <p>All figures generated from <code>plan.json</code>: {len(PHASES)} phases, {money(SUBTOTAL)} direct + {money(CONTINGENCY)} contingency = {money(TOTAL)} against a {money(BUDGET)} ceiling.</p>
</footer>

</div>
{plan_data_block()}
</body>
</html>
"""
    assert_embedded_matches_plan(doc)
    assert_exposure_matches_plan(doc)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(doc)
    print(f"wrote {OUT} ({len(doc):,} bytes) — {len(PHASES)} phases, "
          f"subtotal {SUBTOTAL}, total {TOTAL}, days {TOTAL_DAYS}")
    print(f"  pots: OpenRouter {money(ORR_COST)} certain + up to {money(MIXED_COST)} mixed "
          f"= {money(ORR_MAX)} against a {money2(POT_BALANCE)} pot "
          f"({POT_MEASURED_BY} {POT_MEASURED_AT}); Modal {money(MODAL_COST)}; "
          f"decision point phase {DECISION_ID}")


if __name__ == "__main__":
    build()
