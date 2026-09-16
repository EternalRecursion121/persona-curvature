#!/bin/bash
# Enforce THIS RUN's own $30 Modal cap, which zoo40_meter.sh does not.
#
# EMERGENT-MISALIGNMENT MEDICAL run, 2026-09-11.  Samuel authorised $40 total;
# $30 of it is Modal and the rest is OpenRouter judging, which the meter cannot
# see.  START is the meter reading immediately before this run's first
# container (2026-09-11T16:52:07Z, est_total_spend=$2564.45).
#
# The meter's BUDGET is cumulative across every raise the project has ever made
# (2718.00), and est_total_spend stood at 2564.45 when this run started, so the
# meter would not hard-stop until 153.55 dollars of this run's spend -- five
# times the authorised cap.  The meter is a workspace-wide backstop, not
# a per-run one.
#
# No sibling job was live at START (the meter read containers=0 at
# 2026-09-11T16:52:07Z), but one may appear, and the meter counts every
# container in the workspace.  Attribution here is the same per-tick split by
# container count that `wiki/pages/behaviour/dolci-flag-training.md` used:
# credit (containers - SIB) container-ticks to this run.  SIB is 0 here because
# no other job is running; the syc guard hard-coded SIB = 1, which with a single
# container attributes NOTHING and read $0.00 against a raw delta of $0.35 when
# this guard first ran.  Raise SIB if a sibling appears.
#
# On breach: stop only this run's own units.  It never touches the sibling.
set -u
Q=/home/vibe12/projects/persona-curvature/qwen35
LOG=$Q/phase10_runs/zoo40_meter.log
START=2564.45
CAP=29.00           # stop a dollar short of the authorised $30 Modal share
MINE="zoo-em-train zoo-em-train1 zoo-em-train2 zoo-em-train3 zoo-em-scoreprobe zoo-em-score zoo-em-flat zoo-em-gram zoo-em-colspace zoo-em-eval zoo-em-probe zoo-em-probescore"

while true; do
  read -r att raw <<<"$(python3 - "$LOG" "$START" <<'PY'
import re, sys
log, start = sys.argv[1], float(sys.argv[2])
RATE, INT, SIB = 2.10, 300/3600.0, 0   # SIB = sibling containers per tick
mine = 0.0; last = start
for line in open(log):
    m = re.match(r'\S+\s+containers=(\d+)\s+gpu_h_since_relaunch=[\d.]+\s+est_total_spend=\$([\d.]+)', line)
    if not m:
        continue
    n, sp = int(m.group(1)), float(m.group(2))
    if sp < start:
        continue
    last = sp
    if n:
        mine += INT * RATE * max(0, n - SIB)
print(f"{mine:.2f} {last-start:.2f}")
PY
)"
  live=""
  for s in $MINE; do
    st="$(systemctl is-active $s.service)"
    # Type=oneshot units report "activating" for their whole life, not "active";
    # the syc guard's "= active" test would have seen no live unit and exited
    # immediately, which is exactly what it did here at 16:53:59Z.
    [ "$st" = "active" ] || [ "$st" = "activating" ] && live="$live $s"
  done
  echo "$(date -u +%FT%TZ) attributed=\$$att raw_meter_delta=\$$raw cap=\$$CAP live:$live"
  if [ -z "$live" ]; then
    echo "$(date -u +%FT%TZ) no units of this run left -- guard exiting"
    exit 0
  fi
  if [ "$(echo "$att > $CAP" | bc -l)" = "1" ]; then
    echo "$(date -u +%FT%TZ) !!! attributed \$$att exceeds \$$CAP -- stopping this run's units"
    for s in $live; do sudo systemctl stop $s.service; done
    # Stopping the unit kills the CLI, not the app: every launch here uses
    # `modal run --detach`, and the em_bad training arm is deliberately orphaned
    # from its unit so the other two could run beside it.  Stop the apps too,
    # by id resolved at stop time, and ONLY the ones this run owns
    # (pc-qwen35-phase13-*) -- never a sibling's.
    ids=$(timeout 120 /home/vibe12/cartovenv/bin/modal app list --json 2>/dev/null \
      | python3 -c "import sys,json
try: a=json.load(sys.stdin)
except Exception: sys.exit(0)
for x in a:
    d=x.get('description') or ''
    if d.startswith('pc-qwen35-phase13') and not x.get('stopped_at'):
        print(x['app_id'])" 2>/dev/null)
    for id in $ids; do
      echo "$(date -u +%FT%TZ) stopping app $id"
      timeout 180 /home/vibe12/cartovenv/bin/modal app stop -y "$id" \
        || echo "$(date -u +%FT%TZ) app stop FAILED for $id"
    done
    exit 1
  fi
  sleep 300
done
