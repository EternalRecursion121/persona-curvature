#!/bin/bash
# Enforce THIS RUN's own $35 cap, which zoo40_meter.sh does not.
#
# The meter's BUDGET is cumulative across every raise the project has ever made
# (2688.00), and est_total_spend stood at 2528.40 when this run started, so the
# meter would not hard-stop until 159.60 dollars of this run's spend -- four and
# a half times the authorised cap.  The meter is a workspace-wide backstop, not
# a per-run one.
#
# It also counts a sibling app's container (`pc-qwen35-phase10-sphere-isokl`)
# that this run does not own, so the raw meter delta overstates what this run
# drew.  Attribution here is the same per-tick split by container count that
# `wiki/pages/behaviour/dolci-flag-training.md` used: assume one sibling
# container per tick and credit the rest to this run.
#
# On breach: stop only this run's own units.  It never touches the sibling.
set -u
Q=/home/vibe12/projects/persona-curvature/qwen35
LOG=$Q/phase10_runs/zoo40_meter.log
START=2528.40
CAP=34.00           # stop a dollar short of the authorised 35
MINE="zoo-sycevala zoo-sycevalb zoo-sycevalc zoo-sycevald zoo-sycgram zoo-syctrain zoo-syctrainrest"

while true; do
  read -r att raw <<<"$(python3 - "$LOG" "$START" <<'PY'
import re, sys
log, start = sys.argv[1], float(sys.argv[2])
RATE, INT = 2.10, 300/3600.0
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
        mine += n * INT * RATE * max(0, n - 1) / n
print(f"{mine:.2f} {last-start:.2f}")
PY
)"
  live=""
  for s in $MINE; do
    [ "$(systemctl is-active $s.service)" = "active" ] && live="$live $s"
  done
  echo "$(date -u +%FT%TZ) attributed=\$$att raw_meter_delta=\$$raw cap=\$$CAP live:$live"
  if [ -z "$live" ]; then
    echo "$(date -u +%FT%TZ) no units of this run left -- guard exiting"
    exit 0
  fi
  if [ "$(echo "$att > $CAP" | bc -l)" = "1" ]; then
    echo "$(date -u +%FT%TZ) !!! attributed \$$att exceeds \$$CAP -- stopping this run's units"
    for s in $live; do sudo systemctl stop $s.service; done
    exit 1
  fi
  sleep 300
done
