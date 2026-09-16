#!/bin/bash
# One-line experiment status. Emits ONLY when something has changed since the
# last emit, so a heartbeat that has nothing to say stays silent rather than
# training the reader to ignore it.
Q=/home/vibe12/projects/persona-curvature/qwen35
STATE=$Q/.last_status
now=""
now+="traits:$([ -f $Q/traits_secondary.json ] && echo ok || echo pending) "
now+="consts:$(~/cartovenv/bin/python -c "
import json,os
p='$Q/constitutions.json'
print(sum(1 for v in json.load(open(p)).values() if 'constitution' in v) if os.path.exists(p) else 0)" 2>/dev/null) "
now+="pairs:$(ls $Q/data/*.jsonl 2>/dev/null | wc -l) "
now+="adapters:$(ls $Q/adapters 2>/dev/null | wc -l) "
now+="running:$(ps -eo cmd | grep -cE '[m]odal run|[g]en_pairs|[t]rain') "
# modal_WORKSPACE, not modal_mine. The Modal workspace is SHARED: on 2026-08-19
# it read $0.0179 while this project had used no GPU at all -- another agent's
# probe. The old field name invited a future reader to attribute it to this
# experiment. It is renamed rather than removed because knowing the pot moved is
# still useful; claiming it is not.
#
# UNREADABLE IS NOT ZERO. The old version piped grep into bc, and on an empty
# match bc exits 0 with no output, so the `|| echo '?'` never fired and the field
# rendered BLANK -- a format change, a CLI error or an expired credential would
# all have reported an empty figure that reads as nothing-spent. Verified: empty
# input yields field=[] , not '?'. The discriminator is the table's Cost HEADER,
# which Modal prints even with no rows: header present and no rows is genuinely
# zero; header absent means the report could not be read, which is a different
# fact and must say so.
now+="modal_workspace_today:$(
  rep=$(cd /home/vibe12/projects/persona-curvature && timeout 60 ~/cartovenv/bin/modal billing report --for today 2>/dev/null)
  if ! printf '%s' "$rep" | grep -q 'Cost'; then
    echo 'UNREADABLE'
  else
    amt=$(printf '%s' "$rep" | grep -oE '[0-9]+\.[0-9]{4,}' | paste -sd+ | bc 2>/dev/null)
    echo "${amt:-0}"
  fi)"
prev=$(cat $STATE 2>/dev/null)
if [ "$now" != "$prev" ]; then
  echo "$now" > $STATE
  echo "QWEN35 STATUS CHANGED: $now"
fi
exit 0
