#!/bin/bash
# Spend meter that does NOT depend on Modal billing (which lags many hours and
# read $0.00 all through the 08-25 overnight run while ~$410 was being spent).
# Instead: integrate ACTIVE CONTAINER-MINUTES, which is observable in real time.
MODAL=/home/vibe12/cartovenv/bin/modal
STATE=/home/vibe12/projects/persona-curvature/qwen35/phase10_runs/zoo40_meter.state
LOG=/home/vibe12/projects/persona-curvature/qwen35/phase10_runs/zoo40_meter.log
RATE=2.10          # $/GPU-hour, A100-40GB
PRIOR=46.90        # already drawn from the $300 top-up (resume + eval), per this meter
BUDGET=2718.00     # raised 2026-09-11 by exactly $30.00 for the EMERGENT-MISALIGNMENT
                   # MEDICAL run (zoo-em-*.service): three SFT arms on
                   # ModelOrganismsForEM bad/good medical advice and a length-matched
                   # Dolci-Instruct-SFT arm in the zoo's LoRA-A frame, their first-order
                   # data scoring against 173 directions, cross-Grams, column space, a
                   # Big Five battery and the EM paper's eight free-form questions.
                   # $30 of the $40 Samuel authorised for the run; the other $10 is
                   # OpenRouter judging, which this meter does not see.  ON TOP of every
                   # raise below, for the same reason those are on top of each other:
                   # the meter counts every container in the workspace at the A100 rate,
                   # so a sibling's run draws on the same number and one run's cap does
                   # not cover another's.  Previous value 2688.00.
                   # Was 2688.00, raised 2026-09-10 by exactly $35.00 for the SYCOPHANCY-FORECAST run
                   # (zoo-syc-*.service): six DPO arms on Dolci-Instruct-DPO subsets
                   # chosen by their first-order score along align_sycophantic, plus a
                   # three-part sycophancy battery, the Big Five battery, the previous
                   # run's compliance battery for the length-stratified corrigible arm,
                   # and one cross-Gram.  The cap Samuel set for it on 2026-09-10.
                   # Previous value 2653.00.  ON TOP of every raise below, for the same
                   # reason those are on top of each other: the meter counts every
                   # container in the workspace at the A100 rate, so a sibling's run
                   # draws on the same number and one run's cap does not cover another's.
                   # Was 2653.00, raised 2026-09-10 by exactly $35.00 for the DOLCI-FLAG TRAINING run
                   # (zoo-dolciflag-*.service): five DPO arms on Dolci pairs selected by
                   # the corrigible-negative flag, plus their behavioural eval and two
                   # cross-Grams.  Previous value 2618.00, raised 2026-09-09 by exactly
                   # $15.00 for the PROBE-ADAPTERS run
                   # (zoo-probepairs/zoo-probetrain/zoo-probescore.service): three
                   # new data-failure-mode probe traits (overhedging, padding,
                   # false_certainty) trained in the zoo frame and used to score a
                   # Dolci-Instruct-SFT sample, against an LLM classifier.  The cap
                   # Samuel set for it on 2026-09-09.  ON TOP of every raise below,
                   # for the same reason those are on top of each other: the meter
                   # counts every container in the workspace at the A100 rate, so a
                   # sibling's run draws on the same number and one run's cap does
                   # not cover another's.
                   # Was 2603.00, raised 2026-09-09 by exactly $60.00 for the DOLCI DATA-AUDIT run
                   # (zoo-dolci-*.service): scoring allenai/Dolci-Instruct-DPO and
                   # allenai/Dolci-Instruct-SFT samples against the personality and
                   # alignment directions, the cap Samuel set for it on 2026-09-09.
                   # ON TOP of every raise below, for the same reason those are on top
                   # of each other: the meter counts every container in the workspace
                   # at the A100 rate, so a sibling's run draws on the same number and
                   # one run's cap does not cover another's.
                   # Was 2543.00, raised 2026-09-09 by exactly $25.00 for the RANK-SWEEP run
                   # (zoo-ranksweep.service, experiment S3 of the 2026-09-09 paper
                   # reading: 15 traits retrained at lora_r 1, 4 and 16 with
                   # alpha = 2r and the seed-0 LoRA-A truncated to r rows, then
                   # cross-Grams against the 134 rank-64 stage-one adapters and
                   # the rank-64 second-seed set), the cap Samuel set for it on
                   # 2026-09-09.  ON TOP of the caps below for the same reason
                   # those are on top of each other: the meter counts every
                   # container in the workspace at the A100 rate, so a sibling's
                   # run draws on the same number and one run's cap does not
                   # cover another's.
                   # Was 2518.00, raised 2026-09-09 by exactly $15.00 for the FISHER-METRIC FACTOR
                   # ANALYSIS and MATCHED-DOSE STEERING runs (experiments G1 and G5 of
                   # the 2026-09-09 paper reading): zoo-fishergram.service,
                   # zoo-dosecalib.service, zoo-dosesteer.service.  The cap Samuel set
                   # for that task.  ON TOP of the raise below, for the reason every
                   # raise here is on top of the last: the meter counts every container
                   # in the workspace at the A100 rate whatever the app is named, and a
                   # stop fires hard_stop, which kills EVERY pc-qwen35 app.  Was 2503.00,
                   # raised 2026-09-09 by exactly $5.00 for the REWARD-HACKS GRADIENT-ATOMS
                   # run (experiment G3 of the 2026-09-09 paper reading, added to the
                   # gradient-atoms task by Samuel after G2 was scoped): unsupervised
                   # atoms on the 973 matched School of Reward Hacks rows, extracted
                   # through the same EKFAC projection.  Same unit, zoo-gradatoms.service.
                   # Was 2498.00, raised 2026-09-09 by exactly $30.00 for the GRADIENT-ATOMS run
                   # (zoo-gradatoms.service, experiment G2/G4 of the 2026-09-09
                   # paper reading: per-pair DPO gradients w.r.t. LoRA-B on the
                   # zoo's preference corpus and its two null arms, EKFAC
                   # projection, sparse dictionary learning), the cap Samuel set
                   # for it on 2026-09-09.  ON TOP of the $45/$5/$8/$10 below for
                   # the same reason those are on top of each other: the meter
                   # counts every container in the workspace at the A100 rate, so
                   # a sibling's run draws on the same number and one run's cap
                   # does not cover another's.  NOTE this run asks for an
                   # A100-80GB, which Modal bills above the $2.10 A100-40GB RATE
                   # this meter uses, so the true spend runs above the reading.
                   # Was 2468.00, raised 2026-09-09 by exactly $45.00 for the PERSONA-SLIDERS run
                   # (zoo-sliders.service, experiment S1 of the 2026-09-09 paper
                   # reading: SliderSpace-style LoRAs trained to activation-space
                   # persona targets), the cap Samuel set for it on 2026-09-09.
                   # ON TOP of the $5/$8/$10 below for the same reason those are on
                   # top of each other: the meter counts every container in the
                   # workspace at the A100 rate, so a sibling's run draws on the
                   # same number and one run's cap does not cover another's.
                   # Was 2423.00, raised 2026-09-09 by exactly $5.00 for the COLUMN-SPACE run
                   # (zoo-colspace.service), CPU containers only, the cap Samuel
                   # set for it.  ON TOP of the $8 and $10 below, for the same
                   # reason those are on top of each other: the meter counts
                   # every container in the workspace, at the A100 rate, whatever
                   # the app is named, so a CPU run still draws on the budget and
                   # one run's cap does not cover another's.
                   # Was 2418.00, raised 2026-09-09 by exactly $8.00 for the reward-hacks
                   # DATA-SCORING run (zoo-sorh-datascore.service), the cap Samuel set
                   # for it, against the $300 of further spend he authorised on
                   # 2026-09-08.  It is ON TOP of the $10 below because the two runs
                   # overlap: the meter counts every container, so one run's cap does
                   # not cover the other's, and a stop fires hard_stop, which kills
                   # EVERY pc-qwen35 app -- the sibling's included.
                   # Was 2410.00, raised 2026-09-09 by exactly $10 for the Fisher-norm
                   # run (zoo-fisher.service); Samuel authorised $10 on 2026-09-09.
                   # Was 2400.00, raised 2026-08-29 against a $5k grant. Covers the 34
                   # Lexicon validation traits (~$631) + eval (~$35) on top of the
                   # $956 already spent, with margin. Still a hard stop: the meter
                   # caps spend, it does not create funds.
APPPREFIX=pc-qwen35   # widened 2026-08-29: the steering apps run as
                      # pc-qwen35-steer and did NOT match pc-qwen35-phase10,
                      # so `modal run --detach` steering apps sat outside the
                      # kill switch -- stopping the systemd unit does not stop
                      # a detached app.
# Must match EVERY unit that can spend, not just training. When this was
# "zoo-batch*.service" the meter saw eval as "no active service", exited on
# its own completion condition, and spend went untracked while eval ran.
SERVICEGLOB="zoo-*.service"   # every zoo unit; naming a subset has twice let the
                              # meter exit while a sibling was still spending.
INTERVAL=300
[ -f "$STATE" ] || echo "0" > "$STATE"

# Stop spend for real, and VERIFY it stopped. The previous version called
# `modal app stop --name <desc>`, and there is no --name flag: the call failed
# every time, into a log nobody read, while claiming the budget was enforced.
# `app stop` also prompts for confirmation unless given -y, so an un-verified
# stop could hang forever holding the budget open. Ephemeral app IDs change on
# every relaunch, so the ID is resolved at stop time, never hardcoded.
hard_stop() {
  # Stop EVERY batch, not one named service. Batches overlap (batch 3's tail runs
  # while batch 4 starts), so a stop scoped to a single unit would leave the other
  # one spending. Services first, then the detached Modal apps they launched --
  # `modal run --detach` means killing the CLI does not kill the app.
  for svc in $(systemctl list-units --plain --no-legend $SERVICEGLOB 2>/dev/null | awk '{print $1}'); do
    echo "$(date -u +%FT%TZ)  STOP: systemctl stop $svc" >> $LOG
    sudo systemctl stop "$svc" >> $LOG 2>&1
  done
  local ids
  ids=$(timeout 120 $MODAL app list --json 2>/dev/null \
        | python3 -c "import sys,json
try: a=json.load(sys.stdin)
except Exception: sys.exit(0)
for x in a:
    d=x.get('description') or ''
    if d.startswith('$APPPREFIX') and 'stopped' not in (x.get('state') or '') and not x.get('stopped_at'):
        print(x['app_id'])" 2>/dev/null)
  [ -z "$ids" ] && echo "$(date -u +%FT%TZ)  STOP: no live app matching '$APPPREFIX*'" >> $LOG
  for id in $ids; do
    echo "$(date -u +%FT%TZ)  STOP: stopping app $id" >> $LOG
    timeout 180 $MODAL app stop -y "$id" >> $LOG 2>&1 \
      || echo "$(date -u +%FT%TZ)  STOP: FAILED for $id" >> $LOG
  done
  sleep 30
  local left rc
  left=$(timeout 120 $MODAL container list 2>/dev/null); rc=$?
  if [ $rc -ne 0 ]; then
    echo "$(date -u +%FT%TZ)  STOP: cannot verify -- container list unreadable. SPEND MAY CONTINUE." >> $LOG
  else
    left=$(echo "$left" | grep -c "^│ ta-")
    echo "$(date -u +%FT%TZ)  STOP: verified, $left container(s) remaining" >> $LOG
    [ "$left" != "0" ] && echo "$(date -u +%FT%TZ)  STOP: !!! CONTAINERS STILL RUNNING AFTER STOP" >> $LOG
  fi
}

while true; do
  ts=$(date -u +%FT%TZ)
  # Branch on the CLI's exit code, not on whether the output looks empty.
  # `grep -c` prints 0 and exits 1 on no matches, so a dead CLI and a genuine
  # zero were indistinguishable -- the old check could never fire, and an
  # unreadable count was silently billed as free.
  out=$(timeout 120 $MODAL container list 2>/dev/null); rc=$?
  if [ $rc -ne 0 ]; then
    echo "$ts  CONTAINER COUNT UNREADABLE (modal rc=$rc) -- not treated as zero" >> $LOG
  else
    n=$(echo "$out" | grep -c "^│ ta-")
    acc=$(cat "$STATE")
    acc=$(echo "$acc + $n * $INTERVAL / 3600" | bc -l)
    echo "$acc" > "$STATE"
    spend=$(echo "$PRIOR + $acc * $RATE" | bc -l)
    printf "%s  containers=%s  gpu_h_since_relaunch=%.2f  est_total_spend=\$%.2f / \$%.2f\n" \
      "$ts" "$n" "$acc" "$spend" "$BUDGET" >> $LOG
    if [ "$(echo "$spend > $BUDGET" | bc -l)" = "1" ]; then
      echo "$ts  !!! est spend \$$spend exceeds budget \$$BUDGET -- STOPPING" >> $LOG
      hard_stop
      exit 0
    fi
    # finished? no containers AND the orchestrator is gone
    if [ "$n" = "0" ] && [ -z "$(systemctl list-units --plain --no-legend --state=active $SERVICEGLOB 2>/dev/null)" ]; then
      echo "$ts  service inactive and no containers -- meter exiting" >> $LOG; exit 0
    fi
  fi
  sleep $INTERVAL
done
