#!/usr/bin/env bash
# PHASE 3 NULL-CONTROL ANALYSIS: Gram + decomposition for the two corrupted-data
# arms (shuffled, permuted).  This is the ANALYSIS side; launch_nulls.sh is the
# TRAINING side.  Nothing in this file trains anything.
#
# WHAT THE ARMS MEAN, because the PASS/FAIL only makes sense with it stated:
#   shuffled  -- the preference direction inside each pair is destroyed, so the
#                adapter is trained on the same text with no consistent
#                chosen/rejected signal.
#   permuted  -- the trait->data correspondence is broken, so `extraverted`
#                is trained on some other trait's (intact) pairs.
# Either arm reproducing the real signed factor separation would mean the
# geometry we measured is a property of the training pipeline and not of the
# trait, and the phase-5 result would be void.  That is the stop condition.
#
# ONE MODAL APP PER ARM.  The app name must carry a leading-digit phase number
# immediately after the word "phase" -- `phase3` -- because a sibling agent's
# billing parser reads the leading digits after "phase" out of the app name to
# attribute cost.  An app called `pc-qwen35-gram-shuffled` or
# `pc-qwen35-phaseIII-...` bills as an unknown phase and the arm's cost silently
# lands in nobody's ledger; a hardcoded name is why every phase once billed as
# phase 2.
set -euo pipefail
cd "$(dirname "$0")"

PY=/home/vibe12/cartovenv/bin/python
MODAL=/home/vibe12/cartovenv/bin/modal
export PC_ADAPTER_VOLUME=pc-qwen35-sweep

# THE TWO FILES THAT HOLD THE REAL PHASE-5 RESULT.  They are ~$55 of GPU work
# that cannot be regenerated without re-spending it, and both are the DEFAULT
# output path of the two tools this script calls: gram_on_modal.py writes
# results/gram_sweep.npz when PC_ADAPTER_SUBDIR is empty, and decompose.py
# writes results/decomposition.json when --out is omitted.  So the failure mode
# is not exotic -- it is one forgotten environment variable.  We fingerprint
# them before and after and abort if either moved.
REAL_NPZ=results/gram_sweep.npz
REAL_JSON=results/decomposition.json

fingerprint() {  # fingerprint <file> -> sha256, or the literal word ABSENT
  if [ -e "$1" ]; then sha256sum "$1" | cut -d' ' -f1; else echo ABSENT; fi
}
REAL_NPZ_BEFORE=$(fingerprint "$REAL_NPZ")
REAL_JSON_BEFORE=$(fingerprint "$REAL_JSON")

assert_real_untouched() {  # called after every step, not only at the end,
                           # so a clobber is caught before the NEXT arm runs
                           # and overwrites the evidence of the first one.
  local now_npz now_json
  now_npz=$(fingerprint "$REAL_NPZ")
  now_json=$(fingerprint "$REAL_JSON")
  if [ "$now_npz" != "$REAL_NPZ_BEFORE" ]; then
    echo "FATAL: $REAL_NPZ CHANGED during the null run ($REAL_NPZ_BEFORE -> $now_npz)." >&2
    echo "       A null arm has overwritten the real Gram. Restore it before continuing." >&2
    exit 70
  fi
  if [ "$now_json" != "$REAL_JSON_BEFORE" ]; then
    echo "FATAL: $REAL_JSON CHANGED during the null run ($REAL_JSON_BEFORE -> $now_json)." >&2
    echo "       A null arm has overwritten the real decomposition. Restore it before continuing." >&2
    exit 70
  fi
}

arm() {  # arm <label> <volume-subdir>
  local label="$1" subdir="$2"
  # Derived here, not typed, and derived the same way gram_on_modal.py derives
  # it (results/gram_<SUBDIR>.npz).  A retyped path that drifts from the tool's
  # actual output would make decompose.py read a STALE npz from an earlier arm
  # and report the wrong arm's numbers under this arm's name -- a wrong result
  # that looks completely well-formed.
  local npz="results/gram_${subdir}.npz"
  local out="results/decomposition_${label}.json"

  # A null arm writing to either real-result path is a bug in this script, not
  # a runtime condition; refusing loudly beats discovering it from a diff.
  if [ "$npz" = "$REAL_NPZ" ] || [ "$out" = "$REAL_JSON" ]; then
    echo "FATAL: arm $label resolves to a real-result path ($npz / $out)" >&2
    exit 64
  fi

  echo "=== $label : Gram over /adapters/${subdir} on volume ${PC_ADAPTER_VOLUME} ==="
  echo "    npz  -> $npz"
  echo "    json -> $out"

  PC_ADAPTER_SUBDIR="$subdir" \
  PC_APP_NAME="pc-qwen35-phase3-gram-${label}" \
    "$MODAL" run gram_on_modal.py 2>&1 | tee "phase3_gram_${label}.log"
  assert_real_untouched

  if [ ! -e "$npz" ]; then
    echo "FATAL: $npz was not written; not running decompose.py on a stale or absent Gram." >&2
    exit 71
  fi

  # --out is MANDATORY here.  Its default is results/decomposition.json, i.e.
  # the real result.
  #
  # --runmeta is ALSO mandatory, and this is the subtler of the two.  The null
  # adapters carry the SAME trait names as the real sweep, so the default
  # results/runmeta_sweep.json would match every one of them by name and hand
  # TEST 6 the REAL sweep's reward margins as if they were this arm's.  The
  # nuisance-covariate test would then be run on a covariate from a different
  # experiment and report a confidently wrong number.  Pointing at a per-arm
  # file that does not exist makes decompose.py print TEST 6 UNAVAILABLE, which
  # is the honest state: untested, not ruled out.
  #
  # Both label files are passed so the null decomposition is scored by exactly
  # the same labels as the real one.  These arms cover the 100 PRIMARY traits
  # only, so the 40 secondary labels simply find no adapter and the held-out
  # batch probe (TEST 5) reports "no held-out set" -- correct, not a failure.
  "$PY" decompose.py \
      --npz "$npz" \
      --labels traits_primary.json \
      --labels traits_secondary.json \
      --runmeta "results/runmeta_${label}.json" \
      --out "$out" 2>&1 | tee "phase3_decompose_${label}.log"
  assert_real_untouched
  echo "=== $label done ==="
}

case "${1:-all}" in
  shuffled) arm shuffled data_null_shuffled_p100 ;;
  permuted) arm permuted data_null_permuted_p100 ;;
  all)      arm shuffled data_null_shuffled_p100
            arm permuted data_null_permuted_p100 ;;
  *) echo "usage: $0 {shuffled|permuted|all}" >&2; exit 64 ;;
esac

echo
echo "real result verified untouched:"
echo "  $REAL_NPZ  $REAL_NPZ_BEFORE"
echo "  $REAL_JSON $REAL_JSON_BEFORE"
echo "next: $MODAL run cross_gram_on_modal.py   (seed-paired noise floor)"
echo "then: $PY compare_nulls.py"
