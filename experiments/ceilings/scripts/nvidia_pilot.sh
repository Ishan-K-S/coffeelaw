#!/usr/bin/env bash
# One pilot phase for one model, run inside the container started by scripts/nvidia_env.sh.
# A crashed run (exit 1, e.g. an outage longer than the retries) is retried after a pause; runs
# resume (team runs by content-hash ID, calibration by cached settings). Exit 0 and exit 2 (a
# failed calibration gate, which is a finding) end the loop, as does exit 4 (request cap).
#
#   scripts/nvidia_pilot.sh <model> <ceilings subcommand and options...>
set -uo pipefail
model="$1"
shift
ROOT="${ROOT:-data/pilot}"
for attempt in 1 2 3 4 5 6; do
  echo "=== $(date -u +%FT%TZ) attempt $attempt: ceilings $* --model $model"
  ceilings --root "$ROOT" "$@" --model "$model"
  code=$?
  echo "=== $(date -u +%FT%TZ) exit $code"
  case $code in
    0|2|3|4) exit $code ;;
  esac
  sleep 300
done
exit 1
