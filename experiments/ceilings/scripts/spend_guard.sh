#!/usr/bin/env bash
# Per-phase spending guard for the paid run. Every INTERVAL seconds it reads the key's total usage
# from OpenRouter and compares what this phase has spent (usage now - usage at the phase start)
# with the phase's budget. Over budget: it pauses the phase's containers (docker stop; nothing is
# deleted, every phase resumes from its cache on `docker start`), records the trip and exits, so
# whoever watches can check whether something went wrong or the phase is just close to done.
# It also exits when the phase finishes (its done markers all exist).
#
#   scripts/spend_guard.sh <phase> <budget-dollars> <container-prefix> <done-dir> <n-done>
#   e.g. scripts/spend_guard.sh calibration 10 paidcal_ data/paid/state/calibrated 10
#
# The key is read from KEY_FILE and sent only in a request header through stdin.
set -uo pipefail
cd "$(dirname "$0")/.."
phase="$1"; budget="$2"; prefix="$3"; done_dir="$4"; n_done="$5"
KEY_FILE="${KEY_FILE:-$HOME/coffee_law_paid_key.txt}"
INTERVAL="${INTERVAL:-120}"
state="data/paid/state/budget_${phase}.json"
mkdir -p data/paid/state

usage() {
  printf 'Authorization: Bearer %s\n' "$(tr -d ' \r\n' < "$KEY_FILE")" \
    | curl -s --max-time 30 -H @- https://openrouter.ai/api/v1/key \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["usage"])'
}

start=""
if [ -f "$state" ]; then
  start=$(python3 -c "import json; print(json.load(open('$state'))['start_usage'])")
fi
if [ -z "$start" ]; then
  start=$(usage) || { echo "cannot read usage"; exit 1; }
  printf '{"phase": "%s", "budget": %s, "start_usage": %s, "started": "%s"}\n' \
    "$phase" "$budget" "$start" "$(date -u +%FT%TZ)" > "$state"
fi
echo "guard: phase $phase, budget \$$budget, usage at start \$$start"
while true; do
  now=$(usage) || { sleep "$INTERVAL"; continue; }
  spent=$(python3 -c "print(round($now - $start, 4))")
  if [ "$(ls "$done_dir" 2>/dev/null | wc -l | tr -d ' ')" -ge "$n_done" ]; then
    echo "guard: phase $phase finished; spent \$$spent of \$$budget"
    exit 0
  fi
  if python3 -c "import sys; sys.exit(0 if $spent > $budget else 1)"; then
    ids=$(docker ps -q --filter "name=$prefix")
    [ -n "$ids" ] && docker stop $ids >/dev/null
    printf '{"phase": "%s", "budget": %s, "spent": %s, "tripped": "%s"}\n' \
      "$phase" "$budget" "$spent" "$(date -u +%FT%TZ)" > "data/paid/state/budget_${phase}_tripped.json"
    echo "guard: phase $phase PAUSED at \$$spent (budget \$$budget); containers $prefix* stopped"
    exit 3
  fi
  sleep "$INTERVAL"
done
