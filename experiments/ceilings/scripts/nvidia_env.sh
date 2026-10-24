#!/usr/bin/env bash
# Run a command in a capped ceilings:dev container against NVIDIA build's free endpoints.
# The key is read from KEY_FILE (outside the repository) into the environment and passed to
# Docker by name only (-e NAME), so it never appears in a command line, a log or a file here.
#
#   scripts/nvidia_env.sh <container-name> <command...>
#   e.g. scripts/nvidia_env.sh doc_lightning python scripts/doctor.py --model openai/<name>
#
# DETACH=1 runs the container in the background; RESTART=1 also restarts it automatically.
# CPUS / MEMORY set the caps (default 1 / 1g).
set -euo pipefail
cd "$(dirname "$0")/.."
KEY_FILE="${KEY_FILE:-$HOME/.nvidia_api_key}"
CEILINGS_OPENAI_API_KEY="$(tr -d ' \r\n' < "$KEY_FILE")"
export CEILINGS_OPENAI_API_KEY
export CEILINGS_OPENAI_BASE_URL="https://integrate.api.nvidia.com/v1"
export CEILINGS_OPENAI_PROFILES="${CEILINGS_OPENAI_PROFILES:-configs/nvidia_build.json}"
# The free tier allows about 40 requests per minute per key; split it across the containers
# that run at the same time, with headroom.
export CEILINGS_OPENAI_RPM="${CEILINGS_OPENAI_RPM:-4}"
export CEILINGS_OPENAI_RETRIES="${CEILINGS_OPENAI_RETRIES:-10}"
export CEILINGS_TEMPERATURE="${CEILINGS_TEMPERATURE:-0.7}"
export PYTHONUNBUFFERED=1
name="$1"
shift
envs=()
for v in CEILINGS_OPENAI_API_KEY CEILINGS_OPENAI_BASE_URL CEILINGS_OPENAI_PROFILES \
         CEILINGS_OPENAI_RPM CEILINGS_OPENAI_RETRIES CEILINGS_TEMPERATURE PYTHONUNBUFFERED \
         CEILINGS_REASONING_TOKENS; do
  [ -n "${!v:-}" ] && envs+=(-e "$v")
done
mode=(--rm)
[ "${DETACH:-0}" = 1 ] && mode=(-d --rm)
# RESTART=1: detached and restarted by Docker after a crash, a Docker restart or the Mac waking up
# (every phase resumes from its cache).
[ "${RESTART:-0}" = 1 ] && mode=(-d --restart unless-stopped)
exec docker run "${mode[@]}" --name "$name" --cpus "${CPUS:-1}" --memory "${MEMORY:-1g}" \
  "${envs[@]}" -v "$PWD":/work -w /work ceilings:dev "$@"
