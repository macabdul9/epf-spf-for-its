#!/usr/bin/env bash
# Copy this repo to Babel.
#
#   tuning/sync_to_babel.sh                 copy
#   DRY_RUN=1 tuning/sync_to_babel.sh       list what would be copied; changes nothing
#
# rsync: re-running sends only what changed, and never deletes on Babel.
set -euo pipefail

BABEL_HOST="${BABEL_HOST:-abdulw@login.babel.cs.cmu.edu}"
BABEL_REPO="${BABEL_REPO:-/home/abdulw/epf-spf-its-reimplementation}"
DRY_RUN="${DRY_RUN:-0}"

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Babel logs in by password. Offering keys first makes it disconnect with
# "Too many authentication failures", so keys are not offered at all.
ssh_cmd="ssh -o PubkeyAuthentication=no"

rsync_flags=(-a --human-readable --info=stats1,progress2)
if [[ "$DRY_RUN" == 1 ]]; then
  rsync_flags+=(--dry-run --itemize-changes)
fi

echo "Babel: $BABEL_HOST:$BABEL_REPO"
rsync "${rsync_flags[@]}" \
  -e "$ssh_cmd" \
  --exclude=__pycache__ \
  --exclude=.venv \
  --exclude='*.egg-info' \
  --exclude=.pytest_cache \
  --exclude=.ruff_cache \
  --exclude=runs/ \
  "$repo/" \
  "$BABEL_HOST:$BABEL_REPO/"

if [[ "$DRY_RUN" == 1 ]]; then
  echo "Dry run: nothing was copied."
fi
