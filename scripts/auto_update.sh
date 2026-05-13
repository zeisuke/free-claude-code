#!/usr/bin/env bash
# Auto-update free-claude-code: pull changes, sync deps, restart server if updated.

set -euo pipefail

REPO_DIR="$HOME/HermesAgent/free-claude-code"
PLIST="$HOME/Library/LaunchAgents/ai.hermes.free-proxy.plist"
LOG="$REPO_DIR/update.log"
UV="$HOME/.local/bin/uv"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

cd "$REPO_DIR"

log "--- auto-update start ---"

# Fetch and check for upstream changes
git fetch origin main --quiet
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    log "Already up to date ($LOCAL). Skipping restart."
    exit 0
fi

log "New commits detected: $LOCAL -> $REMOTE"

# Pull changes
git pull --ff-only origin main 2>&1 | tee -a "$LOG"

# Sync dependencies
log "Syncing dependencies..."
"$UV" sync --quiet 2>&1 | tee -a "$LOG"

# Restart server via launchd
log "Restarting ai.hermes.free-proxy..."
launchctl unload "$PLIST" 2>&1 | tee -a "$LOG" || true
sleep 2
launchctl load "$PLIST" 2>&1 | tee -a "$LOG"

sleep 5
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8082/ | grep -q "200\|404\|422"; then
    log "Server is UP on port 8082."
else
    log "WARNING: Server did not respond on port 8082 after restart."
fi

log "--- auto-update done ---"
