#!/bin/bash
# Blackwell tmux session bootstrap — runs ON the host (via attach.sh over ssh).
#
# Creates one detached tmux session with:
#   - window "run"  — plain shell for the owner's first command stream
#   - window "run2" — plain shell for the owner's second command stream
#   - window "mon"  — top pane nvtop, bottom pane htop
#
# Non-interactive: creates the session detached and exits. It never attaches;
# the caller (attach.sh) does that separately.
#
# Idempotent: if $SESSION already exists, this is a no-op. That is the single
# most important property here — it must never destroy a session that already
# has a training run inside it.

set -euo pipefail

# Values arrive as positional arguments from attach.sh (empty means "use the default"), so
# nothing has to be interpolated into a string the remote shell re-parses. Env vars still work
# when this script is run directly on the host.
SESSION="${1:-${SESSION:-qwen27b}}"
REPO_DIR="${2:-${REPO_DIR:-$HOME/Concern-is-All-You-Need}}"
WINDOWS=(run run2 mon)

# Fail fast, before creating anything.
for bin in tmux nvtop htop; do
    if ! command -v "$bin" >/dev/null 2>&1; then
        echo "[ERROR] '$bin' not found. Install it with: sudo apt install -y $bin" >&2
        exit 1
    fi
done

if [ ! -d "$REPO_DIR" ]; then
    echo "[ERROR] REPO_DIR is not a directory: $REPO_DIR" >&2
    echo "        Set it explicitly, e.g. REPO_DIR=~/path/to/repo ./scripts/blackwell/attach.sh" >&2
    exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[INFO] tmux session '$SESSION' already exists — leaving it untouched."
    # An ssh drop mid-bootstrap can leave a session that exists but is missing windows, and
    # the guard above would then skip it forever. Report that rather than repair it: adding
    # windows to a session that may be running training is not worth the risk.
    present="$(tmux list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null || true)"
    for window in "${WINDOWS[@]}"; do
        if ! printf '%s\n' "$present" | grep -qx "$window"; then
            echo "[WARN] window '$window' is missing from '$SESSION'." >&2
            echo "       Add it by hand, or kill the session (tmux kill-session -t $SESSION)" >&2
            echo "       and re-run — but only once you are sure nothing is running in it." >&2
        fi
    done
    exit 0
fi

echo "[INFO] Creating tmux session '$SESSION' in $REPO_DIR ..."

tmux new-session -d -s "$SESSION" -n run -c "$REPO_DIR"
tmux set-option -t "$SESSION" mouse on

tmux new-window -t "$SESSION" -n run2 -c "$REPO_DIR"

# Panes are addressed as {top}/{bottom} rather than .0/.1: pane numbering starts at 1 under a
# `pane-base-index 1` config, and a bad target under `set -e` would abort here and leave a
# half-built session behind.
tmux new-window -t "$SESSION" -n mon -c "$REPO_DIR"
tmux split-window -v -t "$SESSION:mon" -c "$REPO_DIR"
tmux send-keys -t "$SESSION:mon.{top}" 'nvtop' C-m
tmux send-keys -t "$SESSION:mon.{bottom}" 'htop' C-m
tmux select-pane -t "$SESSION:mon.{top}"

tmux select-window -t "$SESSION:run"

echo "[INFO] Session '$SESSION' ready. Attach with:"
echo "  tmux attach -t $SESSION"
