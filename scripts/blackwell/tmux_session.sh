#!/bin/bash
# Blackwell tmux session bootstrap — runs ON the host (via attach.sh over ssh).
#
# Creates one detached tmux session laid out so that attaching lands directly on
# a working view — a shell to type into, with the GPU visible beside it:
#
#   window "run"   — left: shell in $REPO_DIR (active)   right: nvtop
#   window "run2"  — left: shell in $REPO_DIR (active)   right: htop
#
# The monitors live beside the shells rather than in a window of their own. A
# separate "mon" window means the GPU is never on screen while a command runs,
# which is exactly when it needs watching.
#
# Non-interactive: creates the session detached and exits. It never attaches;
# the caller (attach.sh) does that separately.
#
# Idempotent: if $SESSION already exists, this is a no-op unless RECREATE=1 is
# passed, and even then it refuses while any pane is busy. It must never destroy
# a session that has a training run inside it.

set -euo pipefail

# Values arrive as positional arguments from attach.sh (empty means "use the default"), so
# nothing has to be interpolated into a string the remote shell re-parses. Env vars still work
# when this script is run directly on the host.
SESSION="${1:-${SESSION:-qwen27b}}"
REPO_DIR="${2:-${REPO_DIR:-$HOME/Concern-is-All-You-Need}}"
RECREATE="${3:-${RECREATE:-0}}"

# Commands a pane may be running and still count as idle. Anything else — python, a training
# run, an editor — means the session is in use and must not be torn down.
IDLE_COMMANDS=(bash zsh sh fish nvtop htop)

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
    if [ "$RECREATE" != "1" ]; then
        echo "[INFO] tmux session '$SESSION' already exists — leaving it untouched."
        echo "       To rebuild it with the current layout: RECREATE=1 ./scripts/blackwell/attach.sh"
        exit 0
    fi
    busy=""
    while read -r pane_command; do
        for idle in "${IDLE_COMMANDS[@]}"; do
            [ "$pane_command" = "$idle" ] && continue 2
        done
        busy="${busy}${busy:+, }${pane_command}"
    done < <(tmux list-panes -s -t "$SESSION" -F '#{pane_current_command}')
    if [ -n "$busy" ]; then
        echo "[ERROR] Refusing to recreate '$SESSION': panes are running $busy" >&2
        echo "        Let them finish, or kill the session by hand once you are sure." >&2
        exit 1
    fi
    echo "[INFO] Recreating idle session '$SESSION' ..."
    tmux kill-session -t "$SESSION"
fi

echo "[INFO] Creating tmux session '$SESSION' in $REPO_DIR ..."

tmux new-session -d -s "$SESSION" -n run -c "$REPO_DIR"
tmux set-option -t "$SESSION" mouse on

# Panes are addressed as {left}/{right} rather than .0/.1: pane numbering starts at 1 under a
# `pane-base-index 1` config, and a bad target under `set -e` would abort here and leave a
# half-built session behind. The monitor takes the narrower side — nvtop and htop stay readable
# at ~38% while the shell keeps enough width for wrapped log lines.
tmux split-window -h -l 38% -t "$SESSION:run" -c "$REPO_DIR"
tmux send-keys -t "$SESSION:run.{right}" 'nvtop' C-m
tmux select-pane -t "$SESSION:run.{left}"

tmux new-window -t "$SESSION" -n run2 -c "$REPO_DIR"
tmux split-window -h -l 38% -t "$SESSION:run2" -c "$REPO_DIR"
tmux send-keys -t "$SESSION:run2.{right}" 'htop' C-m
tmux select-pane -t "$SESSION:run2.{left}"

tmux select-window -t "$SESSION:run"

echo "[INFO] Session '$SESSION' ready. Attach with:"
echo "  tmux attach -t $SESSION"
