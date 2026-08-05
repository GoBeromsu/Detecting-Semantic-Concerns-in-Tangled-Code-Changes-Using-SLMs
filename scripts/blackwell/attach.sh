#!/bin/bash
# Run on the Mac: get from a cold laptop into the Blackwell tmux session in
# one command.
#
# Two ssh invocations, not one:
#   1. Pipe tmux_session.sh into a non-interactive `bash -s` to create the
#      session (idempotent, detached, no tty needed).
#   2. `ssh -t ... tmux attach` for the actual attach, which needs a real tty.
# They cannot be merged: a single `ssh -t "cmd1 && tmux attach"` would still
# force a tty onto the setup leg, and piping a script into a tty-allocated
# ssh session is unreliable.

set -euo pipefail

HOST="${HOST:-blackwell}"
SESSION="${SESSION:-qwen27b}"
# RECREATE=1 rebuilds an existing session with the current layout. The host script still
# refuses while any pane is busy, so this cannot take down a running job.
RECREATE="${RECREATE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ssh joins its trailing arguments into a single string and hands it to the remote shell to
# parse, so local quoting protects nothing: every value crossing the link is re-split remotely.
# %q-quote each one, and pass them positionally rather than as an inline `VAR=... bash` prefix
# so a value can never terminate the command it is embedded in.
q_session="$(printf '%q' "$SESSION")"
q_repo_dir="$(printf '%q' "${REPO_DIR:-}")"
q_recreate="$(printf '%q' "$RECREATE")"

echo "[INFO] Setting up tmux session '$SESSION' on $HOST ..."
ssh "$HOST" bash -s -- "$q_session" "$q_repo_dir" "$q_recreate" < "$SCRIPT_DIR/tmux_session.sh"

echo "[INFO] Attaching to '$SESSION' on $HOST ..."
exec ssh -t "$HOST" tmux attach -t "$q_session"
