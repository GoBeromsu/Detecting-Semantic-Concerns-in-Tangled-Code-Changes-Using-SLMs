# Blackwell Manual Session Runbook — Qwen3.6-27B LoRA

> **ACCESS POLICY (owner directive).** Every host command from §2 onward is executed by the
> owner personally, at the keyboard, with the owner's own credentials. **No agent ever sshs to
> this host autonomously** — this runbook is a checklist for an owner-run session, not something
> an agent invokes.
>
> Host tuning (freeing memory, adjusting swap) is permitted this session; the three settings that
> must still stay untouched are listed in §13.

Scope is the local model only — this runbook never touches `RQ/GPT/infer.py`, and nothing under
`RQ/SLM/unsloth/` reads `OPENAI_API_KEY`.

Every command is copy-pasted from the actual `argparse` definitions in
`RQ/SLM/unsloth/{train,infer,infer_options,memory,probe,preflight}.py` — no invented flags.
Invoke everything as a module (`python -m RQ.SLM.unsloth.<name>`).

Set once per session:

```bash
EVIDENCE_DIR=.omo/evidence/unsloth
CONFIG=RQ/SLM/unsloth/configs/qwen3_6_27b.yml
HOST_PROFILE=RQ/SLM/configs/hosts/blackwell-rtx-pro-6000.yml
mkdir -p "$EVIDENCE_DIR"
```

All qualification/template/overflow/preflight evidence lives in one directory
(`$EVIDENCE_DIR`) so `train.py`'s evidence checks find everything without extra flags.

---

## 0. Open blockers

Re-verify these before booking a session; the values are point-in-time, not standing truth.

| Blocker | Blocks | Action |
|---|---|---|
| `WANDB_API_KEY` absent from `.env` | §8 (`require_full_credentials`, `train.py:304-305`) | obtain the key and export it (§2) |
| ~~Hub repo `Berom0227/Semantic-Concern-SLM-Qwen3.6-27B-adapter` absent~~ | — | **cleared 2026-08-05**: created public, matching its nine sibling repos; `require_publishable()` verified PASS against the live Hub |
| `t_infer` unbounded | §8 launch decision (§8a) | run the §7a canary after smoke |

Host tuning is available this session (the workstation is otherwise idle), so swap and memory
pressure are **setup items, not gates** — see §4a.

---

## 1. Preconditions (on Mac, before ssh)

- [ ] `pytest __test__/unsloth/` green (all 12 test files).
- [ ] `pytest __test__/` green (legacy 14B contract untouched).
- [ ] `basedpyright RQ/SLM/unsloth/` reports 0 errors / 0 warnings.
- [ ] `git status --short --branch` clean on `main`, carrying both `41263f4` (wandb
      project/run-name wiring — without it the run lands in the wrong project under an
      auto-generated name, §9) and `efeca02` (SEED=43 datasets — without it a seed-42 row
      overflows 16384 and `train.py` aborts with `TrainingDataError` before training, §6).

**Go/no-go:** all four green → proceed to host session. Any red → stop, fix, re-run; do not ssh.

---

## 2. Host session setup

```bash
ssh beomsu@blackwell.tailee178.ts.net hostname
# expected: dcs33979 — preflight.py hard-gates on this exact string (EXPECTED_HOSTNAME)
```

Then on the host:

```bash
ssh beomsu@blackwell.tailee178.ts.net
cd <repo>
git fetch && git checkout main && git pull
git merge-base --is-ancestor 41263f4 HEAD && echo "wandb wiring present"   # must print
git merge-base --is-ancestor efeca02 HEAD && echo "seed-43 dataset present" # must print
git status --porcelain        # must be EMPTY — full training refuses to publish otherwise
uv sync --extra local-gpu     # NOT plain `uv sync` — see below
```

**The `--extra local-gpu` is load-bearing.** Every GPU dependency (`torch`, `unsloth`, `peft`,
`trl`, `outlines`) lives in that optional extra, gated on `sys_platform == 'linux' and
platform_machine == 'x86_64'`. Plain `uv sync` installs **none** of them, and because the ML
stack is imported lazily, §3's go/no-go and §4's zero-GPU probe both still pass — the first
failure would be §5, well after the point where a fail-fast check should have caught it.

Prove the stack actually imports before spending a session on it:

```bash
uv run python -c "import torch, unsloth, peft, trl, outlines; print(torch.__version__, torch.cuda.is_available())"
# must print a cu128 build and True

python -m RQ.SLM.unsloth.infer --help | grep -- --base
# must print — proves the host checkout carries §12's two-arm CLI, not an older single-arm one
```

Credentials — **`train.py` reads `os.environ` directly and never calls `load_dotenv()`**
(unlike the legacy `RQ/SLM/train.py`). A `.env` file alone is **not** enough; the values must be
in the environment of the shell that launches training:

```bash
read -rs HF_HUB_TOKEN && export HF_HUB_TOKEN     # no file written, no shell history
read -rs WANDB_API_KEY && export WANDB_API_KEY
echo "${HF_HUB_TOKEN:+set}" "${WANDB_API_KEY:+set}"   # both must print "set"
```

Or, if the host repo already holds a `.env` you trust: `set -a; source .env; set +a`.

Full training hard-fails without both tokens. §4–§7 need neither.

**Go/no-go:** hostname confirmed, branch clean, `uv sync --extra local-gpu` clean, the import
line printing `True`, `--base` present in `infer --help`, both tokens exported.

---

## 3. Remote session survival (tmux)

The session runs over a Tailscale ssh link — currently DERP-relayed, not a direct connection.
An ssh drop **outside tmux kills the training process**; it is a normal child of the ssh shell
with nothing to reparent it. Every GPU-phase command (§5, §6, §7, §7a, §8, §12) must run inside
tmux. §4 is quick and zero-GPU — bare ssh is fine.

One command from a cold laptop creates the session (if absent) and attaches to it:

```bash
./scripts/blackwell/attach.sh          # run on the Mac, not the host
```

It ssh's to the `blackwell` alias twice on purpose: once non-interactively to pipe
`scripts/blackwell/tmux_session.sh` to the host and build the session detached, then once with
`-t` to attach, which needs a real tty.

The session `qwen27b` holds three windows:

| Window | Contents |
|--------|----------|
| `run`  | shell in the repo — phase commands (§5, §6, §7, §7a, §8, §12) |
| `run2` | second shell in the repo — `--verify-only`, `git status`, log tails |
| `mon`  | `nvtop` (top pane) over `htop` (bottom pane) |

`tmux_session.sh` fails fast before creating anything if `tmux`, `nvtop`, or `htop` is missing
(it prints the exact `sudo apt install -y <pkg>` line) or if the repo directory is wrong. It
defaults to `REPO_DIR=$HOME/Concern-is-All-You-Need`; if the host checkout lives elsewhere,
export `REPO_DIR` and it is forwarded over ssh.

**It is idempotent and never touches an existing session** — re-running `attach.sh` mid-training
detects `qwen27b`, leaves it alone, and attaches. That is the property that makes it safe to run
after a dropped link without thinking about it.

- Switch windows: `Ctrl-b 0/1/2` or `Ctrl-b w`. Move between panes: `Ctrl-b o` or `Ctrl-b` +
  arrows. Detach: `Ctrl-b d` (training keeps running). Reattach: `./scripts/blackwell/attach.sh`;
  never start a second session with the same name mid-run.

**Go/no-go:** the shell about to run a GPU-phase command has non-empty `$TMUX`.

---

## 4. Read-only probe & preflight (zero GPU allocation)

```bash
python -m RQ.SLM.unsloth.probe --output "$EVIDENCE_DIR/probe.json"
cat "$EVIDENCE_DIR/probe.json"
```

`probe.py` only shells out to `nvidia-smi`, reads `/proc/meminfo` and disk usage, and queries
`torch.version.cuda` / `torch.cuda.is_bf16_supported()` — no model load, no CUDA allocation.

Eyeball against the host profile:

| Field | Expected |
|---|---|
| `valid` | `true` |
| `hostname` | `dcs33979` |
| `gpus[0].compute_capability` | `12.0` |
| `gpus[0].memory.total_mib` | `97887` |
| `gpus[0].memory.free_mib` | `>= 96793` |
| `gpus[0].power_limit_w` | `~300` |
| `gpus[0].ecc_mode` | `disabled` (expected — do not enable) |
| `gpus[0].persistence_mode` | `disabled` (expected — do not enable) |
| `torch.cuda` | `12.8` |
| `torch.bf16_supported` | `true` |
| `processes.compute` | `[]` (no compute process holds the GPU) |

`cached_bytes` has no CLI helper — it is a plain required int you supply by hand:

```bash
CACHED_BYTES=$(du -sb ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B 2>/dev/null | cut -f1 || echo 0)
```

```bash
python -m RQ.SLM.unsloth.preflight \
  --config "$CONFIG" \
  --host-profile "$HOST_PROFILE" \
  --probe-json "$EVIDENCE_DIR/probe.json" \
  --cached-bytes "$CACHED_BYTES" \
  --output "$EVIDENCE_DIR/preflight.json"
cat "$EVIDENCE_DIR/preflight.json"
```

`preflight.py` is pure validation over the JSON files above — no torch/CUDA import, zero GPU
allocation. `result.warnings` **will** include `limited_ram` (host RAM is 31 GiB) — expected,
not a failure. `remaining_disk_bytes` must stay ≥ `reserve_bytes` (15 GiB).

**Go/no-go:** `preflight.json` has `"valid": true`. Any `false` → stop, report the failing field, do not proceed to GPU allocation.

### 4a. Host-RAM headroom (setup check, once per session)

The host has 31 GiB RAM. `preflight.py` records this but never hard-gates it (`limited_ram` only
sets `require_peak_rss_measurement`), so check it yourself before the first model-loading phase:

```bash
grep -E 'MemAvailable|SwapTotal|SwapFree' /proc/meminfo
```

Want **`MemAvailable ≥ 10 GiB`** — that is `1.5 × (largest safetensors shard, measured 3.721 GiB)
+ 4 GiB`. The model loads via an accelerate device-map that streams shards to the GPU, so host
RAM holds about one shard plus its in-flight copy, never the full 51.8 GiB. Never size this off
the cache directory total (~51.8 GiB), which would look permanently insufficient.

Short on headroom? Free it: close whatever is holding memory, or add swap — host tuning is
available this session. Re-check and continue; this is a setup step, not a no-go. A *host* OOM
during a load is still the §13 abort, but it is now a recoverable one.

---

## 5. Memory qualification ladder (bounded, child-isolated GPU allocation)

Run inside tmux (§3) — this allocates real GPU memory in a loop and can run for a while. Run it
as a single invocation; splitting it per rung would break `approved_16384`'s binding.

```bash
python -m RQ.SLM.unsloth.memory \
  --config "$CONFIG" \
  --host-profile "$HOST_PROFILE" \
  --output "$EVIDENCE_DIR"
```

No `--preflight` flag needed — it defaults to `<output>/preflight.json`, the file written in §4.
`memory.py` **hard-requires** that file to already validate; it crashes uncaught if it is
missing or stale.

Ladder: exactly `(2048, 4096, 8192, 12288, 16384)`, ascending, one length per run. Each length
relaunches as an isolated **child process**, so a CUDA OOM is caught as `terminal_failure` and
kills only that child; the parent session and earlier measurements survive. On the first
`terminal_failure` the parent stops attempting longer lengths
(`not_attempted_after_boundary`) rather than retrying past a known ceiling.

Output: `$EVIDENCE_DIR/measurements.jsonl` and `$EVIDENCE_DIR/qualification.json`. Full training
refuses to start unless the latter shows:

```json
{"status": "approved_16384", "approved_max_seq_length": 16384, ...}
```

This requires **all five** lengths to pass (finite loss, finite grads on `in_proj_a`/`in_proj_b`,
optimizer state allocated, optimizer step succeeded, VRAM headroom ≥10% post-step) **and** the
preflight evidence to bind by hash. Do not hand-edit `qualification.json` or re-point
`--host-profile`/`--config` after the fact — `train.py` re-verifies every hash.

**Go/no-go:** `status == "approved_16384"`. Anything else (`requires_owner_decision`) → stop, report `first_failure_boundary`, do not bypass by hand-editing evidence.

---

## 6. Template evidence (real GPU model load — not zero-allocation)

Run inside tmux (§3).

```bash
python -m RQ.SLM.unsloth.train --inspect-template --evidence-dir "$EVIDENCE_DIR"
```

Despite the name, `--inspect-template` is *not* read-only. It calls `create_runtime()`, which
runs the real ~51.8 GiB (55,562,855,904 byte) BF16 load onto the GPU purely to read
`tokenizer.chat_template`, then returns without training. Budget real VRAM/time. (This is the
first *persistent, parent-process* load — the §5 ladder children already loaded the model in
isolated subprocesses.)

Writes `$EVIDENCE_DIR/template-inspection.json` (template text + instruction/response mask
strings). No `--host-profile`, no credentials required.

**Overflow evidence (`overflow-rows.json`) is not produced here.** `train.py` has no standalone
dry-render mode — that file is only written as a side effect of an actual `trainer.train()` call
given `--evidence-dir`, i.e. §7. Both smoke and full runs enforce, unconditionally, at 16384:
zero rows exceed the budget (empty exclusion list), **1400** rows retained. Any overflow raises
`TrainingDataError` before anything trains.

**Go/no-go:** `template-inspection.json` exists and is non-empty.

---

## 7. Smoke train

Run inside tmux (§3).

```bash
python -m RQ.SLM.unsloth.train --smoke --max-steps 5 --evidence-dir "$EVIDENCE_DIR"
```

- Requires neither token nor `--host-profile` (the full-run credential/qualification gate is
  skipped under `--smoke`), and trains with `report_to="none"` — no wandb run is created.
- **Writes `$EVIDENCE_DIR/overflow-rows.json`** (empty exclusion list, 1400 retained), which the
  full run requires to already exist.
- Cannot claim qualification evidence even if `--evidence-dir` points at a qualified directory —
  `run_manifest.json`'s `qualification_dir` is forced to `None` for `run_mode: smoke`.
- `verify_adapter()` runs automatically at the end (CPU-only); a failure raises and no upload is
  attempted (smoke never uploads regardless).
- Saved under `outputs/unsloth/Qwen3.6-27B-LoRA/<UTC-timestamp>/adapter/`.

**Record for the §8a projection**, from this one invocation:

| Value | Where |
|---|---|
| `t_step` | final Trainer runtime summary: `train_runtime ÷ completed_optimizer_steps`. Biased HIGH (5 steps include warmup) — conservative; do not correct downward |
| `t_render` | wall-clock of the render phase before training starts (outside the Trainer summary) |
| `t_manifest_verify` | wall-clock of the post-train manifest + verify phase (also outside it) |

**Go/no-go:** exit 0, `run_manifest.json` written, `overflow-rows.json` shows `"final_row_count": 1400` and an empty exclusion list. Any mismatch → stop; do not run §8 on bad overflow evidence.

### 7a. `t_infer` canary (owner-run, immediately after §7)

This does double duty: it bounds `t_infer` for §8a **and** it is the fail-fast canary for §12.
It runs the real `run_evaluation()` — the same adapter load, prompt render, constrained decode,
CSV contract, and failure sidecar §12 uses — just narrowed to 11 rows of one cell. If §12 is
going to break on this host, it breaks here, in three minutes, instead of eleven hours later.

**There is no CLI subset for this**, by design: `infer.py` always runs the fixed canonical sweep
(contexts, message conditions and row count are constants in `results.py:26-28`), and
`infer_options.py` exposes no `--limit` or `--contexts`. No such flag will be added — the
minimal-flag posture is an invariant. So the narrowing is done by constructing
`EvaluationOptions` directly, which is the same frozen dataclass the CLI builds.

Run it from the repo root inside tmux by pasting into `uv run python -`, or save it under
`/tmp` and run `uv run python /tmp/t_infer_bench.py` — never write scratch files into the repo
tree, which must stay clean for §8's manifest check. `output_root` points outside the repo for
the same reason.

The only value you supply by hand is the adapter directory from §7.

```python
# owner-run, in the repo venv on the host, inside tmux (§3)
import csv
import tempfile
from pathlib import Path

from RQ.SLM.unsloth.data import LOCAL_DATA_DIR
from RQ.SLM.unsloth.generation import MAX_NEW_TOKENS
from RQ.SLM.unsloth.infer import run_evaluation
from RQ.SLM.unsloth.infer_options import EvaluationOptions

ADAPTER_DIR = Path("outputs/unsloth/Qwen3.6-27B-LoRA/<UTC-timestamp>/adapter")  # from §7

outcome = run_evaluation(EvaluationOptions(
    config_path=Path("RQ/SLM/unsloth/configs/qwen3_6_27b.yml"),
    adapter_path=ADAPTER_DIR,
    data_source="local",
    data_dir=LOCAL_DATA_DIR,
    contexts=(12288,),            # msg1/12288 is the worst cell on both axes — see below
    message_conditions=(True,),
    seed=42,                      # identical to §12: seed = 42 + row_index
    temperature=0.3,
    max_new_tokens=MAX_NEW_TOKENS,
    limit=11,                     # row 0 is the discarded warm-up
    resume=False,
    output_root=Path(tempfile.gettempdir()) / "t_infer",   # never inside the repo tree
    run_directory=None,
))

with outcome.result_files[0].open(newline="", encoding="utf-8") as handle:
    elapsed = [float(row["inference_time"]) for row in csv.DictReader(handle)]
slowest = max(elapsed[1:])
print("rows", len(elapsed), "failures", outcome.failure_count)
print("slowest_of_10", slowest, "bound", 2 * slowest, "t_infer_hours", 7000 * 2 * slowest / 3600)
```

Record `bound = 2 × slowest_of_10` and `t_infer = 7000 × bound`. **7,000, not 3,500** — §12 runs
the sweep twice, once for the LoRA adapter and once for the unadapted base model.

**Why `msg1/12288` specifically.** It is the worst cell on both axes at once, so a clean result
here bounds every other cell:

- *Latency* — the longest prompt in the sweep, so `slowest_of_10` is an upper bound for all ten
  cells, not a mid-range sample.
- *Prompt budget* — the only place `validate_prompt_budget()` could plausibly fire.
  `add_truncated_commits()` truncates the rendered commit to the context window, so the prompt is
  bounded by `12288 + len(system_prompt) + 128 max_new_tokens` against `max_seq_length: 16384`,
  and the with-message system prompt is the longer of the two. The margin is expected to be
  comfortable — the seed-43 datasets were regenerated precisely so no row overflows the Qwen
  budget (`efeca02`) — so this is confirming an invariant that should already hold, not
  discovering one. If it *does* fire, `failures.jsonl` names the offending rows and §12 is a
  no-go until the dataset is re-examined.

**Go/no-go:** `outcome.failure_count == 0`, `len(elapsed) == 11`, and `bound`/`t_infer` recorded
→ carry into §8a. Any failure count above zero means §12 will bleed rows on this host — diagnose
from `failures.jsonl` before proceeding. Cannot run or raises → `t_infer` stays unbounded ⇒
**§8 NO-GO**; do not substitute an estimate.

---

## 8. Full training

Run inside tmux (§3) — multi-hour, multi-epoch. See §9 for monitoring and §10 for what a crash
does and does not recover.

```bash
python -m RQ.SLM.unsloth.train \
  --evidence-dir "$EVIDENCE_DIR" \
  --host-profile "$HOST_PROFILE"
```

(`--config` defaults to `$CONFIG`; omit `--smoke`/`--max-steps` for the real run — 5 epochs,
batch 1 × grad-accum 8, seq 16384, BF16, SDPA, per `configs/qwen3_6_27b.yml`.)

Preconditions the code itself enforces before touching the GPU:
- `$EVIDENCE_DIR` contains `template-inspection.json`, `overflow-rows.json`,
  `measurements.jsonl`, `qualification.json` (§5–§7) **and** `preflight.json` (§4).
- `qualification.json.status == "approved_16384"` and `approved_max_seq_length == 16384`,
  hash-bound to the current `config.yml` and `host_profile.yml` bytes.
- `HF_HUB_TOKEN` and `WANDB_API_KEY` present in the environment (§2).
- `require_publishable()` — clean Git worktree, and `HfApi.repo_info()` resolves
  `Berom0227/Semantic-Concern-SLM-Qwen3.6-27B-adapter`.

Git provenance is checked twice, and **both calls now gate**. At launch, `require_publishable()`
refuses a dirty worktree and an unreachable adapter repo before the runtime is created — the two
publication preconditions that were otherwise only discovered after training finished. The tree
is then re-derived and enforced again after training completes, right before `build_manifest`, so
a tree that goes dirty *mid-run* still burns the training run and has its manifest refused. Keep
the tree untouched throughout.

**What happens automatically on completion**, all inside this one invocation, in order:
1. `trainer.train()` — one checkpoint per epoch into a *separate* `<timestamp>/checkpoints/`
   directory (§10) — then `save_model()` + tokenizer save to `<timestamp>/adapter/`.
2. `run_manifest.json` written (fails here if git went dirty or evidence is inconsistent).
3. `verify_adapter()` — CPU-only re-check. **If this raises, the run stops and step 4 never runs.**
4. `_upload_adapter()` runs **unconditionally** — there is no separate upload flag or step. It
   calls `HfApi(token=os.environ["HF_HUB_TOKEN"]).upload_folder(...)` to
   `Berom0227/Semantic-Concern-SLM-Qwen3.6-27B-adapter`. The moment full training passes
   verification the adapter is public/updated on the Hub; there is no local-only dry run of a
   full pass.

Optional: `--verify-adapter-path-file <path>` writes the resolved adapter directory to a file.

### 8a. Pre-launch decision

Check immediately before launching, in order:

1. **Publication** — owner authorization to publish, and the Hub repo actually exists (§0). The
   upload is the unconditional last step of this same invocation, so a no-go here is a no-go on
   the whole run. `require_publishable()` now probes both the clean worktree and `repo_info()`
   for the adapter repo *before* the runtime is created, so a missing repo costs seconds rather
   than the whole training run — but it is still cheaper to confirm it here than to be bounced.
2. **Disk** — `df_free − P ≥ 15 GiB`, where `P = missing_model_bytes + 6 × checkpoint_size +
   adapter_size + upload_staging`. The `6` is `save_total_limit: 5` plus one, because
   transformers 5.5.0 writes the new checkpoint *before* rotating. Checkpoints are LoRA-sized,
   not 51.8 GiB base weights — with 5 epochs and `save_strategy: "epoch"`, the limit of 5 keeps
   every epoch's checkpoint, so nothing is ever actually rotated away. (2026-08-03: model fully
   cached, 102.5 GiB available — a > 68 GiB margin.)
3. **Wall-clock** — the session budget is 2박3일; pass rules use the conservative **60 h**, with
   the remaining ~6 h as slack. Project:

   ```
   t_step_effective = t_step × L                    # L = corpus/smoke mean-token-length ratio, floored at 1.0
   T_full           = 875 × t_step_effective ÷ 3600
   T_total          = t_elapsed + t_render + T_full + t_ckpt + t_manifest_verify + t_upload + t_infer
   ```

   Only in the fallback branch, where the long-context allowance is unevidenced, apply the floor
   `t_step_effective = max(t_step × L, 168 s/step)` (the §9 upper band).
   **Pass iff `T_total ≤ 60 h`. Any operand unbounded ⇒ NO-GO** — in particular `t_infer` (§7a,
   which counts **both** §12 arms: 7,000 generations, not 3,500) and `t_upload`
   (`adapter_bytes ÷ measured uplink throughput`).

**Fail ⇒ NO-GO + escalate.** Never autonomously reduce `num_train_epochs`, shorten
`max_seq_length`, or change batching to fit the window. Present the projection and wait.

**Requalification.** Any config edit alters bytes that are hash-bound into the qualification
evidence, staleness-invalidating `approved_16384` so full training refuses to start. An approved
alternative therefore needs fresh authorization, requalification, and a new projection **in a new
session** — never an in-session rescue.

**No retry budget.** No resume-from-checkpoint exists (§10), so `T_total` contains zero restart
headroom. A mid-run crash ends the session.

> **Pre-session flag.** The a-priori upper band alone (≈ 41 h, §9) plus tails plausibly breaches
> 60 h before any measurement is taken. Settle feasibility with the owner **before** booking the
> session, not at the post-smoke gate.

**Go/no-go:** all three checks above pass; exit 0 and the Hub repo shows the new commit. A crash after step 1 but before step 4 means the adapter trained but was **not** published — check which step failed; retrying re-trains from scratch (§10).

---

## 9. Remote monitoring (during full training)

Everything here is read-only. Nothing should write into the repo tree while training runs — the
post-training git-clean re-check (§8, step 2) blocks manifest + upload if the tree goes dirty.

- **wandb** — project `Untangling-Multi-Concern-Commits-with-Small-Language-Models`, run name
  `qwen3.6-27b-semantic-concern-slm-unsloth-lora-<timestamp>` (same UTC timestamp as the adapter
  and checkpoint directories). The run URL is printed to stdout when the Trainer starts. Full
  training only — smoke creates no wandb run (§7).
- **GPU / CPU** — the right-hand monitor panes from §3.
- **ETA from observed steps** — 1400 rows / effective batch 8 → 175 optimizer steps per epoch,
  875 total over 5 epochs, checkpoints at 175/350/525/700/875. Once past step ~30, take the mean
  seconds-per-step from recent `logging_steps: 10` windows and project
  `remaining ≈ (875 − current_step) × s_per_step`. The a-priori band for this host is
  **96–168 s/step (≈ 23–41 h of pure training)** — ~27.8B params × ~8 FLOPs/param/token ×
  ~26.5M tokens at 40–70 effective TFLOPS under the 300 W cap. A measured value far outside that
  band is a signal to check thermals/power first, not a better ETA.
  **Hard stop:** at step ~30 and each epoch boundary, recompute
  `session_elapsed + (875 − current_step) × observed_s_per_step + remaining_tail`; crossing
  **60 h** → stop and escalate, never trade config for time (§8a).
- **Training log** — `train.py` has no log-file writer; Trainer output only goes to the pane's
  stdout. To get something taillable, redirect at launch:

  ```bash
  python -m RQ.SLM.unsloth.train \
    --evidence-dir "$EVIDENCE_DIR" \
    --host-profile "$HOST_PROFILE" \
    2>&1 | tee "$HOME/qwen27b-train-$(date -u +%Y%m%dT%H%M%SZ).log"
  ```

  then from another pane: `tail -f "$HOME/qwen27b-train-<timestamp>.log"`. Keeping it under
  `$HOME` leaves the repo tree untouched with certainty.

**Go/no-go:** informational only. If `nvidia-smi` shows the process gone and wandb has stopped logging, the run has stopped — check the pane's exit status and proceed to §10 or §11.

---

## 10. Checkpointing & crash recovery

`save_strategy: "epoch"` with `save_total_limit: 5` — one checkpoint per epoch, and with
`num_train_epochs: 5` the limit is never actually reached, so all five survive. Checkpoints do
**not** land inside the final adapter directory:

```
outputs/unsloth/Qwen3.6-27B-LoRA/<UTC-timestamp>/checkpoints/checkpoint-<step>/   # Trainer output_dir
outputs/unsloth/Qwen3.6-27B-LoRA/<UTC-timestamp>/adapter/                         # final save_model() target
```

`create_trainer(...)` receives the former as `SFTConfig`'s `output_dir`. The split exists because
`build_manifest`/`verify_adapter` reject any entry under the adapter root that is not a regular
file (the `evidence/` subdirectory is the one exception), so a `checkpoint-<step>/` directory
there would fail that check. `outputs/` is gitignored, so none of it dirties the tree.

**No resume flag exists.** `parse_args()` defines no `--resume-from-checkpoint`, and
`build_sft_kwargs()` never sets `resume_from_checkpoint` on `SFTConfig`. If the full run crashes
for any reason, the checkpoints are **not loadable by anything in this CLI today**. Recovery is:
confirm the process is dead (§9), then restart the whole §8 invocation, which starts a fresh
`<timestamp>` directory. Per-epoch checkpointing only bounds wasted compute at one epoch; it does
not make a crashed run resumable. Combined with §8a's zero retry budget, a crash past roughly the
halfway point **ends the session**.

**Do not hand-delete checkpoints mid-run** — they belong to the running Trainer process, which
prunes them itself. Remove `checkpoints/` only after the run has finished or is confirmed dead.
They are never verified, never uploaded; the deliverable is `adapter/`.

**Go/no-go:** informational for a healthy run. On crash: verify dead via §9, do not attempt to resume, restart cleanly from §8 as a new-session decision.

---

## 11. Adapter verification (CPU-only)

Verification already runs automatically inside §7 and §8. Use this to re-check independently —
after copying the adapter elsewhere, or before trusting one someone else produced:

```bash
python -m RQ.SLM.unsloth.train --verify-adapter <adapter_dir>
```

This takes an early-return branch — no training config validation, no GPU, no credentials. It
still loads `--config` (defaults to `$CONFIG`) to cross-check `config.yml` bytes against the
adapter's captured evidence copy.

Equivalent standalone form that also writes a JSON report:

```bash
python -m RQ.SLM.unsloth.adapter --adapter-dir <adapter_dir> --config "$CONFIG" --output <report.json>
```

**Go/no-go:** exits 0 with no `ContractError`. Any failure means the adapter is not trustworthy — do not upload or use it; report the failing field.

---

## 12. Inference

Run inside tmux (§3). **This phase has two arms, and both are required** — the experiment is a
paired base-vs-LoRA ablation on the same seed-43 split, so a LoRA number without its base
counterpart answers nothing.

```bash
# arm 1 — the fine-tuned adapter
python -m RQ.SLM.unsloth.infer --adapter <adapter_dir> --config "$CONFIG"

# arm 2 — the same sweep against the unadapted base tower
python -m RQ.SLM.unsloth.infer --base --config "$CONFIG"
```

Exactly one of `--adapter` or `--base` is required unless `--verify-only` is used; supplying both
or neither is refused before any GPU work, so a mistyped adapter path can never silently degrade
into a base run and be written up as a fine-tuned result. `--data-source`/`--data` (`local`
default or `hub`) selects the test split source; `--output` (default `results`) is the results
root.

The canonical sweep is **fixed in code**, not flag-configurable: contexts
`[12288, 8192, 4096, 2048, 1024]` × message conditions `(without, with)` = 10 result files per
arm, `seed=42`, `temperature=0.3`, `max_new_tokens=128`. On the adapter arm `validate_adapter()`
re-runs first; if it fails, inference never loads the model. The base arm skips that check
entirely — there is no adapter to validate.

The two arms write to separate model trees, so neither can overwrite the other:

```
results/Qwen3.6-27B-LoRA/<run-timestamp>/msg{0,1}/{12288,8192,4096,2048,1024}_zs.csv
results/Qwen3.6-27B-LoRA/<run-timestamp>/{failures.jsonl,run_identity.json}
results/Qwen3.6-27B/<run-timestamp>/msg{0,1}/{12288,8192,4096,2048,1024}_zs.csv
results/Qwen3.6-27B/<run-timestamp>/{failures.jsonl,run_identity.json}
```

**A failed row costs one row, not the run.** A typed `GenerationError`/`ModelOutputError` is
appended to `failures.jsonl` and the sweep continues to the next row. The process exits `1` and
prints the resume command when any row failed, so a non-zero exit here means "gaps to fill", not
"start over".

Resume — re-attempts exactly the rows missing from each CSV, keyed by SHA rather than by row
count, so it fills gaps left by skipped failures instead of assuming an unbroken prefix. It is
safe to run repeatedly: a regenerated row is appended at the end of its CSV, so resume matches
rows by SHA membership rather than by position and a cell repaired by one resume is still read
correctly by the next. Each repaired cell is then rewritten back into test-split order via an
atomic `os.replace`, so a finished CSV always reads in source order — downstream analysis pairs
models by row position (`RQ/analysis/compare_models.py`), and a permanently appended row would
compare one model's row against another model's commit:

```bash
python -m RQ.SLM.unsloth.infer --adapter <adapter_dir> --config "$CONFIG" \
  --resume --run-directory results/Qwen3.6-27B-LoRA/<run-timestamp>
```

Resume requires `--run-directory` and an existing `failures.jsonl` in it, so it can never
silently mint a fresh timestamped directory and half-populate it. It also refuses a run directory
belonging to the *other* arm: both arms share a SHA order, so without that check a `--base`
resume would append cleanly into the LoRA tree and blend two models into one CSV undetectably.

Beyond the arm name, resume compares a `run_identity.json` written when the run started. It
digests the adapter's file contents, the config bytes, the ordered test SHAs, and every sweep
parameter (`seed`, `temperature`, `max_new_tokens`, contexts, message conditions). Every LoRA
checkpoint maps to the same tree name, so the arm check alone cannot see the likelier mistake —
resuming against a *different* one of the five checkpoints `save_total_limit: 5` leaves on disk
(§10), or against an edited config. Either would blend two experiments into one CSV that still
finalizes as canonical. If resume reports an identity mismatch, do not delete the file to force
it through: point `--adapter`/`--config` back at the exact inputs the run started with, or start
a fresh run.

In the other direction, a *fresh* run refuses any directory that already holds a
`run_identity.json` or a result CSV, and tells you to use `--resume`. It does not refuse a
directory that merely exists: a process killed between creating the directory and writing its
first sidecar leaves an empty shell that holds no results, and re-running the same command
simply reinitializes it. Nothing is ever deleted to make that happen.

GPU-free completeness check, run once per arm after the fact:

```bash
python -m RQ.SLM.unsloth.infer --verify-only --output results/Qwen3.6-27B-LoRA/<run-timestamp>
python -m RQ.SLM.unsloth.infer --verify-only --output results/Qwen3.6-27B/<run-timestamp>
```

This requires all ten files present with exactly 350 semantically valid rows each, or it raises
`FinalizationError`. It requires `failures.jsonl` to **exist** but does not require it to be
empty: the sidecar is append-only provenance, so a transient failure that a later `--resume`
regenerated successfully must not permanently disqualify the run. The 350-rows-per-file check is
the stronger gate and is what actually certifies completeness.

**Scheduling:** start §12 only when the remaining window covers `t_infer` alone (earlier tail
components have already elapsed), and only once `t_infer` is bounded (§7a) — remembering that
§7a's `t_infer` already counts **both** arms. Unlike training, an interrupted sweep **is**
resumable, so an overrun here defers cleanly: stop, record, continue in a later authorized
session.

**Go/no-go:** both arms have all 10 CSVs at 350 rows, both `--verify-only` invocations exit 0,
and any rows recorded in either `failures.jsonl` have since been regenerated by `--resume`.

---

## 13. Abort criteria & host safety

Any of the following → **stop, report, do not improvise on the host**:

- `preflight.json` fails any gate.
- The memory ladder fails below 16384 (`status == "requires_owner_decision"`).
- Disk free dips toward the 15 GiB reserve during model download/caching.
- **Host RAM OOM during model load.** If the *host* OOMs — not a CUDA OOM inside a ladder child —
  stop the phase, free memory or add swap (§4a), and restart the phase. Do not let a run limp
  along under memory pressure.
- Any need to modify **ECC mode, persistence mode, or power cap**. Unlike swap, these three are
  pinned in the host profile (`ecc: disabled`, `persistence_mode: disabled`, `power_cap_w: 300`)
  and validated by `preflight.py`'s drift check — changing one invalidates the preflight evidence
  and, for the power cap, the §9 ETA band derived from 300 W. Leave them alone even if a warning
  suggests otherwise.
- Tolerate, do not kill, the 4 known display processes (`Xorg`, `gnome-shell`, `rustdesk`, `obs`)
  — expected on this workstation, and preflight already accounts for them via
  `vram_free_observed_mib` rather than `vram_total_mib`.

---

## 14. Known risks

- **sm_120 kernel support is prototype-tier.** PyTorch 2.7 ships Blackwell support with
  CUDA 12.8 wheels and Triton 3.3; the host runs torch 2.7.1+cu128 + triton 3.3.1, and the
  pipeline pins `attn_implementation: "sdpa"`, so PyTorch-core SDPA is on the critical path. It
  is documented but unexercised here — §6's first persistent load is the confirmation point, and
  a kernel failure there is stop-and-report.
- **Unsloth API fit is settled.** Qwen3.5/3.6 text-only SFT uses `FastLanguageModel` +
  16-bit load + explicit `target_modules`; `FastModel` is for MoE. This pipeline is text-only with
  12 explicit targets and `mtp|visual` excluded, so there is no auto-target risk.
- **Inference VRAM is bounded, inference wall-clock is not.** KV-cache and recurrent state at
  16384 are bounded (full-attention KV ≈ 1.0 GiB plus MiB-scale DeltaNet state), but
  `max_new_tokens` caps tokens, not time — hence the §7a measurement.

The full 2026-08-03 readiness audit backing these — drift tables, shard measurements, source
citations — lives in `.omo/evidence/readiness-audit-20260803T170844Z/` (gitignored, machine-local).
