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
| Hub repo `Berom0227/Semantic-Concern-SLM-Qwen3.6-27B-adapter` absent (404 @ 2026-08-03) | §8's unconditional post-train upload | create the repo before launching §8 |
| `t_infer` unbounded | §8 launch decision (§8a) | run the §7a measurement after smoke |

Host tuning is available this session (the workstation is otherwise idle), so swap and memory
pressure are **setup items, not gates** — see §4a.

---

## 1. Preconditions (on Mac, before ssh)

- [ ] `pytest __test__/unsloth/` green (all 14 files).
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
uv sync
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

**Go/no-go:** hostname confirmed, branch clean, `uv sync` clean, both tokens exported.

---

## 3. Remote session survival (tmux)

The session runs over a Tailscale ssh link — currently DERP-relayed, not a direct connection.
An ssh drop **outside tmux kills the training process**; it is a normal child of the ssh shell
with nothing to reparent it. Every GPU-phase command (§5, §6, §7, §7a, §8, §12) must run inside
tmux. §4 is quick and zero-GPU — bare ssh is fine.

```bash
tmux new -s qwen27b       # first time
tmux attach -t qwen27b    # after any disconnect/reconnect
```

On first creation, build the run layout — left pane for phase commands, right column stacked
with a GPU monitor over a CPU monitor:

```bash
tmux split-window -h -l '38%'   # right column; tmux ≥ 3.1 — on older, -p 38
tmux split-window -v -l '50%'   # split it top/bottom; on older, -p 50
tmux select-pane -t 0           # back to the left (run) pane
```

- Top-right: `nvtop` — or `watch -n 30 nvidia-smi` if absent (installs are out of scope here).
- Bottom-right: `htop` — or `top` if absent.
- Move between panes: `Ctrl-b o` or `Ctrl-b` + arrows. Detach: `Ctrl-b d` (training keeps
  running). Reattach: `tmux attach -t qwen27b`; never start a second session with the same name
  mid-run.

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

### 7a. `t_infer` measurement (owner-run, immediately after §7)

**There is no CLI subset for this**, by design: `infer.py` always runs the fixed canonical sweep
(contexts, message conditions and row count are constants in `results.py:26-28`), and
`infer_options.py` exposes no `--limit` or `--contexts`. No such flag will be added — the
minimal-flag posture is an invariant. So this is an owner-run snippet against the package's own
generation API, recorded by hand like `cached_bytes` in §4.

Run it from the repo root inside tmux by pasting into `uv run python -`, or save under `$TMPDIR`
and run `uv run python "$TMPDIR/t_infer_bench.py"` — never write scratch files into the repo
tree, which must stay clean for §8's manifest check.

The only value you supply by hand is the adapter directory from §7. Everything else comes from
the pinned config and the canonical split, so the timed generation is exactly the work
`msg0/12288_zs.csv`'s first row does — same prompt, same context, same seed.

```python
# owner-run, in the repo venv on the host, inside tmux (§3)
import time
from pathlib import Path

import pandas as pd

from RQ.SLM.unsloth.config import load_config
from RQ.SLM.unsloth.data import load_split
from RQ.SLM.unsloth.generation import GenerationRequest, PeftLoadRequest, load_backend
from RQ.main import add_truncated_commits
from utils.prompt import get_prompt_by_type

ADAPTER_DIR = Path("outputs/unsloth/Qwen3.6-27B-LoRA/<UTC-timestamp>/adapter")  # from §7

config = load_config(Path("RQ/SLM/unsloth/configs/qwen3_6_27b.yml"))
rows = load_split("local", "test")                       # 350 canonical test rows
frame = pd.DataFrame([dict(row.as_mapping()) for row in rows])
commit = str(add_truncated_commits(                      # same render as §12 msg0/12288
    frame, context_window=12288, include_message=False
)["truncated_commit"].tolist()[0])
system_prompt = get_prompt_by_type("Zero-shot", False)

backend = load_backend(PeftLoadRequest(
    model_id=config.model.id,
    revision=config.model.revision,
    adapter_path=ADAPTER_DIR,
    max_seq_length=config.training.max_seq_length,
))

elapsed = []
for index in range(11):                                  # index 0 is the discarded warm-up
    started = time.perf_counter()
    _ = backend.generate(GenerationRequest(system_prompt, commit, seed=42))
    elapsed.append(time.perf_counter() - started)
slowest = max(elapsed[1:])
print("slowest_of_10", slowest, "bound", 2 * slowest, "t_infer_hours", 3500 * 2 * slowest / 3600)
```

`seed=42` reproduces §12's row 0 exactly (`options.seed + row_index`). Record
`bound = 2 × slowest_of_10` and `t_infer = 3500 × bound` (3,500 = the sweep size).

**Go/no-go:** `bound` and `t_infer` recorded → carry into §8a. Cannot run or raises → `t_infer` stays unbounded ⇒ **§8 NO-GO**; do not substitute an estimate.

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

Git provenance is computed twice but only the second call gates: at launch it is captured and
discarded (`_ = _provenance("full")`, informational); the tree is re-derived and **enforced**
after training completes, right before `build_manifest`. So a tree that goes dirty mid-run still
burns the whole training run, then has its manifest refused. Keep the tree untouched throughout.

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
   the whole run; a repo that does not exist fails *after* the multi-hour training.
2. **Disk** — `df_free − P ≥ 15 GiB`, where `P = missing_model_bytes + 3 × checkpoint_size +
   adapter_size + upload_staging`. The `3` is `save_total_limit: 2` plus one, because
   transformers 5.5.0 writes the new checkpoint *before* rotating. Checkpoints are LoRA-sized,
   not 51.8 GiB base weights. (2026-08-03: model fully cached, 102.5 GiB available — a > 68 GiB
   margin.)
3. **Wall-clock** — the session budget is 2박3일; pass rules use the conservative **60 h**, with
   the remaining ~6 h as slack. Project:

   ```
   t_step_effective = t_step × L                    # L = corpus/smoke mean-token-length ratio, floored at 1.0
   T_full           = 875 × t_step_effective ÷ 3600
   T_total          = t_elapsed + t_render + T_full + t_ckpt + t_manifest_verify + t_upload + t_infer
   ```

   Only in the fallback branch, where the long-context allowance is unevidenced, apply the floor
   `t_step_effective = max(t_step × L, 168 s/step)` (the §9 upper band).
   **Pass iff `T_total ≤ 60 h`. Any operand unbounded ⇒ NO-GO** — in particular `t_infer` (§7a)
   and `t_upload` (`adapter_bytes ÷ measured uplink throughput`).

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

`save_strategy: "epoch"` with `save_total_limit: 2` — one checkpoint per epoch, auto-pruned to
the 2 most recent. Checkpoints do **not** land inside the final adapter directory:

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

Run inside tmux (§3).

```bash
python -m RQ.SLM.unsloth.infer --adapter <adapter_dir> --config "$CONFIG"
```

`--adapter` is required unless `--verify-only` is used. `--data-source`/`--data` (`local` default
or `hub`) selects the test split source; `--output` (default `results`) is the results root.

The canonical sweep is **fixed in code**, not flag-configurable: contexts
`[12288, 8192, 4096, 2048, 1024]` × message conditions `(without, with)` = 10 result files,
`seed=42`, `temperature=0.3`, `max_new_tokens=128`. `validate_adapter()` re-runs first; if it
fails, inference never loads the model.

```
results/Qwen3.6-27B-LoRA/<run-timestamp>/msg0/{12288,8192,4096,2048,1024}_zs.csv
results/Qwen3.6-27B-LoRA/<run-timestamp>/msg1/{12288,8192,4096,2048,1024}_zs.csv
results/Qwen3.6-27B-LoRA/<run-timestamp>/failures.jsonl
```

Resume an interrupted run:

```bash
python -m RQ.SLM.unsloth.infer --adapter <adapter_dir> --config "$CONFIG" \
  --resume --run-directory <results/Qwen3.6-27B-LoRA/<run-timestamp>>
```

GPU-free completeness check after the fact:

```bash
python -m RQ.SLM.unsloth.infer --verify-only --output <results/Qwen3.6-27B-LoRA/<run-timestamp>>
```

This requires exactly 350 successful rows per file and an empty `failures.jsonl`, or it raises
`FinalizationError`.

**Scheduling:** start §12 only when the remaining window covers `t_infer` alone (earlier tail
components have already elapsed), and only once `t_infer` is bounded (§7a). Unlike training, an
interrupted sweep **is** resumable, so an overrun here defers cleanly: stop, record, continue in
a later authorized session.

**Go/no-go:** all 10 CSVs present, `failures.jsonl` empty, `--verify-only` exits 0.

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
