# Blackwell Manual Session Runbook — Qwen3.6-27B LoRA

> **ACCESS POLICY (owner directive, 2026-08-02).** The Blackwell host is accessed **only with
> the owner's explicit permission, per action.** Current standing state is **READ-ONLY**: at
> most probe/preflight-class inspection (§4) is currently permitted — no model downloads, no
> writes to the host, no GPU allocation, no package installs — until the owner explicitly lifts
> this, phase by phase. **No automated agent ever sshs to this host.** Every host command in
> this runbook, from §2 onward, is executed by the owner personally, at the keyboard, with the
> owner's own credentials. This runbook is a checklist for that owner-run session, not something
> any agent invokes autonomously.

Owner-run, step-by-step. Every command below is copy-pasted from the actual `argparse`
definitions in `RQ/SLM/unsloth/{train,infer,infer_options,memory,probe,preflight}.py` —
no invented flags. Invoke everything as a module (`python -m RQ.SLM.unsloth.<name>`); the
flat-script bootstrap bugs in `train.py`/`infer.py`/`memory.py`'s self-relaunch (bare
`python RQ/SLM/unsloth/<name>.py` dying with `ModuleNotFoundError`) were fixed for all three
files in commit `893f2ad` and do not affect `-m` invocation either way.

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

## 1. Preconditions (on Mac, before ssh)

- [ ] `pytest __test__/unsloth/` green (all 14 files).
- [ ] `pytest __test__/` green (legacy 14B contract untouched).
- [ ] `basedpyright RQ/SLM/unsloth/` reports 0 errors / 0 warnings.
- [ ] `git status --short --branch` clean on a branch that contains commit `41263f4` (wandb
      project/run-name wiring) — currently `feat/wandb-run-wiring`. The earlier
      `feat/unsloth-qwen36-27b-local-lora` branch was merged into `main` via PR #11 (`d96bd91`)
      but does **not** contain `41263f4`; training from it lands the wandb run in the wrong
      project with an auto-generated name (see §9).

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
git fetch && git checkout feat/wandb-run-wiring
git merge-base --is-ancestor 41263f4 HEAD && echo "wandb wiring present"  # must print the message
git status --porcelain        # must be EMPTY — full training refuses to publish otherwise
uv sync
```

Credentials — **`train.py` reads `os.environ` directly and never calls `load_dotenv()`**
(confirmed: no `dotenv` import in `RQ/SLM/unsloth/train.py`, unlike the legacy `RQ/SLM/train.py`).
A `.env` file alone is **not** enough; export into the shell used to launch training:

```bash
set -a; source .env; set +a
echo "${HF_HUB_TOKEN:+set}" "${WANDB_API_KEY:+set}"   # both must print "set"
```

Full training hard-fails (`require_full_credentials`) without both `HF_HUB_TOKEN` and
`WANDB_API_KEY` in the process environment. Smoke training needs neither.

**Go/no-go:** Owner authorization for this phase confirmed? Hostname confirmed, branch clean, `uv sync` clean, both tokens exported → proceed.

---

## 3. Remote session survival (tmux)

The whole session runs over a Tailscale ssh link. An ssh drop **outside tmux kills the training
process** — it is a normal child of the ssh shell, with nothing to reparent it. Every GPU-phase
command below (§5 memory ladder, §6 template evidence, §7 smoke train, §8 full training, §12
inference) must run inside tmux on the host, not directly over the raw ssh session. §4
(probe/preflight) is quick and zero-GPU — bare is fine, tmux not required.

```bash
command -v tmux   # confirm it exists before relying on it (Ubuntu 22.04 ships it by default)
```

Start a named session for this run, or reattach after any reconnect:

```bash
tmux new -s qwen27b       # first time
tmux attach -t qwen27b    # after any disconnect/reconnect
```

On first creation, build the run layout: left pane for running the phase commands, right
column stacked with a GPU monitor on top and a CPU monitor below:

```bash
tmux split-window -h -l '38%'   # right column (~38% width); tmux ≥ 3.1 — on older, -p 38
tmux split-window -v -l '50%'   # split the right column top/bottom; on older, -p 50
tmux select-pane -t 0           # back to the left (run) pane
```

- Top-right pane: `nvtop` — or `watch -n 30 nvidia-smi` if nvtop is absent (installs are out
  of scope on this host).
- Bottom-right pane: `htop` — or `top` if absent.
- Move between panes: `Ctrl-b o`, or `Ctrl-b` + arrow keys.
- Detach without killing anything inside (training keeps running): `Ctrl-b d`. Reattach with
  `tmux attach -t qwen27b` — the panes come back exactly as they were; never start a second
  session with the same name mid-run.

**Go/no-go:** `tmux` present; session created or reattached; the shell about to run a GPU-phase
command is inside a tmux pane (`echo $TMUX` non-empty). If not, `tmux new -s qwen27b` first —
never launch the memory ladder, `train.py`, or `infer.py` directly over the bare ssh shell.

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

`cached_bytes` has no CLI helper — it is a manual integer preflight needs to compute the
missing-download size. First run, nothing cached yet:

```bash
CACHED_BYTES=0
# if the model was already partially/fully pulled into the HF cache, measure it instead:
# CACHED_BYTES=$(du -sb ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B 2>/dev/null | cut -f1 || echo 0)
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
allocation. Eyeball `result.valid == true`. `result.warnings` **will** include `limited_ram`
(host RAM is 31 GiB, ≤ the 31 GiB threshold) — this is a known, expected warning, not a
failure; it sets `require_peak_rss_measurement: true`, which the memory ladder already
captures per-length. `remaining_disk_bytes` must stay ≥ `reserve_bytes` (15 GiB).

**Go/no-go:** `preflight.json` has `"valid": true`. Any `false` → stop, report the failing field, do not proceed to GPU allocation.

---

## 5. Memory qualification ladder (bounded, child-isolated GPU allocation)

Run inside tmux (§3) — this allocates real GPU memory in a loop and can run for a while.

```bash
python -m RQ.SLM.unsloth.memory \
  --config "$CONFIG" \
  --host-profile "$HOST_PROFILE" \
  --output "$EVIDENCE_DIR"
```

No `--preflight` flag needed — it defaults to `<output>/preflight.json`, i.e. the file just
written in step 4. `memory.py` **hard-requires** that file to already validate; it crashes
uncaught if it is missing or stale, so step 4 must precede this.

Ladder: exactly `(2048, 4096, 8192, 12288, 16384)`, ascending, one length per run. Each length
relaunches `memory.py` as an isolated **child process** (`subprocess.run`) — a CUDA OOM in the
child raises there and is caught as `terminal_failure`, killing only that child; the parent
session and any earlier successful measurements survive. On the first `terminal_failure` the
parent stops attempting longer lengths (marks them `not_attempted_after_boundary`) rather than
retrying past a known ceiling.

Output: `$EVIDENCE_DIR/measurements.jsonl` (one line per length) and
`$EVIDENCE_DIR/qualification.json`. Full training refuses to start unless
`qualification.json` shows:

```json
{"status": "approved_16384", "approved_max_seq_length": 16384, ...}
```

This requires **all five** lengths to pass (finite loss, finite grads on `in_proj_a`/`in_proj_b`,
optimizer state allocated, optimizer step succeeded, VRAM headroom ≥10% post-step) **and** the
preflight evidence to bind by hash. Do not hand-edit `qualification.json` or re-point
`--host-profile`/`--config` after the fact — `train.py` re-verifies every hash before trusting it.

**Go/no-go:** Owner authorization for this phase confirmed? `qualification.json.status == "approved_16384"`. Anything else (`requires_owner_decision`) → stop, report `first_failure_boundary`, do not attempt to bypass by hand-editing evidence.

---

## 6. Template evidence (real GPU model load — not zero-allocation)

Run inside tmux (§3) — this loads the full model onto the GPU.

```bash
python -m RQ.SLM.unsloth.train --inspect-template --evidence-dir "$EVIDENCE_DIR"
```

**Correction vs. the original plan**: despite the name, `--inspect-template` is *not*
read-only. It calls `create_runtime()`, which runs
`unsloth.FastLanguageModel.from_pretrained(...)` — the real ~51.8 GiB (55,562,855,904 byte)
BF16 model load onto the GPU — purely to read `tokenizer.chat_template`, then returns without
training. Budget real VRAM/time for this step; it is the first real model load of the session.

Writes `$EVIDENCE_DIR/template-inspection.json` (template text + instruction/response mask
strings). No `--host-profile`, no credentials required for this flag.

**Overflow evidence (`overflow-rows.json`) is not produced here.** `train.py` has no
standalone "dry render" mode — the file is only written as a side effect of an actual
`trainer.train()` call (smoke or full) invoked with `--evidence-dir`. That happens in the next
step. Both invocations enforce, unconditionally, at the default 16384 sequence length:

> zero rows exceed the 16384-token budget (empty exclusion list), **1400** rows retained.

Any overflow raises `TrainingDataError` and stops the run before it trains anything.

**Go/no-go:** Owner authorization for this phase confirmed? `template-inspection.json` exists and is non-empty. Proceed to smoke train.

---

## 7. Smoke train

Run inside tmux (§3).

```bash
python -m RQ.SLM.unsloth.train --smoke --max-steps 5 --evidence-dir "$EVIDENCE_DIR"
```

- Requires neither `HF_HUB_TOKEN`/`WANDB_API_KEY` nor `--host-profile` (the full-run
  credential/qualification gate is skipped when `--smoke` is set).
- Trains with `report_to="none"` — no wandb run is created and no `WANDB_API_KEY` is read, so
  this step is safe to run on a host with no wandb credentials configured.
- **This is the step that writes `$EVIDENCE_DIR/overflow-rows.json`** (empty exclusion list,
  1400 retained — see §6). Passing the same `$EVIDENCE_DIR` here pre-populates the file the
  full run will require to already exist.
- Cannot claim qualification evidence even if `--host-profile`/`--evidence-dir` point at a
  qualified directory — `run_manifest.json`'s `qualification_dir` is forced to `None` for
  `run_mode: smoke` (enforced in `build_manifest`).
- `verify_adapter()` runs automatically at the end (CPU-only, content-addressed) — if it fails
  the run raises, no upload is attempted (smoke never uploads regardless).
- Saved under `outputs/unsloth/Qwen3.6-27B-LoRA/<UTC-timestamp>/adapter/`.

**Go/no-go:** Owner authorization for this phase confirmed? Exit 0, `run_manifest.json` written, `$EVIDENCE_DIR/overflow-rows.json` shows `"final_row_count": 1400` and an empty exclusion list. Any mismatch → stop; do not proceed to full training with bad overflow evidence.

---

## 8. Full training

Run inside tmux (§3) — this is the multi-hour, multi-epoch run; do not run it over a bare ssh
shell. See §9 for how to monitor it remotely and §10 for what a crash mid-run does and does not
recover.

```bash
python -m RQ.SLM.unsloth.train \
  --evidence-dir "$EVIDENCE_DIR" \
  --host-profile "$HOST_PROFILE"
```

(`--config` defaults to `$CONFIG` already; omit `--smoke` and `--max-steps` for the real run —
5 epochs, batch 1 × grad-accum 8, seq 16384, BF16, SDPA, per `configs/qwen3_6_27b.yml`.)

Preconditions the code itself enforces before touching the GPU:
- `$EVIDENCE_DIR` contains `template-inspection.json`, `overflow-rows.json`,
  `measurements.jsonl`, `qualification.json` (all from §5–§7) **and** `preflight.json` (§4).
- `qualification.json.status == "approved_16384"` and `approved_max_seq_length == 16384`,
  hash-bound to the current `config.yml` and `host_profile.yml` bytes.
- `HF_HUB_TOKEN` and `WANDB_API_KEY` present in the shell environment (§2).
- The wandb run for this training appears in project
  `Untangling-Multi-Concern-Commits-with-Small-Language-Models` with run name
  `qwen3.6-27b-semantic-concern-slm-unsloth-lora-<timestamp>` (the same UTC timestamp used for
  the adapter and checkpoint directories).
- Git provenance is computed twice, but only one call gates anything: at launch it's captured
  for informational purposes only (the result is discarded, `_ = _provenance("full")` — nothing
  enforces on it); the tree is re-derived and **enforced** only after training completes, right
  before `build_manifest` writes the manifest. Practically: if the tree goes dirty mid-run, the
  whole training run still happens, but `build_manifest` refuses to write the manifest
  afterward. Keep the tree untouched for the full duration.

**What happens automatically on completion** (all inside this one invocation, in order):
1. `trainer.train()` — writing a checkpoint once per epoch to a *separate*
   `outputs/unsloth/Qwen3.6-27B-LoRA/<timestamp>/checkpoints/` directory (see §10; this does not
   touch the adapter directory) — then `save_model()` + tokenizer save to
   `outputs/unsloth/Qwen3.6-27B-LoRA/<timestamp>/adapter/`.
2. `run_manifest.json` written (fails here if git went dirty or evidence is inconsistent).
3. `verify_adapter()` — CPU-only re-check of the just-written adapter. **If this raises, the run stops here and the next step never executes.**
4. Because this is not `--smoke`, `_upload_adapter()` runs **unconditionally and automatically** — there is no separate "upload" flag or step. It calls `huggingface_hub.HfApi(token=os.environ["HF_HUB_TOKEN"]).upload_folder(folder_path=<adapter_dir>, repo_id="Berom0227/Semantic-Concern-SLM-Qwen3.6-27B-adapter", repo_type="model")`. The moment full training finishes and passes verification, the adapter is public/updated on the Hub — there is no local-only "dry run" of a full training pass.

Duration is unknown — this is the first real run at this scale on this host. Do not estimate;
observe (§9).

Optional: `--verify-adapter-path-file <path>` writes the resolved adapter directory to a file for scripting the inference step below.

**Go/no-go:** Owner authorization for this phase confirmed? Exit 0 and the Hub repo `Berom0227/Semantic-Concern-SLM-Qwen3.6-27B-adapter` shows the new commit. A crash after step 1 but before step 4 means the adapter trained but was **not** published — check which step failed before retrying; retrying re-trains from scratch (§10 — there is no resume flag, only bounded-loss checkpoints).

---

## 9. Remote monitoring (during full training)

Everything here is read-only observation. Nothing should write into the repo tree while
training runs — the post-training git-clean re-check (§8, step 2) blocks manifest + upload if
the tree goes dirty mid-run.

- **wandb** — project `Untangling-Multi-Concern-Commits-with-Small-Language-Models` (from
  `configs/qwen3_6_27b.yml`, set into `WANDB_PROJECT` before trainer construction), run name
  `qwen3.6-27b-semantic-concern-slm-unsloth-lora-<timestamp>` — `<experiment_name>-<timestamp>`,
  the same UTC timestamp used for the adapter and checkpoint directories (§8, §10). The run URL
  is printed to stdout when the Trainer starts; it's also under that project in the wandb web
  UI. This only applies to full training — smoke training passes `report_to="none"` and creates
  no wandb run at all (§7).
- **GPU / CPU** — in the right-hand monitor panes of the §3 layout, read-only: `nvtop`
  (top-right; fallback `watch -n 30 nvidia-smi`) and `htop` (bottom-right; fallback `top`).
- **ETA from observed steps** — the step count is fixed up front: 1400 rows / effective batch
  8 → 175 optimizer steps per epoch, 875 total over 5 epochs, with a checkpoint at each epoch
  boundary (steps 175/350/525/700/875). Once the Trainer log has passed step ~30, take the
  mean seconds-per-step from the recent `logging_steps: 10` windows and project
  `remaining ≈ (875 − current_step) × s_per_step`. The a-priori model for this host puts
  s_per_step around 96–168 s (≈ 23–41 h of pure training; ~27.8B params × ~8 FLOPs/param/token
  × ~26.5M tokens at 40–70 effective TFLOPS under the 300 W Max-Q cap). A measured value far
  outside that band is a signal to check thermals/power first, not a better ETA.
- **Training log** — `train.py` has no built-in log-file writer; Trainer's console output
  (loss/step every `logging_steps: 10`) only goes to the tmux pane's stdout/stderr. To get
  something taillable from outside the run pane, redirect at launch instead of relying on
  scrollback, e.g.:

  ```bash
  python -m RQ.SLM.unsloth.train \
    --evidence-dir "$EVIDENCE_DIR" \
    --host-profile "$HOST_PROFILE" \
    2>&1 | tee "$HOME/qwen27b-train-$(date -u +%Y%m%dT%H%M%SZ).log"
  ```

  then, from any other pane: `tail -f "$HOME/qwen27b-train-<timestamp>.log"`. (`outputs/` is
  gitignored, so a log written under it wouldn't dirty the tree either, but keeping it under
  `$HOME` keeps the repo tree untouched with certainty and needs no extra reasoning about what's
  ignored.)

**Go/no-go:** informational only — nothing here gates progression. If `nvidia-smi` shows the
process gone and wandb has stopped logging steps, the run has stopped (crashed or finished);
check the exit status of the tmux pane running `train.py` and proceed to §10 or §11 accordingly.

---

## 10. Checkpointing & crash recovery

`configs/qwen3_6_27b.yml` sets `training.save_strategy: "epoch"` (`save_total_limit: 2`) — the
Trainer writes one checkpoint per epoch (5 epochs total), automatically pruning to keep at most
the 2 most recent.

**Verified against the current code** (`train.py:_checkpoint_dir`/`_adapter_dir`/`run`,
`runtime.py:create_trainer`/`build_sft_kwargs`): checkpoints do **not** land inside the final
adapter directory. `run()` builds two separate paths per training run and passes only the first
to the Trainer:

```
outputs/unsloth/Qwen3.6-27B-LoRA/<UTC-timestamp>/checkpoints/checkpoint-<step>/   # Trainer output_dir — periodic, epoch checkpoints
outputs/unsloth/Qwen3.6-27B-LoRA/<UTC-timestamp>/adapter/                         # final trainer.save_model() target
```

`checkpoints/` is what `create_trainer(runtime, examples, config, checkpoint_dir, ...)` passes
as `SFTConfig`'s `output_dir`, so HF Trainer's default `checkpoint-<step>/` subdirectories land
there, never under `adapter/`. This split exists precisely because `build_manifest`/
`verify_adapter` reject any entry under the adapter root that isn't a regular file (the
`evidence/` subdirectory is the one allowed exception) — a `checkpoint-<step>/` directory
appearing under `adapter/` would fail that check. `outputs/` is gitignored (`.gitignore:12`), so
none of this — checkpoints or the final adapter files — ever dirties the git tree the way §8's
step 2 cares about.

**No resume flag exists — say so plainly.** `RunArguments`/`parse_args()` defines no
`--resume-from-checkpoint` (or equivalent), and `build_sft_kwargs()` never sets
`resume_from_checkpoint` on `SFTConfig`. If the full run crashes for any reason (ssh drop outside
tmux, host OOM, CUDA error), the checkpoints sitting under `checkpoints/checkpoint-<step>/` are
**not loadable by anything in this CLI today**. Recovery is: confirm the process is actually
dead (§9), then **restart the whole `train.py` invocation from §8**, which starts a fresh
`<timestamp>` run directory from scratch. Per-epoch checkpointing only bounds how much compute a
crash wastes (at most one epoch); it does not make a crashed run resumable.

**Do not hand-delete checkpoints mid-run.** They belong to the running Trainer process
(`save_total_limit: 2` already prunes older ones on its own); deleting the checkpoint Trainer is
actively writing, or the directory it's writing into, can corrupt or crash the in-progress run.
Only remove `checkpoints/` after the run has fully finished — successfully or confirmed dead —
and you no longer need it.

**Checkpoints are not the deliverable.** The validated, uploaded artifact is `adapter/` (§11,
§8 step 3–4) — that is what gets published to the Hub. `checkpoints/` exists purely as crash
insurance for the duration of one run; it is never verified, never uploaded, and has no life
beyond helping you judge how far a crashed run got.

**Go/no-go:** informational for a healthy run. On crash: verify dead via §9, do not attempt to
resume, restart cleanly from §8.

---

## 11. Adapter verification (CPU-only, before trusting anything)

Verification already ran automatically inside §7 and §8. Use this to re-check independently
(e.g. after copying the adapter directory elsewhere, or before trusting an adapter someone else
produced):

```bash
python -m RQ.SLM.unsloth.train --verify-adapter <adapter_dir>
```

When `--verify-adapter` is passed, `train.py` takes an early-return branch — no config
validation for training, no GPU, no credentials needed. It still loads `--config` (defaults to
`$CONFIG`) to cross-check `config.yml` bytes against the adapter's captured evidence copy.

Equivalent standalone form that also writes a JSON report:

```bash
python -m RQ.SLM.unsloth.adapter --adapter-dir <adapter_dir> --config "$CONFIG" --output <report.json>
```

**Go/no-go:** exits 0 with no `ContractError`. Any failure means the adapter directory is not
trustworthy — do not upload/use it for inference; report the failing field.

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

Results land at:

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

GPU-free completeness check after the fact (no model load at all):

```bash
python -m RQ.SLM.unsloth.infer --verify-only --output <results/Qwen3.6-27B-LoRA/<run-timestamp>>
```

This requires exactly 350 successful rows per result file and an empty `failures.jsonl`, or it raises `FinalizationError`.

**Go/no-go:** Owner authorization for this phase confirmed? All 10 CSVs present, `failures.jsonl` empty, `--verify-only` exits 0.

---

## 13. Abort criteria & host safety

Any of the following → **stop, report, do not improvise on the host**:
- Preflight (`$EVIDENCE_DIR/preflight.json`) fails any gate.
- The memory ladder fails below 16384 (`qualification.json.status == "requires_owner_decision"`).
- Disk free dips toward the 15 GiB reserve during model download/caching.
- **Host RAM OOM during model load** (31 GiB RAM + ~1 GiB swap vs. ~51.8 GiB BF16 weights). If
  the *host* OOMs (not a CUDA OOM inside the child ladder), abort immediately — do not add
  swap, do not tweak the host. Report and wait for guidance.
- Any need to modify ECC mode, persistence mode, power cap, or swap. These are pinned in the
  host profile (`ecc: disabled`, `persistence_mode: disabled`, `power_cap_w: 300`,
  `swap_mib: 980`) precisely so nothing on the host changes underneath the run — never touch
  them, even if a warning suggests it would help.
- Tolerate, do not kill, the 4 known display processes (`Xorg`, `gnome-shell`, `rustdesk`,
  `obs`) — they're expected on this workstation and preflight already accounts for them via
  `vram_free_observed_mib` rather than `vram_total_mib`.

---

## Deviations from the original plan (found while reading the code)

1. **Template and overflow evidence are two separate commands, not one.** The plan's step 5
   assumed `--inspect-template --evidence-dir` produces both the template inspection *and* the
   token-overflow evidence. In the actual code, `--inspect-template` only writes
   `template-inspection.json` and returns early; `overflow-rows.json` is only ever written as a
   side effect of a real `trainer.train()` call (smoke or full) given `--evidence-dir`. The
   runbook splits this into §6 (template) and §7 (smoke, which produces overflow evidence).
2. **`--inspect-template` is not zero-GPU.** It calls `create_runtime()`, which does a real
   `unsloth.FastLanguageModel.from_pretrained(...)` load of the full ~51.8 GiB BF16 model onto
   the GPU just to read the tokenizer's chat template. Only `probe.py` and `preflight.py` are
   truly zero-allocation.
3. **`.env` is not auto-loaded by `RQ/SLM/unsloth/train.py`.** Unlike the legacy
   `RQ/SLM/train.py`/`RQ/SLM/infer.py` (which call `load_dotenv()`), the unsloth `train.py`
   reads `os.environ` directly with no `dotenv` import. `HF_HUB_TOKEN`/`WANDB_API_KEY` must be
   exported into the shell (`set -a; source .env; set +a`), not just present in the file.
4. **Full training auto-uploads to the Hub with no separate "upload" step or flag.** The plan
   treated "adapter upload to Hugging Face" as its own pipeline stage; in the code it is the
   unconditional last step of the same `train.py` invocation used for full training (runs
   whenever `--smoke` is absent, immediately after `verify_adapter()` succeeds). There is no
   way to do a full training run without it also publishing to
   `Berom0227/Semantic-Concern-SLM-Qwen3.6-27B-adapter` on success.
5. **`cached_bytes` for `preflight.py` has no code-provided helper.** It's a plain required
   `int` argument; the owner must supply it by hand (0, or a `du -sb` measurement of the HF
   cache directory for this model).
6. **Git provenance is computed twice for full runs, but only the second call gates anything.**
   `_provenance("full")` runs at launch, but its result is discarded (`_ = _provenance("full")`
   in `train.py:run()`) — it enforces nothing, it's informational only at that point.
   `build_manifest` re-derives clean status via a fresh `git status --porcelain` *after* training
   completes, right before writing the manifest, and that call is the one that actually blocks —
   a tree that goes dirty mid-training still blocks manifest write (and thus upload) even though
   training itself already ran. Practical guidance is unchanged: keep the tree untouched for the
   full duration.
7. **Per-epoch checkpoints write to a directory separate from the final adapter save**, added by
   a concurrent lane after the first version of this runbook. `train.py` now defines
   `_checkpoint_dir(config, timestamp)` = `<adapter_dir_root>/<timestamp>/checkpoints/`, distinct
   from `_adapter_dir(config, timestamp)` = `<adapter_dir_root>/<timestamp>/adapter/`, and passes
   the former (not the latter) to `create_trainer(...)` as the Trainer's `output_dir`. This keeps
   HF Trainer's `checkpoint-<step>/` subdirectories out of the path `build_manifest`/
   `verify_adapter` validate, both of which reject any non-regular-file entry under the adapter
   root except the `evidence/` subdirectory — confirmed by reading `train.py` and `runtime.py` at
   the moment this section was written, not assumed from the plan.
8. **No resume-from-checkpoint CLI flag exists anywhere in this package.** `RunArguments`/
   `parse_args()` has no `--resume-from-checkpoint` flag, and `build_sft_kwargs()` never sets
   `resume_from_checkpoint` on `SFTConfig`. The plan's "training progress must be recoverable"
   directive is satisfied only in the bounded sense that per-epoch checkpoints cap wasted compute
   at under one epoch on a crash — a crashed full run must still be relaunched from scratch
   (§8/§10); the checkpoints under `checkpoints/checkpoint-<step>/` are not currently loadable by
   any command in this package.
