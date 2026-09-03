# RQ/SLM — TWO INDEPENDENT MODEL PATHS

Two paths coexist here. They share no configs and never import each other. Confusing them is the #1 hazard.

## LEGACY 14B (frozen — do not touch)
- Files: `train.py`, `infer.py`, `convert_to_gguf.py`; configs `configs/{qwen,phi,gpt_oss}.yml`, `sweep.yaml`, `deepspeed.json`.
- Produced the **arXiv preprint (2601.21298)** results on Sheffield Stanage SLURM (A100/H100). Kept for reproducibility.
- `train.py`, `infer.py`, `convert_to_gguf.py` are **hash-locked** (`__test__/fixtures/slm/protected-files.sha256`) + AST-characterized (`__test__/test_slm_legacy_contract.py`). Keep byte-identical.
- Invoke flat: `python RQ/SLM/train.py --config RQ/SLM/configs/qwen.yml`.

## CURRENT MANUSCRIPT — Qwen3.6-27B (`unsloth/` package)
- Isolated vertical slice. Invoke as modules: `python -m RQ.SLM.unsloth.{train,infer,memory}`.
- Config: `unsloth/configs/qwen3_6_27b.yml` (single source of truth). Host facts: `configs/hosts/blackwell-rtx-pro-6000.yml`.
- Trains via Unsloth, evaluates via Transformers+PEFT. **No merged model, no GGUF.** Adapter stays unmerged.
- Targets ONE local Blackwell workstation GPU. Stanage A100/H100 is inaccessible / out of scope for 27B.

### Module map (`unsloth/`)
| Module | Role |
|--------|------|
| `config.py` | validated config + host profile load; rejects non-BF16/quant/flash-attn/left-pad/packing |
| `data.py` | split load, row validation, chat rendering, canonical hashing |
| `runtime.py` | lazy Unsloth/TRL constructors, response-only mask |
| `probe.py` | read-only host-fact capture → probe JSON that `preflight` consumes |
| `preflight.py` | read-only host-fact validation before allocation |
| `memory.py` (+`_memory_worker.py`, `_memory_types.py`) | per-length qualification, child-isolated |
| `adapter.py` | CPU-only content-addressed adapter validation |
| `generation.py` / `results.py` / `infer_options.py` | constrained JSON gen, result schema/finalization, infer CLI opts |
| `train.py` / `infer.py` | training + BF16 PEFT inference/verify CLIs |
| `_types.py` | shared Protocols/TypedDicts/aliases (import symbols from HERE, not re-export modules) |

### Invariants (enforced in code — do not weaken)
- **1400** retained rows: zero rows exceed 16384 tokens under the Qwen3.6-27B tokenizer, so the exclusion list is empty. Any other count at 16384 is an error.
- Response-only supervision: labels = JSON object + EOS only; manifest `objective: response_only_json_eos`.
- pad token real AND ≠ EOS, else hard fail (no EOS-pad fallback).
- BF16 throughout; rank 32 / alpha 48 / dropout 0.05; 12 targets; `mtp|visual` excluded.
- `autocast_adapter_dtype=False`; TRL 0.20 real HF `Dataset` + `SFTConfig(max_length, dataset_text_field, packing=False)`.
- Outlines **1.3.2** API (`from_transformers` + `Generator` + `JsonSchema`); NOT `JSONLogitsProcessor`.
- Memory ladder exactly `(2048,4096,8192,12288,16384)`; `approved_16384` only when all pass + preflight hashes bind.
- `max_seq_length` is config-only (16384). There is NO CLI override flag — the guard lives in `config.py`.

### Status
Phase A (CPU: config/preflight/data/adapter/results/orchestration) done, covered by `__test__/unsloth/`.
Phase B **has been executed** on the Blackwell host: training + the SEED=43 inference sweep produced three archived repeats under `results/Qwen3.6-27B-LoRA/` (each with `run_identity.json` binding `adapter_sha256`/`config_sha256`), aggregated into `avg_result/`. `RQ/analysis/config.yaml` reads from these.
The gate still applies to every new run: training refuses to start until `memory.py` records `approved_16384` whose config/host/measurement hashes still match on disk. Local `outputs/unsloth/` dirs on a non-GPU machine are empty scaffolding — adapters live on the GPU host.

### DO NOT
- Point 27B tooling at legacy configs, or vice versa.
- Re-add removed CLI knobs (lr/rank/alpha/device-index/max-seq-length overrides) — minimal-flag posture is intentional.
- Add `cast`/`Any`/`# type: ignore` — the slice is basedpyright 0/0.
