# RQ/SLM: Small Language Models

Two independent paths are preserved:

| Path | Purpose | Entry points |
|---|---|---|
| Local Unsloth | Qwen3.6-27B BF16, unmerged LoRA adapter | `python -m RQ.SLM.unsloth.{preflight,train,memory,infer}` |
| Legacy | Published Qwen3-14B/GGUF results | `RQ/SLM/{train,infer,convert_to_gguf}.py` and protected legacy shells |

The legacy path is unchanged. The local path is the `RQ/SLM/unsloth/` package.

## Local Unsloth configuration and invariants

The pinned config is `RQ/SLM/unsloth/configs/qwen3_6_27b.yml`; the host profile remains `RQ/SLM/configs/hosts/blackwell-rtx-pro-6000.yml`.

- Qwen/Qwen3.6-27B is text-only, BF16, SDPA, unmerged LoRA (rank 32, alpha 48, dropout 0.05).
- The config owns the 16384 maximum sequence length; no CLI can override it. Config validation rejects an unapproved deviation.
- Training is right-padded, unpacked, response-only JSON+EOS supervision.
- The 1400-row train split excludes exactly row 1326 (`fc26be9a7f99e5f0d7db53cc53714dfd0c512a269fb3bd22ea3e7bdc12b742ba`, 16816 tokens), retaining 1399 rows.
- Generation requires a real pad token distinct from EOS. BF16/CUDA checks and Outlines 1.3.2 guards remain active.

Install the local GPU extra on Linux x86_64:

```bash
uv sync --frozen --python 3.12 --extra local-gpu
```

## Local Unsloth workflow

All entrypoints use module invocation.

### Preflight

```bash
python -m RQ.SLM.unsloth.preflight \
  --config RQ/SLM/unsloth/configs/qwen3_6_27b.yml \
  --host-profile RQ/SLM/configs/hosts/blackwell-rtx-pro-6000.yml \
  --probe-json <captured-probe.json> --cached-bytes 0 \
  --output .omo/evidence/unsloth/preflight.json
```

Preflight validates live Blackwell facts before GPU allocation.

### Template inspection and smoke training

```bash
python -m RQ.SLM.unsloth.train --inspect-template --evidence-dir .omo/evidence/unsloth
python -m RQ.SLM.unsloth.train --smoke --max-steps 2 --evidence-dir .omo/evidence/unsloth
```

Training flags are `--config`, `--dataset-source {local,hub}`, `--smoke`, `--max-steps`, `--inspect-template`, `--evidence-dir`, `--verify-adapter`, `--host-profile`, and `--verify-adapter-path-file`. The adapter directory comes from the config, not a flag. Saved adapters are validated automatically.

### Memory qualification

```bash
python -m RQ.SLM.unsloth.memory \
  --config RQ/SLM/unsloth/configs/qwen3_6_27b.yml \
  --host-profile RQ/SLM/configs/hosts/blackwell-rtx-pro-6000.yml \
  --lengths 2048 4096 8192 12288 16384 --output .omo/evidence/unsloth
```

The canonical ladder is `(2048, 4096, 8192, 12288, 16384)`; only a complete successful ladder produces `approved_16384`.

### BF16 PEFT evaluation and verification

```bash
python -m RQ.SLM.unsloth.infer \
  --adapter outputs/unsloth/<run>/adapter \
  --config RQ/SLM/unsloth/configs/qwen3_6_27b.yml --output results
python -m RQ.SLM.unsloth.infer --verify-only --output results/Qwen3.6-27B-LoRA/<run>
```

Inference flags are `--config`, `--adapter`, `--data-source {local,hub}`, `--resume`, `--run-directory`, `--verify-only`, and `--output`. Resume requires an explicit run directory. Evaluation uses the fixed canonical sweep and semantically finalizes CSV results.

## Package files

| File | Role |
|---|---|
| `unsloth/config.py` | validated config and host profile |
| `unsloth/data.py` | dataset validation and response-only rendering |
| `unsloth/runtime.py` | lazy BF16 Unsloth/TRL runtime |
| `unsloth/train.py` | training, manifest, adapter validation |
| `unsloth/infer.py` | BF16 PEFT evaluation and verification |
| `unsloth/memory.py` | isolated memory qualification |
| `unsloth/preflight.py` | live host preflight |
| `unsloth/adapter.py` | adapter contract validation |
| `unsloth/results.py` | result schema and finalization |
| `unsloth/generation.py` | constrained generation |
| `unsloth/infer_options.py` | GPU-free inference options |
| `unsloth/_types.py`, `unsloth/_memory_types.py`, `unsloth/_memory_worker.py` | private shared support |

## Legacy path

The published-results path remains unchanged:

```bash
python RQ/SLM/train.py --config RQ/SLM/configs/qwen.yml
python RQ/SLM/infer.py
python RQ/SLM/convert_to_gguf.py --model qwen
```
