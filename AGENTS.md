# CONCERN-IS-ALL-YOU-NEED — EXPERIMENT CODEBASE

**Generated:** 2026-07-31 · own git repo · branch `feat/unsloth-qwen36-27b-local-lora`

## OVERVIEW
Python research code (MSc dissertation) for detecting multiple semantic concerns in tangled code commits with SLMs. Covers dataset construction, GPT-4.1 + Qwen inference, LoRA fine-tuning, unified RQ1–4 analysis, and a Streamlit result explorer.

## STRUCTURE
```
Concern-is-All-You-Need/
├── app.py                    # Streamlit entrypoint (uses visual_eval/)
├── RQ/
│   ├── GPT/infer.py          # GPT-4.1 zero-shot inference → results/gpt/
│   ├── SLM/                  # TWO paths — see RQ/SLM/AGENTS.md
│   │   ├── train.py infer.py convert_to_gguf.py   # LEGACY 14B (frozen, hash-locked)
│   │   ├── configs/          # legacy configs + hosts/ profile
│   │   └── unsloth/          # NEW Qwen3.6-27B local BF16 LoRA package
│   ├── analysis/             # unified RQ1–4 analysis — see RQ/analysis/AGENTS.md
│   └── main.py               # token-truncation helper (unrelated to analysis/)
├── datasets/                 # 3-stage dataset pipeline — see datasets/AGENTS.md
├── utils/                    # eval.py prompt.py model.py llms/ (lazy-import facade)
├── visual_eval/              # Streamlit dashboard components
├── scripts/hpc/stanage-slurm/ # FROZEN Stanage HPC SLURM *.sh + environment.yml (14B paper results)
├── results/                  # GENERATED outputs only (timestamped) — never hand-edit
├── supplementary/            # paper's extended result CSVs
└── __test__/                 # pytest (CPU-only); unsloth/ subtree + fixtures/
```

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Qwen3.6-27B local training/inference | `RQ/SLM/unsloth/` (invoke `python -m RQ.SLM.unsloth.{train,infer,memory}`) |
| Published-paper 14B path | `RQ/SLM/{train,infer,convert_to_gguf}.py` (frozen) |
| Dataset build/tangle/validate | `datasets/scripts/` |
| Run analysis | `RQ/analysis/run.py --rq 1 2 3 4 \| --all \| --list` |
| Shared metrics/prompts/LLM clients | `utils/` |
| Concern-type enum, DF columns | `utils/llms/constant.py` (`COMMIT_TYPES`, `DEFAULT_DF_COLUMNS`) |

## COMMANDS
```bash
uv sync                          # deps (extras: --extra dev, --extra hpc, --extra local-gpu)
streamlit run app.py             # result explorer
uv run pytest __test__/ -q       # CPU-only suite (matches CI)
python RQ/analysis/run.py --rq 1 # analysis dispatcher
```

## CONVENTIONS
- Dependency manager: `uv` (uv.lock committed). Python pinned 3.12 (`.python-version`); pyproject floor `>=3.10` — do NOT use 3.11+/3.12-only syntax (e.g. `type X = ...` statements) in shipped code.
- Env vars: `OPENAI_API_KEY` (GPT inference), `HF_HUB_TOKEN` (HuggingFace), `WANDB_API_KEY` (27B full run).
- Wheel packages: `utils`, `visual_eval` only (hatchling).
- CI (`.github/workflows/ci.yml`): uv venv + `.[dev]` + pytest on push/PR to main.
- Type checking: basedpyright must be 0 errors / 0 warnings on the `unsloth/` slice; no `cast`/`Any`/`# type: ignore` suppressions.

## ANTI-PATTERNS (THIS REPO)
- Do NOT trust `CLAUDE.md` — stale: says analysis entry is `main.py` (actual `run.py`), calls RQ2 "efficiency" (actual RQ2 = commit-message impact, RQ4 = efficiency), and omits the `unsloth/` 27B path entirely. This AGENTS.md + README are authoritative.
- LoRA config is rank=32, alpha=48 (CLAUDE.md's rank=64/alpha=128 is wrong).
- `results/` is generated output — regenerate via runs, never hand-edit.
- Legacy 14B files (`RQ/SLM/train.py`, `infer.py`, `convert_to_gguf.py`) + 8 `scripts/hpc/stanage-slurm/*.sh` are hash-locked in `__test__/fixtures/slm/protected-files.sha256`. Keep byte-identical.
- Multiple `.venv` dirs may exist — use the root one via `uv`.
