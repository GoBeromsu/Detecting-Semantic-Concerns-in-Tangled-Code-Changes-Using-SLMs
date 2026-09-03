# DETECTING MULTIPLE SEMANTIC CONCERNS IN TANGLED CODE COMMITS — REPLICATION PACKAGE

**Generated:** 2026-09-03 · own git repo · branch `main`

## OVERVIEW
Replication package for arXiv:2601.21298 (Koh, Walkinshaw, Shin — preprint, under review). The main part of the work was carried out at the University of Sheffield. Python research code covering dataset construction, GPT-4.1 + Qwen inference, LoRA fine-tuning, unified RQ1–4 analysis, and a Streamlit result explorer.

## STRUCTURE
```
Concern-is-All-You-Need/
├── app.py                    # Streamlit entrypoint (uses visual_eval/)
├── RQ/
│   ├── GPT/infer.py          # GPT-4.1 zero-shot inference → results/gpt*/
│   ├── SLM/                  # TWO paths — see RQ/SLM/AGENTS.md
│   │   ├── train.py infer.py convert_to_gguf.py   # arXiv v1 14B path (frozen, hash-locked)
│   │   ├── configs/          # legacy configs + hosts/ profile
│   │   └── unsloth/          # Qwen3.6-27B local BF16 LoRA package (+ RUNBOOK.md)
│   ├── analysis/             # unified RQ1–4 analysis — see RQ/analysis/AGENTS.md
│   └── main.py               # header-preserving token truncation (unrelated to analysis/)
├── datasets/                 # 3-stage dataset pipeline — see datasets/AGENTS.md
│   └── scripts/structural_validity/  # diff-level colocation + structural checks
├── utils/                    # eval.py prompt.py model.py llms/ (lazy-import facade)
├── visual_eval/              # Streamlit dashboard components
├── scripts/hpc/stanage-slurm/ # FROZEN Stanage HPC SLURM *.sh + environment.yml (arXiv v1)
├── scripts/blackwell/        # local 27B GPU-host tmux/attach helpers
├── results/                  # GENERATED outputs only (timestamped) — never hand-edit
├── supplementary/            # revised manuscript's 27B result + CI tables
└── __test__/                 # pytest (CPU-only); unsloth/ subtree + fixtures/
```

## CURRENT MANUSCRIPT VS ARXIV V1
| Arm | Model | Result dirs |
|-----|-------|-------------|
| Current manuscript (reported tables) | Qwen3.6-27B BF16 LoRA, local GPU host | `results/{gpt-seed43,Qwen3.6-27B,Qwen3.6-27B-LoRA}/` + `supplementary/` |
| arXiv v1 provenance | Qwen3-14B on Stanage SLURM | `results/{gpt,Qwen,Qwen3-14B-LoRA}/` |

`RQ/analysis/config.yaml` points at the **current 27B/seed43 arm**, so `results/analysis/` and `supplementary/` carry Qwen3.6 results. The public arXiv v1 text still describes 14B; reproduce that arm by re-pointing the `models:` block, not by editing outputs.

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Qwen3.6-27B training/inference | `RQ/SLM/unsloth/` (invoke `python -m RQ.SLM.unsloth.{train,infer,memory}`) |
| arXiv v1 14B path | `RQ/SLM/{train,infer,convert_to_gguf}.py` (frozen) |
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
- Env vars: `OPENAI_API_KEY` (GPT inference), `HF_HUB_TOKEN` (HuggingFace), `WANDB_API_KEY` (27B runs).
- Wheel packages: `utils`, `visual_eval` only (hatchling).
- CI (`.github/workflows/ci.yml`): uv venv + `.[dev]` + `pytest __test__/ -v` on push/PR to main.
- Type checking: basedpyright must be 0 errors / 0 warnings on the `unsloth/` slice; no `cast`/`Any`/`# type: ignore` suppressions.
- README is the public replication-package front door: paper link, authorship, Sheffield attribution, and the arXiv citation live there and must stay in sync with this file.

## ANTI-PATTERNS (THIS REPO)
- LoRA config is rank=32, alpha=48 (dropout 0.05 on the 27B arm).
- RQ numbering: RQ1 concern count · RQ2 commit message · RQ3 token budget · RQ4 efficiency. Analysis entry point is `run.py`, not `main.py`.
- `results/` and `results/analysis/` are generated output — regenerate via runs, never hand-edit.
- 14B files (`RQ/SLM/train.py`, `infer.py`, `convert_to_gguf.py`) + 8 `scripts/hpc/stanage-slurm/*.sh` are hash-locked in `__test__/fixtures/slm/protected-files.sha256`. Keep byte-identical.
- Multiple `.venv` dirs may exist — use the root one via `uv`.
- `CLAUDE.md` is only a pointer to this file; do not grow a second, drifting instruction set there.
