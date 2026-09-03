# RQ/analysis — UNIFIED RQ1–4 DISPATCHER

Meta-runner that orchestrates 17 analysis scripts from one config. Not a library — everything flows through `run.py`.

## Entry point
```bash
python RQ/analysis/run.py --rq 1 2 3 4   # selected RQs
python RQ/analysis/run.py --all          # every RQ
python RQ/analysis/run.py --list         # list without running
```
`run.py` reads `config.yaml`, then for each script spawns `python -m RQ.analysis.RQ<N>.<module>` as a subprocess. Module name derives from each entry's `file` field (drop `.py`).

## config.yaml = single source of truth
- `common`: `context_lengths=[1024,2048,4096,8192,12288]`, `default_context=12288`, results/output bases.
- `models`: GPT-4.1, Qwen, QwenFT with CSV path templates `msg{msg}/{context}_zs.csv`. Currently bound to the revised manuscript's 27B arm: `results/gpt-seed43`, `results/Qwen3.6-27B`, and `results/Qwen3.6-27B-LoRA`.
- `rq1–4`: name, description, per-script `{file, models/csv_files, required_columns}`.

## The four RQs (mind the numbering)
| RQ | Question | Primary column |
|----|----------|----------------|
| RQ1 | impact of concern **count** | `hamming_loss` × `concern_count` |
| RQ2 | impact of **commit message** | `hamming_loss` × `with_message` (0/1) |
| RQ3 | token-budget **robustness** | `hamming_loss` × `context_len` (sweep) |
| RQ4 | inference **efficiency** | `inference_time` × count/tokens/message |

Shared helpers: `plot_utils.py` (styling, `display_model_name`), `stats_utils.py` (Wilcoxon/Mann-Whitney), `summary_ci.py` (commit-level 95% CI for manuscript tables), `data_aggregation/{aggregate,average}_experiments.py`.

## Inputs / outputs
- Reads model CSVs from `results/<model>/.../msg{0,1}/<context>_zs.csv` (QwenFT uses an aggregated CSV).
- Writes to `results/analysis/RQ<N>/`.
- `supplementary/` mirrors the revised manuscript's generated 27B mean/CI tables. The arXiv v1 14B arm remains under `results/{gpt,Qwen,Qwen3-14B-LoRA}`.

## DO NOT
- Add a script to `config.yaml` without confirming `run.py`'s dispatch resolves its module path.
- Hand-edit `results/analysis/` — regenerate by re-running the RQ.
- Assume dataset imbalance — all RQs rely on the 1/7 per-type uniformity guaranteed by `datasets/`.
