# Stanage SLURM Scripts (Frozen)

These are the frozen SLURM job scripts used on the University of Sheffield's
Stanage HPC cluster (A100/H100) to produce the published Qwen3-14B paper
results. They are kept as-is for reproducibility and are **not** used by the
Qwen3.6-27B Unsloth path.

The Qwen3.6-27B local training/inference path runs on a local Blackwell
workstation (not HPC) directly via:

```bash
python -m RQ.SLM.unsloth.probe
python -m RQ.SLM.unsloth.train
python -m RQ.SLM.unsloth.memory
python -m RQ.SLM.unsloth.infer
```

with no SLURM/wrapper scripts involved.
