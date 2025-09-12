#!/bin/bash
#SBATCH --job-name=infer
#SBATCH --time=20:00:00
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bkoh3@sheffield.ac.uk

set -euo pipefail

MODEL_TYPE=${1:-"Phi"}
[[ "$MODEL_TYPE" =~ ^(Phi|Qwen)$ ]] || { echo "MODEL_TYPE must be Phi or Qwen"; exit 1; }

echo "MODEL_TYPE=${MODEL_TYPE}"
echo "Starting inference job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK, Memory: $SLURM_MEM_PER_NODE MB"

# Ensure logs directory exists
mkdir -p logs

module purge
module load GCCcore/12.3.0
module load CUDA/12.4.0
module load Anaconda3/2022.05

# Activate environment using 'source activate' per Stanage rules
echo "🔧 Activating venv..."
source activate venv

# Start periodic GPU utilization logging (every 60 s)
GPU_LOG="logs/gpu_usage_${SLURM_JOB_ID}_${MODEL_TYPE}.csv"
echo "timestamp,power.draw[W],gpu.util[%],mem.util[%],mem.used[MiB]" > "$GPU_LOG"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,utilization.memory,memory.used --format=csv,noheader >> "$GPU_LOG"
    sleep 60
  done
) &
GPU_MON_PID=$!

# Ensure GPU logger is terminated on script exit
cleanup() {
  kill "$GPU_MON_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "🚀 Starting inference at $(date)"
python -u RQ/SLM/infer.py --model "${MODEL_TYPE,,}" | tee -a "logs/infer_output_${SLURM_JOB_ID}_${MODEL_TYPE}.log"

echo "✅ Inference completed at $(date)"

source deactivate


