#!/bin/bash
#SBATCH --job-name=phi4_infer_hf
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/phi4_infer_%j.out
#SBATCH --error=logs/phi4_infer_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bkoh3@sheffield.ac.uk

# Sheffield HPC Stanage - Inference (Hugging Face / GGUF) for Phi-4
# Spec matches fine_tuning/run_training.sh; only job name and entrypoint differ

echo "Starting Phi-4 inference job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK, Memory: $SLURM_MEM_PER_NODE MB"

# Ensure logs directory exists
mkdir -p logs

module purge
module load GCCcore/12.3.0
module load CUDA/12.1.1
module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Configure cache directories on fastdata to avoid filling home quota
export FASTDATA_BASE="/mnt/parscratch/users/$USER"
export HF_HOME="$FASTDATA_BASE/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Return to original directory
cd "$SLURM_SUBMIT_DIR"

# Activate environment using 'source activate' per Stanage rules
echo "🔧 Activating phi4_env..."
source activate phi4_env

# Start periodic GPU utilization logging (every 60 s)
GPU_LOG="logs/gpu_usage_${SLURM_JOB_ID}.csv"
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
python -u RQ/Phi/infer_huggingface.py | tee -a "logs/phi4_infer_output_${SLURM_JOB_ID}.log"

echo "✅ Inference completed at $(date)"

source deactivate


