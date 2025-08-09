#!/bin/bash
#SBATCH --job-name=phi4_infer_hf
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
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

# Activate environment using 'source activate' per Stanage rules
echo "🔧 Activating phi4_env..."
source activate phi4_env

echo "🚀 Starting inference at $(date)"
uv run python RQ/Phi/infer_huggingface.py

echo "✅ Inference completed at $(date)"

source deactivate


