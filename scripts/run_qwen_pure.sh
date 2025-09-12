#!/bin/bash
#SBATCH --job-name=qwen-pure-gguf
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --output=logs/qwen_pure_%j.out
#SBATCH --error=logs/qwen_pure_%j.err

set -euo pipefail

mkdir -p logs

module purge
module load GCCcore/12.3.0
module load CUDA/12.4.0
module load Anaconda3/2022.05

echo "Activating venv..."
source activate venv

export FASTDATA_BASE="/mnt/parscratch/users/$USER"
export LLAMA_CPP_DIR="$FASTDATA_BASE/llama.cpp"
export TMPDIR="${TMPDIR:-/tmp/qwen_pure_$$}"

echo "Running Qwen pure GGUF conversion..."
python RQ/SLM/convert_qwen_pure.py

echo "Done!"