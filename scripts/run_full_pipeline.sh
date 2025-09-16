#!/bin/bash
#SBATCH --job-name=full-pipeline
#SBATCH --time=34:00:00
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bkoh3@sheffield.ac.uk

set -euo pipefail

MODEL_TYPE=${1:-"Phi"}
[[ "$MODEL_TYPE" =~ ^(Phi|Qwen)$ ]] || { echo "MODEL_TYPE must be Phi or Qwen"; exit 1; }

echo "=========================================="
echo "FULL PIPELINE: ${MODEL_TYPE}"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "=========================================="

# Store the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ==================== PHASE 1: TRAINING ====================
echo ""
echo "=========================================="
echo "PHASE 1: TRAINING"
echo "=========================================="
echo "Starting ${MODEL_TYPE} training at $(date)"

# Execute training script directly
bash "${SCRIPT_DIR}/run_training.sh" "${MODEL_TYPE}"
TRAINING_EXIT=$?

if [ $TRAINING_EXIT -ne 0 ]; then
    echo "❌ Training failed with exit code: $TRAINING_EXIT"
    exit 1
fi

echo "✅ Training completed successfully at $(date)"

# ==================== PHASE 2: GGUF CONVERSION ====================
echo ""
echo "=========================================="
echo "PHASE 2: GGUF CONVERSION"
echo "=========================================="
echo "Starting GGUF conversion at $(date)"

# Execute GGUF conversion script directly
bash "${SCRIPT_DIR}/run_gguf_conversion.sh" "${MODEL_TYPE}"
CONVERSION_EXIT=$?

if [ $CONVERSION_EXIT -ne 0 ]; then
    echo "❌ GGUF conversion failed with exit code: $CONVERSION_EXIT"
    exit 1
fi

echo "✅ GGUF conversion completed successfully at $(date)"

# ==================== PHASE 3: INFERENCE ====================
echo ""
echo "=========================================="
echo "PHASE 3: INFERENCE"
echo "=========================================="
echo "Starting inference at $(date)"

# Execute inference script directly (convert model type to lowercase for inference)
MODEL_PARAM=$(echo "$MODEL_TYPE" | tr '[:upper:]' '[:lower:]')
bash "${SCRIPT_DIR}/run_infer_huggingface.sh" "${MODEL_PARAM}"
INFERENCE_EXIT=$?

if [ $INFERENCE_EXIT -ne 0 ]; then
    echo "❌ Inference failed with exit code: $INFERENCE_EXIT"
    exit 1
fi

echo "✅ Inference completed successfully at $(date)"

# ==================== FINAL SUMMARY ====================
echo ""
echo "=========================================="
echo "COMPLETE PIPELINE SUMMARY"
echo "=========================================="
echo "Model Type: ${MODEL_TYPE}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Training Status: ✅ Success"
echo "GGUF Conversion Status: ✅ Success"
echo "Inference Status: ✅ Success"
echo "Pipeline completed at: $(date)"
echo ""

# Display job statistics
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,State,ExitCode

echo ""
echo "🎉 Full pipeline completed successfully!"