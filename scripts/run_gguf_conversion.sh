#!/bin/bash
#SBATCH --job-name=convert-gguf
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/convert_%j.out
#SBATCH --error=logs/convert_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bkoh3@sheffield.ac.uk

# Sheffield HPC Stanage - CPU-only GGUF Conversion for Phi-4 Fine-tuned Model
# Convert merged LoRA model to GGUF format and upload to Hugging Face

echo "Starting GGUF conversion job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK, Memory: $SLURM_MEM_PER_NODE MB"


mkdir -p logs

module purge
module load GCCcore/12.3.0
module load CUDA/12.4.0
module load Anaconda3/2022.05
module load CMake/3.26.3-GCCcore-12.3.0

if ! conda env list | grep -q "venv"; then
    echo "❌ venv not found!"
    echo "Please run setup_env.sh first to create the environment:"
    echo "sbatch setup_env.sh"
    exit 1
fi

# Activate environment using HPC-compatible method
echo "🔧 Activating venv..."
source activate venv

# Environment variables
export CUDA_VISIBLE_DEVICES=0
export FASTDATA_BASE="/mnt/parscratch/users/$USER"

# Temporary workspace  
export TMPDIR="${TMPDIR:-/tmp/gguf_conversion_$$}"
mkdir -p "$TMPDIR"
echo "📁 Temporary workspace: $TMPDIR"

export LLAMA_CPP_DIR="$FASTDATA_BASE/llama.cpp"
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "❌ llama.cpp not found at $LLAMA_CPP_DIR"
    echo "Please run setup_env.sh first to build llama.cpp:"
    echo "sbatch scripts/setup_env.sh"
    exit 1
else
    echo "✅ llama.cpp found at $LLAMA_CPP_DIR"
fi

# Find project root using git
PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)

if [ -z "$PROJECT_ROOT" ]; then
    echo "❌ Could not find git repository root!"
    echo "Make sure you're running this script from within the git repository."
    echo ""
    echo "💡 Try:"
    echo "  cd /path/to/Concern-is-All-You-Need"
    echo "  sbatch scripts/run_gguf_conversion.sh"
    exit 1
fi

echo "✅ Found project root: $PROJECT_ROOT"

# Change to project root directory
cd "$PROJECT_ROOT"

# Set converter script path
CONVERTER="$PROJECT_ROOT/RQ/SLM/convert_to_gguf.py"

if [ ! -f "$CONVERTER" ]; then
    echo "❌ Could not locate conver_to_gguf.py at: $CONVERTER"
    exit 1
fi

echo "✅ Found converter script: $CONVERTER"

# Determine model type from first argument or default to phi
MODEL_TYPE=${1:-"phi"}

# Run GGUF conversion with model type
echo "🚀 Starting GGUF conversion for ${MODEL_TYPE} at $(date)"
python "$CONVERTER" --model "${MODEL_TYPE,,}" | tee -a "logs/convert_output_${SLURM_JOB_ID}.log"

conversion_exit_code=$?

if [ $conversion_exit_code -eq 0 ]; then
    echo "🎉 GGUF conversion completed successfully!"
    echo "📊 File sizes in GGUF directory:"
    ls -lh "$TMPDIR/gguf_output/" 2>/dev/null || echo "GGUF files uploaded to HF Hub"
else
    echo "❌ GGUF conversion failed with exit code: $conversion_exit_code"
    exit 1
fi

echo "✅ GGUF conversion completed at $(date)"

# Cleanup temporary workspace
echo "🧹 Cleaning up temporary workspace: $TMPDIR"
rm -rf "$TMPDIR"

# Deactivate environment
source deactivate

echo "✅ GGUF conversion job completed!" 