#!/bin/bash
#SBATCH --job-name=phi4-gguf-convert
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/phi4_gguf_convert_%j.out
#SBATCH --error=logs/phi4_gguf_convert_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bkoh3@sheffield.ac.uk

# Sheffield HPC Stanage - CPU-only GGUF Conversion for Phi-4 Fine-tuned Model
# Convert merged LoRA model to GGUF format and upload to Hugging Face

echo "Starting GGUF conversion process..."

# Create logs directory
mkdir -p logs

# Setup environment - CPU-only configuration
module purge
module load GCCcore/12.3.0
# module load CUDA/12.1.1
module load Anaconda3/2022.05
# module load cuDNN/8.9.2.26-CUDA-12.1.1
module load CMake/3.26.3-GCCcore-12.3.0

# Check if phi4_env exists (should be created by setup_env.sh)
if ! conda env list | grep -q "phi4_env"; then
    echo "❌ phi4_env not found!"
    echo "Please run setup_env.sh first to create the environment:"
    echo "sbatch setup_env.sh"
    exit 1
fi

# Activate environment using HPC-compatible method
echo "🔧 Activating phi4_env..."
source activate phi4_env

# Environment variables - HF Hub delegation
export CUDA_VISIBLE_DEVICES=0

# Temporary workspace
export TMPDIR="${TMPDIR:-/tmp/gguf_conversion_$$}"
mkdir -p "$TMPDIR"
echo "📁 Temporary workspace: $TMPDIR"

# llama.cpp location (environment dependent)
export LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "❌ llama.cpp not found at $LLAMA_CPP_DIR"
    echo "💡 Set LLAMA_CPP_DIR environment variable or install to ~/llama.cpp"
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
CONVERTER="$PROJECT_ROOT/RQ/Phi/conver_to_gguf.py"

if [ ! -f "$CONVERTER" ]; then
    echo "❌ Could not locate conver_to_gguf.py at: $CONVERTER"
    exit 1
fi

echo "✅ Found converter script: $CONVERTER"
# Run GGUF conversion
echo "🚀 Starting GGUF conversion..."
python "$CONVERTER"

conversion_exit_code=$?

if [ $conversion_exit_code -eq 0 ]; then
    echo "🎉 GGUF conversion completed successfully!"
    echo "📊 File sizes in GGUF directory:"
    ls -lh "$TMPDIR/gguf_output/" 2>/dev/null || echo "GGUF files uploaded to HF Hub"
else
    echo "❌ GGUF conversion failed with exit code: $conversion_exit_code"
    exit 1
fi

# Cleanup temporary workspace
echo "🧹 Cleaning up temporary workspace: $TMPDIR"
rm -rf "$TMPDIR"

# Deactivate environment
source deactivate

echo "✅ GGUF conversion job completed!" 