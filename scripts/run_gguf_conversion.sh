#!/bin/bash
#SBATCH --job-name=phi4-gguf-convert
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/phi4_gguf_convert_%j.out
#SBATCH --error=logs/phi4_gguf_convert_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bkoh3@sheffield.ac.uk

# Sheffield HPC Stanage - GGUF Conversion for Phi-4 Fine-tuned Model
# Convert merged LoRA model to GGUF format and upload to Hugging Face

echo "Starting GGUF conversion process..."

# Create logs directory
mkdir -p logs

# Setup environment - Match setup_env.sh configuration
module purge
module load GCCcore/12.3.0
module load CUDA/12.1.1
module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1
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

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export FASTDATA_BASE="/mnt/parscratch/users/$USER"

# Ensure FASTDATA_BASE directory exists
mkdir -p "$FASTDATA_BASE"

# Clone llama.cpp if not exists
LLAMA_CPP_DIR="$FASTDATA_BASE/llama.cpp"
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "📦 Cloning llama.cpp..."
    cd "$FASTDATA_BASE"
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    
    # Install Python requirements
    pip install -r requirements.txt
    
    # Build llama.cpp using CMake (new method)
    echo "🔨 Building llama.cpp with CMake..."
    
    # CMake build configuration with CUDA support
    cmake -B build -DGGML_CUDA=ON
    cmake --build build --config Release -j4
    
    if [ $? -eq 0 ]; then
        echo "✅ llama.cpp built successfully"
        
        # Verify build outputs exist
        echo "🔍 Verifying build outputs..."
        ls -la build/bin/
        
        # Check for essential binaries
        REQUIRED_BINARIES=("llama-quantize" "llama-cli")
        MISSING_BINARIES=()
        
        for binary in "${REQUIRED_BINARIES[@]}"; do
            if [ ! -f "build/bin/$binary" ]; then
                MISSING_BINARIES+=("$binary")
            else
                echo "✅ Found: build/bin/$binary"
            fi
        done
        
        if [ ${#MISSING_BINARIES[@]} -gt 0 ]; then
            echo "❌ Missing required binaries: ${MISSING_BINARIES[*]}"
            echo "Build may have failed or incomplete"
            exit 1
        fi
        
        echo "✅ All required binaries found"
    else
        echo "❌ llama.cpp build failed"
        exit 1
    fi
else
    echo "✅ llama.cpp already exists"
fi

# Return to fine_tuning directory
cd "$SLURM_SUBMIT_DIR"

# Load environment variables if .env exists
if [ -f ".env" ]; then
    echo "📄 Loading environment variables from .env"
    source .env
fi

# Run GGUF conversion
echo "🚀 Starting GGUF conversion..."
python conver_to_gguf.py

conversion_exit_code=$?

if [ $conversion_exit_code -eq 0 ]; then
    echo "🎉 GGUF conversion completed successfully!"
    echo "📊 File sizes in GGUF directory:"
    ls -lh "$FASTDATA_BASE/models/gguf/"
else
    echo "❌ GGUF conversion failed with exit code: $conversion_exit_code"
    exit 1
fi

# Deactivate environment
source deactivate

echo "✅ GGUF conversion job completed!" 