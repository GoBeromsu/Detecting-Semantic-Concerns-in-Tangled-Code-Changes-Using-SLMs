#!/bin/bash
#SBATCH --job-name=setup-env
#SBATCH --time=0:30:00
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/setup_env_%j.out
#SBATCH --error=logs/setup_env_%j.err

# Sheffield HPC Stanage - Centralized Environment Setup
# - Uses scripts/environment.yml for base Conda env
# - Uses uv and pyproject.toml extras (.[hpc]) for application deps
# - Installs CUDA-specific PyTorch separately to avoid conflicts
# - Keeps flash-attn install explicit with --no-build-isolation
# - Activates with 'source activate' (HPC module requirement)
# Reference: https://docs.hpc.shef.ac.uk/en/latest/stanage/software/apps/python.html

set -euo pipefail

echo "Setting up HPC environment using uv and pyproject extras..."

mkdir -p logs

# Resolve repository root using Git
REPO_DIR="$(git rev-parse --show-toplevel 2>/dev/null || { echo "ERROR: Not in a Git repository"; exit 1; })"
echo "Using Git repository root: $REPO_DIR"

module purge
module load GCCcore/12.3.0
module load CUDA/12.1.1
module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load CMake/3.26.3-GCCcore-12.3.0

# Create or reuse conda environment from YAML
if conda env list | grep -q "phi4_env"; then
    echo "Found existing conda env phi4_env. Reusing."
else
    echo "Creating conda env from scripts/environment.yml..."
    conda env create -f "$REPO_DIR/scripts/environment.yml" || { echo "Failed to create conda env"; exit 1; }
fi

# Activate via 'source activate' (required when Anaconda is provided via module)
echo "Activating conda env (source activate)..."
source activate phi4_env

# Use uv for fast, reproducible pip installs
echo "Installing uv..."
python -m pip install --upgrade uv

# Install CUDA-specific PyTorch separately (avoid generic CPU wheel)
echo "Installing CUDA-specific PyTorch..."
python -m pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn explicitly; --no-build-isolation ensures env compilers are used
echo "Installing flash-attn (no build isolation)..."
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation

# Install application dependencies from pyproject extras (.[hpc])
cd "$REPO_DIR"
uv pip install -e ".[hpc]"

echo "Installing GPU-enabled llama-cpp-python (CUDA 12.1, SM_90)..."
uv pip uninstall llama-cpp-python
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON" \
uv pip install llama-cpp-python --no-binary llama-cpp-python -v

echo "Environment setup completed. To activate later: source activate phi4_env"

source deactivate


