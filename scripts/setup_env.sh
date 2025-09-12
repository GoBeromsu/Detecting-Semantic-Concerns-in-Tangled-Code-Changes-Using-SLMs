#!/bin/bash
#SBATCH --job-name=setup-env
#SBATCH --time=0:30:00
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1 # https://docs.hpc.shef.ac.uk/en/latest/stanage/GPUComputingStanage.html#gsc.tab=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
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

REPO_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
echo "[INFO] Using repository root: $REPO_DIR"

module purge
module load GCCcore/12.3.0
module load CUDA/12.4.0
module load Anaconda3/2022.05
module load CMake/3.26.3-GCCcore-12.3.0

if ! conda env list | grep -q "venv"; then
    echo "[INFO] Creating new conda env 'venv'..."
    conda env create -f "$REPO_DIR/scripts/environment.yml"
fi
source activate venv

export TMPDIR=$PWD/tmp
mkdir -p $TMPDIR

echo "Installing uv..."
python -m pip install --upgrade uv

# Install CUDA-specific PyTorch separately (avoid generic CPU wheel)
echo "Installing CUDA-specific PyTorch..."
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# Install flash-attn explicitly; --no-build-isolation ensures env compilers are used
echo "Installing flash-attn (no build isolation)..."
python -m pip install flash-attn --no-build-isolation

# Install application dependencies from pyproject extras (.[hpc])
cd "$REPO_DIR"
uv pip install -e ".[hpc]"

uv pip uninstall llama-cpp-python
rm -rf ~/.cache/uv/ ~/.cache/pip/
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=90" \
python -m pip install llama-cpp-python --no-binary llama-cpp-python --no-cache-dir -v

python -m pip install deepspeed accelerate

echo "Environment setup completed. To activate later: source activate venv"
source deactivate