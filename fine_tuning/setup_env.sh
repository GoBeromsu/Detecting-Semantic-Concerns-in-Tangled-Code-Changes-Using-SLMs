#!/bin/bash
#SBATCH --job-name=phi4-env-setup
#SBATCH --time=0:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/phi4_env_setup_%j.out
#SBATCH --error=logs/phi4_env_setup_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bkoh3@sheffield.ac.uk

# Sheffield HPC Stanage - Environment Setup for Phi-4 Fine-tuning
# Multi-Concern Commit Classification with Phi-4

echo "Setting up Phi-4 LoRA fine-tuning environment..."

# Create logs directory
mkdir -p logs

module purge
module load GCCcore/12.3.0
module load CUDA/12.1.1
module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load CMake/3.26.3-GCCcore-12.3.0

# Check if environment exists and reuse if available
if conda env list | grep -q "phi4_env"; then
    echo "♻️ Found existing phi4_env, reusing it..."
else
    echo "🏗️ Creating new conda environment..."
    if ! conda env create -f environment.yml; then
        echo "❌ Failed to create conda environment. Exiting..."
        exit 1
    fi
fi

# Activate environment using 'source activate' instead of 'conda activate'
# Sheffield HPC requirement: Due to Anaconda being installed as a module,
# must use 'source' command instead of 'conda' when activating environments
# Reference: https://docs.hpc.shef.ac.uk/en/latest/stanage/software/apps/python.html
echo "🔧 Activating phi4_env..."
source activate phi4_env

# Install packaging and ninja
echo "📦 Installing packaging and ninja..."
pip install packaging ninja

# Install PyTorch with CUDA 12.1.1 support
echo "📦 Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn with no build isolation
# Install flash-attn version 2.7.4.post1 without build isolation to ensure compatibility with existing environment
echo "📦 Installing flash-attn..."
pip install flash-attn==2.7.4.post1 --no-build-isolation

# Step 4: Install remaining ML dependencies
 echo "📦 Installing ML dependencies..."
pip install -r requirements.txt

echo "✅ Environment setup completed successfully!"
echo "To activate the environment manually: source activate phi4_env"

source deactivate 