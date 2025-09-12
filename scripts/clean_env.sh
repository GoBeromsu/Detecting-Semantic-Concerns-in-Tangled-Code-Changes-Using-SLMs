#!/bin/bash
#SBATCH --job-name=clean-env
#SBATCH --time=0:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/clean_env_%j.out
#SBATCH --error=logs/clean_env_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bkoh3@sheffield.ac.uk

# Create logs directory
mkdir -p logs

module purge
module load GCCcore/12.3.0
module load Anaconda3/2022.05

# Remove existing environment if exists
if conda env list | grep -q "venv"; then
    echo "🗑️ Removing existing venv..."
    conda remove -n venv --all -y
    echo "✅ Environment venv removed successfully!"
else
    echo "ℹ️ Environment venv not found. Nothing to clean."
fi

echo "🧹 Cleanup completed!" 