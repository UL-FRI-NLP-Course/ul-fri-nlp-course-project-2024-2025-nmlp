#!/bin/bash
#SBATCH --job-name=9b-inference
#SBATCH --output=outputs/9b-inf_%j.log
#SBATCH --error=errors/9b-inf_%j.log
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# module purge
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate gams

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gams


echo "Starting inference job"
echo "Running on node: $(hostname)"

# Ensure Hugging Face cache uses scratch if needed
# export TRANSFORMERS_CACHE=$SLURM_TMPDIR/hf_cache

# Run your script
srun python inference_with_fine_tunning.py

echo "Job finished"