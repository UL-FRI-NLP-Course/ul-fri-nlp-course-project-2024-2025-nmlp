#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --output=outputs/eval_output_%j.log
#SBATCH --error=errors/eval_error_%j.log
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --nodes=1

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gams

echo "Testing evaluation"

srun python evaluation.py