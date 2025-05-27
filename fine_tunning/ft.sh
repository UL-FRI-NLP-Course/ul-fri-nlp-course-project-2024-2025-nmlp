#!/bin/bash
#SBATCH --job-name=fine_tune_gams
#SBATCH --output=outputs/ft_gams_%j.log
#SBATCH --error=errors/ft_gams_%j.log
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1

# module purge
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate gams

module purge
module load Python/3.12 bzip2
source /d/hpc/projects/onj_fri/nmlp/venv/bin/activate


echo "Starting GaMS fine-tuning job"
echo "Running on node: $(hostname)"

# Ensure Hugging Face cache uses scratch if needed
# export TRANSFORMERS_CACHE=$SLURM_TMPDIR/hf_cache

python -c "import transformers; print('✅ Transformers is installed')"

# Run your script
python gams.py

echo "Job finished"