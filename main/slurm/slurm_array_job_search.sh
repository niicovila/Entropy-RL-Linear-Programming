#!/bin/bash
#SBATCH --job-name=qreps-experiments
#SBATCH -p medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-1
#SBATCH --output=samples_cartpole_elbe_exp_2/arrayJob_%A_%a.out
#SBATCH --error=samples_cartpole_elbe_exp_2/arrayJob_%A_%a.err

module load "Miniconda3/4.9.2"
eval "$(conda shell.bash hook)"
conda activate qreps
poetry install
set -x

export WANDB_API_KEY=<your key>

poetry run python -u tuning/random_search_elbe.py ${SLURM_ARRAY_TASK_ID} ../data/elbe/cartpole/samples_cartpole_elbe_exp_2