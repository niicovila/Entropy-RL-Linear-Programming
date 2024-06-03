#!/bin/bash
#SBATCH --job-name=qreps
#SBATCH -p medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-999
#SBATCH --output=samples_acro_elbe_exp_5/arrayJob_%A_%a.out
#SBATCH --error=samples_acro_elbe_exp_5/arrayJob_%A_%a.err

module load "Miniconda3/4.9.2"
eval "$(conda shell.bash hook)"
conda activate qreps
poetry install
set -x

poetry run python -u ray_tune/random_search_elbe_exp_2.py ${SLURM_ARRAY_TASK_ID} samples_acro_elbe_exp_5