#!/bin/bash
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=8G
#SBATCH --time=36:00:00
#SBATCH --job-name=test
#SBATCH --array=0-4
#SBATCH --output=logs/output_test_%a_%j.log
#SBATCH --account=def-lantoine
module load python/3.9.6
source ~/nick/bin/activate
python train.py test $SLURM_ARRAY_TASK_ID
