#!/bin/bash
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=8G
#SBATCH --time=18:00:00
#SBATCH --job-name=20
#SBATCH --array=0-4
#SBATCH --output=20_%a_%j.log
module load python/3.9.6
source ~/nick/bin/activate
python train.py 20 $SLURM_ARRAY_TASK_ID
