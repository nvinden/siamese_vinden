#!/bin/bash
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=8G
#SBATCH --time=36:00:00
#SBATCH --job-name=hard_emb25_lay16
#SBATCH --array=0-4
#SBATCH --output=logs/output_hard_emb25_lay16_%a_%j.log
#SBATCH --account=def-lantonie
module load python/3.9.6
source ~/nick/bin/activate
python train.py hard_emb25_lay16 $SLURM_ARRAY_TASK_ID
