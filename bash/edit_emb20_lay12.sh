#!/bin/bash
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=8G
#SBATCH --time=36:00:00
#SBATCH --job-name=edit_emb20_lay12
#SBATCH --array=0-4
#SBATCH --output=logs/output_edit_emb20_lay12_%a_%j.log
#SBATCH --account=def-lantonie
module load python/3.9.6
source ~/nick/bin/activate
python train.py edit_emb20_lay12 $SLURM_ARRAY_TASK_ID
