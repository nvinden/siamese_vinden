#!/bin/bash
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=8G
#SBATCH --time=36:00:00
#SBATCH --job-name=emb50_rat11_gru_bi0_lay1
#SBATCH --array=0-4
#SBATCH --output=logs/output_emb50_rat11_gru_bi0_lay1_%a_%j.log
#SBATCH --account=def-lantonie
module load python/3.9.6
source ~/nick/bin/activate
python train.py emb50_rat11_gru_bi0_lay1 $SLURM_ARRAY_TASK_ID
