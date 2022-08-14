#!/bin/bash
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=8G
#SBATCH --time=36:00:00
#SBATCH --job-name=emb100_rat11_lstm_bi1_lay2
#SBATCH --array=0-4
#SBATCH --output=logs/output_emb100_rat11_lstm_bi1_lay2_%a_%j.log
#SBATCH --account=def-lantonie
module load python/3.9.6
source ~/nick/bin/activate
python train.py emb100_rat11_lstm_bi1_lay2 $SLURM_ARRAY_TASK_ID
