#!/bin/bash
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=8G
#SBATCH --time=15:00:00
#SBATCH --job-name=edit_emb5_lay4
#SBATCH --array=0-4
#SBATCH --output=logs/gru_soundex_%a_%j.log
#SBATCH --account=def-lantonie
module load python/3.9.6
source ~/nick/bin/activate
python train.py gru_soundex $SLURM_ARRAY_TASK_ID
