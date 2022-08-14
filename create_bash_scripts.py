import os

path_to = "/home/nick/Projects/Research/siamese_vinden/configs"
dir_names = [name for name in os.listdir(path_to) if os.path.isfile(os.path.join(path_to, name))]

for name in dir_names:
    name = name.replace(".json", "")
    out_str = f"#!/bin/bash\n#SBATCH --gres=gpu:p100:1\n#SBATCH --mem=8G\n#SBATCH --time=36:00:00\n#SBATCH --job-name={name}\n#SBATCH --array=0-4\n#SBATCH --output=logs/output_{name}_%a_%j.log\n#SBATCH --account=def-lantonie\nmodule load python/3.9.6\nsource ~/nick/bin/activate\npython train.py {name} $SLURM_ARRAY_TASK_ID\n"

    out_str = out_str.replace("\t", "")

    text_file = open(f"bash/{name}.sh", "w")
    n = text_file.write(out_str)
    text_file.close()
