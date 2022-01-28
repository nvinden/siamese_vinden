import os
import pandas as pd

dir_stem = "logs/" #"./"

log_list = list()

out = dict()

for file in os.listdir(dir_stem):
    if file.endswith(".log"):
        log_list.append(os.path.join(dir_stem, file))

for file in log_list:
    if "output" not in file:
        continue

    with open(file, "r") as f:
        last_line = f.read().splitlines()[-1]

    run_name = file[12 : -15]

    if "Time" in last_line:
        value = float(last_line[6:]) / (60 * 60)
    else:
        value = 36.0

    if run_name not in out:
        out[run_name] = [value]
    else:
        out[run_name][0] += value

df = pd.DataFrame.from_dict(out)
df = df.T
df = df.div(5)

df.to_csv("time.csv")

print("test")
    

    
