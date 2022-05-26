import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, f1_score
from statistics import mean

from collections import defaultdict

import pandas as pd

import os

path_to = "results/"
dir_names = [name for name in os.listdir(path_to) if os.path.isdir(os.path.join(path_to, name))]

all_info = list()

for dir in dir_names:
    scores = defaultdict(list)
    threshold_list = []
    for k in range(5):
        path = os.path.join(path_to, dir, f"test_k{k}.csv")

        if not os.path.isfile(path):
            continue

        df = pd.read_csv(path, usecols = ["model_score", "label"])
        df = df.astype({"label": int, "model_score": float})

        label = df['label'].to_numpy().astype(np.int8)
        model_score = df['model_score'].to_numpy()

        precision, recall, thresholds = precision_recall_curve(label, model_score)

        f_score_list = list()
        for pre, rec in zip(precision, recall):
            lower = pre + rec
            if lower == 0.0:
                lower = 0.0000000001
            
            f_score_list.append(2 * (pre * rec) / lower)

        threshold_index = np.argmax(np.array(f_score_list))
        threshold = thresholds[threshold_index]

        f_score = max(f_score_list)
        scores['f_score'].append(f_score)

        yp = (model_score >= threshold).astype(bool)
        yt = label.astype(bool)

        threshold_list.append(threshold)

        scores['tp'].append(np.count_nonzero(yt & yp))
        scores['tn'].append(np.count_nonzero(np.logical_not(yt) & np.logical_not(yp)))
        scores['fp'].append(np.count_nonzero(np.logical_not(yt) & yp))
        scores['fn'].append(np.count_nonzero(yt & np.logical_not(yp)))

    temp = {k: mean(v) for k, v in scores.items()}
    temp["name"] = dir
    for i, threshold in enumerate(threshold_list):
        temp[f"t{i}"] = threshold
    all_info.append(temp)

df = pd.DataFrame(all_info)
df.to_csv("Excel_Measures.csv")
        
