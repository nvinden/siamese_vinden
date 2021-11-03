import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
from pyjarowinkler import distance as jw

from dataset import RDataset
from process import load_json_config, emb2str

import random
import time
import os

def create_jw():
    DATASET_CONFIG, TRAIN_CONFIG, MODEL_KWARGS = load_json_config("configs/run01.json")
    dataset = RDataset(DATASET_CONFIG)

    if os.path.isfile("data/JW_negatives.csv") and os.path.isfile("data/JW_used_negatives.npy"):
        df = pd.read_csv("data/JW_negatives.csv")

        used_dict = np.load("data/JW_used_negatives.npy", allow_pickle = True).item()
    else:
        column_names = ["jw_score", "word1", "word2", "index"]
        df = pd.DataFrame(columns = column_names)

        used_dict = dict()

    master = dataset.master_dataset.table
    master_len = len(master)

    master_cross_product = master_len ** 2
    pair_dict = dataset.pair_dict.item()

    total_pairs = 0
    start_time = time.time()

    while(True):
        rn = random.randint(0, master_cross_product)

        if rn in used_dict:
            print("Same Index Found...")
            continue

        index1 = int(rn // master_len)
        index2 = int(rn % master_len)

        if index1 == index2:
            print("Same Index1 and Index2...")
            continue

        emb1 = master[index1]
        emb2 = master[index2]

        word1 = emb2str(emb1)
        word2 = emb2str(emb2)

        if word1 in pair_dict and word2 in pair_dict and pair_dict[word1] == word2:
            print(f"Pairs found: {word1} {word2}")
            continue

        used_dict[rn] = -1
        jw_distance = jw.get_jaro_distance(word1, word2, winkler=True, scaling=0.1)

        df = df.append({"jw_score": jw_distance, "word1": word1, "word2": word2, "index": rn}, ignore_index = True)
        total_pairs += 1

        if total_pairs % 50000 == 0:
            print(f"{total_pairs} negatives found in {time.time() - start_time}")

            np.save("data/JW_used_negatives.npy", used_dict, allow_pickle = True)

            df = df.sort_values(by = ["jw_score"], ascending = False, ignore_index = True)
            df.to_csv("data/JW_negatives.csv")

            start_time = time.time()

if __name__ == '__main__':
    create_jw()