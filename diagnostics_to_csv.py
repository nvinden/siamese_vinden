import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd

import time
import os
from datetime import datetime
from pyjarowinkler import distance as jw

from os import listdir
from os.path import isfile, join

from dataset import SiamesePairsDataset, SiameseMasterDataset
from model import Siamese
from process import save_data, load_data, load_json_config
from train import test_on_test_set

def diagnostics_to_csv():
    path = "saves"
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    columns = ["Run Name", "Pair Loss", "Master Loss", "Avg Loss", "Pair Accuracy", "Master Accuracy", "Avg Accuracy"]
    csv = pd.DataFrame(columns = columns)
    csv.astype({'Pair Loss': 'float64'}).dtypes
    csv.astype({'Master Loss': 'float64'}).dtypes
    csv.astype({'Avg Loss': 'float64'}).dtypes
    csv.astype({'Pair Accuracy': 'float64'}).dtypes
    csv.astype({'Master Accuracy': 'float64'}).dtypes
    csv.astype({'Avg Accuracy': 'float64'}).dtypes

    for save_name in onlyfiles:
        torch.manual_seed(0)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"CURRENT DEVICE: {device}")

        save_file = os.path.join("saves", str(save_name))
        json_file = os.path.join("configs", str(save_name) + ".json")

        DATASET_CONFIG, TRAIN_CONFIG, MODEL_KWARGS = load_json_config(json_file)

        #CREATING DATASETS
        pair_ds = SiamesePairsDataset(DATASET_CONFIG)
        master_ds = SiameseMasterDataset(DATASET_CONFIG)

        batch_size = TRAIN_CONFIG['batch_size']

        ttv_split = DATASET_CONFIG["ttv_split"]
        ttv_split_pair = [int(len(pair_ds) * elem) for elem in ttv_split]
        ttv_split_master = [len(master_ds) - ttv_split_pair[1] * 2, ttv_split_pair[1] * 2]

        train_pair, test_pair = torch.utils.data.random_split(pair_ds, [ttv_split_pair[0], ttv_split_pair[1]])
        train_master, test_master = torch.utils.data.random_split(master_ds, [ttv_split_master[0], ttv_split_master[1]])

        pair_loader_train = DataLoader(train_pair, batch_size = TRAIN_CONFIG['batch_size'], shuffle = True, drop_last = True)
        pair_loader_test = DataLoader(test_pair, batch_size = TRAIN_CONFIG['batch_size'], shuffle = True, drop_last = True)

        master_loader_train = DataLoader(train_master, batch_size = 2 * TRAIN_CONFIG['batch_size'], shuffle = True, drop_last = True)
        master_loader_test = DataLoader(test_master, batch_size = 2 * TRAIN_CONFIG['batch_size'], shuffle = True, drop_last = True)

        #LOADING FROM SAVE OR CREATING NEW DATA
        if not os.path.isfile(save_file):
            print("NO FILE FOUND")
            exit()
        else:
            start_epoch, model, optim, scheduler, log_list = load_data(save_file, TRAIN_CONFIG, MODEL_KWARGS)

        pair_accuracy, master_accuracy, pair_accuracy_jw, master_accuracy_jw = test_on_test_set(model, pair_loader_test, master_loader_test)
        average_accuracy = (pair_accuracy + master_accuracy) / 2
        average_accuracy_jw = (pair_accuracy_jw + master_accuracy_jw) / 2

        pl = log_list['pair_loss'][-1]
        ml = log_list['master_loss'][-1]
        al = log_list['avg_loss'][-1]

        if torch.is_tensor(pl):
            pl = pl.detach().numpy() 
        if torch.is_tensor(ml):
            ml = ml.detach().numpy() 
        if torch.is_tensor(al):
            al = al.detach().numpy() 

        csv = csv.append({"Run Name": save_name, "Pair Loss": pl, "Master Loss": ml, "Avg Loss": al, "Pair Accuracy": pair_accuracy, "Master Accuracy": master_accuracy, "Avg Accuracy": average_accuracy}, ignore_index = True)

    csv.to_csv("run_diagnostics.csv")

if __name__ == '__main__':
    diagnostics_to_csv()
