import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd

import time
import os
from datetime import datetime
from pyjarowinkler import distance as jw

from dataset import SiamesePairsDataset, SiameseMasterDataset
from model import Siamese
from process import save_data, load_data, add_to_log_list, load_json_config, print_log_list_diagnostics, emb2str

def diagnose(save_name):
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
        print("NO SAVE FILE FOUND")
        exit()
    else:
        start_epoch, model, optim, scheduler, log_list = load_data(save_file, TRAIN_CONFIG, MODEL_KWARGS)

    print_log_list_diagnostics(log_list)

    criterion = nn.MSELoss()

    model = model.to(device)
    criterion = criterion.to(device)

    model.eval()
    start_time = time.time()

    column_names = ["Distance Score", "JW Distance", "Name0", "Name1"]

    false_pair_list = pd.DataFrame(columns = column_names)
    false_master_list = pd.DataFrame(columns = column_names)
    true_pair_list = pd.DataFrame(columns = column_names)
    true_master_list = pd.DataFrame(columns = column_names)

    false_pair_list.astype({'Distance Score': 'float64'}).dtypes
    false_master_list.astype({'Distance Score': 'float64'}).dtypes
    true_pair_list.astype({'Distance Score': 'float64'}).dtypes
    true_master_list.astype({'Distance Score': 'float64'}).dtypes

    false_pair_list.astype({'JW Distance': 'float64'}).dtypes
    false_master_list.astype({'JW Distance': 'float64'}).dtypes
    true_pair_list.astype({'JW Distance': 'float64'}).dtypes
    true_master_list.astype({'JW Distance': 'float64'}).dtypes

    for batch_no, (pair_data, master_data) in enumerate(zip(pair_loader_test, master_loader_test)):
        #OPTIMIZING ON PAIR
        pair0 = pair_data['name0']
        pair1 = pair_data['name1']
        pair0.to(device)
        pair1.to(device)

        target_pair = torch.zeros([len(pair0)], dtype = torch.float, device = device)

        out_pair, _ = model(pair0, pair1)
        pair_fails = out_pair >= 0.5

        for pair_no, pair_truth in enumerate(pair_fails):
            dist = out_pair[pair_no].item()

            name0 = pair0[pair_no]
            name0 = emb2str(name0)

            name1 = pair1[pair_no]
            name1 = emb2str(name1)

            jw_distance = jw.get_jaro_distance(name0, name1, winkler=True, scaling=0.1)

            if pair_truth:
                false_pair_list = false_pair_list.append({"Distance Score": dist, "JW Distance": jw_distance, "Name0": name0, "Name1": name1}, ignore_index = True)
            else:
                true_pair_list = true_pair_list.append({"Distance Score": dist, "JW Distance": jw_distance, "Name0": name0, "Name1": name1}, ignore_index = True)

        #OPTIMIZING ON MASTER
        master0 = master_data['name'][0:batch_size]
        master1 = master_data['name'][batch_size:]
        master0.to(device)
        master1.to(device)

        target_master = torch.ones([len(pair0)], dtype = torch.float, device = device)

        out_master, _ = model(master0, master1)
        master_fails = out_master <= 0.5

        for master_no, master_truth in enumerate(master_fails):
            dist = out_master[master_no].item()

            name0 = master0[master_no]
            name0 = emb2str(name0)

            name1 = master1[master_no]
            name1 = emb2str(name1)

            jw_distance = jw.get_jaro_distance(name0, name1, winkler=True, scaling=0.1)

            if master_truth:
                false_master_list = false_master_list.append({"Distance Score": dist, "JW Distance": jw_distance, "Name0": name0, "Name1": name1}, ignore_index = True)
            else:
                true_master_list = true_master_list.append({"Distance Score": dist, "JW Distance": jw_distance, "Name0": name0, "Name1": name1}, ignore_index = True)

    
    false_pair_list = false_pair_list.sort_values(by=['Distance Score'], ascending = False)
    false_master_list = false_master_list.sort_values(by=['Distance Score'])
    true_pair_list = true_pair_list.sort_values(by=['Distance Score'])
    true_master_list = true_master_list.sort_values(by=['Distance Score'], ascending = False)

    false_pair_csv = os.path.join("data", save_name, "false_pair.csv")
    false_master_csv = os.path.join("data", save_name, "false_master.csv")
    true_pair_csv = os.path.join("data", save_name, "true_pair.csv")
    true_master_csv = os.path.join("data", save_name, "true_master.csv")

    if not os.path.isdir(os.path.join("data", save_name)):
        os.mkdir(os.path.join("data", save_name))

    false_pair_list.to_csv(false_pair_csv)
    false_master_list.to_csv(false_master_csv)
    true_pair_list.to_csv(true_pair_csv)
    true_master_list.to_csv(true_master_csv)

if __name__ == '__main__':
    config_list = ["run_with_attention_1", ]

    log_file_name = "log.txt"
    debug = True

    print("GPU Available: " + str(torch.cuda.is_available()))

    with open(log_file_name, "a") as log:
        for config in config_list:
            start_time = time.time()

            diagnose(config)