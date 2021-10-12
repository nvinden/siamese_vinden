import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import time
import os
from datetime import datetime

from dataset import SiamesePairsDataset, SiameseMasterDataset
from model import Siamese
from process import save_data, load_data, add_to_log_list, load_json_config

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

    train_pair = pair_ds[:ttv_split_pair[0]]
    test_pair = pair_ds[ttv_split_pair[0]:]

    train_master = master_ds[:ttv_split_master[0]]
    test_master = master_ds[ttv_split_master[0]:]

    assert len(train_pair['name0']) == ttv_split_pair[0]
    assert len(test_pair['name0']) == ttv_split_pair[1]
    assert len(train_master['name']) == ttv_split_master[0]
    assert len(test_master['name']) == ttv_split_master[1]

    pair_loader_train = DataLoader(train_pair, batch_size = TRAIN_CONFIG['batch_size'], shuffle = True, drop_last = True)
    pair_loader_test = DataLoader(test_pair, batch_size = TRAIN_CONFIG['batch_size'], shuffle = True, drop_last = True)

    master_loader_train = DataLoader(train_master, batch_size = 2 * TRAIN_CONFIG['batch_size'], shuffle = True, drop_last = True)
    master_loader_test = DataLoader(test_master, batch_size = 2 * TRAIN_CONFIG['batch_size'], shuffle = True, drop_last = True)

    #LOADING FROM SAVE OR CREATING NEW DATA
    if not os.path.isfile(save_file):
        model = Siamese(TRAIN_CONFIG, MODEL_KWARGS)

        optim = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['lr'])
        log_list = {}

        start_epoch = 0
    else:
        start_epoch, model, optim, log_list = load_data(save_file, TRAIN_CONFIG, MODEL_KWARGS)

    criterion = nn.MSELoss()

    model = model.to(device)
    criterion = criterion.to(device)

    model.eval()
    start_time = time.time()

    for test in pair_loader_test:
        print("TEST 1")

    for test in master_loader_test:
        print("TEST 2")

    for batch_no, (pair_data, master_data) in enumerate(zip(pair_loader_test, master_loader_test)):
        #OPTIMIZING ON PAIR
        pair0 = pair_data['name0']
        pair1 = pair_data['name1']
        pair0.to(device)
        pair1.to(device)

        target_pair = torch.zeros([len(pair0)], dtype = torch.float, device = device)

        out_pair, _ = model(pair0, pair1)

        #OPTIMIZING ON MASTER
        master0 = master_data['name'][0:batch_size]
        master1 = master_data['name'][batch_size:]
        master0.to(device)
        master1.to(device)

        target_master = torch.ones([len(pair0)], dtype = torch.float, device = device)

        out_master, _ = model(master0, master1)

if __name__ == '__main__':
    config_list = ["run02", ]

    log_file_name = "log.txt"
    debug = True

    print("GPU Available: " + str(torch.cuda.is_available()))

    with open(log_file_name, "a") as log:
        for config in config_list:
            start_time = time.time()

            diagnose(config)