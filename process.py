import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Siamese

import json

def save_data(path, epoch, model, optimizer, log_list):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'log_list': log_list
    }, path)

    print(f"Saved run successfully at {path}")

def load_data(path, TRAIN_CONFIG, MODEL_KWARGS):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.load(path)

    epoch = data['epoch']
    log_list = data['log_list']

    model = Siamese(TRAIN_CONFIG, MODEL_KWARGS).to(device)
    model.load_state_dict(data['model_state_dict'])

    optim = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['lr'])
    optim.load_state_dict(data['optimizer_state_dict'])

    print(f"Loaded run successfully from {path}")

    return epoch, model, optim, log_list

def add_to_log_list(log_list, pair_loss, master_loss, avg_loss, pair_test = None, master_test = None, avg_test = None):
    if not "pair_loss" in log_list:
        log_list["pair_loss"] = list()
    if not "master_loss" in log_list:
        log_list["master_loss"] = list()
    if not "avg_loss" in log_list:
        log_list["avg_loss"] = list()
    if not "pair_test" in log_list:
        log_list["pair_test"] = list()
    if not "master_test" in log_list:
        log_list["master_test"] = list()
    if not "avg_test" in log_list:
        log_list["avg_test"] = list()
        
    log_list["pair_loss"].append(pair_loss)
    log_list["master_loss"].append(master_loss)
    log_list["avg_loss"].append(avg_loss)

    if pair_loss is not None:
        log_list["pair_test"].append(pair_loss)
    if master_loss is not None:
        log_list["master_test"].append(master_loss)
    if avg_loss is not None:
        log_list["avg_test"].append(avg_loss)

def load_json_config(path):
    with open(path) as f:
        data = json.load(f)

    return data["DATASET_CONFIG"], data["TRAIN_CONFIG"], data["MODEL_KWARGS"]
