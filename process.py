import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Siamese

import json
from random import randrange

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

def create_pretrained_vectors(model, embeddings):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    embeddings_in = torch.clone(embeddings)
    embeddings_truth = torch.clone(embeddings)
    replace_list = list()

    for i, curr_name in enumerate(embeddings_in):
        str_len = [torch.nonzero(row) for row in curr_name]
        
        char_index = 0
        while str_len[char_index].nelement() != 0:
            char_index += 1

        start_row = torch.zeros([curr_name.shape[1]], device = device)
        start_row[model.START] = 1

        end_row = torch.zeros([curr_name.shape[1]], device = device)
        end_row[model.END] = 1

        curr_name = torch.cat([start_row.unsqueeze(0), curr_name])
        curr_name[char_index + 1] = end_row

        curr_name = curr_name[:-1]

        embeddings_truth[i] = curr_name

        index_to_replace = randrange(char_index) + 1
        replace_list.append(index_to_replace)

        curr_name[index_to_replace] = torch.ones(curr_name[index_to_replace].shape, device = device)

        embeddings_in[i] = curr_name

    return embeddings_in, embeddings_truth, replace_list