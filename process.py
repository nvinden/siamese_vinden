import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Siamese

import json
from random import randrange

def save_data(path, epoch, model, optimizer, scheduler, log_list, dataset):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'log_list': log_list,
            'dataset': dataset
            
    }, path)

    print(f"Saved run successfully at {path}")

def load_data(path, TRAIN_CONFIG, MODEL_KWARGS):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.load(path, map_location = torch.device(device))

    epoch = data['epoch']
    log_list = data['log_list']

    model = Siamese(TRAIN_CONFIG, MODEL_KWARGS).to(device)
    model.load_state_dict(data['model_state_dict'])

    optim = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['lr'])
    optim.load_state_dict(data['optimizer_state_dict'])

    ds = data['dataset']

    if 'scheduler_state_dict' in data:
        scheduler = torch.optim.lr_scheduler.StepLR(optim, TRAIN_CONFIG['scheduler_step_size'], gamma=TRAIN_CONFIG['scheduler_gamma'])
        scheduler.load_state_dict(data['scheduler_state_dict'])
    else:
        scheduler = None

    print(f"Loaded run successfully from {path}")

    return epoch, model, optim, scheduler, log_list, ds

def add_to_log_list(log_list, loss, accuracy = None):
    if not "loss" in log_list:
        log_list["loss"] = list()
    if not "accuracy" in log_list:
        log_list["accuracy"] = list()
        
    log_list["loss"].append(loss)

    if accuracy is not None:
        log_list["accuracy"].append(accuracy)
        
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

def print_log_list_diagnostics(log_list):
    print("losses: pair, master, average")
    for i, (p, m, a) in enumerate(zip(log_list['pair_loss'], log_list['master_loss'], log_list['avg_loss'])):
        print(i + 1, "{:.5f}".format(p.item()), "{:.5f}".format(m.item()), "{:.5f}".format(a.item()))

    print("accuracy: pair, master, average")
    for i, (p, m, a) in enumerate(zip(log_list['pair_test'], log_list['master_test'], log_list['avg_test'])):
        print(i + 1, "{:.5f}".format(p.item()), "{:.5f}".format(m.item()), "{:.5f}".format(a.item()))

def emb2str(emb):
    word = ""
    for char in emb:
        char = char.item()
        if char >= 30:
            continue
        word = word + chr(char + 97)
    return word

def str2emb(string, string_pad = 30):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_table = torch.full(size = (string_pad, ), fill_value = 32, device = device, dtype = torch.uint8)
    torch_table[0] = 30
    for j in range(1, len(string) + 1):
        cha = string[j - 1]
        torch_table[j] = ord(cha) - 97
    j += 1
    torch_table[j] = 31

    return torch_table

def contrastive_loss(v_i, v_j, y, m : float = 1.0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cs = nn.CosineSimilarity(dim=1, eps=1e-8)

    sim = cs(v_i, v_j)
    sim = sim.to(device)
    y = y.to(device)

    l = 0.5 * y * (sim ** 2)
    maxs = torch.max(torch.zeros([sim.shape[0]], device = device, requires_grad = False, dtype = torch.float), m - sim)
    r = 0.5  * (1 - y) * (maxs ** 2)

    return torch.mean(l + r)