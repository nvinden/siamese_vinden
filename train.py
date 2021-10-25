import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import time
import os
from datetime import datetime
from pyjarowinkler import distance as jw
import sys

from dataset import SiamesePairsDataset, SiameseMasterDataset, EmbeddingsMasterList, RDataset
from model import Siamese
from process import save_data, load_data, add_to_log_list, load_json_config, emb2str, contrastive_loss

def train(save_name):
    torch.manual_seed(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"CURRENT DEVICE: {device}")

    save_file = os.path.join("saves", str(save_name))
    json_file = os.path.join("configs", str(save_name) + ".json")

    DATASET_CONFIG, TRAIN_CONFIG, MODEL_KWARGS = load_json_config(json_file)

    ttv_splits = DATASET_CONFIG['ttv_split']

    #LOADING FROM SAVE OR CREATING NEW DATA
    if not os.path.isfile(save_file):
        model = Siamese(TRAIN_CONFIG, MODEL_KWARGS)

        optim = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=TRAIN_CONFIG["scheduler_step_size"], gamma=TRAIN_CONFIG["scheduler_gamma"])
        log_list = {}

        ds = RDataset(DATASET_CONFIG)

        start_epoch = 0
    else:
        start_epoch, model, optim, scheduler, log_list, ds = load_data(save_file, TRAIN_CONFIG, MODEL_KWARGS)

        if 'epoch_reset' in TRAIN_CONFIG and TRAIN_CONFIG['epoch_reset']:
            start_epoch = 0
            optim = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['lr'])
            scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=TRAIN_CONFIG["scheduler_step_size"], gamma=TRAIN_CONFIG["scheduler_gamma"])

    criterion = contrastive_loss

    ds = RDataset(DATASET_CONFIG)

    model = model.to(device)

    for epoch in range(start_epoch, TRAIN_CONFIG["n_epochs"]):
        model.train()
        model.requires_grad_()
        start_time = time.time()

        #locking parameters of the encoder for first number of epochs
        if TRAIN_CONFIG["A_name"] in ["attention", ] and "attention_lock_encoder_epochs" in TRAIN_CONFIG:
            epoch_lock = TRAIN_CONFIG["attention_lock_encoder_epochs"]
            if epoch > epoch_lock:
                model.A_function.training = False
            else:
                model.A_function.training = True

        total_epoch_loss = 0

        ds.mode = "train"
        for batch_no, data in enumerate(ds):
            #OPTIMIZING ON PAIR
            optim.zero_grad()

            n0 = data['emb0']
            n1 = data['emb1']
            label = data['label']

            n0.requires_grad = False
            n1.requires_grad = False
            label.requires_grad = False

            n0.to(device)
            n1.to(device)
            label.to(device)

            name_similarity, (v_i, v_j) = model(n0, n1)

            loss = criterion(v_i, v_j, label)
            loss.backward()
            optim.step()

            #ADDING TO DIAGNOSTICS
            total_epoch_loss += loss.item()

        scheduler.step()

        if epoch >= 10:
            print("Embedding...")
            ds.embeddings.embed_all(model)
            print("Embedding done...")

            print("Adding to dataset...")
            n_added = ds.add_to_dataset()
            print(f"{n_added} entries added")

        #PRINTING DIAGNOSTICS
        total_epoch_loss /= (batch_no + 1)

        print(f"\nEpoch {epoch + 1}:")
        print(f"          Loss: {total_epoch_loss}")
        if (epoch + 1) % 10 == 0:
            save_data(save_file, epoch, model, optim, scheduler, log_list, ds)
            #accuracy = test_on_test_set(model, test_dl)
            #add_to_log_list(log_list, total_epoch_pair_loss, total_epoch_master_loss, total_epoch_average_loss, pair_accuracy, master_accuracy, average_accuracy, pair_accuracy_jw, master_accuracy_jw, average_accuracy_jw)
            #print(f"          Test: {accuracy}")
            #save_data(save_file, epoch, model, optim, scheduler, log_list)
        else:
            pass
            #add_to_log_list(log_list, total_epoch_pair_loss, total_epoch_master_loss, total_epoch_average_loss)

        print(f" TIME: {time.time() - start_time} seconds")

    return total_epoch_loss, accuracy

def test_on_test_set(model, test_dl):
    jw_k = 0.7
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_accuracy = 0
    total_accuracy_jw = 0
    criterion = contrastive_loss
    with torch.no_grad():
        for batch_no, data in enumerate(test_dl):
            n0 = data['n0']
            n1 = data['n1']
            label = data['label']

            n0.requires_grad = False
            n1.requires_grad = False
            label.requires_grad = False

            n0 = n0.to(device)
            n1 = n1.to(device)
            label = label.to(device)

            name_similarity, (v_i, v_j) = model(n0, n1)
            loss = criterion(v_i, v_j, label)

            total_accuracy += loss.item()

    total_accuracy /= (batch_no + 1)

    return total_accuracy

if __name__ == '__main__':
    config_list = ["run01", ]

    log_file_name = "log.txt"
    debug = True

    print("GPU Available: " + str(torch.cuda.is_available()))

    with open(log_file_name, "a") as log:
        for config in config_list:
            start_time = time.time()

            if debug == True:
                train_loss, test_loss = train(config)
                text_out = f"{datetime.now()}\n{config}: training complete\nTime: {time.time() - start_time}\nTrain Loss: {train_loss} \
                        \nTest Loss: {test_loss}"
            else:
                try:
                    train_loss, test_loss = train(config)
                    text_out = f"{datetime.now()}\n{config}: training complete\nTime: {time.time() - start_time}\nTrain Loss: {train_loss} \
                            \nTest Loss: {test_loss}"
                except Exception as e:
                    text_out = f"{datetime.now()}\n{config}: training failure\n{str(e)}\n"

            print("\n" + text_out)
            log.write(text_out + "\n")