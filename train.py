import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd

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
    criterion = contrastive_loss

    model = model.to(device)

    #preparing saving directory
    result_directory = os.path.join("results", save_name)
    if not os.path.isdir(result_directory):
        os.mkdir(result_directory)
    if not os.path.isdir(os.path.join(result_directory, "train")):
        os.mkdir(os.path.join(result_directory, "train"))
    if not os.path.isdir(os.path.join(result_directory, "test")):
        os.mkdir(os.path.join(result_directory, "test"))

    #training loop
    for epoch in range(start_epoch, TRAIN_CONFIG["n_epochs"]):
        model.train()
        model.requires_grad_()
        start_time = time.time()

        total_epoch_loss = 0
        total_pairs = len(ds)

        data_save_condition = (epoch % 5 == 0)

        if data_save_condition:
            model_dict = dict()
            dict_index = 0

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

            # Saving dat
            if data_save_condition:
                name1_list = [emb2str(i) for i in n0]
                name2_list = [emb2str(i) for i in n1]

                for n1, n2, mod, lab in zip(name1_list, name2_list, name_similarity, label):
                    model_dict[dict_index] = {"name1": n1, "name2": n2, "model_score": mod.item(), "label": lab.item()}
                    dict_index += 1

            #ADDING TO DIAGNOSTICS
            total_epoch_loss += loss.item()

        if data_save_condition:
            path_train = os.path.join(result_directory, "train", f"epoch{str(epoch).zfill(3)}.csv")
            path_test = os.path.join(result_directory, "test", f"epoch{str(epoch).zfill(3)}.csv")
            df = pd.DataFrame.from_dict(model_dict, "index")
            df.to_csv(path_train)

            # Run on test set as well
            accuracy = save_test_list(model, ds, path_test)

            save_data(save_file, epoch + 1, model, optim, scheduler, log_list, ds)

        scheduler.step()

        #PRINTING DIAGNOSTICS
        total_epoch_loss /= (batch_no + 1)

        print(f"\nEpoch {epoch + 1}:")
        print(f"          Loss: {total_epoch_loss}")
        '''
        if (epoch + 1) % 10 == 0:
            save_data(save_file, epoch + 1, model, optim, scheduler, log_list, ds)
            accuracy = save_test_list(model, ds, save_name)
            add_to_log_list(log_list, total_epoch_loss, accuracy)
            print(f"          Test: {accuracy}")

            ds.mode = "train"

            ds.shuffle_ds()
        else:
            add_to_log_list(log_list, total_epoch_loss)
            ds.shuffle_ds()
        '''

        '''
        if (epoch + 1) % 3 == 0:
            print("Embedding...")
            ds.embeddings.embed_all(model)
            print("Embedding done...")

            print("Adding to dataset...")
            n_added, pairs_found, already_found = ds.add_to_dataset()
            ds.embeddings.embeddings = None
            print(f"{n_added} entries added, {pairs_found} pairs found, {already_found} already found...")
        '''

        print(f"trained on {total_pairs} pairs")
        print(f" TIME: {time.time() - start_time} seconds")

    return 0, 0

def save_test_list(model, ds, save_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_dict = dict()
    dict_index = 0

    original_mode = ds.mode
    model.eval()
    criterion = contrastive_loss

    total_accuracy = 0
    with torch.no_grad():
        ds.mode = "test"
        for batch_no, data in enumerate(ds):
            if len(data) == 0:
                continue

            n0 = data['emb0']
            n1 = data['emb1']
            label = data['label']

            n0.requires_grad = False
            n1.requires_grad = False
            label.requires_grad = False

            n0 = n0.to(device)
            n1 = n1.to(device)
            label = label.to(device)

            name_similarity, (v_i, v_j) = model(n0, n1)
            loss = criterion(v_i, v_j, label)

            for i in range(len(n0)):
                name1 = emb2str(n0[i])
                name2 = emb2str(n1[i])
                model_score = name_similarity[i].item()
                curr_label = label[i].item()

                model_dict[dict_index] = {"name1": name1, "name2": name2, "model_score": model_score, "label": curr_label}
                dict_index += 1

            total_accuracy += loss.item()
        
    ds.mode = original_mode
        
    df = pd.DataFrame.from_dict(model_dict, "index")
    df.to_csv(save_name)

    total_accuracy /= (batch_no + 1)

    return total_accuracy

def test_on_test_set(model, ds):
    original_mode = ds.mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_accuracy = 0
    criterion = contrastive_loss
    with torch.no_grad():
        ds.mode = "test"
        for batch_no, data in enumerate(ds):
            n0 = data['emb0']
            n1 = data['emb1']
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
        
    ds.mode = original_mode

    total_accuracy /= (batch_no + 1)

    return total_accuracy

if __name__ == '__main__':
    config_list = ["step2_41_50_k1", "step2_41_50_k2", "step2_41_50_k3", "step2_41_50_k4"]

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