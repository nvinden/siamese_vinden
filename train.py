import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

import time
import os
from datetime import datetime
import sys
import random

from sklearn.metrics import precision_recall_curve

from dataset import RDataset
from model import Siamese
from process import save_data, load_data, load_json_config, emb2str, contrastive_loss

np.random.seed(41)
torch.manual_seed(1608)
random.seed(55)


def train_full_k(save_name):
    for k in range(5):
        train(save_name, k)

def train(save_name, k):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"CURRENT DEVICE: {device}")

    save_file = os.path.join("saves", str(save_name) + "_k" + str(k))
    json_file = os.path.join("configs", str(save_name) + ".json")

    DATASET_CONFIG, TRAIN_CONFIG, MODEL_KWARGS = load_json_config(json_file)

    #LOADING FROM SAVE OR CREATING NEW DATA
    if not os.path.isfile(save_file):
        model = Siamese(TRAIN_CONFIG, MODEL_KWARGS)

        optim = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=TRAIN_CONFIG["scheduler_step_size"], gamma=TRAIN_CONFIG["scheduler_gamma"])
        log_list = {}

        ds = RDataset(DATASET_CONFIG, TRAIN_CONFIG, k, dimensions = MODEL_KWARGS['hidden_size'])

        start_epoch = 0
        f_score_val_best = 0
    else:
        start_epoch, model, optim, scheduler, log_list, ds, f_score_val_best = load_data(save_file, TRAIN_CONFIG, MODEL_KWARGS)
    criterion = contrastive_loss

    model = model.to(device)

    #preparing saving directory
    result_directory = os.path.join("results", save_name)
    if not os.path.isdir(result_directory):
        os.mkdir(result_directory)
    if not os.path.isdir(os.path.join(result_directory, "train")):
        os.mkdir(os.path.join(result_directory, "train"))
    if not os.path.isdir(os.path.join(result_directory, "val")):
        os.mkdir(os.path.join(result_directory, "val"))

    #training loop
    for epoch in range(start_epoch, TRAIN_CONFIG["n_epochs"]):
        model.train()
        model.requires_grad_()
        start_time = time.time()

        total_epoch_loss = 0
        total_pairs = len(ds)

        data_save_condition = ((epoch % 5 == 0) or epoch == TRAIN_CONFIG["n_epochs"] - 1) and epoch != 0
        embedding_condition = ((epoch % 10 == 0)) and epoch != 0 and epoch != TRAIN_CONFIG["n_epochs"]

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
                    model_dict[dict_index] = {"name1": n1, "name2": n2, "model_score": mod.item(), "label": lab.item(), "k": k}
                    dict_index += 1

            #ADDING TO DIAGNOSTICS
            total_epoch_loss += loss.item()

        if data_save_condition:
            if not os.path.isdir(os.path.join(result_directory, "train", str(k))):
                os.mkdir(os.path.join(result_directory, "train", str(k)))
            if not os.path.isdir(os.path.join(result_directory, "val", str(k))):
                os.mkdir(os.path.join(result_directory, "val", str(k)))

            path_train = os.path.join(result_directory, "train", str(k), f"epoch{str(epoch).zfill(3)}.csv")
            path_val = os.path.join(result_directory, "val", str(k), f"epoch{str(epoch).zfill(3)}.csv")

            #SAVING TRAIN CSV
            df = pd.DataFrame.from_dict(model_dict, "index")
            df.to_csv(path_train)

            # Run on test set as well
            f_score_val = save_list(model, ds, path_val, k, "val")
            if f_score_val > f_score_val_best:
                f_score_val_best = f_score_val
                best_save_file = os.path.join("saves", str(save_name) + "_k" + str(k) + "_BEST")
                save_data(best_save_file, epoch + 1, model, optim, scheduler, log_list, ds, f_score_val_best)
                print(f"Saved a new best with f-score")
            print(f"F-Score: {f_score_val}")

            save_data(save_file, epoch + 1, model, optim, scheduler, log_list, ds, f_score_val_best)

        scheduler.step()

        #PRINTING DIAGNOSTICS
        total_epoch_loss /= (batch_no + 1)
        print(f"Epoch {epoch + 1}:")
        print(f"          Loss: {total_epoch_loss}")

        if embedding_condition:
            print("Embedding...")
            ds.embeddings.embed_all(model)
            print("Embedding done...")

            print("Adding to dataset...")
            n_added, pairs_found, already_found = ds.add_to_dataset()
            ds.embeddings.embeddings = None
            print(f"{n_added} entries added, {pairs_found} pairs found, {already_found} already found...")


        print(f"trained on {total_pairs} pairs")
        print(f" TIME: {time.time() - start_time} seconds")

    print(f"Finished training {save_name} on k = {k}")

    #SAVING LIST ON BEST
    best_save_file = os.path.join("saves", str(save_name) + "_k" + str(k) + "_BEST")
    _, best_model, _, _, _, _, _  = load_data(best_save_file, TRAIN_CONFIG, MODEL_KWARGS)
    path_test = os.path.join(result_directory, f"test_k{k}.csv")
    f_score = save_list(best_model, ds, path_test, k, "test")

    print(f"Finished testing with f-score of {f_score}")

    return 0, 0

def save_list(model, ds, save_name, k, set_type : str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_dict = dict()
    dict_index = 0

    original_mode = ds.mode
    model.eval()

    with torch.no_grad():
        ds.mode = set_type
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

            name_similarity, (_, _) = model(n0, n1)

            for i in range(len(n0)):
                name1 = emb2str(n0[i])
                name2 = emb2str(n1[i])
                model_score = name_similarity[i].item()
                curr_label = label[i].item()

                model_dict[dict_index] = {"name1": name1, "name2": name2, "model_score": model_score, "label": curr_label, "k": k}
                dict_index += 1
        
    ds.mode = original_mode
        
    df = pd.DataFrame.from_dict(model_dict, "index")
    df.to_csv(save_name)

    label = df["label"].to_numpy()
    model_score = df["model_score"].to_numpy()

    precision, recall, _ = precision_recall_curve(label, model_score)

    f_score_list = list()
    for pre, rec in zip(precision, recall):
      lower = pre + rec
      if lower == 0.0:
        lower = 0.00000001
      
      f_score_list.append(2 * (pre * rec) / lower)

    f_score = max(f_score_list)

    return f_score

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

def main():
    arg = sys.argv
    if len(arg) != 3:
        print("Error: Must have 2 command line arguments")
        return
    
    config_file = arg[1]
    k_number = arg[2]

    try:
        config_file = str(config_file)
        k_number = int(k_number)

        if k_number < 0 or k_number > 4:
            raise Exception
    except Exception:
        print("Error: config must be a string, and k must be an integer")
        return

    log_file_name = "log.txt"
    debug = True

    print("GPU Available: " + str(torch.cuda.is_available()))

    with open(log_file_name, "a") as log:
        start_time = time.time()

        train(config_file, k_number)
        text_out = f"{datetime.now()}\n{config_file}: training complete\nTime: {time.time() - start_time}"
        print("\n" + text_out)
        log.write(text_out + "\n")


if __name__ == '__main__':
    main()