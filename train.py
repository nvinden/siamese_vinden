import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import time
import os
from datetime import datetime

from dataset import SiamesePairsDataset, SiameseMasterDataset
from model import Siamese
from process import save_data, load_data, add_to_log_list, load_json_config, emb2str

def train(save_name):
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
        model = Siamese(TRAIN_CONFIG, MODEL_KWARGS)

        optim = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['lr'])
        log_list = {}

        start_epoch = 0
    else:
        start_epoch, model, optim, log_list = load_data(save_file, TRAIN_CONFIG, MODEL_KWARGS)

        if TRAIN_CONFIG['epoch_reset']:
            start_epoch = 0
            optim = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['lr'])

    criterion = nn.MSELoss()

    model = model.to(device)
    criterion = criterion.to(device)

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

        total_epoch_pair_loss = 0
        total_epoch_master_loss = 0

        for batch_no, (pair_data, master_data) in enumerate(zip(pair_loader_train, master_loader_train)):
            #OPTIMIZING ON PAIR
            optim.zero_grad()

            pair0 = pair_data['name0']
            pair1 = pair_data['name1']
            pair0.to(device)
            pair1.to(device)

            #print(emb2str(pair0[0]), emb2str(pair1[0]))

            target_pair = torch.zeros([len(pair0)], dtype = torch.float, device = device)

            out_pair, _ = model(pair0, pair1)

            loss_pair = criterion(target_pair, out_pair)
            loss_pair.backward()
            optim.step()

            #OPTIMIZING ON MASTER
            optim.zero_grad()

            master0 = master_data['name'][0:batch_size]
            master1 = master_data['name'][batch_size:]

            master0.to(device)
            master1.to(device)

            target_master = torch.ones([len(pair0)], dtype = torch.float, device = device)

            out_master, _ = model(master0, master1)

            loss_master = criterion(target_master, out_master)
            loss_master.backward()
            optim.step()

            #print(emb2str(master0[0]), emb2str(master1[0]))

            #ADDING TO DIAGNOSTICS
            total_epoch_pair_loss += loss_pair.item()
            total_epoch_master_loss += loss_master.item()

        #PRINTING DIAGNOSTICS
        total_epoch_pair_loss /= (batch_no + 1)
        total_epoch_master_loss /= (batch_no + 1)
        total_epoch_average_loss = (total_epoch_pair_loss + total_epoch_master_loss) / 2

        print(f"\nEpoch {epoch + 1}:")
        print(f"    Pair Loss: {total_epoch_pair_loss}")
        print(f"  Master Loss: {total_epoch_master_loss}")
        print(f"     Avg Loss: {total_epoch_average_loss}")

        if (epoch + 1) % 10 == 0:
            pair_accuracy, master_accuracy = test_on_test_set(model, pair_loader_test, master_loader_test)
            average_accuracy = (pair_accuracy + master_accuracy) / 2
            add_to_log_list(log_list, total_epoch_pair_loss, total_epoch_master_loss, total_epoch_average_loss, pair_accuracy, master_accuracy, average_accuracy)
            print(f"    Pair Test: {pair_accuracy}")
            print(f"  Master Test: {master_accuracy}")
            print(f"     Avg Test: {average_accuracy}")

            #save_data(save_file, epoch, model, optim, log_list)
        else:
            add_to_log_list(log_list, total_epoch_pair_loss, total_epoch_master_loss, total_epoch_average_loss)

        print(f" TIME: {time.time() - start_time} seconds")

    return total_epoch_pair_loss, total_epoch_master_loss, total_epoch_average_loss, pair_accuracy, master_accuracy, average_accuracy

def test_on_test_set(model, pair_loader_test, master_loader_test):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_pair_mse = 0
    total_master_mse = 0
    for batch_no, (pair, master) in enumerate(zip(pair_loader_test, master_loader_test)):
        pair0 = pair['name0']
        pair1 = pair['name1']
        pair0 = pair0.to(device)
        pair1 = pair1.to(device)

        master0 = master['name'][0:len(pair0)]
        master1 = master['name'][len(pair0):]

        man_pair, _ =  model(pair0, pair1)
        man_master, _ = model(master0, master1)

        pair_sum = torch.sum(man_pair <= 0.5)
        master_sum = torch.sum(man_master >= 0.5)

        total_pair_mse += pair_sum.item() / pair0.shape[0]
        total_master_mse += master_sum.item() / pair0.shape[0]
    total_pair_mse /= (batch_no + 1)
    total_master_mse /= (batch_no + 1)

    return total_pair_mse, total_master_mse

if __name__ == '__main__':
    config_list = ["pretrained_encoder_phase2_116", ]

    log_file_name = "log.txt"
    debug = True

    print("GPU Available: " + str(torch.cuda.is_available()))

    with open(log_file_name, "a") as log:
        for config in config_list:
            start_time = time.time()

            if debug == True:
                total_epoch_pair_loss, total_epoch_master_loss, total_epoch_average_loss, pair_accuracy, master_accuracy, total_accuracy = train(config)
                text_out = f"{datetime.now()}\n{config}: training complete\nTime: {time.time() - start_time}\nPair Loss: {total_epoch_pair_loss} \
                        \nMaster Loss: {total_epoch_master_loss}\nAverage Loss: {total_epoch_average_loss}\nPair Accuracy: {pair_accuracy}\n\
                        Master Accuracy: {master_accuracy}\nAverage Accuracy: {total_accuracy}\n"
            else:
                try:
                    total_epoch_pair_loss, total_epoch_master_loss, total_epoch_average_loss, pair_accuracy, master_accuracy, total_accuracy = train(config)
                    text_out = f"{datetime.now()}\n{config}: training complete\nTime: {time.time() - start_time}\nPair Loss: {total_epoch_pair_loss} \
                        \nMaster Loss: {total_epoch_master_loss}\nAverage Loss: {total_epoch_average_loss}\nPair Accuracy: {pair_accuracy}\n\
                        Master Accuracy: {master_accuracy}\nAverage Accuracy: {total_accuracy}\n"
                except Exception as e:
                    text_out = f"{datetime.now()}\n{config}: training failure\n{str(e)}\n"

            print("\n" + text_out)
            log.write(text_out + "\n")