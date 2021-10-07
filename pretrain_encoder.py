import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import time
import os
from datetime import datetime

from dataset import PretrainDataset
from model import Siamese
from process import save_data, load_data, add_to_log_list, load_json_config, create_pretrained_vectors

def train(save_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"CURRENT DEVICE: {device}")

    save_file = os.path.join("saves", str(save_name))
    json_file = os.path.join("configs", str(save_name) + ".json")

    DATASET_CONFIG, TRAIN_CONFIG, MODEL_KWARGS = load_json_config(json_file)

    #CREATING DATASETS
    ds = PretrainDataset(DATASET_CONFIG)
    batch_size = TRAIN_CONFIG['batch_size']
    dl = DataLoader(ds, batch_size = TRAIN_CONFIG['batch_size'], shuffle = True, drop_last = True)

    #LOADING FROM SAVE OR CREATING NEW DATA
    if not os.path.isfile(save_file):
        model = Siamese(TRAIN_CONFIG, MODEL_KWARGS)

        optim = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['lr'])
        log_list = {}

        start_epoch = 0
    else:
        start_epoch, model, optim, log_list = load_data(save_file, TRAIN_CONFIG, MODEL_KWARGS)


    #decimal chance each character is used in the syllabus
    char_percentage =  [0.08123, 0.02908, 0.03494, 0.03206, 0.12481, 0.01547, 0.02865, \
            0.03937, 0.05963, 0.00245, 0.02553, 0.06997, 0.03321, 0.07405, 0.05787, 0.01605, \
            0.00093, 0.08325, 0.05979, 0.0524, 0.02932, 0.00847, 0.01545, 0.00112, 0.01805, \
            0.00682, 0, 0, 0, 0, 0, 0]
    char_percentage = torch.FloatTensor(char_percentage)
    char_percentage.requires_grad_(False)
    criterion = nn.NLLLoss(weight = char_percentage)

    model = model.to(device)
    criterion = criterion.to(device)

    encoder = model.A_function
    embedding_func = model.embedding_function

    for epoch in range(start_epoch, TRAIN_CONFIG["n_epochs"]):
        model.train()
        model.requires_grad_()
        start_time = time.time()

        total_epoch_loss = 0

        for batch_no, data in enumerate(dl):
            #OPTIMIZING ON PAIR
            optim.zero_grad()

            name = data['name']
            name.to(device)

            embeddings = embedding_func(name)

            embeddings_in, embeddings_truth, replace_list = create_pretrained_vectors(model, embeddings)

            result = encoder(embeddings_in)
            result = F.log_softmax(result, dim = 2)

            inp = [result[inp_batch_no, character] for inp_batch_no, character in enumerate(replace_list)]
            inp = torch.stack(inp, dim = 0)

            if torch.cuda.is_available():
                target = torch.cuda.LongTensor(replace_list)
            else:
                target = torch.LongTensor(replace_list)

            loss = criterion(inp, target)

            #ADDING TO DIAGNOSTICS
            total_epoch_loss += loss.item()

        total_epoch_loss /= (batch_no + 1)

        #PRINTING DIAGNOSTICS
        '''
        total_epoch_pair_loss /= (batch_no + 1)
        total_epoch_master_loss /= (batch_no + 1)
        total_epoch_average_loss = (total_epoch_pair_loss + total_epoch_master_loss) / 2

        print(f"\nEpoch {epoch + 1}:")
        print(f"    Pair Loss: {total_epoch_pair_loss}")
        print(f"  Master Loss: {total_epoch_master_loss}")
        print(f"     Avg Loss: {total_epoch_average_loss}")
        '''
        print(f"\nEpoch {epoch + 1}:")
        print(f"         Loss: {total_epoch_loss}")

        '''
        if (epoch + 1) % 10 == 0:
            pair_accuracy, master_accuracy = test_on_test_set(model, pair_loader_test, master_loader_test)
            average_accuracy = (pair_accuracy + master_accuracy) / 2
            add_to_log_list(log_list, total_epoch_pair_loss, total_epoch_master_loss, total_epoch_average_loss, pair_accuracy, master_accuracy, average_accuracy)
            print(f"    Pair Test: {pair_accuracy}")
            print(f"  Master Test: {master_accuracy}")
            print(f"     Avg Test: {average_accuracy}")

            save_data(save_file, epoch, model, optim, log_list)
        else:
            add_to_log_list(log_list, total_epoch_pair_loss, total_epoch_master_loss, total_epoch_average_loss)
        '''

        save_data(save_file, epoch, model, optim, log_list)

        print(f" TIME: {time.time() - start_time} seconds")

    return total_epoch_loss

if __name__ == '__main__':
    config_list = ["pretrained_encoder", ]

    log_file_name = "log.txt"
    debug = True

    print("GPU Available: " + str(torch.cuda.is_available()))

    with open(log_file_name, "a") as log:
        for config in config_list:
            start_time = time.time()

            if debug == True:
                total_epoch_pair_loss, total_epoch_master_loss, total_epoch_average_loss = train(config)
                text_out = f"{datetime.now()}\n{config}: training complete\nTime: {time.time() - start_time}\nPair Loss: {total_epoch_pair_loss} \
                    \nMaster Loss: {total_epoch_master_loss}\nAverage Loss: {total_epoch_average_loss}\n"
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