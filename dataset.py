import torch
import torch.nn as nn
import os
import csv
import numpy as np

from torch.utils.data import Dataset

class SiamesePairsDataset(Dataset):
    def __init__(self, config, reprocess : bool = False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_root = config['data_root']
        self.string_pad = config['string_pad']
        self.table = list()

        unprocessed_pair_file_path = os.path.join(self.data_root, "records25k_data.tsv")
        processed_pair_file_path = os.path.join(self.data_root, "records25k_data_processed.pt")
        #Reprosses records25k_data into records25k_data_processed
        if not os.path.isfile(processed_pair_file_path) or reprocess == True:
            with open(unprocessed_pair_file_path) as fd:
                reader = csv.reader(fd, delimiter="\t", quotechar='"')
                for line in reader:
                    pair = line[:2]

                    assert pair[0].islower()
                    assert pair[1].islower()

                    assert pair[0].isalpha()
                    assert pair[1].isalpha()

                    pair[0] = "<" + pair[0] + ">"
                    pair[1] = "<" + pair[1] + ">"

                    pair[0] = pair[0].ljust(self.string_pad, "*")
                    pair[1] = pair[1].ljust(self.string_pad, "*")

                    assert len(pair[0]) <= self.string_pad
                    assert len(pair[1]) <= self.string_pad

                    self.table.append(pair)
                
                table = np.array(self.table, dtype = str)

                torch_table = torch.zeros((len(table), 2, self.string_pad), device = self.device, dtype = torch.uint8)
                for i, word in enumerate(table):
                    for k, word_spec in enumerate(word):
                        for j, char in enumerate(word_spec):
                            if char == "<":
                                torch_table[i, k, j] = 30
                            elif char == ">":
                                torch_table[i, k, j] = 31
                            elif char == "*":
                                torch_table[i, k, j] = 32
                            else:
                                torch_table[i, k, j] = ord(char) - 97

                torch_table = torch_table[torch.randperm(torch_table.size()[0])]

                self.table = torch_table
                
                torch.save(self.table, processed_pair_file_path)
        #Loading preprcessed numpy saved datafile
        else:
            self.table = torch.load(processed_pair_file_path)

        self.table.to(self.device)

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, idx):
        name0 = self.table[idx, 0]
        name1 = self.table[idx, 1]

        return {"name0": name0, "name1": name1}

class SiameseMasterDataset(Dataset):
    def __init__(self, config, reprocess : bool = False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_root = config['data_root']
        self.string_pad = config['string_pad']
        self.table = list()

        unprocessed_master_file_path = os.path.join(self.data_root, "records_surnames_counts_250k.tsv")
        processed_master_file_path = os.path.join(self.data_root, "records_surnames_counts_250k_processed.pt")

        #Reprosses records25k_data into records25k_data_processed
        if not os.path.isfile(processed_master_file_path) or reprocess == True:
            with open(unprocessed_master_file_path) as fd:
                reader = csv.reader(fd, delimiter="\t", quotechar='"')
                for line in reader:
                    name = line[0]

                    assert name.islower()
                    assert name.isalpha()

                    name = "<" + name + ">"

                    name = name.ljust(self.string_pad, "*")

                    assert len(name) <= self.string_pad

                    self.table.append(name)
                
                table = np.array(self.table, dtype = str)

                torch_table = torch.zeros((len(table), self.string_pad), device = self.device, dtype = torch.uint8)
                for i, word in enumerate(table):
                    for j, char in enumerate(word):
                        if char == "<":
                            torch_table[i, j] = 30
                        elif char == ">":
                            torch_table[i, j] = 31
                        elif char == "*":
                            torch_table[i, j] = 32
                        else:
                            torch_table[i, j] = ord(char) - 97

                torch_table = torch_table[torch.randperm(torch_table.size()[0])]

                self.table = torch_table
                
                torch.save(self.table, processed_master_file_path)
        #Loading preprcessed numpy saved datafile
        else:
            self.table = torch.load(processed_master_file_path)
        
        self.table.to(self.device)

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, idx):
        name = self.table[idx]

        return {"name": name}

class PretrainDataset(Dataset):
    def __init__(self, config, reprocess : bool = False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_root = config['data_root']
        self.string_pad = config['string_pad']
        self.table = list()

        processed_master_file_path = os.path.join(self.data_root, "records_surnames_counts_250k_processed.pt")
        processed_pair_file_path = os.path.join(self.data_root, "records25k_data_processed.pt")
        processed_pretrain_file_path = os.path.join(self.data_root, "pretrain_dataset.pt")

        pair_table = torch.load(processed_pair_file_path)
        master_table = torch.load(processed_master_file_path)
        
        pair_table = torch.cat([pair_table[:, 0, :], pair_table[:, 1, :]], dim = 0)

        self.table = torch.cat([master_table, pair_table])

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, idx):
        name = self.table[idx]

        return {"name": name}