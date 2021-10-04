import torch
import torch.nn as torch
import os
import csv
import numpy as np

from torch.utils.data import Dataset

class SiamesePairsDataset(Dataset):
    def __init__(self, config, reprocess : bool = False):
        self.data_root = config['data_root']
        self.string_pad = config['string_pad']
        self.table = list()

        unprocessed_pair_file_path = os.path.join(self.data_root, "records25k_data.tsv")
        processed_pair_file_path = os.path.join(self.data_root, "records25k_data_processed.npy")
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
                
                self.table = np.array(self.table, dtype = str)

                np.save(processed_pair_file_path, self.table)
        #Loading preprcessed numpy saved datafile
        else:
            self.table = np.load(processed_pair_file_path)

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, idx):
        name0 = self.table[idx, 0]
        name1 = self.table[idx, 1]

        return {"name0": name0, "name1": name1}

class SiameseMasterDataset(Dataset):
    def __init__(self, config, reprocess : bool = False):
        self.data_root = config['data_root']
        self.string_pad = config['string_pad']
        self.table = list()

        unprocessed_master_file_path = os.path.join(self.data_root, "records_surnames_counts_250k.tsv")
        processed_master_file_path = os.path.join(self.data_root, "records_surnames_counts_250k_processed.npy")

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
                
                self.table = np.array(self.table, dtype = str)

                np.save(processed_master_file_path, self.table)
        #Loading preprcessed numpy saved datafile
        else:
            self.table = np.load(processed_master_file_path)

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, idx):
        name = self.table[idx]

        return {"name": name}