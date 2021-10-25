import torch
import torch.nn as nn
import os
import csv
import numpy as np
import random
import json

from process import emb2str, str2emb
from annoy import AnnoyIndex

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

class RDataset(Dataset):
    def __init__(self, config, reprocess : bool = False):
        self.splits = config["ttv_split"]
        self.initial_set_negatives = config['initial_set_negatives']
        self.batch_size = config["batch_size"]

        assert sum(self.splits) == 1.0

        data_root = config["data_root"]
        self.data_root = data_root

        R_god_file = os.path.join(data_root, "R_GOD.npy")

        ds_master_pair_config = {
            "data_root": config["data_root"],
            "string_pad": config["string_pad"],
        }
        self.pair_dataset = SiamesePairsDataset(ds_master_pair_config)
        self.master_dataset = SiameseMasterDataset(ds_master_pair_config)

        self.mode = "train"

        pair_dict_save = os.path.join(data_root, "pair_dict.npy")
        if os.path.isfile(pair_dict_save):
            self.pair_dict = np.load(pair_dict_save, allow_pickle = True)
        else:
            self.pair_dict = self._build_pair_dict()
            np.save(pair_dict_save, self.pair_dict) 

        if not os.path.isfile(R_god_file) or reprocess:
            self.train_table = dict()
            self.test_table = dict()
            self.val_table = dict()

            self.pair_length = len(self.pair_dataset)
            self.master_length = len(self.master_dataset)

            break_points_pair = [int(self.pair_length * self.splits[0]), int(self.pair_length * (self.splits[0] + self.splits[1])), int(self.pair_length * (self.splits[0] + self.splits[1] + self.splits[2]))]
            break_points_master = [int(self.master_length * self.splits[0]), int(self.master_length * (self.splits[0] + self.splits[1])), int(self.master_length * (self.splits[0] + self.splits[1] + self.splits[2]))]

            self.train_table['pairs'] = self.pair_dataset.table[0:break_points_pair[0]].clone()
            self.train_table['master'] = self.master_dataset.table[0:break_points_master[0]].clone()

            self.test_table['pairs'] = self.pair_dataset.table[break_points_pair[0]:break_points_pair[1]].clone()
            self.test_table['master'] = self.master_dataset.table[break_points_master[0]:break_points_master[1]].clone()

            self.val_table['pairs'] = self.pair_dataset.table[break_points_pair[1]:break_points_pair[2]].clone()
            self.val_table['master'] = self.master_dataset.table[break_points_master[1]:break_points_master[2]].clone()

            total_list = (self.splits, self.train_table, self.test_table, self.val_table)
            np.save(R_god_file, total_list, allow_pickle = True)
        else:
            total_list = np.load(R_god_file, allow_pickle = True)
            self.splits, self.train_table, self.test_table, self.val_table = total_list

            assert self.splits == config['ttv_split']

        self.master_cart_product_length_train = len(self.train_table['master']) ** 2
        self.master_cart_product_length_test = len(self.test_table['master']) ** 2
        self.master_cart_product_length_val = len(self.val_table['master']) ** 2

        self.train_ds = list()
        self.test_ds = list()
        self.val_ds = list()

        self.train_used = dict()
        self.test_used = dict()
        self.val_used = dict()

        self.initialize(reprocess)

        self.embeddings = EmbeddingsMasterList(self.pair_dataset, self.master_dataset)

    def set_mode(self, mode : str):
        self.mode = "train"

    def _build_pair_dict(self):
        pair_dict = dict()

        for pair in self.pair_dataset:
            name0 = emb2str(pair["name0"])
            name1 = emb2str(pair["name1"])

            pair_dict[name0] = name1
            pair_dict[name1] = name0

        return pair_dict

    def get_negative_pair(self, idx = None, unique = True):
        if self.mode == "train":
            ds = self.train_table['master']
            cart_length = self.master_cart_product_length_train
            used_dict = self.train_used
        elif self.mode == "test":
            ds = self.test_table['master']
            cart_length = self.master_cart_product_length_test
            used_dict = self.test_used
        elif self.mode == "val":
            ds = self.val_table['master']
            cart_length = self.master_cart_product_length_val
            used_dict = self.val_used
        else:
            return -1

        #if idx is included, return the pair associated with that index
        if isinstance(idx, int) and idx >= 0 and idx < cart_length:
            index1 = int(idx / len(ds))
            index2 = int(idx % len(ds))
            return (ds[index1], ds[index2])
        #if idx is not included return a random pair
        else:         
            while True:
                index = random.randint(0, cart_length)

                index1 = int(index / len(ds))
                index2 = int(index % len(ds))

                name1 = emb2str(ds[index1])
                name2 = emb2str(ds[index2])

                if index in used_dict and used_dict[name1] == name2 and unique:
                    continue
                elif name1 in self.pair_dict and name2 in self.pair_dict and self.pair_dict[name1] == name2:
                    continue
                else:
                    break

            return (ds[index1], ds[index2])

    def get_positive_pair(self, idx = None, unique = True):
        if self.mode == "train":
            ds = self.train_table['pairs']
            length = ds.shape[0]
            used_dict = self.train_used
        elif self.mode == "test":
            ds = self.test_table['pairs']
            length = ds.shape[0]
            used_dict = self.test_used
        elif self.mode == "val":
            ds = self.val_table['pairs']
            length = ds.shape[0]
            used_dict = self.val_used
        else:
            return -1
        
        if isinstance(idx, int) and idx >= 0 and idx < length:
            return (ds[idx, 0], ds[idx, 1])
        else:
            while True:
                index = random.randint(0, length)

                name0 = emb2str(ds[idx, 0])
                name1 = emb2str(ds[idx, 1])

                if index in used_dict and used_dict[name0] == name1 and unique:
                    continue
                else:
                    break

            return (ds[index, 0], ds[index, 1])

    def get_ds(self):
        if self.mode == "train":
            ds = self.train_table
        elif self.mode == "test":
            ds = self.test_table
        elif self.moce == "val":
            ds = self.val_table
        else:
            return -1

        return ds

    def get_entry(self, idx, positive : bool, unique : bool = True):
        if positive:
            pair = self.get_positive_pair(idx, unique = unique)
            label = 1.0
        else:
            pair = self.get_negative_pair(idx, unique = unique)
            label = 0.0

        entry = {"emb0": pair[0], "emb1": pair[1], "label": label}
        return entry

    def initialize(self, reprocess : bool = False):
        save_file = os.path.join(self.data_root, "initialized.json")
        orignal_mode = self.mode
        if not os.path.isfile(save_file) or reprocess:
            self.mode = "train"
            for idx in range(len(self.train_table['pairs'])):
                entry = self.get_entry(idx, True, unique = True)
                self.train_ds.append(entry)
                name0 = emb2str(entry['emb0'])
                name1 = emb2str(entry['emb1'])
                self.train_used[name0] = name1
                self.train_used[name1] = name0
            for idx in range(self.initial_set_negatives):
                entry = self.get_entry(None, False, unique = True)
                self.train_ds.append(entry)
                name0 = emb2str(entry['emb0'])
                name1 = emb2str(entry['emb1'])
                self.train_used[name0] = name1
                self.train_used[name1] = name0
            self.shuffle_ds()
            
            self.mode = "test"
            for idx in range(len(self.test_table['pairs'])):
                if idx in self.pair_test_used:
                    continue
                entry = self.get_entry(idx, True, unique = True)
                self.test_ds.append(entry)
                name0 = emb2str(entry['emb0'])
                name1 = emb2str(entry['emb1'])
                self.test_used[name0] = name1
                self.test_used[name1] = name0
            for idx in range(len(self.test_table['pairs'])):
                entry = self.get_entry(None, False, unique = True)
                self.test_ds.append(entry)
                name0 = emb2str(entry['emb0'])
                name1 = emb2str(entry['emb1'])
                self.test_used[name0] = name1
                self.test_used[name1] = name0
            self.shuffle_ds()

            self.mode = "val"
            for idx in range(len(self.val_table['pairs'])):
                if idx in self.pair_val_used:
                    continue
                entry = self.get_entry(idx, True, unique = True)
                self.test_ds.append(entry)
                name0 = emb2str(entry['emb0'])
                name1 = emb2str(entry['emb1'])
                self.val_used[name0] = name1
                self.val_used[name1] = name0
            for idx in range(len(self.val_table['pairs'])):
                entry = self.get_entry(None, False, unique = True)
                self.val_ds.append(entry)
                name0 = emb2str(entry['emb0'])
                name1 = emb2str(entry['emb1'])
                self.val_used[name0] = name1
                self.val_used[name1] = name0
            self.shuffle_ds()

            self.mode = orignal_mode

            self.save_ds()
        else:
            self.load_ds()

    def save_ds(self):
        save_file = os.path.join(self.data_root, "initialized.json")
        with open(save_file, 'w') as fout:
            json.dump({"train": self.train_ds, "test": self.test_ds, "val": self.val_ds, \
                "train_used": self.train_used, "test_used": self.test_used, "val_used": self.val_used}, \
                fout, cls = TorchArrayEncoder)

    def load_ds(self):
        save_file = os.path.join(self.data_root, "initialized.json")
        with open(save_file, "r") as fin:
            decoded = json.load(fin)

            self.train_ds = decoded['train']
            self.test_ds = decoded['test']
            self.val_ds = decoded['val']

            self.train_used = decoded["train_used"]
            self.test_used = decoded["test_used"]
            self.val_used = decoded["val_used"]

            self.train_ds = [{"emb0": torch.tensor(entry["emb0"]), "emb1": torch.tensor(entry["emb1"]), "label": entry["label"]} for entry in self.train_ds]
            self.test_ds = [{"emb0": torch.tensor(entry["emb0"]), "emb1": torch.tensor(entry["emb1"]), "label": entry["label"]} for entry in self.test_ds]
            self.val_ds = [{"emb0": torch.tensor(entry["emb0"]), "emb1": torch.tensor(entry["emb1"]), "label": entry["label"]} for entry in self.val_ds]

    def shuffle_ds(self):
        if self.mode == "train":
            ds = self.train_ds
        elif self.mode == "test":
            ds = self.test_ds
        elif self.mode == "val":
            ds = self.val_ds
        else:
            return -1

        random.shuffle(ds)

    def __len__(self):
        if self.mode == "train":
            return len(self.train_ds)
        elif self.mode == "test":
            return len(self.test_ds)
        elif self.mode == "val":
            return len(self.val_ds)
        else:
            return -1

    def __getitem__(self, idx):
        if self.mode == "train":
            return self.train_ds[idx]
        elif self.mode == "test":
            return self.test_ds[idx]
        elif self.mode == "val":
            return self.val_ds[idx]
        else:
            return -1

    def __iter__(self):
        if self.mode == "train":
            self.n = 0
        elif self.mode == "test":
            self.n = 0
        elif self.mode == "val":
            self.n = 0
        else:
            return -1

        return self

    def __next__(self):
        if self.n + self.batch_size <= len(self):
            if self.mode == "train":
                entries =  self.train_ds[self.n:self.n + self.batch_size]
            elif self.mode == "test":
                entries =  self.test_ds[self.n:self.n + self.batch_size]
            elif self.mode == "val":
                entries =  self.val_ds[self.n:self.n + self.batch_size]
            else:
                return -1

            self.n += self.batch_size
            entries_concat = dict()
            for row in entries:
                for key in row.keys():
                    if not torch.is_tensor(row[key]):
                        row[key] = torch.tensor(row[key])

                    if key not in entries_concat:
                        entries_concat[key] = row[key].unsqueeze(0)
                    else:
                        entries_concat[key] = torch.cat([entries_concat[key], row[key].unsqueeze(0)], dim = 0)
            return entries_concat
        else:
            raise StopIteration

    def add_to_dataset(self):
        if self.mode == "train":
            ds = self.train_ds
            used = self.train_used
        elif self.mode == "test":
            ds = self.test_ds
            used = self.test_used
        elif self.mode == "val":
            ds = self.val_ds
            used = self.val_used
        else:
            return -1

        n_added = 0

        index2name = self.embeddings.index2name
        name2index = self.embeddings.name2index
        pair_dict = self.pair_dict.item()
        pair_dict_keys = list(pair_dict)

        for i in range(len(pair_dict_keys)):
            name = pair_dict_keys[i]
            name_idx = name2index[name]

            nn_idx = self.embeddings.get_nn(name_idx)
            nn_name = index2name[nn_idx]

            if name in pair_dict and not nn_name == pair_dict[name]:
                if not(name in used and used[name] == nn_name):
                    ds.append({"emb0": str2emb(name), "emb1": str2emb(nn_name), "label": 0.0})
                    used[name] = nn_name
                    used[nn_name] = name
                    n_added += 1
            elif name in pair_dict:
                print(f"Pair Retrieved {name} {nn_name}")

        return n_added

class EmbeddingsMasterList():
    def __init__(self, pair_dataset, master_dataset, trees = 40, dimensions = 50):
        self.pair_dataset = pair_dataset
        self.master_dataset = master_dataset

        self.trees = trees
        self.dimensions = dimensions
        self.embeddings = None

        if os.path.isfile("data/index2name.npy") and os.path.isfile("data/name2index.npy"):
            self.index2name = np.load("data/index2name.npy", allow_pickle = True).item()
            self.name2index = np.load("data/name2index.npy", allow_pickle = True).item()
        else:
            self.name2index, self.index2name = self._build_dict()

    def embed_all(self, model):
        self.embeddings = AnnoyIndex(self.dimensions, 'euclidean')
        model.eval()
        for i in range(len(self.index2name)):
            name = self.index2name[i]
            embedded_name = str2emb(name, string_pad = model.n_tokens).unsqueeze(0)
            v = model(embedded_name).squeeze(0)
            self.embeddings.add_item(i, v)

            if i % 5000 == 0:
                print(f"Embedded {i} names")

        self.embeddings.build(self.trees)
        
        model.train()

    def get_nn(self, index):
        return self.embeddings.get_nns_by_item(index, 2)[1]

    def _build_dict(self):
        index = 0

        name2index = dict()

        for row in self.master_dataset:
            name2index[emb2str(row["name"])] = index
            index += 1

        index2name = {v: k for k, v in name2index.items()}

        np.save("data/index2name.npy", index2name) 
        np.save("data/name2index.npy", name2index)

        return name2index, index2name
    
class TorchArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if torch.is_tensor(obj):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)