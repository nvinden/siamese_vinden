import torch
import torch.nn as nn
import os
import csv
import numpy as np
import random
import json
import pandas as pd

from process import emb2str, str2emb
from annoy import AnnoyIndex

from torch.utils.data import Dataset

class SiamesePairsDataset(Dataset):
    def __init__(self, config, reprocess : bool = False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_root = config['data_root']
        self.string_pad = config['string_pad']
        self.pair_file = config["pair_file"]
        self.table = list()

        unprocessed_pair_file_path = os.path.join(self.data_root, self.pair_file + ".tsv")
        processed_pair_file_path = os.path.join(self.data_root, self.pair_file + "_processed.pt")
        #Reprosses records25k_data into records25k_data_processed
        if not os.path.isfile(processed_pair_file_path) or reprocess == True:
            with open(unprocessed_pair_file_path) as fd:
                reader = csv.reader(fd, delimiter = "\t", quotechar='"')
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
        random.seed(160855)

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
    def __init__(self, config, train_config, k, dimensions, reprocess : bool = False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.partition_data_name = config["partition_data_name"] + ".json"
        self.k = config["k"]
        self.kth = k
        self.initial_random_negatives = config['initial_random_negatives']
        self.initial_jeremy_negatives = config['initial_jeremy_negatives']
        self.test_random_negatives = config['test_random_negatives']
        self.test_jeremy_negatives = config['test_jeremy_negatives']
        self.batch_size = config["batch_size"]

        self.random_mutability = config['random_mutability']
        self.jeremy_mutability = config['jeremy_mutability']

        if "hard_neg_cap" in train_config:
            self.hard_neg_cap = train_config["hard_neg_cap"]
        else:
            self.hard_neg_cap = None

        if "hard_neg_search_cap" in train_config:
            self.hard_neg_search_cap = train_config["hard_neg_search_cap"]
        else:
            self.hard_neg_search_cap = None

        data_root = config["data_root"]
        self.data_root = data_root

        if "pair_file" not in config:
            config["pair_file"] = "records25k_data"

        ds_master_pair_config = {
            "data_root": config["data_root"],
            "string_pad": config["string_pad"],
            "pair_file": config["pair_file"]
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

        self.training_phase = config['training_phase'] if 'training_phase' in config else 1

        #CREATING TRAINING AND TEST SPLITS ON KTH PARTITION
        self.pair_length = len(self.pair_dataset)

        self.k_length = self.pair_length // self.k

        self.partitions = dict()
        self.partitions["positives"] = [self.pair_dataset.table[i:i + self.k_length] for i in range(0, self.pair_length, self.k_length)]

        if len(self.partitions["positives"]) > self.k:
            self.partitions["positives"].pop()

        self.master_cart_product_length = len(self.master_dataset) ** 2

        self.ds = list()
        self.used = dict()

        self.initialize(reprocess)
        self.create_train_test_sets()

        self.embeddings = EmbeddingsMasterList(self.pair_dataset, self.master_dataset, dimensions = dimensions)
        self.emb_idx = 0

    def create_train_test_sets(self):
        kth = self.kth
        val_kth = self.kth - 1 if self.kth - 1 >= 0 else self.k - 1
        ini = self.partitions

        #creating test set
        self.test_ds = list()
        for i, row in enumerate(ini["positives"][kth]):
            self.test_ds.append({"emb0": row[0], "emb1": row[1], "label": 1.0, "mutable": 0, "type": 0})
        for i, row in enumerate(ini["random"][kth]):
            if i >= self.test_random_negatives:
                break
            m = 0 if i / self.test_random_negatives >= self.random_mutability else 1
            self.test_ds.append({"emb0": row[0], "emb1": row[1], "label": 0.0, "mutable": m, "type": 1})
        for i, row in enumerate(ini["jeremy"][kth]):
            if i >= self.test_jeremy_negatives:
                break
            m = 0 if i / self.test_jeremy_negatives >= self.jeremy_mutability else 1
            self.test_ds.append({"emb0": row[0], "emb1": row[1], "label": 0.0, "mutable": m, "type": 2}) 

        self.val_ds = list()
        if self.training_phase == 1:    
            for i, row in enumerate(ini["positives"][val_kth]):
                self.val_ds.append({"emb0": row[0], "emb1": row[1], "label": 1.0, "mutable": 0, "type": 0})
            for i, row in enumerate(ini["random"][val_kth]):
                if i >= self.test_random_negatives:
                    break
                m = 0 if i / self.test_random_negatives >= self.random_mutability else 1
                self.val_ds.append({"emb0": row[0], "emb1": row[1], "label": 0.0, "mutable": m, "type": 1})
            for i, row in enumerate(ini["jeremy"][val_kth]):
                if i >= self.test_jeremy_negatives:
                    break
                m = 0 if i / self.test_jeremy_negatives >= self.jeremy_mutability else 1
                self.val_ds.append({"emb0": row[0], "emb1": row[1], "label": 0.0, "mutable": m, "type": 2})

        #creating train set
        self.train_ds = list()
        for k_no in range(self.k):
            if k_no == kth or (k_no == val_kth and self.training_phase == 1):
                continue

            for i, row in enumerate(ini["positives"][k_no]):
                self.train_ds.append({"emb0": row[0], "emb1": row[1], "label": 1.0, "mutable": 0, "type": 0})
            for i, row in enumerate(ini["random"][k_no]):
                shortest = min(len(ini["random"][k_no]), self.initial_random_negatives)
                if i >= self.initial_random_negatives:
                    break
                m = 0 if i / shortest >= self.random_mutability else 1
                self.train_ds.append({"emb0": row[0], "emb1": row[1], "label": 0.0, "mutable": m, "type": 1})
            for i, row in enumerate(ini["jeremy"][k_no]):
                shortest = min(len(ini["jeremy"][k_no]), self.initial_jeremy_negatives)
                if i >= self.initial_jeremy_negatives:
                    break
                m = 0 if i / shortest >= self.jeremy_mutability else 1
                self.train_ds.append({"emb0": row[0], "emb1": row[1], "label": 0.0, "mutable": m, "type": 2}) 

        random.shuffle(self.test_ds)
        random.shuffle(self.train_ds)
        random.shuffle(self.val_ds)

        #self.pairs_in_test_set = self._create_pairs_in_test_set(ini["positives"][k_no])
    
    def _create_pairs_in_test_set(self, positives):
        out = dict()

        for row in positives:
            for word in row:
                word = emb2str(word)

                out[word] = -1
        
        return out

    def initialize(self, filename, reprocess : bool = False):
        save_file = os.path.join(self.data_root, self.partition_data_name)
        if not os.path.isfile(save_file) or reprocess:
            total_random_grab = self.initial_random_negatives if self.initial_random_negatives > self.test_random_negatives else self.test_random_negatives
            total_random_grab *= 5
            total_jeremy_grab = self.initial_jeremy_negatives if self.initial_jeremy_negatives > self.test_jeremy_negatives else self.test_jeremy_negatives
            total_jeremy_grab *= 5

            self.partitions["random"] = self.get_random_list(total_random_grab)
            self.partitions["jeremy"] = self.get_jeremy_list(total_jeremy_grab)

            random_k_length = len(self.partitions["random"]) // self.k
            jeremy_k_length = len(self.partitions["jeremy"]) // self.k

            if random_k_length != 0:
                self.partitions["random"] = [self.partitions["random"][i:i + random_k_length] for i in range(0, len(self.partitions["random"]), random_k_length)]
            else:
                self.partitions["random"] = [[] for i in range(5)]

            if jeremy_k_length != 0:
                self.partitions["jeremy"] = [self.partitions["jeremy"][i:i + jeremy_k_length] for i in range(0, len(self.partitions["jeremy"]), jeremy_k_length)]
            else:
                self.partitions["jeremy"] = [[] for i in range(5)]

            self.save_ds()
        else:
            self.load_ds()

    def get_jeremy_list(self, length):
        out = torch.empty(0)

        table = pd.read_csv("data/pairs.csv")

        i = 0
        while i < length:
            index = random.randint(0, len(table))

            name1 = table.iloc[index, 0]
            name2 = table.iloc[index, 1]

            emb1 = str2emb(name1)
            emb2 = str2emb(name2)

            embs = torch.cat([emb1.unsqueeze(0), emb2.unsqueeze(0)], dim = 0)

            used_key = name1 + "_" + name2

            if self.already_used(name1, name2):
                continue
            elif self.are_pairs(name1, name2):
                continue
            else:
                if len(out) == 0:
                    out = embs.unsqueeze(0)
                else:
                    out = torch.cat([out, embs.unsqueeze(0)], dim = 0)
                
                self.add_to_used(name1, name2)
                i += 1
        return out

    def get_random_list(self, length):
        out = torch.empty(0)
        ds = self.master_dataset.table
        cart_length = len(ds) ** 2
        i = 0
        while i < length:
            index = random.randint(0, cart_length)

            index1 = int(index / len(ds))
            index2 = int(index % len(ds))

            name1 = emb2str(ds[index1])
            name2 = emb2str(ds[index2])

            embs = torch.cat([ds[index1].unsqueeze(0), ds[index2].unsqueeze(0)], dim = 0)

            if self.already_used(name1, name2):
                continue
            elif self.are_pairs(name1, name2):
                continue
            else:
                if len(out) == 0:
                    out = embs.unsqueeze(0)
                else:
                    out = torch.cat([out, embs.unsqueeze(0)], dim = 0)
                
                self.add_to_used(name1, name2)
                i += 1
        return out

    def add_to_used(self, name1, name2):
        if name1 not in self.used:
            self.used[name1] = list()
        if name2 not in self.used:
            self.used[name2] = list()

        self.used[name1].append(name2)
        self.used[name2].append(name1)
    
    def already_used(self, name1, name2):
        if name1 in self.used and name2 in self.used[name1]:
            return True
        elif name2 in self.used and name1 in self.used[name2]:
            return True
        
        return False

    def are_pairs(self, name1, name2):
        if name1 in self.pair_dict and name2 in self.pair_dict[name1]:
            return True
        elif name2 in self.pair_dict and name1 in self.pair_dict[name2]:
            return True
        
        return False

    def set_mode(self, mode : str):
        self.mode = mode

    def _build_pair_dict(self):
        pair_dict = dict()

        for pair in self.pair_dataset:
            name0 = emb2str(pair["name0"])
            name1 = emb2str(pair["name1"])

            if name0 not in pair_dict:
                pair_dict[name0] = list()
            if name1 not in pair_dict:
                pair_dict[name1] = list()
            
            if name1 not in pair_dict[name0]:
                pair_dict[name0].append(name1)
            if name0 not in pair_dict[name1]:
                pair_dict[name1].append(name0)

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

    def save_ds(self):
        save_file = os.path.join(self.data_root, self.partition_data_name)
        with open(save_file, 'w') as fout:
            json.dump({"positives": self.partitions["positives"], "random": self.partitions["random"], "jeremy": self.partitions["jeremy"], \
                "used": self.used}, \
                fout, cls = TorchArrayEncoder)

    def load_ds(self):
        save_file = os.path.join(self.data_root, self.partition_data_name)
        with open(save_file, "r") as fin:
            decoded = json.load(fin)

            self.partitions['positives'] = torch.tensor(decoded['positives'])
            self.partitions['random'] = torch.tensor(decoded['random'])
            self.partitions['jeremy'] = torch.tensor(decoded['jeremy'])

            self.used = decoded["used"]

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
        if self.n + self.batch_size <= len(self.train_ds):
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
                        row[key] = torch.tensor(row[key], device = self.device)

                    val = row[key].unsqueeze(0).to(self.device)
                    if key not in entries_concat:
                        entries_concat[key] = val
                    else:
                        entries_concat[key] = torch.cat([entries_concat[key], val], dim = 0)
            return entries_concat
        else:
            raise StopIteration

    def _remove_mutable(self):
        new_list = list()
        for entry in self.train_ds:
            if entry['mutable'] == 0:
                new_list.append(entry)

        self.train_ds = new_list

    def add_to_dataset(self):
        self._remove_mutable()
        
        n_added = 0
        n_pairs_found = 0
        n_already_used = 0

        index2name = self.embeddings.index2name
        name2index = self.embeddings.name2index

        table = self.master_dataset.table

        hard_neg_list = list()

        if self.hard_neg_search_cap is not None:
            max_iterations = self.hard_neg_search_cap
        else:
            max_iterations = len(table)

        count = 0
        while True:
            name_emb = table[self.emb_idx]
            name = emb2str(name_emb)

            name_idx = name2index[name]

            nn_idx = self.embeddings.get_nn(name_idx)
            nn_name = index2name[nn_idx]
            nn_name_emb = str2emb(nn_name)

            if not self.are_pairs(name, nn_name):
                if not self.already_used(name, nn_name):
                    score = self.embeddings.get_distance(self.emb_idx, nn_idx)
                    hard_neg_list.append({"name0": name_emb, "name1": nn_name_emb, "score": score, "type": 3})
                    n_added += 1
                else:
                    n_already_used += 1
            else:
                n_pairs_found += 1

            self.emb_idx += 1

            if self.emb_idx >= len(table):
                self.emb_idx = 0

            count += 1

            if count >= max_iterations:
                break

        hard_neg_list = pd.DataFrame(hard_neg_list)
        hard_neg_list = hard_neg_list.sort_values(by=['score'], ascending = True, ignore_index = True)

        if self.hard_neg_cap is not None and self.hard_neg_cap < len(hard_neg_list):
            hard_neg_list = hard_neg_list.head(self.hard_neg_cap)

        hard_neg_list = hard_neg_list.to_dict('records')
        for rec in hard_neg_list:
            self.train_ds.append({"emb0": rec["name0"].type(torch.int64), "emb1": rec["name1"].type(torch.int64), "label": 0.0, "mutable": 1, "type": 3})

        random.shuffle(self.train_ds)

        return n_added, n_pairs_found, n_already_used

    def get_types_in_train_set(self):
        out_str = ""

        sets_list = [self.train_ds, self.val_ds, self.test_ds]
        sets_name = ["Train", "  Val", " Test"]
        
        for ttv_set, ttv_name in zip(sets_list, sets_name):
            types_list = [0] * 5

            for entry in ttv_set:
                if "type" in entry:
                    types_list[entry["type"]] += 1
                else:
                    types_list[4] += 1

            out_str += f"{ttv_name} -> Positives: {types_list[0]}, Random: {types_list[1]}, Jeremy: {types_list[2]}, Hard: {types_list[3]}, Uninitialized: {types_list[4]}, Total: {sum(types_list)}"

            if ttv_name != " Test":
                out_str += "\n"

        return out_str


class EmbeddingsMasterList():
    def __init__(self, pair_dataset, master_dataset, trees = 40, dimensions = 50):
        self.pair_dataset = pair_dataset
        self.master_dataset = master_dataset

        self.trees = trees
        self.dimensions = dimensions
        self.embeddings = None

        self.i = 0

        if os.path.isfile("data/index2name.npy") and os.path.isfile("data/name2index.npy"):
            self.index2name = np.load("data/index2name.npy", allow_pickle = True).item()
            self.name2index = np.load("data/name2index.npy", allow_pickle = True).item()
        else:
            self.name2index, self.index2name = self._build_dict()

    def embed_all(self, model):
        self.embeddings = AnnoyIndex(self.dimensions, 'euclidean')
        model.eval()
        for i in range(len(self.master_dataset)):
            embedded_name = self.master_dataset[i]['name'].unsqueeze(0)
            v = model(embedded_name).squeeze(0)
            self.embeddings.add_item(i, v)

        self.embeddings.build(self.trees)

        #self.embeddings.load("test.ann")
        
        model.train()

    def get_nn(self, index, number = 1):
        return self.embeddings.get_nns_by_item(index, number + 1)[1]
    
    def get_distance(self, i, j):
        return self.embeddings.get_distance(i, j)

    def _build_dict(self):
        index = 0

        name2index = dict()

        for row in self.master_dataset:
            name2index[emb2str(row["name"])] = index
            index += 1

        index2name = {v: k for k, v in name2index.items()}
        #/home/nvinden/projects/def-lantonie/jfoxcrof
        np.save("data/index2name.npy", index2name) 
        np.save("data/name2index.npy", name2index)

        return name2index, index2name
    
class TorchArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if torch.is_tensor(obj):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)