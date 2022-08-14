import torch
import torch.nn as nn
import torch.nn.functional as F

from names_dataset import NameDataset

from metaphone import doublemetaphone
import jellyfish

import numpy as np

import math

class Siamese(nn.Module):
    def __init__(self, config, model_kwargs):
        super(Siamese, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.batch_size = 32

        self.A_name = config["A_name"]
        self.embedding_type = config["embedding_type"]
        self.n_tokens = config["n_tokens"]

        assert self.A_name in ["lstm", "gru", "attention"]
        assert self.embedding_type in ["one_hot", "normal"]

        self.phonetic_method = config["phonetic_method"] if "phonetic_method" in config else "none"

        if self.A_name in ["lstm", "gru"]:
            self.bidirectional = model_kwargs["bidirectional"]
            self.input_dim = model_kwargs["input_size"]
            self.hidden_dim = model_kwargs["hidden_size"]
        else:
            layer_config = model_kwargs['TransformerEncoderLayer']
            self.input_dim = layer_config['d_model']
            self.hidden_dim = model_kwargs['hidden_dim']

        if self.embedding_type == "one_hot":
            self.embedding_function = self.one_hot_encode
            assert self.n_tokens >= 24
        elif self.embedding_type == "normal":
            self.embedding_function = self.normal_encode

        if self.A_name == "lstm":
            self.A_function = nn.LSTM(**model_kwargs, device = self.device)
        elif self.A_name == "gru":
            self.A_function = nn.GRU(**model_kwargs)
        elif self.A_name == "attention":
            TransformerEncoderLayer_KWARGS = model_kwargs['TransformerEncoderLayer']
            TransformerEncoder_KWARGS = model_kwargs['TransformerEncoder']
            layer = nn.TransformerEncoderLayer(**TransformerEncoderLayer_KWARGS)
            self.A_function = nn.TransformerEncoder(layer, TransformerEncoder_KWARGS['num_layers'])
            self.mask = self.generate_square_subsequent_mask(self.n_tokens)

        if self.A_name in ["lstm", "gru"] and self.bidirectional:
            if self.phonetic_method == "none":
                self.bidirectional_linear = nn.Linear(self.n_tokens * 2 * self.hidden_dim, self.hidden_dim)
            else:
                self.bidirectional_linear = nn.Linear((2 * self.n_tokens + 1) * 2 * self.hidden_dim, self.hidden_dim)
            self.n_layers = model_kwargs["num_layers"]
        elif self.A_name in ["attention", ]:
            self.attention_linear = nn.Linear(self.input_dim * self.n_tokens, self.hidden_dim)

        self.START = self.input_dim
        self.END = self.input_dim + 1
        self.PAD = self.input_dim + 2
        
        self.preddef_on = False
        if "predef_weights" in config and config['predef_weights'] == True:
            self.preddef_on = True

            self.nd = NameDataset()
            country_list = self.nd.get_country_codes(alpha_2=False)
            self.country2idx = {country.name : idx  for idx, country in enumerate(country_list)}

            alpha_to_names = {count.alpha_2 : count.name for count in country_list}
            test = self.nd.get_top_names(n = 20000, use_first_names = False)
            self.n_names_per_country = {alpha_to_names[key]: len(value) for key, value in test.items()}

            self.preddef_input_length = 2 * len(alpha_to_names)
            self.preddef_linear = nn.Sequential(
                nn.Linear(self.preddef_input_length, 1000),
                nn.Sigmoid(),
                nn.Linear(1000, 500),
                nn.Sigmoid(),
                nn.Linear(500, 250),
                nn.Sigmoid(),
                nn.Linear(250, 2 * 2 * self.n_layers * self.hidden_dim),
                nn.Sigmoid()
            )

    def generate_square_subsequent_mask(self, sz: int):
        return torch.triu(torch.ones(sz, sz, device = self.device) * float('-inf'), diagonal=1)

    def one_hot_encode(self, seq):
        out_vector = torch.zeros([seq.shape[0], self.n_tokens, self.input_dim], dtype = torch.float, device = self.device)

        for i, word in enumerate(seq):
            buffer = 0
            for j, char in enumerate(word):
                char = char.item()
                if char == self.END:
                    break
                elif char == self.START:
                    buffer += 1
                    continue
                out_vector[i, j - buffer, char] = 1

        return out_vector

    def normal_encoder(self, seq):
        pass

    def manhatten_distance(self, h_0, h_1):
        out = torch.abs(h_0 - h_1)
        out = torch.sum(out, dim = 1)
        out = torch.exp(-out)

        return out

    def cosine_similarity(self, h_0, h_1):
        return nn.functional.cosine_similarity(h_0, h_1)

    def emb2str(self, emb):
        word = ""
        for char in emb:
            char = char.item()
            if char >= 30:
                continue
            word = word + chr(char + 97)
        return word

    def str2emb(self, string, string_pad = 30):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch_table = torch.full(size = (string_pad, ), fill_value = self.PAD, device = device, dtype = torch.uint8)
        
        if len(string) == 0:
            torch_table[0] = self.START
            torch_table[1] = self.END

        torch_table[0] = self.START
        for j in range(1, len(string) + 1):
            cha = string[j - 1]
            if cha.isdigit():
                torch_table[j] = int(cha) + 26
            else:
                torch_table[j] = ord(cha) - 97
        j += 1
        torch_table[j] = self.END

        return torch_table
    
    def create_predef_input(self, word1):
        word1 = [self.emb2str(word) for word in word1]
        search_components = [self.nd.search(word)['last_name'] for word in word1]

        out = torch.full([len(word1), len(self.country2idx) * 2], -1.0, device = self.device, dtype = torch.float)

        for i, batch in enumerate(search_components):
            if batch is None:
                continue

            for country, value in batch['country'].items():
                if value is None:
                    continue
                country_id = self.country2idx[country]
                out[i, country_id] = value

            for country, value in batch['rank'].items():
                if value is None:
                    continue
                country_id = self.country2idx[country]
                n_name_in_curr_country = self.n_names_per_country[country]
                value = 1 - (value / n_name_in_curr_country)
                out[i, country_id + len(self.country2idx)] = value

        return out
    
    def predef_forward(self, inputs):
        default_out = self.preddef_linear(inputs)

        break_point = default_out.shape[1] // 2

        h_0 = default_out[:, 0:break_point]
        c_0 = default_out[:, break_point:]

        h_0 = h_0.view(self.batch_size, 2 * self.n_layers, self.hidden_dim)
        c_0 = c_0.view(self.batch_size, 2 * self.n_layers, self.hidden_dim)

        h_0 = torch.permute(h_0, (1, 0, 2)).contiguous()
        c_0 = torch.permute(c_0, (1, 0, 2)).contiguous()

        return h_0, c_0

    def get_phonetic(self, seq, method):
        words = [self.emb2str(emb) for emb in seq]
        
        if method == 'soundex':
            phonetic_list = [jellyfish.soundex(word).lower() for word in words]
        elif method == 'metaphone':
            phonetic_list = [doublemetaphone(word)[0].lower() for word in words]
        else:
            print("this is an error if you reach here")
            return None

        phonetic_emb_list = [self.str2emb(word) for word in phonetic_list]
        phonetic_emb_list = torch.stack(phonetic_emb_list, dim = 0)
        phonetic_full_emb = self.embedding_function(phonetic_emb_list)

        return phonetic_full_emb

    def forward(self, seq0, *argv):
        self.batch_size = len(seq0)

        two_vars = False

        for arg in argv:
            if torch.is_tensor(arg):
                seq1 = arg
                two_vars = True

        seq0_embedded = self.embedding_function(seq0)

        if self.phonetic_method != "none":
            seq0_phonetic = self.get_phonetic(seq0, self.phonetic_method)
            seq0_embedded = torch.cat([seq0_embedded, torch.ones([seq0_embedded.shape[0], 1, self.input_dim], device = self.device), seq0_phonetic], dim = 1)

        if two_vars:
            assert len(seq0) == len(seq1)
            seq1_embedded = self.embedding_function(seq1)

            if self.phonetic_method != "none":
                seq1_phonetic = self.get_phonetic(seq1, self.phonetic_method)
                seq1_embedded = torch.cat([seq1_embedded, torch.ones([seq1_embedded.shape[0], 1, self.input_dim], device = self.device), seq1_phonetic], dim = 1)

        #TAKES EMBEDDED SEQUENCES AND TURNS THEM INTO 2 SIMILARITY VECTORS
        #   Normal lstm and gru functions
        #   Designed from https://www.researchgate.net/publication/307558687_Siamese_Recurrent_Architectures_for_Learning_Sentence_Similarity
        if self.A_name in ["lstm", "gru"] and not self.bidirectional:
            h_0, _ = self.A_function(seq0_embedded)
            indexes_of_end_h0 = (seq0 == self.END).nonzero()[:, 1]
            h_0_out = [h_0[i, indexes_of_end_h0[i] - 2].unsqueeze(0) for i in range(len(h_0))]
            h_0_out = torch.cat(h_0_out, dim = 0)

            if two_vars:
                h_1, _ = self.A_function(seq1_embedded)
                indexes_of_end_h1 = (seq1 == self.END).nonzero()[:, 1]
                h_1_out = [h_1[i, indexes_of_end_h1[i] - 2].unsqueeze(0) for i in range(len(h_1))]
                h_1_out = torch.cat(h_1_out, dim = 0)

        #   Bidirectional lstm and gru functions
        #   Designed from https://aclanthology.org/W16-1617.pdf
        elif self.A_name in ["lstm", "gru"] and self.bidirectional:
            if self.preddef_on:
                seq0_predef_input = self.create_predef_input(seq0)
                h_first, c_first = self.predef_forward(seq0_predef_input)
                if self.A_name == "lstm":
                    h_0, _ = self.A_function(seq0_embedded, (h_first, c_first))
                elif self.A_name == "gru":
                    h_0, _ = self.A_function(seq0_embedded, h_first)
            else:
                h_0, _ = self.A_function(seq0_embedded)

            h_0 = h_0.reshape(self.batch_size, -1)
            h_0_out = self.bidirectional_linear(h_0)

            if two_vars:
                if self.preddef_on:
                    seq1_predef_input = self.create_predef_input(seq1)
                    h_first, c_first = self.predef_forward(seq1_predef_input)
                    if self.A_name == "lstm":
                        h_1, _ = self.A_function(seq1_embedded, (h_first, c_first))
                    elif self.A_name == "gru":
                        h_1, _ = self.A_function(seq1_embedded, h_first)
                else:
                    h_1, _ = self.A_function(seq1_embedded)
                h_1, _ = self.A_function(seq1_embedded)
                h_1 = h_1.reshape(self.batch_size, -1)
                h_1_out = self.bidirectional_linear(h_1)

        #   Attention based encoder
        elif self.A_name == "attention":
            h_0 = self.A_function(seq0_embedded, mask = self.mask)
            h_0 = torch.sigmoid(h_0)
            h_0 = h_0.reshape(self.batch_size, -1)
            h_0_out = self.attention_linear(h_0)
            h_0_out = torch.sigmoid(h_0_out)

            if two_vars:
                h_1 = self.A_function(seq1_embedded, mask = self.mask)
                h_1 = torch.sigmoid(h_1)
                h_1 = h_1.reshape(self.batch_size, -1)
                h_1_out = self.attention_linear(h_1)
                h_1_out = torch.sigmoid(h_1_out)

        if two_vars:
            distance = self.cosine_similarity(h_0_out, h_1_out)
            distance = (distance + 1) / 2

            return distance, (h_0_out, h_1_out)
        else:
            return h_0_out

