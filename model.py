import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.START = 30
        self.END = 31
        self.PAD = 32

        assert self.A_name in ["lstm", "gru", "attention"]
        assert self.embedding_type in ["one_hot", "normal"]

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

        if self.A_name in ["lstm", "gru"] and self.bidirectional:
            self.bidirectional_linear = nn.Linear(self.n_tokens * 2 * self.hidden_dim, self.hidden_dim)
        elif self.A_name in ["attention", ]:
            self.attention_linear = nn.Linear(self.input_dim * self.n_tokens, self.hidden_dim)

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

    def forward(self, seq0, seq1):
        self.batch_size = len(seq0)

        assert len(seq0) == len(seq1)

        seq0_embedded = self.embedding_function(seq0)
        seq1_embedded = self.embedding_function(seq1)

        #TAKES EMBEDDED SEQUENCES AND TURNS THEM INTO 2 SIMILARITY VECTORS
        #   Normal lstm and gru functions
        #   Designed from https://www.researchgate.net/publication/307558687_Siamese_Recurrent_Architectures_for_Learning_Sentence_Similarity
        if self.A_name in ["lstm", "gru"] and not self.bidirectional:
            h_0, _ = self.A_function(seq0_embedded)
            h_1, _ = self.A_function(seq1_embedded)

            indexes_of_end_h0 = (seq0 == self.END).nonzero()[:, 1]
            indexes_of_end_h1 = (seq1 == self.END).nonzero()[:, 1]

            h_0_out = [h_0[i, indexes_of_end_h0[i] - 2].unsqueeze(0) for i in range(len(h_0))]
            h_1_out = [h_1[i, indexes_of_end_h1[i] - 2].unsqueeze(0) for i in range(len(h_1))]

            h_0_out = torch.cat(h_0_out, dim = 0)
            h_1_out = torch.cat(h_1_out, dim = 0)

        #   Bidirectional lstm and gru functions
        #   Designed from https://aclanthology.org/W16-1617.pdf
        elif self.A_name in ["lstm", "gru"] and self.bidirectional:
            h_0, _ = self.A_function(seq0_embedded)
            h_1, _ = self.A_function(seq1_embedded)

            h_0 = h_0.reshape(self.batch_size, -1)
            h_1 = h_1.reshape(self.batch_size, -1)

            h_0_out = self.bidirectional_linear(h_0)
            h_1_out = self.bidirectional_linear(h_1)
        #   Attention based encoder
        elif self.A_name == "attention":
            h_0 = self.A_function(seq0_embedded)
            h_1 = self.A_function(seq1_embedded)

            h_0 = h_0.reshape(self.batch_size, -1)
            h_1 = h_1.reshape(self.batch_size, -1)

            h_0_out = self.attention_linear(h_0)
            h_1_out = self.attention_linear(h_1)

        distance = self.manhatten_distance(h_0_out, h_1_out)
        
        return distance, (h_0_out, h_1_out)

