#!/usr/bin/env python

# wujian@2018

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


class TorchRNN(nn.Module):
    def __init__(self,
                 feature_dim,
                 rnn="lstm",
                 num_layers=1,
                 hidden_size=896,
                 dropout=0.0,
                 bidirectional=False):
        super(TorchRNN, self).__init__()
        RNN = rnn.upper()
        supported_rnn = {"LSTM": nn.LSTM, "RNN": nn.RNN, "GRU": nn.GRU}
        if RNN not in supported_rnn:
            raise RuntimeError("unknown RNN type: {}".format(RNN))
        self.rnn = supported_rnn[RNN](
            feature_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional)
        self.output_dim = hidden_size if not bidirectional else hidden_size * 2

    def forward(self, x, squeeze=False, total_length=None):
        """
        Accept tensor([N]xTxF) or PackedSequence object
        """
        is_packed = isinstance(x, PackedSequence)
        # extend dim when inference
        if not is_packed:
            if x.dim() not in [2, 3]:
                raise RuntimeError(
                    "RNN expect input dim as 2 or 3, got {:d}".format(x.dim()))
            if x.dim() != 3:
                x = th.unsqueeze(x, 0)
        x, _ = self.rnn(x)
        # using unpacked sequence
        # x: NxTxD
        if is_packed:
            x, _ = pad_packed_sequence(
                x, batch_first=True, total_length=total_length)
        if squeeze:
            x = th.squeeze(x)
        return x


class Nnet(th.nn.Module):
    def __init__(self,
                 feats_dim,
                 num_bins=257,
                 num_spks=2,
                 rnn_conf=None,
                 non_linear="relu",
                 dropout=0.0):
        super(Nnet, self).__init__()
        support_non_linear = {
            "relu": F.relu,
            "sigmoid": th.sigmoid,
            "tanh": th.tanh
        }
        if non_linear not in support_non_linear:
            raise ValueError(
                "Unsupported non-linear type:{}".format(non_linear))
        self.num_spks = num_spks
        self.num_bins = num_bins

        self.rnn = TorchRNN(feats_dim, **rnn_conf)
        self.drops = th.nn.Dropout(p=dropout)
        self.linear = th.nn.ModuleList([
            th.nn.Linear(self.rnn.output_dim, num_bins)
            for _ in range(self.num_spks)
        ])
        self.non_linear = support_non_linear[non_linear]

    def forward(self, x, train=True):
        x = self.rnn(x)
        # using unpacked sequence
        # x: N x T x D
        x = self.drops(x)
        m = []
        for linear in self.linear:
            y = linear(x)
            y = self.non_linear(y)
            if not train:
                y = y.view(-1, self.num_bins)
            m.append(y)
        return m