# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        # Initialization here...
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm_num_layers = lstm_num_layers
        self.device = device

        self.LSTM = nn.LSTM(
            input_size =self.vocabulary_size,
            hidden_size=self.lstm_num_hidden,
            num_layers=self.lstm_num_layers, 
            batch_first = False # (seq, batch, input) if true (batch, seq, input)
        ).to(self.device)

        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size).to(self.device)

    def forward(self, x, state=None):
        # Implementation here...
        output, state = self.LSTM(x, state)

        return self.linear(output), state
