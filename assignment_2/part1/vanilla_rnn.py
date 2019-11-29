################################################################################
# MIT License
#
# Copyright (c) 2019
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

import torch
import torch.nn as nn
import numpy as np
import sys
################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device):
        super(VanillaRNN, self).__init__()
        # Initialization here ...

        self.seq_length = seq_length
        self.device = device
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.num_hidden = num_hidden
        
        self.W_hx = nn.Parameter(torch.Tensor(num_hidden, input_dim), requires_grad=True)
        self.W_hh = nn.Parameter(torch.Tensor(num_hidden, num_hidden), requires_grad=True)
        self.W_hy = nn.Parameter(torch.Tensor(num_hidden, num_classes), requires_grad=True)

        self.grad_hidden_list = []
        
         # Xavier bound 
        bound = np.sqrt(1 / num_hidden)
        # print('bound for xavier: ', bound)
        for param in self.parameters():
            nn.init.uniform_(param, -bound, bound)
        
        self.b_h = nn.Parameter(torch.zeros(num_hidden, 1), requires_grad=True)
        self.b_p = nn.Parameter(torch.zeros(num_classes, 1), requires_grad=True)

        self.tanh = nn.Tanh()


    def forward(self, x):
        # Implementation here ...
        
        hidden = torch.zeros((self.num_hidden, x.shape[0]), requires_grad=True).to(self.device)
        for step in range(self.seq_length):
           
            hidden = self.tanh(self.W_hx @ x[:,step].reshape(1, -1)  + self.W_hh @ hidden + self.b_h)
            h = torch.zeros((self.num_hidden, x.shape[0]), requires_grad=True).to(self.device)
            hidden = h + hidden
            self.grad_hidden_list.append(h)

        out = self.W_hy.t() @ hidden + self.b_p

        return out.t()