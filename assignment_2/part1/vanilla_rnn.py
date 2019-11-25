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

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...

        self.seq_length = seq_length
        self.device = device
        self.num_classes = num_classes
        self.input_dim = input_dim
        
        mean = 0.0
        std = 1e-4
        
        self.W_hx = nn.Parameter(torch.Tensor(num_hidden, input_dim).normal_(mean=mean, std=std)).to(self.device)
        self.W_hh = nn.Parameter(torch.Tensor(num_hidden, num_hidden).normal_(mean=mean, std=std)).to(self.device)
        self.W_hy = nn.Parameter(torch.Tensor(num_hidden, num_classes).normal_(mean=mean, std=std)).to(self.device)

        self.hidden = torch.zeros((num_hidden, 1), requires_grad=False).to(self.device)

        self.b_h = nn.Parameter(torch.zeros((num_hidden, 1)))
        self.b_p = nn.Parameter(torch.zeros((num_classes, 1)))

        self.tanh = nn.Tanh()
        # print(f"self.hidden {self.hidden.shape}, input_x_weight {self.W_hx.shape}, hidden_layer {self.W_hh.shape} ,Output {self.W_hy.shape}")
        # sys.exit(0)


    def forward(self, x):
        # Implementation here ...
        
        hidden = self.hidden.to(self.device)
        # inputs = (np.arange(self.num_classes) == x[..., None].cpu().detach().numpy()).astype(int)
        # print(f'x shape {x.shape} ') #test {x[:, : -1].shape}, new dim {x[...,None].shape}')
        # sys.exit(0)
        # x = x[..., None]
        for step in range(self.seq_length):
            # print(f'x shape {x.shape} step shape {x[:,step].shape} wrong? {x[step,:].shape}')
            # sys.exit(0)
            hidden = self.tanh(self.W_hx @ x[:,step].reshape(1, -1) + self.W_hh @ hidden + self.b_h)
        out = self.W_hy.t() @ hidden + self.b_p

        return out.t()