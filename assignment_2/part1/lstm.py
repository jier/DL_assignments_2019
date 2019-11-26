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

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes,batch_size, device='cpu'):
        super(LSTM, self).__init__()

        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.batch_size = batch_size

      
        # hidden layer, no backprop
        self.h = torch.zeros(num_hidden, batch_size).to(device)
        self.c = torch.zeros((num_hidden,batch_size)).to(device)

        # weights
        # W_f, W_i, W_o same dim as C_t (hidden) x (x)
        # input modulation gate
        self.W_gx = nn.Parameter(torch.Tensor(input_dim, num_hidden),requires_grad=True).to(device)
        self.W_gh = nn.Parameter(torch.Tensor(num_hidden, num_hidden),requires_grad=True).to(device)

        # input gate
        self.W_ix = nn.Parameter(torch.Tensor(input_dim, num_hidden),requires_grad=True).to(device)
        self.W_ih = nn.Parameter(torch.Tensor(num_hidden, num_hidden),requires_grad=True).to(device)

        # forget gate
        self.W_fx = nn.Parameter(torch.Tensor(input_dim, num_hidden),requires_grad=True).to(device)
        self.W_fh = nn.Parameter(torch.Tensor(num_hidden, num_hidden),requires_grad=True).to(device)

        # output gate
        self.W_ox = nn.Parameter(torch.Tensor(input_dim, num_hidden),requires_grad=True).to(device)
        self.W_oh = nn.Parameter(torch.Tensor(num_hidden, num_hidden),requires_grad=True).to(device)

        self.W_ph = nn.Parameter(torch.Tensor(num_hidden, num_classes),requires_grad=True).to(device)

        # Xavier bound 
        bound = np.sqrt(1 / num_hidden)
        # print('bound for xavier: ', bound)
        for param in self.parameters():
			# nn.init.orthogonal_(param)
            nn.init.uniform_(param, -bound, bound)

        # biases
        self.b_g = nn.Parameter(torch.zeros(num_hidden), requires_grad=True).to(device) # input modulation gate
        self.b_i = nn.Parameter(torch.zeros(num_hidden),requires_grad=True).to(device) # input gate
        # http://proceedings.mlr.press/v37/jozefowicz15.pdf to set forget gates to ones
        self.b_f = nn.Parameter(torch.ones(num_hidden),requires_grad=True).to(device) # forget gate
        self.b_o = nn.Parameter(torch.zeros(num_hidden),requires_grad=True).to(device) # output gate
        self.b_p = nn.Parameter(torch.zeros(num_classes),requires_grad=True).to(device)#output

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        # Implementation here ...
        h = self.h
        c = self.c
        for step in range(self.seq_length):
            x_step = x[:,step].reshape(self.batch_size, self.input_dim)
            # LSTM GATES updates
            g_t = self.tanh(x_step @ self.W_gx + h @ self.W_gh + self.b_g)
            i_t = self.sigmoid(x_step @ self.W_ix + h @ self.W_ih + self.b_i)
            f_t = self.sigmoid(x_step @ self.W_fx + h @ self.W_fh + self.b_f)
            o_t = self.sigmoid(x_step @ self.W_ox + h @self.W_oh + self.b_o)

            c = g_t * i_t + c * f_t
            h = self.tanh(c) * o_t

        out = h @ self.W_ph + self.b_p

        # NOTE: nn.CrossEntropyLoss() already has a softmax layer
        # out = self.softmax(p_t)
        return out