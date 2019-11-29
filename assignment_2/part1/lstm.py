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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes,batch_size, device):
        super(LSTM, self).__init__()

        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.device = device

      
        # hidden layer, no backprop
        self.h = torch.zeros(num_hidden, batch_size).to(device)
        self.grad_hidden_list = []
        self.c = torch.zeros((num_hidden,batch_size)).to(device)

        # weights
        # W_f, W_i, W_o same dim as C_t (hidden) x (x)
        # input modulation gate
        self.W_gx = nn.Parameter(torch.Tensor(input_dim, num_hidden),requires_grad=True)
        self.W_gh = nn.Parameter(torch.Tensor(num_hidden, num_hidden),requires_grad=True)

        # input gate
        self.W_ix = nn.Parameter(torch.Tensor(input_dim, num_hidden),requires_grad=True)
        self.W_ih = nn.Parameter(torch.Tensor(num_hidden, num_hidden),requires_grad=True)

        # forget gate
        self.W_fx = nn.Parameter(torch.Tensor(input_dim, num_hidden),requires_grad=True)
        self.W_fh = nn.Parameter(torch.Tensor(num_hidden, num_hidden),requires_grad=True)

        # output gate
        self.W_ox = nn.Parameter(torch.Tensor(input_dim, num_hidden),requires_grad=True)
        self.W_oh = nn.Parameter(torch.Tensor(num_hidden, num_hidden),requires_grad=True)

        self.W_ph = nn.Parameter(torch.Tensor(num_hidden, num_classes),requires_grad=True)

        # Xavier bound 
        bound = np.sqrt(1 / num_hidden)
        # print('bound for xavier: ', bound)
        for param in self.parameters():
            nn.init.uniform_(param, -bound, bound)

        # biases
        self.b_g = nn.Parameter(torch.zeros(1,num_hidden), requires_grad=True) # input modulation gate
        self.b_i = nn.Parameter(torch.zeros(1,num_hidden),requires_grad=True) # input gate
        # http://proceedings.mlr.press/v37/jozefowicz15.pdf to set forget gates to ones
        self.b_f = nn.Parameter(torch.ones(1,num_hidden),requires_grad=True) # forget gate
        self.b_o = nn.Parameter(torch.zeros(1,num_hidden),requires_grad=True) # output gate
        self.b_p = nn.Parameter(torch.zeros(1,num_classes),requires_grad=True)#output

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        # Implementation here ...
        h = self.h
        # print(h.shape)
        c = self.c
        for step in range(self.seq_length):
            x_step = x[:,step].unsqueeze(0)
            # LSTM GATES updates
            g_t = self.tanh(x_step @ self.W_gx + torch.t(h) @ self.W_gh + self.b_g)
            i_t = self.sigmoid(x_step @ self.W_ix + torch.t(h) @ self.W_ih + self.b_i)
            f_t = self.sigmoid(x_step @ self.W_fx + torch.t(h) @ self.W_fh + self.b_f)
            o_t = self.sigmoid(x_step @ self.W_ox + torch.t(h) @self.W_oh + self.b_o)
            c = g_t * i_t + c * f_t
            h = self.tanh(c) * o_t
            # print(x_step.shape, self.W_ox.shape, torch.t(h).shape, self.W_oh.shape, self.b_o.shape)
            h_ = torch.zeros(self.num_hidden, self.batch_size).to(self.device)
            h = h_ + h 
            self.grad_hidden_list.append(h_)
        
        # print(h.shape, self.W_ph.shape, self.b_p.shape)
        out = torch.t(h) @ self.W_ph + self.b_p
        # print(out.shape)
        # sys.exit(0)
        return out