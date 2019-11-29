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
import sys
import os
sys.path.append("..") 
import argparse
import time
from datetime import datetime
import numpy as np
import csv 
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM

# You may want to look into tensorboard for logging
from torch.utils.tensorboard import SummaryWriter

################################################################################

def grads_over_time(config):

    assert config.model_type in ('RNN', 'LSTM')
    if config.tensorboard:
        writer = SummaryWriter(config.summary + datetime.now().strftime("%Y%m%d-%H%M%S"))
    elif config.record_plot:
        CSV_DIR = config.csv
        if not os.path.isfile(CSV_DIR):
            f = open(CSV_DIR, 'w')
            writer = csv.writer(f)
            writer.writerow(['model_type', 'step', 'input_length', 'accuracy', 'loss'])
            f.close()
    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    if config.model_type=='RNN':

        model = VanillaRNN(config.input_length,
                config.input_dim,
                config.num_hidden,
                config.num_classes,
                config.batch_size,
                device=device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    elif config.model_type=='LSTM':

        model = LSTM(config.input_length,
                config.input_dim,
                config.num_hidden,
                config.num_classes,
                config.batch_size,
                device=device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    
    model.to(device)
    # Initialize the dataset and data loader (note the +1)
    # torch.manual_seed(42)
    # np.random.seed(42)
    dataset = PalindromeDataset(config.input_length+1)
    # Setup the loss 
    criterion = torch.nn.CrossEntropyLoss()

    # Add more code here ...
   
    # Add more code here ...0
    batch_inputs, batch_targets = dataset[0]
    batch_inputs = torch.from_numpy(batch_inputs).unsqueeze(0).to(device)
    batch_targets = torch.from_numpy(np.array([batch_targets])).to(device)

    out = model.forward(batch_inputs)
    loss = criterion(out, batch_targets)
    loss.backward()
    ############################################################################
    # QUESTION: what happens here and why?
    ############################################################################
    torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
    ############################################################################
    optimizer.zero_grad()
    optimizer.step()
    gradient_list = []
    for hidden_grad in model.grad_hidden_list:
        # print(torch.norm(hidden_grad.grad).item())
        gradient_list.append(torch.norm(hidden_grad.grad, p=2).item())

    

    return  gradient_list
    


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence') 
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps') 
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    # Debug material
    parser.add_argument('--csv', type=str, default='loss_accuracy.csv')
    parser.add_argument('--summary', type=str, default='runs/RNN', help='Specify where to write out tensorboard summaries')
    parser.add_argument('--tensorboard', type=int, default=0, help='Use tensorboard for one run, default do not show')
    parser.add_argument('--record_plot', type=int, default=0, help='Useful when training to save csv data to plot')
    config = parser.parse_args()

    # Train the model
    gradients = grads_over_time(config)
    config.model_type = 'LSTM'
    gradients_LSTM = grads_over_time(config)
    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(gradients, label="RNN")
    ax.plot(gradients_LSTM, label="LSTM")
    ax.set_xlabel('Step')
    ax.set_ylabel('Gradients')
    ax.set_yscale('log')
    ax.grid()
    ax.legend()

    plt.savefig('hidden_gradients.pdf', format='pdf')
    
