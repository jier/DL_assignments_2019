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
import torch
from torch.utils.data import DataLoader

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM

# You may want to look into tensorboard for logging
from torch.utils.tensorboard import SummaryWriter

################################################################################

def train(config):

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
                config.gradient_check,
                device=device)

    elif config.model_type=='LSTM':

        model = LSTM(config.input_length,
                config.input_dim,
                config.num_hidden,
                config.num_classes,
                config.batch_size,
                config.gradient_check,
                device=device)
    
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    
    model.to(device)
    # Initialize the dataset and data loader (note the +1)
    # torch.manual_seed(42)
    # np.random.seed(42)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    acc_check = []
    # Setup the loss 
    criterion = torch.nn.CrossEntropyLoss()


    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        # Add more code here ...
        optimizer.zero_grad()
        # Add more code here ...0
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        out = model.forward(batch_inputs)
        # print(f'forward output {out.shape}, batch input shape {batch_inputs.shape}, batch_targets.shape {batch_targets.shape}')
        loss = criterion(out, batch_targets)
        loss.backward()
        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        optimizer.step()

        predictions = out.argmax(dim=-1)
        accuracy = (predictions == batch_targets).float().mean()
        acc_check.append(accuracy.detach().cpu().float())
        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Model type {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, config.model_type, examples_per_second,
                    accuracy, loss
            ))
            if config.tensorboard:
                writer.add_scalar('training_loss', loss, step)
                writer.add_scalar('accuracy', accuracy, step)
            elif config.record_plot:
                with open(CSV_DIR, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([config.model_type, step, config.input_length, accuracy.item(), loss.item()])

        
        if  loss <= 1e-3 and not all([ i is 1.0 for i in acc_check[5:] if len(acc_check) >=5]) :
            break
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/96553
            break

    return accuracy
    


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=5, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence') # 1
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps') #10000
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    # Debug material
    parser.add_argument('--csv', type=str, default='loss_accuracy.csv')
    parser.add_argument('--summary', type=str, default='runs/RNN', help='Specify where to write out tensorboard summaries')
    parser.add_argument('--tensorboard', type=int, default=0, help='Use tensorboard for one run, default do not show')
    parser.add_argument('--record_plot', type=int, default=0, help='Useful when training to save csv data to plot')
    parser.add_argument('--gradient_check', type=int, default=0, help='Set to 1 to only record gradients')
    config = parser.parse_args()

    # Train the model
    train(config)