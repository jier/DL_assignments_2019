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

import os
import sys
sys.path.append("..") 
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

################################################################################

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

      # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length )  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size,
                 lstm_num_hidden=config.lstm_num_hidden, lstm_num_layers=config.lstm_num_layers, device=config.device )  # fixme

  

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    for epoch in range(config.epochs):
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()

            #######################################################
            # Add more code here ...
            #######################################################
            optimizer.zero_grad()
            # Set to float LongTensor output dtype of one_hot produces internal error for forward
            batch_inputs = torch.nn.functional.one_hot(batch_inputs, num_classes=dataset.vocab_size).float().to(device)
            
            batch_targets = batch_targets.to(device)
            out, _ = model.forward(batch_inputs)

            #Expected size 64 x 87 x 30 got 64 x 30 x 87 to compute with 64 x 30
            loss = criterion(out.permute(0,2,1), batch_targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
            optimizer.step()

            predictions = out.argmax(dim=-1)
            accuracy = (predictions == batch_targets).float().mean()

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if step % config.print_every == 0:

                print("[{}] Train Step {:04d}/{:04d}, Epoch {:d} Batch Size = {}, Examples/Sec = {:.2f}, "
                    "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                        int(config.train_steps), epoch, config.batch_size, examples_per_second,
                        accuracy, loss
                ))

            if step % config.sample_every == 0:
                # Generate some sentences by sampling from the model
                generate_sentence(model, config, dataset)
                # pass

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

    print('Done training.')

def generate_sentence(model, config, dataset):
    def generate_sequence(model, sample, seq_length, temp):
        
        state = None
        sequences = []
        for _ in range(config.seq_length):
            sequences.append(sample.item())
            # sample need to be long size datatype to support one hot torch operation 
            sample = torch.nn.functional.one_hot(sample.long(),num_classes=dataset.vocab_size).float()

            if state is None:
                output, state = model.forward(sample)
            else:
                output, state = model.forward(sample, state)

            # print(f'output shape before flatten {output.shape}')
            output = output.reshape(-1)
            # print(f'output shape after flatten {output.shape}')
           
            softmax = model.softmax(output * (1 / temp))
            # Encounter NaN at distribution not useful!!
            # softmax_ = output.data.view(-1).div(temp).exp()
            # print(f'diff softmax  allclose {np.allclose(softmax, softmax_)} ')
            # sys.exit(0)

            sample = softmax.multinomial(1).reshape([1, 1])
        return sequences

    with torch.no_grad():
        sample_sentence = np.random.randint(0, dataset.vocab_size, size=(1,1))
        sample_sentence = torch.from_numpy(sample_sentence).float()
  
        gen_sequence = generate_sequence(model, sample_sentence, config.seq_length, config.temp)
        sentence = dataset.convert_to_string(gen_sequence)
        print(f'------GENERATED SENTENCE------- {sentence}\n')

 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--epochs', type=int, default=20, help='How many epochs needed to help LSTM converges')
    parser.add_argument('--temp', type=int, default=1e-3, help='Temperature to sample during softmax')

    config = parser.parse_args()

    # Train the model
    train(config)
