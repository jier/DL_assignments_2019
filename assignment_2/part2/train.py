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
import operator
from functools import reduce

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel
from torch.utils.tensorboard import SummaryWriter
from preprocess import preprocess
################################################################################

def train(config):

    if config.tensorboard:
        writer = SummaryWriter(config.summary + datetime.now().strftime("%Y%m%d-%H%M%S"))
    # Initialize the device which to run the model on
    device = torch.device(config.device)

      # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length )  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size,
                 lstm_num_hidden=config.lstm_num_hidden, lstm_num_layers=config.lstm_num_layers, device=config.device )  

  

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
                if config.tensorboard:
                    writer.add_scalar('training_loss', loss, step)
                    writer.add_scalar('accuracy', accuracy, step)

            if step % config.sample_every == 0:
                # Generate some sentences by sampling from the model
                # print(f'shape state {state[1].shape}')
                # sys.exit(0)
                generate_sentence(step, model, config, dataset)
                # pass

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                print('Done training.')
                break

    

def generate_sentence(step, model, config, dataset):

    def generate_sequence(model, sample, seq_length, temp, input_sentence=[]):
        
        state = None
        # Gather only the last character of the sentence to generate a new sentence
        ones = torch.ones(sample.shape[1]).reshape(1, -1)
        ones[:,0] = sample[:,-1]
        sentence_char = ones
        start = 1

        # To avoid ovewriting given input sentences
        if len(input_sentence) is not 0:
            start = len(input_sentence)

        for i in range(start, config.desired_seq_length + len(input_sentence)):
            # sample need to be long size datatype to support one hot torch operation 
            sample = torch.nn.functional.one_hot(sentence_char.long(), num_classes=dataset.vocab_size).float().to(config.device)

            if state is None:
                output, state = model.forward(sample)
            else:
                output, state = model.forward(sample, state)
   
            output = output.permute(1, 0, 2)

            if config.temp is None:
                # Greedy 
                prediction = output[i-1]
                prediction = prediction.argmax(dim=-1)
                # Use in the loop 2D list otherwise dimension issue in forward expected 1 x B  x I  otherwise B x I
                sentence_char[:,i] = prediction

            else:
                # Temperature
                softmax_prob = torch.nn.functional.softmax(output[i -1]  / temp, dim=1) 
                prediction = torch.multinomial(softmax_prob,1)[0]
                sentence_char[:,i] = prediction

        # indices needs to be int otherwise Keyerror is raised
        return sentence_char[0].int()

    with torch.no_grad():
        if not config.input_sentence:
            sample_sentence = np.random.randint(0, dataset.vocab_size, size=(1, config.desired_seq_length))
            sample_sentence = torch.from_numpy(sample_sentence).float()
            gen_sequence = generate_sequence(model, sample_sentence, config.seq_length, config.temp, [])
        else:
            input_chars = torch.tensor([dataset._char_to_ix[char] for char in config.input_sentence]).unsqueeze(0)
            sample_sentence = np.random.randint(0, dataset.vocab_size, size=(1, config.desired_seq_length + len(input_chars)) )
            sample_sentence = sample_sentence[0:len(input_chars)]
            sample_sentence = torch.from_numpy(sample_sentence).float()
            gen_sequence = generate_sequence(model, sample_sentence, config.seq_length, config.temp, input_chars)
    
        
        # Write to numpy because dataset expect numpy array dtype
        sentence = dataset.convert_to_string(gen_sequence.detach().cpu().numpy())
   
        print("---------GENERATED SENTENCE---------------------\n",file=open(config.sentence_file, "a"))
        print(f' {sentence}\n', file=open(config.sentence_file, "a"))
        print(f'GENERATED SENTENCE at step {step} ------- {sentence}\n', file=open(config.sentence_file, "a"))

 ################################################################################
 ################################################################################

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
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
    parser.add_argument('--temp', type=float, default=None, help='Temperature to sample during softmax') # 1e-3
    parser.add_argument('--desired_seq_length', type=int, default=50, help='Length of an input sequence to generate')
    parser.add_argument('--sentence_file', type=str, default="gen_text_nieuw.txt", help='Length of an input sequence')
    parser.add_argument('--input_sentence', type=str, default="", help='User input sentence to starts prediction every sentence in lower caption')
    parser.add_argument('--summary', type=str, default='runs/Sentences', help='Specify where to write out tensorboard summaries')
    parser.add_argument('--tensorboard', type=int, default=0, help='Use tensorboard for one run, default do not show')

    config = parser.parse_args()
    config.txt_file  = preprocess(config.txt_file)
    if config.sentence_file:
        print(f"------------------FROM TEXT FILE {config.txt_file}--------------------------------------",file=open(config.sentence_file, "a")) 
    # Train the model
    train(config)
