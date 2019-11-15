"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
import matplotlib.pyplot as plt
# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5500
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.

  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch

  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  accuracy = (predictions.argmax(-1) == targets.argmax(1)).float().mean().item()
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model.

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  torch.manual_seed(42)
  ########################
  # PUT YOUR CODE HERE  #
  #######################
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  cifar10 = cifar10_utils.get_cifar10(data_dir=FLAGS.data_dir)
  train_data = cifar10['train']

  n_channels = train_data.images.shape[0]
  n_classes = train_data.labels.shape[1]

  net = ConvNet(n_channels, n_classes)
  net.to(device)

  params = net.parameters()
  optimizer = torch.optim.Adam(params, lr=FLAGS.learning_rate)
  criterion = torch.nn.CrossEntropyLoss()
  rloss = 0
  train_acc_plot = []
  test_acc_plot = []
  loss_train = []
  loss_test = []

  print(f'[DEBUG] start training.... Max steps {FLAGS.max_steps}')

  for i in range(0, FLAGS.max_steps):
    x, y = cifar10['train'].next_batch(FLAGS.batch_size)
    x, y = torch.from_numpy(x).float().to(device) , torch.from_numpy(y).float().to(device)
    out = net.forward(x)
    loss = criterion(out, y.argmax(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    rloss += loss.item()

    if i % FLAGS.eval_freq == 0:
      train_accuracy =  accuracy(out, y)
      with torch.no_grad():
        test_accuracys, test_losses = [] ,[]
        for j in range(0, FLAGS.max_steps):
          test_x, test_y = cifar10['test'].next_batch(FLAGS.batch_size)
          test_x, test_y = torch.from_numpy(test_x).float().to(device) , torch.from_numpy(test_y).float().to(device)

          test_out  = net.forward(test_x)
          test_loss = criterion(test_out, test_y.argmax(1))
          test_accuracy = accuracy(test_out, test_y)
          if device == 'cpu':
            test_losses.append(test_loss)
          else:
            test_losses.append(test_loss.cpu().data.numpy())

          test_accuracys.append(test_accuracy)
        t_acc = np.array(test_accuracys).mean()
        t_loss = np.array(test_losses).mean()
        train_acc_plot.append(train_accuracy)
        test_acc_plot.append(t_acc)
        loss_train.append(rloss/(i + 1))
        loss_test.append(t_loss)
        print(f"iter {i}, train_loss_avg {rloss/(i + 1)}, test_loss_avg {t_loss}, train_acc {train_accuracy}, test_acc_avg {t_acc}")
  print('[DEBUG] Done training')
  if FLAGS.plot:
    print('[DEBUG] Start plotting...')
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(np.arange(len(train_acc_plot)), train_acc_plot, label='training')
    ax1.plot(np.arange(len(test_acc_plot)), test_acc_plot, label='testing')
    ax1.set_title('Training evaluation with batch size '+str(FLAGS.batch_size)+'\n learning rate '+str(FLAGS.learning_rate)+ '\n best accuracy '+str(max(test_acc_plot) )
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax2.plot(np.arange(len(loss_train)), loss_train, label='Train Loss')
    ax2.plot(np.arange(len(loss_test)), loss_test, label='Test Loss')
    ax2.set_title('Loss evaluation')
    ax2.set_ylabel('Loss')
    ax2.legend()
    plt.xlabel('Iteration')
    plt.savefig('convnet.png')
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--plot', type=int, default=0,
                      help='Visualise model with plots, default do not plot')
  FLAGS, unparsed = parser.parse_known_args()

  main()
