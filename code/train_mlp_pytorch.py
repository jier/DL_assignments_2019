"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02
OPTIMIZER_DEFAULT = 'SGD'

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
  Performs training and evaluation of MLP model.

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  torch.manual_seed(42)
  # torch.backends.cudnn.deterministic = True
  # torch.backends.cudnn.benchmark = False

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Get negative slope parameter for LeakyReLU
  neg_slope = FLAGS.neg_slope
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # print("[DEBUG], Device ", device)

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  cifar10 = cifar10_utils.get_cifar10(data_dir=FLAGS.data_dir)
  train_data = cifar10['train']

  # 60000 x 3 x 32 x32 -> 60000 x 3072, input vector 3072
  n_inputs = train_data.images.reshape(train_data.images.shape[0], -1).shape[1]
  n_hidden = dnn_hidden_units
  n_classes = train_data.labels.shape[1]

  # print(f"[DEBUG] n_inputs {n_inputs}, n_classes {n_classes}")

  model = MLP(n_inputs, n_hidden, n_classes, FLAGS.neg_slope)
  model.to(device)

  params = model.parameters()

  if FLAGS.optimizer == 'Adam':
    optimizer = torch.optim.Adam(params, lr=FLAGS.learning_rate)
  elif FLAGS.optimizer == 'Adamax':
    optimizer = torch.optim.Adamax(params, lr=FLAGS.learning_rate)
  elif FLAGS.optimizer == 'Adagrad':
    optimizer = torch.optim.Adagrad(params, lr=FLAGS.learning_rate)
  elif FLAGS.optimizer == 'Adadelta':
    optimizer = torch.optim.Adadelta(params, lr=FLAGS.learning_rate)
  elif FLAGS.optimizer == 'SparseAdam':
    optimizer = torch.optim.SparseAdam(params, lr=FLAGS.learning_rate)
  else:
    optimizer = torch.optim.SGD(params,lr=FLAGS.learning_rate)


  criterion = torch.nn.CrossEntropyLoss()
  rloss = 0
  best_accuracy = 0
  # print('[DEBUG] start training')

  for i in range(0, FLAGS.max_steps):
    x, y = cifar10['train'].next_batch(FLAGS.batch_size)
    x, y = torch.from_numpy(x).float().to(device) , torch.from_numpy(y).float().to(device)
    x = x.reshape(x.shape[0], -1)

    out = model.forward(x)
    loss = criterion.forward(out, y.argmax(1))
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

          test_x = test_x.reshape(test_x.shape[0], -1)

          test_out  = model.forward(test_x)
          test_loss = criterion(test_out, test_y.argmax(1))
          test_accuracy = accuracy(test_out, test_y)
          if device == 'cpu':
            test_losses.append(test_loss)
          else:
            test_losses.append(test_loss.cpu().data.numpy())

          test_accuracys.append(test_accuracy)
        t_acc = np.array(test_accuracys).mean()
        # if device == 'cpu':
        t_loss = np.array(test_losses).mean()

        # print(f"iter {i}, train_loss_avg {rloss/(i + 1)}, test_loss_avg {t_loss}, train_acc {train_accuracy}, test_acc_avg {t_acc}")
        if t_acc > best_accuracy:
          best_accuracy = t_acc

  # print(f"Best Accuracy {best_accuracy}",flush=True)
  print(best_accuracy)



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
  # print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
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
  parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
                      help='Negative slope parameter for LeakyReLU')
  parser.add_argument('--optimizer', type=str, default=OPTIMIZER_DEFAULT,
                      help='Optimizer needed to run the network')
  FLAGS, unparsed = parser.parse_known_args()

  main()
