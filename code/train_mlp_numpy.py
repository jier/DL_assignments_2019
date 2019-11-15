"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

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
  # accuracy = total_correct / total(batch size)
  accuracy = (predictions.argmax(1) == targets.argmax(1)).sum() / predictions.shape[0]
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

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Get negative slope parameter for LeakyReLU
  neg_slope = FLAGS.neg_slope

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
  train_data = cifar10['train']

  # 60000 x 3 x 32 x32 -> 60000 x 3072, input vector 3072
  n_inputs = train_data.images.reshape(train_data.images.shape[0], -1).shape[1]
  n_hidden = dnn_hidden_units
  n_classes = train_data.labels.shape[1]

  print(f"n_inputs {n_inputs}, n_classes {n_classes}")
  net = MLP(n_inputs, n_hidden, n_classes, neg_slope=neg_slope)
  loss = CrossEntropyModule()
  train_acc_plot = []
  test_acc_plot = []
  loss_train = []
  loss_test = []
  rloss = 0
  print('[DEBUG] start training')
  for i in range(0, FLAGS.max_steps):
    x, y = cifar10['train'].next_batch(FLAGS.batch_size)
    x = x.reshape(x.shape[0], -1)

    out = net.forward(x)
    loss_forward = loss.forward(out, y)

    loss_grad = loss.backward(out, y)

    net.backward(loss_grad)

    for n in net.net:
      if hasattr(n, 'params'):
        n.params['weight'] = n.params['weight'] - FLAGS.learning_rate* n.grads['weight']
        n.params['bias'] = n.params['bias'] - FLAGS.learning_rate * n.grads['bias']

    rloss += loss_forward
    if  i % FLAGS.eval_freq == 0:
      train_accuracy = accuracy(out, y)

      testX, testY = cifar10['test'].images, cifar10['test'].labels
      testX = testX.reshape(testX.shape[0], -1)

      testOut = net.forward(testX)
      testLoss = loss.forward(testOut, testY)

      test_accuracy = accuracy(testOut, testY)
      train_acc_plot.append(train_accuracy)
      test_acc_plot.append(test_accuracy)
      loss_train.append(rloss/(i + 1))
      loss_test.append(testLoss)
      print(f'iter {i}, avg loss train {rloss/(i + 1)}, test loss {testLoss}, train acc {train_accuracy}, test acc {test_accuracy}')
  if FLAGS.plot:
    print('Start plotting...')
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(np.arange(len(train_acc_plot)), train_acc_plot, label='training')
    ax1.plot(np.arange(len(test_acc_plot)), test_acc_plot, label='testing')
    ax1.set_title('Training evaluation with batch size '+str(FLAGS.batch_size)+'\n learning rate '+str(FLAGS.learning_rate) )
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax2.plot(np.arange(len(loss_train)), loss_train, label='Train Loss')
    ax2.plot(np.arange(len(loss_test)), loss_test, label='Test Loss')
    ax2.set_title('Loss evaluation')
    ax2.set_ylabel('Loss')
    ax2.legend()
    plt.xlabel('Iteration')
    plt.savefig('numpy.png')
    # plt.show()

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
  parser.add_argument('--plot', type=int, default=0,
                      help='Visualise model with plots, default do not plot')
  FLAGS, unparsed = parser.parse_known_args()

  main()
