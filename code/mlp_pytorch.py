"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes, neg_slope):
    """
    Initializes MLP object.

    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
      neg_slope: negative slope parameter for LeakyReLU

    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(MLP, self).__init__()

    self.n_classes = n_classes
    self.neg_slope = neg_slope
    self.n_inputs = n_inputs

    self.n_hidden = [self.n_inputs] + n_hidden
    self.net = []
    for i, h in enumerate(self.n_hidden[:-1]):
      self.net.extend([nn.Linear(self.n_hidden[i], self.n_hidden[i + 1]), nn.LeakyReLU(negative_slope=self.neg_slope)])

    # No softmax, included in CrossEntropyLoss
    self.net.extend([nn.Linear(self.n_hidden[-1],self.n_classes)])
    self.net = nn.Sequential(*self.net)
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through
    several layer transformations.

    Args:
      x: input to the network
    Returns:
      out: outputs of the network

    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    for net in self.net:
      x = net(x)

    out = x
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
