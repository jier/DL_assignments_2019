"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data.
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module.

    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and
    std = 0.0001. Initialize biases self.params['bias'] with 0.

    Also, initialize gradients with zeros.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.params = {'weight': np.random.normal(scale=1e-4, size=(in_features,out_features)), 'bias': np.zeros(out_features)}
    self.grads = {'weight':np.zeros((in_features,out_features)) , 'bias': np.zeros(out_features)}

    ########################
    # END OF YOUR CODE    #
    #######################



  def forward(self, x):
    """
    Forward pass.

    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.x = x
    #print(f'{x.shape}, {self.params["weight"].shape}')
    out =  x @ self.params['weight']  + self.params['bias']
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to
    layer parameters in self.grads['weight'] and self.grads['bias'].
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    self.grads['weight'] = self.x.T @ dout
    self.grads['bias'] = np.sum(dout, axis=0, keepdims=True)
    dx = dout @  self.params['weight'].T
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class LeakyReLUModule(object):
  """
  Leaky ReLU activation module.
  """
  def __init__(self, neg_slope):
    """
    Initializes the parameters of the module.

    Args:
      neg_slope: negative slope parameter.

    TODO:
    Initialize the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.neg_slope = neg_slope
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.

    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    out = ((x > 0) * x) + ((x <= 0 ) * x * self.neg_slope)
    self.leakyX = x
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = np.ones_like(self.leakyX)
    dx[self.leakyX <= 0] = self.neg_slope
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx * dout


class SoftMaxModule(object):
  """
  Softmax activation module.
  """

  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    y = np.exp(x - np.max(x))
    out = y/np.sum(y, axis=1, keepdims=True)
    self.softmax = out
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.
    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # BxC,  CxC -> BxCxC
    S_diag = np.einsum('bi, ik->bik', self.softmax, np.eye(self.softmax.shape[1]))

    # BxC , BxC -> BxCxC

    S_squared = np.einsum('bi,bj->bij', self.softmax, self.softmax)
    dS = S_diag -S_squared

    # BxCxC, BxC -> BxC
    dx = np.einsum('ijk, ik->ij', dS, dout)

    #######################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """

  def forward(self, x, y):
    """
    Forward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss

    TODO:
    Implement forward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    x = np.clip(x, 1e-20, np.inf) # prevent division by zero
    out = - np.sum(y * np.log(x)) /y.shape[0] # normalise data
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    x = np.clip(x, 1e-20, np.inf)
    dx  = -y/x / y.shape[0]
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
