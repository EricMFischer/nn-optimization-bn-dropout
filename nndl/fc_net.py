import numpy as np
import pdb

from .layers import *
from .layer_utils import *

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """
  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize all parameters of the network in the self.params dictionary.
    #   The weights and biases of layer 1 are W1 and b1; and in general the 
    #   weights and biases of layer i are Wi and bi. The
    #   biases are initialized to zero and the weights are initialized
    #   so that each parameter has mean 0 and standard deviation weight_scale.
    #
    #   BATCHNORM: Initialize the gammas of each layer to 1 and the beta
    #   parameters to zero.  The gamma and beta parameters for layer 1 should
    #   be self.params['gamma1'] and self.params['beta1'].  For layer 2, they
    #   should be gamma2 and beta2, etc. Only use batchnorm if self.use_batchnorm 
    #   is true and DO NOT batch normalize the output scores.
    # ================================================================ #
    
    # N,D,H1,H2,C = 2,15,20,30,10
    dims = [input_dim] + hidden_dims + [num_classes] # (15,20,30,10)
    for i in range(self.num_layers):
        layer_num = str(i + 1)
        input_dimension = dims[i] # D,H1,H2: 15,20,30
        current_dim = dims[i + 1] # H1,H2,C: 20,30,10

        weights = np.random.normal(0, weight_scale, (input_dimension, current_dim))
        self.params['W' + layer_num] = weights
        self.params['b' + layer_num] = np.zeros(current_dim)
        
        # BATCH NORMALIZATION
        if self.use_batchnorm and i < len(hidden_dims):
            self.params['beta' + layer_num] = np.zeros(current_dim)
            self.params['gamma' + layer_num] = np.ones(current_dim)
    
    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in np.arange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.
    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    # ================================================================ #
    #   Implement the forward pass of the FC net and store the output
    #   scores as the variable "scores".
    #
    #   BATCHNORM: If self.use_batchnorm is true, insert a bathnorm layer
    #   between the affine_forward and relu_forward layers.  You may
    #   also write an affine_batchnorm_relu() function in layer_utils.py.
    #
    #   DROPOUT: If dropout is non-zero, insert a dropout layer after
    #   every ReLU layer.
    # ================================================================ #
    num_hidden_layers = self.num_layers - 1
    h_caches = {}

    h = X
    for i in range(num_hidden_layers):
        layer_num = str(i + 1)
        W = self.params['W' + layer_num]
        b = self.params['b' + layer_num]
        
        # BATCH NORMALIZATION
        # batchnorm_forward layer needs to be inserted between each affine and relu
        # layer (except in the output layer) in a forward pass computation in loss
        if self.use_batchnorm:
            beta = self.params['beta' + layer_num]
            gamma = self.params['gamma' + layer_num]
            # with batch normalization we need to keep track of running means and
            # variances. pass self.bn_params[0] to forward pass of first batch
            # normalization layer, etc.
            bn_params = self.bn_params[i]
            h, h_cache = affine_bn_relu_forward(h, W, b, beta, gamma, bn_params)
        else:
            h, h_cache = affine_relu_forward(h, W, b)
        # if dropout is non-zero, insert a dropout layer after every ReLU layer
        if self.use_dropout:
            h, dropout_cache = dropout_forward(h, self.dropout_param)
            h_cache = (*h_cache, dropout_cache)

        # updates cache
        h_caches['h' + layer_num] = h_cache
    
    # scores computation for last scores layer
    scores_layer_num = str(self.num_layers)
    scores_W = self.params['W' + scores_layer_num]
    scores_b = self.params['b' + scores_layer_num]
    scores, scores_cache = affine_forward(h, scores_W, scores_b)

    # If test mode return early
    if mode == 'test':
      return scores

    loss = 0.0
    grads = {}
    # ================================================================ #
    #   Implement the backwards pass of the FC net and store the gradients
    #   in the grads dict, so that grads[k] is the gradient of self.params[k]
    #   Be sure your L2 regularization includes a 0.5 factor.
    #
    #   BATCHNORM: Incorporate the backward pass of the batchnorm.
    #
    #   DROPOUT: Incorporate the backward pass of dropout.
    # ================================================================ #
    loss, dscores = softmax_loss(scores, y)
    for i in range(self.num_layers):
        layer_num = str(i + 1)
        W = self.params['W' + layer_num]
        loss += 0.5 * self.reg * np.sum(W**2)
    
    # backward propagation for scores layer
    dh, dW, db = affine_backward(dscores, scores_cache) # upstream deriv, cache
    scores_layer_num = str(self.num_layers)
    scores_W = self.params['W' + scores_layer_num]
    grads['b' + scores_layer_num] = db
    grads['W' + scores_layer_num] = dW + self.reg * scores_W
    
    # backward propagation for all hidden layers
    for i in range(num_hidden_layers, 0, -1):
        layer_num = str(i)
        layer_cache = h_caches['h' + layer_num]
        if self.use_dropout:
            dropout_cache = layer_cache[-1]
            dh = dropout_backward(dh, dropout_cache)
            h_caches['h' + layer_num] = layer_cache[:-1] # remove dropout cache
            layer_cache = h_caches['h' + layer_num]

        if self.use_batchnorm:
            dh, dW, db, dbeta, dgamma = affine_bn_relu_backward(dh, layer_cache)
            grads['beta' + layer_num] = dbeta
            grads['gamma' + layer_num] = dgamma
        else:
            dh, dW, db = affine_relu_backward(dh, layer_cache) # upstream d, cache
        grads['b' + layer_num] = db
        grads['W' + layer_num] = dW + self.reg * self.params['W' + layer_num]
    
    return loss, grads
