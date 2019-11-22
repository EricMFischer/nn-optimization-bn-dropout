from .layers import *

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def affine_relu_forward(x, w, b):
  """
  Convenience layer that performs an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)

  cache = (fc_cache, relu_cache)
  return out, cache

def affine_bn_relu_forward(x, w, b, beta, gamma, bn_params):
  """
  Convenience layer that performs an affine transform followed by a batch
  normalization (except in the output layer) followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - beta: Shift paremeter of shape (D,)
  - gamma: Scale parameter of shape (D,)
  - bn_params: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var: Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_params)
  out, relu_cache = relu_forward(bn)

  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache

def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache

  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)

  return dx, dw, db

def affine_bn_relu_backward(dout, cache):
  """
  Backward pass for the affine-bn-relu convenience layer
  """
  fc_cache, bn_cache, relu_cache = cache

  da = relu_backward(dout, relu_cache)
  dbn, dgamma, dbeta = batchnorm_backward(da, bn_cache)
  dh, dw, db = affine_backward(dbn, fc_cache)

  return dh, dw, db, dbeta, dgamma
