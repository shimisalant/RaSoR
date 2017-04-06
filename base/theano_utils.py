import numpy as np
import theano
import theano.tensor as tt


floatX = theano.config.floatX


def cast_floatX_np(n):
  return np.asarray(n, dtype=floatX)


def cast_floatX(x):
  return tt.cast(x, floatX)


def get_shared_floatX(value, name):
  return theano.shared(cast_floatX_np(value), name)


def gpu_int32(name, x_val, return_shared_var=False):
  # theano trick: stored as float32 on GPU (if we're using a GPU), and cast back to int32
  assert x_val.dtype == np.int32
  shared_var = get_shared_floatX(x_val, name)
  cast_shared_var = tt.cast(shared_var, 'int32')
  cast_shared_var.underlying_shared_var = shared_var
  return (cast_shared_var, shared_var) if return_shared_var else cast_shared_var


def clip_sqrt(x):
  return tt.sqrt(tt.clip(x, 0.0, np.inf))


def softmax_columns_with_mask(x, mask, allow_none=False):
  assert x.ndim == 2
  assert mask.ndim == 2
  # for numerical stability
  x *= mask
  x -= x.min(axis=0, keepdims=True)
  x *= mask
  x -= x.max(axis=0, keepdims=True)
  e_x = mask * tt.exp(x)
  sums = e_x.sum(axis=0, keepdims=True)
  if allow_none:
    sums += tt.eq(sums, 0)
  y = e_x / sums      # every column must have at least one non-masked non-zero element
  return y


def softmax_depths_with_mask(x, mask):
  assert x.ndim == 3
  assert mask.ndim == 3
  # for numerical stability
  x *= mask
  x -= x.min(axis=2, keepdims=True)
  x *= mask
  x -= x.max(axis=2, keepdims=True)
  e_x = mask * tt.exp(x)
  sums = e_x.sum(axis=2, keepdims=True)
  y = e_x / (sums + (tt.eq(sums,0)))
  y *= mask
  return y


def argmax_with_mask(x, mask):
  assert x.ndim == 2
  assert mask.ndim == 2
  x_min = x.min(axis=1, keepdims=True)
  x = mask * x + (1 - mask) * x_min
  return x.argmax(axis=1)

