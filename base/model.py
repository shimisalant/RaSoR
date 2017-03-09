
import logging
import os
import numbers
import cPickle

import numpy as np

import theano
import theano.tensor as tt
from theano.ifelse import ifelse

from base.utils import namer, verify_dir_exists
from base.theano_utils import floatX, cast_floatX_np, get_shared_floatX


class BaseModel(object):

  def init_start(self, config):
    self._params = {}
    self._is_training = tt.iscalar('is_training')
    self._np_rng = np.random.RandomState(config.seed // 2 + 123)
    if config.device == 'cpu':
      from theano.tensor.shared_randomstreams import RandomStreams          # works on cpu
    else:
      from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams # works on gpu
    self._theano_rng = RandomStreams(config.seed // 2 + 321)
    self._init_scale = config.init_scale
    self._pre_epoch_hooks = []

  def register_pre_epoch_hook(self, func, for_train=False, for_eval=False):
    assert for_train or for_eval
    self._pre_epoch_hooks.append((func, for_train, for_eval))

  def invoke_pre_epoch_hooks(self, is_train=False, is_eval=False):
    assert is_train or is_eval
    for func, for_train, for_eval in self._pre_epoch_hooks:
      if (for_train and is_train) or (for_eval and is_eval):
        func()

  ################### Constructing shared vars ###################

  def get_param_init(self, shape, init_scheme, init_scale=None):
    if isinstance(init_scheme, numbers.Number):
      value = np.full(shape, float(init_scheme))
    elif init_scheme == 'identity':
      assert len(shape) == 2 and shape[0] == shape[1]
      value = np.eye(shape[0])
    elif init_scheme == 'uniform':
      scale = init_scale or self._init_scale
      value = self._np_rng.uniform(low=-scale, high=scale, size=shape)
    elif init_scheme == 'gaussian':
      scale = init_scale or self._init_scale
      value = self._np_rng.normal(loc=0., scale=scale, size=shape)
    elif init_scheme == 'glorot_uniform':
      assert len(shape) == 2
      s = np.sqrt(6.0 / (shape[0] + shape[1]))
      value = self._np_rng.uniform(low=-s, high=s, size=shape)
    elif init_scheme == 'orthogonal':
      assert len(shape) == 2
      u, _, v = np.linalg.svd(self._np_rng.normal(0.0, 1.0, shape), full_matrices=False)
      #assert u.shape == shape
      value = u if u.shape == shape else v
      scale = init_scale or 1.1
      value *= scale
    else:
      raise AssertionError('unrecognized init scheme')
    return value

  def make_param_from_value(self, name, value):
    if name in self._params:
      param = self._params[name];
      if value.shape != param.get_value().shape:
        raise AssertionError('parameter {} re-use attempt with mis-matching shapes: '
          'existing shape {}, requested shape {}'.format(
            name, param.get_value().shape, value.shape))
      return param
    param = get_shared_floatX(value, name)
    self._params[name] = param
    return param

  def make_param(self, name, shape, init_scheme, init_scale=None):
    value = self.get_param_init(shape, init_scheme, init_scale)
    return self.make_param_from_value(name, value)

  def make_concat_param(self, name, shapes, init_schemes, axis):
    if len(shapes) != len(init_schemes):
      raise AssertionError('number of shapes and number of init schemes are incompatible')
    if len(set([shape[:axis] + shape[axis+1:] for shape in shapes])) != 1:
      raise AssertionError('all shapes should be identical on all axes except given axis')
    val = np.concatenate([self.get_param_init(shape, init_scheme)
      for shape, init_scheme in zip(shapes, init_schemes)], axis=axis)
    w = self.make_param_from_value(name, val)
    return w

  ################### I/O ###################

  def save(self, filename):
    logging.getLogger().info('Saving model weights to {}'.format(filename))
    verify_dir_exists(filename)
    param_dict = {name : param.get_value() for name, param in self._params.iteritems()}
    with open(filename, 'wb') as f:
      cPickle.dump(param_dict, f, protocol=cPickle.HIGHEST_PROTOCOL)

  def load_if_exists(self, filename, allow_mismatch=False):
    if not os.path.isfile(filename):
      return False
    logger = logging.getLogger()
    logger.info('Loading model weights found in {}'.format(filename))
    with open(filename, 'rb') as f:
      param_dict = cPickle.load(f)
      param_names, loaded_param_names = set(self._params.keys()), set(param_dict.keys())
      if param_names != loaded_param_names:
        msg = ('Parameter names loaded from {} do not match model\'s parameter names.\n'
          'param names only found in model: {}\n'
          'param names only found in loaded model: {}').format(
            filename, param_names.difference(loaded_param_names),
            loaded_param_names.difference(param_names))
        if allow_mismatch:
          logger.info(msg)
          param_dict = {param_name: param_dict[param_name] for param_name \
            in param_names.intersection(loaded_param_names)}
        else:
          raise AssertionError(msg)
      for name, value in param_dict.iteritems():
        self._params[name].set_value(value)
    return True

  ################### Dropout ###################

  def get_dropout_noise(self, shape, dropout_p):
    if dropout_p == 0:
      return 1
    keep_p = 1 - dropout_p
    return cast_floatX_np(1. / keep_p) * self._theano_rng.binomial(
      size=shape, p=keep_p, n=1, dtype=floatX)

  def apply_dropout_noise(self, x, noise):
    return ifelse(self._is_training, noise * x, x)

  def dropout(self, x, dropout_p):
    return self.apply_dropout_noise(x, self.get_dropout_noise(x.shape, dropout_p))

  ################### Misc ###################

  def get_param_sizes(self):
    param_sizes = {name: param.get_value().size for name, param in self._params.iteritems()}
    return sum(param_sizes.values()), param_sizes

  ################### Simple layers ###################

  def linear(self, name, x, input_dim, output_dim, with_bias=True, w_init='uniform', bias_init=0):
    # x                 (..., input_dim)
    n = namer(name)
    W = self.make_param(n('W'), (input_dim, output_dim), w_init)
    y = tt.dot(x, W)     # (..., output_dim)
    if with_bias:
      b = self.make_param(n('b'), (output_dim,), bias_init)
      y += b
    return y
        
  def ff(self, name, x, dims, activation, dropout_ps, **kwargs):
    assert len(dims) >= 2
    if dropout_ps:
      if isinstance(dropout_ps, numbers.Number):
        dropout_ps = [dropout_ps] * (len(dims) - 1)
      else:
        assert len(dropout_ps) == len(dims) - 1
    n = namer(name)
    h = x
    if activation == 'relu':
      f = tt.nnet.relu
    elif activation == 'sigmoid':
      f = tt.nnet.sigmoid
    elif activation == 'tanh':
      f = tt.tanh
    else:
      raise AssertionError('unrecognized activation function')
    for i, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
      if dropout_ps:
        h = self.dropout(h, dropout_ps[i])
      h = f(self.linear(n('l%d' % (i+1)), h, input_dim, output_dim, **kwargs))
    return h

  ################### LSTM ###################

  def stacked_bi_lstm(self, name,
    x, x_mask, num_layers, input_dim, hidden_dim, drop_x, drop_h, **kwargs):
    n = namer(name)
    h = x
    for l in range(1, num_layers+1):
      h = self.bi_lstm(n('l%d' % l),
        h, x_mask, input_dim if l == 1 else 2*hidden_dim, hidden_dim, drop_x, drop_h, **kwargs)
    return h    # (timesteps, batch_size, 2*hidden_dim)

  def bi_lstm(self, name, x, x_mask, input_dim, hidden_dim, drop_x, drop_h, **kwargs):
    n = namer(name)
    fwd_h = self.lstm(n('fwd'),
      x, x_mask, input_dim, hidden_dim, drop_x, drop_h, backward=False, **kwargs)
    bck_h = self.lstm(n('bck'),
      x, x_mask, input_dim, hidden_dim, drop_x, drop_h, backward=True, **kwargs)
    bi_h = tt.concatenate([fwd_h, bck_h], axis=2)     # (timesteps, batch_size, 2*hidden_dim)
    return bi_h

  def lstm(self, name,
    x, x_mask,
    input_dim, hidden_dim,
    drop_x, drop_h,
    backward=False, couple_i_and_f=False, learn_initial_state=False,
    tie_x_dropout=True, sep_x_dropout=False,
    sep_h_dropout=False,
    w_init='uniform', u_init='orthogonal', forget_bias_init=1, other_bias_init=0):
    """Customizable uni-directional LSTM layer.
    Handles masks, can learn initial state, input and forget gate can be coupled,
    with recurrent dropout, no peephole connections.
    Args:
      x:                    Theano tensor, shape (timesteps, batch_size, input_dim)
      x_mask:               Theano tensor, shape (timesteps, batch_size)
      input_dim:            int, dimension of input vectors
      hidden_dim:           int, dimension of hidden state
      drop_x:               float, dropout rate to apply to inputs
      drop_h:               float, dropout rate to apply to hidden state
      backward:             boolean, whether to recur over timesteps in reveresed order
      couple_i_and_f:       boolean, whether to have input gate = 1 - forget gate
      learn_initial_state:  boolean, whether to have initial cell state and initial hidden state
                            as learnt parameters
      tie_x_dropout:        boolean, whether to have the same dropout masks across timesteps
                            for input
      sep_x_dropout:        boolean, if True dropout is applied over weights of lin. trans. of
                            input; otherwise it is applied over input activations
      sep_h_dropout:        boolean, if True dropout is applied over weights of lin. trans. of
                            hidden state; otherwise it is applied over hidden state activations
      w_init:               string, initialization scheme for weights of lin. trans. of input
      u_init:               string, initialization scheme for weights of lin. trans. of hidden state
      forget_bias_init:     string, initialization scheme for forget gate's bias
      other_bias_init:      string, initialization scheme for other biases
    Note:
      Proper variational dropout (Gal 2015) is:
        tie_x_dropout=True, sep_x_dropout=True, sep_h_dropout=True
      A faster alternative is:
        tie_x_dropout=True, sep_x_dropout=False, sep_h_dropout=False
    Returns:
      h:                    Theano variable, recurrent hidden states at each timestep,
                            shape (timesteps, batch_size, hidden_dim)
    """
    n = namer(name)
    timesteps, batch_size = x.shape[0], x.shape[1]

    num_non_lin = 3 if couple_i_and_f else 4
    num_gates = num_non_lin - 1

    W = self.make_concat_param(n('W'),            # (input_dim, [3|4]*hidden_dim)
      num_non_lin*[(input_dim, hidden_dim)], num_non_lin*[w_init], axis=1)
    b = self.make_concat_param(n('b'),            # ([3|4]*hidden_dim,)
      num_non_lin*[(hidden_dim,)], [forget_bias_init] + num_gates*[other_bias_init], axis=0)
    U = self.make_concat_param(n('U'),            # (hidden_dim, [3|4]*hidden_dim)
      num_non_lin*[(hidden_dim, hidden_dim)], num_non_lin*[u_init], axis=1) 

    if not sep_x_dropout:
      if tie_x_dropout:
        x = self.apply_dropout_noise(x, self.get_dropout_noise((batch_size, input_dim), drop_x))
      else:
        x = self.dropout(x, drop_x)
      lin_x = tt.dot(x, W) + b                    # (timesteps, batch_size, [3|4]*hidden_dim)
    else:
      if tie_x_dropout:
        x_for_f = self.apply_dropout_noise(
          x, self.get_dropout_noise((batch_size, input_dim), drop_x))
        x_for_o = self.apply_dropout_noise(
          x, self.get_dropout_noise((batch_size, input_dim), drop_x))
        if num_gates == 3:
          x_for_i = self.apply_dropout_noise(
            x, self.get_dropout_noise((batch_size, input_dim), drop_x))
        x_for_g = self.apply_dropout_noise(
          x, self.get_dropout_noise((batch_size, input_dim), drop_x))
      else:
        x_for_f = self.dropout(x, drop_x)
        x_for_o = self.dropout(x, drop_x)
        if num_gates == 3:
          x_for_i = self.dropout(x, drop_x)
        x_for_g = self.dropout(x, drop_x)
      lin_x_tensors = [tt.dot(x_for_f, W[:,:hidden_dim]),
        tt.dot(x_for_o, W[:,hidden_dim:2*hidden_dim])]
      if num_gates == 3:
        lin_x_tensors.append(tt.dot(x_for_i, W[:,2*hidden_dim:3*hidden_dim]))
      lin_x_tensors.append(tt.dot(x_for_g, W[:,num_gates*hidden_dim:]))
      lin_x = tt.concatenate(lin_x_tensors, axis=2) + b # (timesteps, batch_size, [3|4]*hidden_dim)

    def step_fn(lin_x_t, x_mask_t, h_tm1, c_tm1, h_noise, U):
      # lin_x_t       (batch_size, [3|4]*hidden_dim)
      # x_mask_t      (batch_size, 1)
      # h_tm1         (batch_size, hidden_dim)
      # c_tm1         (batch_size, hidden_dim)
      # h_noise       (batch_size, [1|3|4]*hidden_dim)
      #               1 if not sep_h_dropout, otherwise: 3 or 4 depending on num_non_lin
      # U             (hidden_dim, [3|4]*hidden_dim)

      if not sep_h_dropout:
        h_tm1 = self.apply_dropout_noise(h_tm1, h_noise)
        lin_h_tm1 = tt.dot(h_tm1, U)                    # (batch_size, [3|4]*hidden_dim)
      else:
        h_tm1_for_f = self.apply_dropout_noise(h_tm1, h_noise[:,:hidden_dim])
        h_tm1_for_o = self.apply_dropout_noise(h_tm1, h_noise[:,hidden_dim:2*hidden_dim])
        if num_gates == 3:
          h_tm1_for_i = self.apply_dropout_noise(h_tm1, h_noise[:,2*hidden_dim:3*hidden_dim])
        h_tm1_for_g = self.apply_dropout_noise(h_tm1, h_noise[:,num_gates*hidden_dim:])
        lin_h_tm1_tensors = [tt.dot(h_tm1_for_f, U[:,:hidden_dim]),
          tt.dot(h_tm1_for_o, U[:,hidden_dim:2*hidden_dim])]
        if num_gates == 3:
          lin_h_tm1_tensors.append(tt.dot(h_tm1_for_i, U[:,2*hidden_dim:3*hidden_dim]))
        lin_h_tm1_tensors.append(tt.dot(h_tm1_for_g, U[:,num_gates*hidden_dim:]))
        lin_h_tm1 = tt.concatenate(lin_h_tm1_tensors, axis=1)             # (batch_size, [3|4]*hidden_dim)

      lin = lin_x_t + lin_h_tm1                                           # (batch_size, [3|4]*hidden_dim)

      gates = tt.nnet.sigmoid(lin[:, :num_gates*hidden_dim])              # (batch_size, [3|4]*hidden_dim)
      f_gate = gates[:, :hidden_dim]                                      # (batch_size, hidden_dim)
      o_gate = gates[:, hidden_dim:2*hidden_dim]                          # (batch_size, hidden_dim)
      i_gate = gates[:, 2*hidden_dim:] if num_gates == 3 else 1 - f_gate  # (batch_size, hidden_dim)
      g = tt.tanh(lin[:, num_gates*hidden_dim:])                          # (batch_size, hidden_dim)

      c_t = f_gate * c_tm1 + i_gate * g
      h_t = o_gate * tt.tanh(c_t)

      h_t = tt.switch(x_mask_t, h_t, h_tm1)
      c_t = tt.switch(x_mask_t, c_t, c_tm1)

      return h_t, c_t
      # end of step_fn

    if learn_initial_state:
      h0 = self.make_param(n('h0'), (hidden_dim,), 0)
      c0 = self.make_param(n('c0'), (hidden_dim,), 0)
      batch_h0 = tt.extra_ops.repeat(h0[None,:], batch_size, axis=0)
      batch_c0 = tt.extra_ops.repeat(c0[None,:], batch_size, axis=0)
    else:
      batch_h0 = batch_c0 = tt.zeros((batch_size, hidden_dim))

    x_mask = tt.shape_padright(x_mask)    # (timesteps, batch_size, 1)

    original_x_mask = x_mask
    if backward:
      lin_x = lin_x[::-1]
      x_mask = x_mask[::-1]

    h_noise = self.get_dropout_noise(
      (batch_size, hidden_dim if not sep_h_dropout else num_non_lin*hidden_dim), drop_h)

    results, _ = theano.scan(step_fn,
        sequences = [lin_x, x_mask],
        outputs_info = [batch_h0, batch_c0],
        non_sequences = [h_noise, U],
        name = n('scan'))

    h = results[0]    # (timesteps, batch_size, hidden_dim)
    if backward:
      h = h[::-1]
    h *= original_x_mask
    return h

