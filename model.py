
import logging

import numpy as np
import theano
import theano.tensor as tt

from theano.ifelse import ifelse
from theano.compile.nanguardmode import NanGuardMode

from base.utils import namer
from base.theano_utils import (cast_floatX, get_shared_floatX, gpu_int32,
  softmax_columns_with_mask, softmax_depths_with_mask)
from base.model import BaseModel
from base.optimizer import AdamOptimizer


def get_model(config, data):
  logger = logging.getLogger()
  logger.info('Building model...')
  model = Model(config, data)
  total_size, params_sizes = model.get_param_sizes()
  sorted_param_sizes = sorted(params_sizes.items())
  logger.info('Done building model. Total number of parameters: {}. Sizes:\n{}\n'.format(total_size,
    '\n'.join('\t{:<8d} {:s}'.format(p_size, p_name) for p_name, p_size in sorted_param_sizes)))
  return model


class Model(BaseModel):
  def __init__(self, config, data):
    self.init_start(config)
    # cuda optimized batched dot product
    batched_dot = tt.batched_dot if config.device == 'cpu' else theano.sandbox.cuda.blas.batched_dot

    ###################################################
    # Load all data onto GPU
    ###################################################

    emb_val = data.word_emb_data.word_emb                                               # (voc size, emb_dim)
    first_known_word = data.word_emb_data.first_known_word
    assert config.emb_dim == emb_val.shape[1]
    assert first_known_word > 0
    emb_val[:first_known_word] = 0 
    if config.learn_single_unk:
      first_unknown_word = data.word_emb_data.first_unknown_word
      known_emb = get_shared_floatX(emb_val[:first_unknown_word], 'known_emb')          # (num known words, emb_dim)
      single_unk_emb = self.make_param('single_unk_emb', (config.emb_dim,), 'uniform')  # (emb_dim,)
      emb = tt.concatenate([known_emb, tt.shape_padleft(single_unk_emb)], axis=0)       # (num known words + 1, emb_dim)
    else:
      emb = get_shared_floatX(emb_val, 'emb')                                           # (voc size, emb_dim)

    if config.is_train:
      trn_ds_vec = data.trn.vectorized
      trn_ctxs, trn_ctx_masks, trn_ctx_lens = _gpu_sequences('trn_ctxs', trn_ds_vec.ctxs, trn_ds_vec.ctx_lens)
      trn_qtns, trn_qtn_masks, trn_qtn_lens = _gpu_sequences('trn_qtns', trn_ds_vec.qtns, trn_ds_vec.qtn_lens)
      trn_anss = _gpu_trn_answers('trn_anss', trn_ds_vec.anss, trn_ds_vec.num_anss, config.max_ans_len)

      dev_ds_vec = data.dev.vectorized
      dev_ctxs, dev_ctx_masks, dev_ctx_lens = _gpu_sequences('dev_ctxs', dev_ds_vec.ctxs, dev_ds_vec.ctx_lens)
      dev_qtns, dev_qtn_masks, dev_qtn_lens = _gpu_sequences('dev_qtns', dev_ds_vec.qtns, dev_ds_vec.qtn_lens)
      dev_anss, dev_ans_masks = _gpu_dev_answers('dev_anss', dev_ds_vec.anss, dev_ds_vec.num_anss, config.max_ans_len)
    else:
      tst_ds_vec = data.tst.vectorized
      tst_ctxs, tst_ctx_masks, tst_ctx_lens = _gpu_sequences('tst_ctxs', tst_ds_vec.ctxs, tst_ds_vec.ctx_lens)
      tst_qtns, tst_qtn_masks, tst_qtn_lens = _gpu_sequences('tst_qtns', tst_ds_vec.qtns, tst_ds_vec.qtn_lens)

    ###################################################
    # Map input given to interface functions to an actual mini batch
    ###################################################

    in_sample_idxs = tt.ivector('in_sample_idxs')           # (batch_Size,)
    in_ctxs = tt.imatrix('in_ctxs')                         # (num samples in dataset, max_p_len of dataset)
    in_ctx_masks = tt.imatrix('in_ctx_masks')               # (num samples in dataset, max_p_len of dataset)
    in_ctx_lens = tt.ivector('in_ctx_lens')                 # (num samples in dataset,)
    in_qtns = tt.imatrix('in_qtns')                         # (num samples in dataset, max_q_len of dataset)
    in_qtn_masks = tt.imatrix('in_qtn_masks')               # (num samples in dataset, max_q_len of dataset)
    in_qtn_lens = tt.ivector('in_qtn_lens')                 # (num samples in dataset,)
    batch_size = in_sample_idxs.size

    p_lens = in_ctx_lens[in_sample_idxs]                    # (batch_size,)
    max_p_len = p_lens.max()
    p = in_ctxs[in_sample_idxs][:,:max_p_len].T             # (max_p_len, batch_size)
    p_mask = in_ctx_masks[in_sample_idxs][:,:max_p_len].T   # (max_p_len, batch_size)
    float_p_mask = cast_floatX(p_mask)

    q_lens = in_qtn_lens[in_sample_idxs]                    # (batch_size,)
    max_q_len = q_lens.max()
    q = in_qtns[in_sample_idxs][:,:max_q_len].T             # (max_q_len, batch_size)
    q_mask = in_qtn_masks[in_sample_idxs][:,:max_q_len].T   # (max_q_len, batch_size)
    float_q_mask = cast_floatX(q_mask)

    if config.is_train:
      in_trn_anss = tt.ivector('in_trn_anss')               # (num samples in train dataset,)
      in_dev_anss = tt.imatrix('in_dev_anss')               # (num samples in test dataset, max_num_ans of test dataset)
      in_dev_ans_masks = tt.imatrix('in_dev_ans_masks')     # (num samples in test dataset, max_num_ans of test dataset)

      trn_a = in_trn_anss[in_sample_idxs]                   # (batch_size,)
      dev_a = in_dev_anss[in_sample_idxs]                   # (batch_size, max_num_ans)
      dev_a_mask = in_dev_ans_masks[in_sample_idxs]         # (batch_size, max_num_ans)

    ###################################################
    # RaSoR
    ###################################################

    ############ embed words

    p_emb = emb[p]        # (max_p_len, batch_size, emb_dim)
    q_emb = emb[q]        # (max_q_len, batch_size, emb_dim)

    ############ q indep

    # (max_q_len, batch_size, 2*hidden_dim)
    q_indep_h = self.stacked_bi_lstm('q_indep_lstm', q_emb, float_q_mask,
      config.num_bilstm_layers, config.emb_dim, config.hidden_dim,
      config.lstm_drop_x, config.lstm_drop_h,
      couple_i_and_f = config.lstm_couple_i_and_f,
      learn_initial_state = config.lstm_learn_initial_state,
      tie_x_dropout = config.lstm_tie_x_dropout,
      sep_x_dropout = config.lstm_sep_x_dropout,
      sep_h_dropout = config.lstm_sep_h_dropout,
      w_init = config.lstm_w_init,
      u_init = config.lstm_u_init,
      forget_bias_init = config.lstm_forget_bias_init,
      other_bias_init = config.default_bias_init)

    ff_dim = config.ff_dims[-1]
    # (max_q_len, batch_size, ff_dim)     # contains junk where masked
    q_indep_ff = self.ff('q_indep_ff', q_indep_h, [2*config.hidden_dim] + config.ff_dims,
      'relu', config.ff_drop_x, bias_init=config.default_bias_init)
    if config.extra_drop_x:
      q_indep_ff = self.dropout(q_indep_ff, config.extra_drop_x)
    w_q = self.make_param('w_q', (ff_dim,), 'uniform')
    q_indep_scores = tt.dot(q_indep_ff, w_q)                                  # (max_q_len, batch_size)
    q_indep_weights = softmax_columns_with_mask(q_indep_scores, float_q_mask) # (max_q_len, batch_size)
    q_indep = tt.sum(tt.shape_padright(q_indep_weights) * q_indep_h, axis=0)  # (batch_size, 2*hidden_dim)
    
    ############ q aligned

    if config.q_aln_ff_tie:
      q_align_ff_p_name = q_align_ff_q_name = 'q_align_ff'
    else:
      q_align_ff_p_name = 'q_align_ff_p'
      q_align_ff_q_name = 'q_align_ff_q'
    # (max_p_len, batch_size, ff_dim)     # contains junk where masked
    q_align_ff_p = self.ff(q_align_ff_p_name, p_emb, [config.emb_dim] + config.ff_dims,
      'relu', config.ff_drop_x, bias_init=config.default_bias_init)
    # (max_q_len, batch_size, ff_dim)     # contains junk where masked
    q_align_ff_q = self.ff(q_align_ff_q_name, q_emb, [config.emb_dim] + config.ff_dims,
      'relu', config.ff_drop_x, bias_init=config.default_bias_init)

    # http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.batched_dot
    # https://groups.google.com/d/msg/theano-users/yBh27AJGq2E/vweiLoXADQAJ
    q_align_ff_p_shuffled = q_align_ff_p.dimshuffle((1,0,2))                    # (batch_size, max_p_len, ff_dim)
    q_align_ff_q_shuffled = q_align_ff_q.dimshuffle((1,2,0))                    # (batch_size, ff_dim, max_q_len)
    q_align_scores = batched_dot(q_align_ff_p_shuffled, q_align_ff_q_shuffled)  # (batch_size, max_p_len, max_q_len)

    p_mask_shuffled = float_p_mask.dimshuffle((1,0,'x'))      # (batch_size, max_p_len, 1)
    q_mask_shuffled = float_q_mask.dimshuffle((1,'x',0))      # (batch_size, 1, max_q_len)
    pq_mask = p_mask_shuffled * q_mask_shuffled               # (batch_size, max_p_len, max_q_len)

    q_align_weights = softmax_depths_with_mask(q_align_scores, pq_mask)         # (batch_size, max_p_len, max_q_len)
    q_emb_shuffled = q_emb.dimshuffle((1,0,2))                                  # (batch_size, max_q_len, emb_dim)
    q_align = batched_dot(q_align_weights, q_emb_shuffled)                      # (batch_size, max_p_len, emb_dim)
    
    ############ p star

    q_align_shuffled = q_align.dimshuffle((1,0,2))            # (max_p_len, batch_size, emb_dim)
    q_indep_repeated = tt.extra_ops.repeat(                   # (max_p_len, batch_size, 2*hidden_dim)
      tt.shape_padleft(q_indep), max_p_len, axis=0)
    p_star = tt.concatenate(                                  # (max_p_len, batch_size, 2*emb_dim + 2*hidden_dim)
      [p_emb, q_align_shuffled, q_indep_repeated], axis=2)

    ############ passage-level bi-lstm

    # (max_p_len, batch_size, 2*hidden_dim)
    p_level_h = self.stacked_bi_lstm('p_level_lstm', p_star, float_p_mask,
      config.num_bilstm_layers, 2*config.emb_dim+2*config.hidden_dim, config.hidden_dim,
      config.lstm_drop_x, config.lstm_drop_h,
      couple_i_and_f = config.lstm_couple_i_and_f,
      learn_initial_state = config.lstm_learn_initial_state,
      tie_x_dropout = config.lstm_tie_x_dropout,
      sep_x_dropout = config.lstm_sep_x_dropout,
      sep_h_dropout = config.lstm_sep_h_dropout,
      w_init = config.lstm_w_init,
      u_init = config.lstm_u_init,
      forget_bias_init = config.lstm_forget_bias_init,
      other_bias_init = config.default_bias_init)

    ############ span scores

    if config.sep_stt_end_drop:
      p_level_h_for_stt = self.dropout(p_level_h, config.ff_drop_x)
      p_level_h_for_end = self.dropout(p_level_h, config.ff_drop_x)
    else:
      p_level_h_for_stt = p_level_h_for_end = self.dropout(p_level_h, config.ff_drop_x)

    # Having a single FF hidden layer allows to compute the FF over the concatenation
    # of span-start-hidden-state and span-end-hidden-state by operating the linear transformation
    # separately over each (more efficient).
    assert len(config.ff_dims) == 1
    # (max_p_len, batch_size, ff_dim)
    p_stt = self.linear('p_stt', p_level_h_for_stt, 2*config.hidden_dim, ff_dim, bias_init=config.default_bias_init)
    # (max_p_len, batch_size, ff_dim)
    p_end = self.linear('p_end', p_level_h_for_end, 2*config.hidden_dim, ff_dim, with_bias=False)

    p_end_zero_padded = tt.concatenate(                                 # (max_p_len+max_ans_len-1, batch_size, ff_dim)
      [p_end, tt.zeros((config.max_ans_len-1, batch_size, ff_dim))], axis=0)    
    p_max_ans_len_range = tt.shape_padleft(                             # (1, max_ans_len)
      tt.arange(config.max_ans_len))
    p_offsets = tt.shape_padright(tt.arange(max_p_len))                 # (max_p_len, 1)
    p_end_idxs = p_max_ans_len_range + p_offsets                        # (max_p_len, max_ans_len)
    p_end_idxs_flat = p_end_idxs.flatten()                              # (max_p_len*max_ans_len,)

    p_ends = p_end_zero_padded[p_end_idxs_flat]                         # (max_p_len*max_ans_len, batch_size, ff_dim)
    p_ends = p_ends.reshape(                                            # (max_p_len, max_ans_len, batch_size, ff_dim)
      (max_p_len, config.max_ans_len, batch_size, ff_dim))

    p_stt_shuffled = p_stt.dimshuffle((0,'x',1,2))                      # (max_p_len, 1, batch_size, ff_dim)

    p_stt_end_lin = p_stt_shuffled + p_ends                             # (max_p_len, max_ans_len, batch_size, ff_dim)
    p_stt_end = tt.nnet.relu(p_stt_end_lin)                             # (max_p_len, max_ans_len, batch_size, ff_dim)

    w_a = self.make_param('w_a', (ff_dim,), 'uniform')
    span_scores = tt.dot(p_stt_end, w_a)                                # (max_p_len, max_ans_len, batch_size)

    ############ span masks

    p_lens_shuffled = p_lens.dimshuffle('x','x',0)                      # (1, 1, batch_size)
    p_end_idxs_shuffled = p_end_idxs.dimshuffle(0,1,'x')                # (max_p_len, max_ans_len, 1)
    span_masks = tt.lt(p_end_idxs_shuffled, p_lens_shuffled)            # (max_p_len, max_ans_len, batch_size)

    span_scores_reshaped = span_scores.dimshuffle((2,0,1)).reshape(     # (batch_size, max_p_len*max_ans_len)
      (batch_size, -1))
    span_masks_reshaped = span_masks.dimshuffle((2,0,1)).reshape(       # (batch_size, max_p_len*max_ans_len)
      (batch_size, -1))
    span_masks_reshaped = cast_floatX(span_masks_reshaped)

    if config.is_train:

      ############ loss

      # (batch_size,), (batch_size), (batch_size,)
      trn_xents, trn_accs, trn_a_hat = _single_answer_classification(
        span_scores_reshaped, span_masks_reshaped, trn_a)
      trn_loss, trn_acc = trn_xents.mean(), trn_accs.mean()

      # (batch_size,), (batch_size), (batch_size,), (batch_size), (batch_size,)
      dev_min_xents, dev_prx_xents, dev_max_accs, dev_prx_accs, dev_a_hat = _multi_answer_classification(
        span_scores_reshaped, span_masks_reshaped, dev_a, dev_a_mask)
      dev_min_loss, dev_prx_loss, dev_max_acc, dev_prx_acc = \
        dev_min_xents.mean(), dev_prx_xents.mean(), dev_max_accs.mean(), dev_prx_accs.mean()
      dev_ans_hat_start_word_idxs, dev_ans_hat_end_word_idxs = \
        _tt_ans_idx_to_ans_word_idxs(dev_a_hat, config.max_ans_len)

      ############ optimization

      opt = AdamOptimizer(config, trn_loss, self._params.values())
      updates = opt.get_updates()
      trn_global_grad_norm = opt.get_global_grad_norm()
      self.get_lr_value = lambda : opt.get_lr_value()

      ############ interface

      train_givens = {
          self._is_training : np.int32(1), 
          in_ctxs: trn_ctxs,
          in_ctx_masks: trn_ctx_masks,
          in_ctx_lens: trn_ctx_lens,
          in_qtns: trn_qtns,
          in_qtn_masks: trn_qtn_masks,
          in_qtn_lens: trn_qtn_lens,
          in_trn_anss: trn_anss}
      dev_givens = {
          self._is_training : np.int32(0), 
          in_ctxs: dev_ctxs,
          in_ctx_masks: dev_ctx_masks,
          in_ctx_lens: dev_ctx_lens,
          in_qtns: dev_qtns,
          in_qtn_masks: dev_qtn_masks,
          in_qtn_lens: dev_qtn_lens,
          in_dev_anss: dev_anss,
          in_dev_ans_masks: dev_ans_masks}

      self.train = theano.function(
        [in_sample_idxs],
        [trn_loss, trn_acc, trn_global_grad_norm],
        givens = train_givens,
        updates = updates)
        #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))

      self.eval_dev = theano.function(
        [in_sample_idxs],
        [dev_min_loss, dev_prx_loss, dev_max_acc, dev_prx_acc, dev_ans_hat_start_word_idxs, dev_ans_hat_end_word_idxs],
        givens = dev_givens,
        updates = None)

    else:   # config.is_train = False

      tst_a_hat = _no_answer_classification(span_scores_reshaped, span_masks_reshaped)    # (batch_size,)
      tst_ans_hat_start_word_idxs, tst_ans_hat_end_word_idxs = _tt_ans_idx_to_ans_word_idxs(
        tst_a_hat, config.max_ans_len)

      tst_givens = {
          self._is_training : np.int32(0), 
          in_ctxs: tst_ctxs,
          in_ctx_masks: tst_ctx_masks,
          in_ctx_lens: tst_ctx_lens,
          in_qtns: tst_qtns,
          in_qtn_masks: tst_qtn_masks,
          in_qtn_lens: tst_qtn_lens}

      self.eval_tst = theano.function(
        [in_sample_idxs],
        [tst_ans_hat_start_word_idxs, tst_ans_hat_end_word_idxs],
        givens = tst_givens,
        updates = None)

    # __init__ end
# Model end


###################################################
# Variable-length data to GPU matrices and masks
###################################################

def _gpu_sequences(name, seqs_val, lens):
  assert seqs_val.dtype == lens.dtype == np.int32
  num_samples, max_seq_len = seqs_val.shape
  assert len(lens) == num_samples
  assert max(lens) == max_seq_len

  gpu_seqs = gpu_int32(name, seqs_val)
  seq_masks_val = np.zeros((num_samples, max_seq_len), dtype=np.int32)
  for i, sample_len in enumerate(lens):
    seq_masks_val[i,:sample_len] = 1
    assert np.all(seqs_val[i,:sample_len] > 0)
    assert np.all(seqs_val[i,sample_len:] == 0)
  gpu_seq_masks = gpu_int32(name + '_masks', seq_masks_val)
  gpu_lens = gpu_int32(name + '_lens', lens)
  return gpu_seqs, gpu_seq_masks, gpu_lens


def _np_ans_word_idxs_to_ans_idx(ans_start_word_idx, ans_end_word_idx, max_ans_len):
  # all arguments are concrete ints
  assert ans_end_word_idx - ans_start_word_idx + 1 <= max_ans_len
  return ans_start_word_idx * max_ans_len + (ans_end_word_idx - ans_start_word_idx)

def _tt_ans_idx_to_ans_word_idxs(ans_idx, max_ans_len):
  # ans_idx theano int32 variable (batch_size,)
  # max_ans_len concrete int
  ans_start_word_idx = ans_idx // max_ans_len
  ans_end_word_idx = ans_start_word_idx + ans_idx % max_ans_len
  return ans_start_word_idx, ans_end_word_idx


def _gpu_trn_answers(name, anss, num_anss, max_ans_len):
  assert anss.dtype == num_anss.dtype == np.int32
  num_samples, max_num_ans, _ = anss.shape
  assert len(num_anss) == num_samples
  assert max_num_ans == 1 and all(num_ans == 1 for num_ans in num_anss)

  anss_val = np.zeros(num_samples, dtype=np.int32)
  for i in range(num_samples):
    ans_start_word_idx = anss[i,0,0]
    ans_end_word_idx = anss[i,0,1]
    anss_val[i] = _np_ans_word_idxs_to_ans_idx(ans_start_word_idx, ans_end_word_idx, max_ans_len)
  gpu_anss = gpu_int32(name, anss_val)
  return gpu_anss


def _gpu_dev_answers(name, anss, num_anss, max_ans_len):
  assert anss.dtype == num_anss.dtype == np.int32
  num_samples, max_num_ans, _ = anss.shape
  assert len(num_anss) == num_samples

  anss_val = np.zeros((num_samples, max_num_ans), dtype=np.int32)
  ans_masks_val = np.zeros((num_samples, max_num_ans), dtype=np.int32)
  for i in range(num_samples):
    for j in range(num_anss[i]):
      ans_start_word_idx = anss[i,j,0]
      ans_end_word_idx = anss[i,j,1]
      anss_val[i,j] = _np_ans_word_idxs_to_ans_idx(ans_start_word_idx, ans_end_word_idx, max_ans_len)
      ans_masks_val[i,j] = 1
  gpu_anss = gpu_int32(name, anss_val)
  gpu_ans_masks = gpu_int32(name + '_masks', ans_masks_val)
  return gpu_anss, gpu_ans_masks


###################################################
# Classification
###################################################

def _single_answer_classification(x, x_mask, y):
  # x       float32 (num_samples, num_classes)    scores i.e. logits
  # x_mask  int32   (num_samples, num_classes)    score masks (each sample has a variable number of classes)
  # y       int32   (num_samples,)                target classes i.e. ground truth answers (given as class indices)
  assert x.ndim == x_mask.ndim == 2
  assert y.ndim == 1

  x *= x_mask
  # substracting min needed since all non masked-out elements of a row may be negative.
  x -= x.min(axis=1, keepdims=True)             # (num_samples, num_classes)
  x *= x_mask                                   # (num_samples, num_classes)
  y_hat = tt.argmax(x, axis=1)                  # (num_samples,)
  accs = cast_floatX(tt.eq(y_hat, y))           # (num_samples,)

  x -= x.max(axis=1, keepdims=True)             # (num_samples, num_classes)
  x *= x_mask                                   # (num_samples, num_classes)
  exp_x = tt.exp(x)                             # (num_samples, num_classes)
  exp_x *= x_mask                               # (num_samples, num_classes)

  sum_exp_x = exp_x.sum(axis=1)                 # (num_samples,)
  log_sum_exp_x = tt.log(sum_exp_x)             # (num_samples,)

  x_star = x[tt.arange(x.shape[0]), y]          # (num_samples,)
  xents = log_sum_exp_x - x_star                # (num_samples,)

  return xents, accs, y_hat


def _multi_answer_classification(x, x_mask, y, y_mask):
  # x       float32 (num_samples, num_classes)    scores i.e. logits
  # x_mask  int32   (num_samples, num_classes)    score masks (each sample has a variable number of classes)
  # y       int32   (num_samples, num_answers)    target classes i.e. ground truth answers (given as class indices)
  # y_mask  int32   (num_samples, num_answers)    target classes masks (each sample has a variable number of answers)
  assert x.ndim == x_mask.ndim == y.ndim == y_mask.ndim == 2
  num_samples = x.shape[0]
  num_classes = x.shape[1]
  num_answers = y.shape[1]

  x *= x_mask
  y *= y_mask
  # substracting min needed since all non masked-out elements of a row may be negative.
  x -= x.min(axis=1, keepdims=True)                     # (num_samples, num_classes)
  x *= x_mask                                           # (num_samples, num_classes)
  y_hat = tt.argmax(x, axis=1)                          # (num_samples,)
  accs = y_mask * tt.eq(tt.shape_padright(y_hat), y)    # (num_samples, num_answers)
  # did we correctly predict any of the ground truth answers?
  max_accs = cast_floatX(tt.max(accs, axis=1))          # (num_samples,)
  # proxy for comparing against train set: did we correctly predict first answer?
  prx_accs = accs[:, 0]                                 # (num_samples,)

  x -= x.max(axis=1, keepdims=True)                     # (num_samples, num_classes)
  x *= x_mask                                           # (num_samples, num_classes)
  exp_x = tt.exp(x)                                     # (num_samples, num_classes)
  exp_x *= x_mask                                       # (num_samples, num_classes)

  sum_exp_x = exp_x.sum(axis=1, keepdims=True)          # (num_samples, 1)
  log_sum_exp_x = tt.log(sum_exp_x)                     # (num_samples, 1)

  x_flat = x.flatten()                                  # (num_samples * num_classes,)

  shifts = tt.shape_padright(                           # (num_samples, 1)
    tt.arange(0, num_samples*num_classes, num_classes))
  y_shifted = y + shifts                                # (num_samples, num_answers)
  y_shifted_flat = y_shifted.flatten()                  # (num_samples * num_answers,)
  x_stars = x_flat[y_shifted_flat]                      # (num_samples * num_answers,)
  x_stars = x_stars.reshape((num_samples, num_answers))

  xents = log_sum_exp_x - x_stars                       # (num_samples, num_answers)
  xents *= cast_floatX(y_mask)                          # (num_samples, num_answers)

  # place max xent in masked out places, so we can find the min xent (otherwise we'll find the zeros as mins).
  xents_max = tt.max(xents, axis=1, keepdims=True)      # (num_samples, 1)
  min_xents = tt.min(                                   # (num_samples,)  # min cross-entropy over all answers
    xents + (1-cast_floatX(y_mask))*xents_max, axis=1)
  # proxy for comparing against train set: cross-entropy with first answer
  prx_xents = xents[:, 0]                               # (num_samples,)

  return min_xents, prx_xents, max_accs, prx_accs, y_hat


def _no_answer_classification(x, x_mask):
  # x       float32 (num_samples, num_classes)    scores i.e. logits
  # x_mask  int32   (num_samples, num_classes)    score masks (each sample has a variable number of classes)
  assert x.ndim == x_mask.ndim == 2
  x *= x_mask
  # substracting min needed since all non masked-out elements of a row may be negative.
  x -= x.min(axis=1, keepdims=True)             # (num_samples, num_classes)
  x *= x_mask                                   # (num_samples, num_classes)
  y_hat = tt.argmax(x, axis=1)                  # (num_samples,)
  return y_hat

