import logging

import numpy as np
import theano
import theano.tensor as tt

from theano.ifelse import ifelse
from theano.compile.nanguardmode import NanGuardMode

from base.utils import namer
from base.theano_utils import (cast_floatX, get_shared_floatX, gpu_int32,
  softmax_columns_with_mask, softmax_depths_with_mask, argmax_with_mask)
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

    trn_ctxs, trn_ctx_masks, trn_ctx_lens, trn_qtns, trn_qtn_masks, trn_qtn_lens, trn_qtn_ctx_idxs, \
      trn_anss, trn_ans_stts, trn_ans_ends = _gpu_dataset('trn', data.trn, config)

    dev_ctxs, dev_ctx_masks, dev_ctx_lens, dev_qtns, dev_qtn_masks, dev_qtn_lens, dev_qtn_ctx_idxs, \
      dev_anss, dev_ans_stts, dev_ans_ends = _gpu_dataset('dev', data.dev, config)

    tst_ctxs, tst_ctx_masks, tst_ctx_lens, tst_qtns, tst_qtn_masks, tst_qtn_lens, tst_qtn_ctx_idxs, \
      tst_anss, tst_ans_stts, tst_ans_ends = _gpu_dataset('tst', data.tst, config)

    ###################################################
    # Map input given to interface functions to an actual mini batch
    ###################################################

    qtn_idxs = tt.ivector('qtn_idxs')                           # (batch_bize,)
    batch_size = qtn_idxs.size

    dataset_ctxs = tt.imatrix('dataset_ctxs')                   # (num contexts in dataset, max_p_len of dataset)
    dataset_ctx_masks = tt.imatrix('dataset_ctx_masks')         # (num contexts in dataset, max_p_len of dataset)
    dataset_ctx_lens = tt.ivector('dataset_ctx_lens')           # (num contexts in dataset,)
    dataset_qtns = tt.imatrix('dataset_qtns')                   # (num questions in dataset, max_q_len of dataset)
    dataset_qtn_masks = tt.imatrix('dataset_qtn_masks')         # (num questions in dataset, max_q_len of dataset)
    dataset_qtn_lens = tt.ivector('dataset_qtn_lens')           # (num questions in dataset,)
    dataset_qtn_ctx_idxs = tt.ivector('dataset_qtn_ctx_idxs')   # (num questions in dataset,)
    dataset_anss = tt.ivector('dataset_anss')                   # (num questions in dataset,)
    dataset_ans_stts = tt.ivector('dataset_ans_stts')           # (num questions in dataset,)
    dataset_ans_ends = tt.ivector('dataset_ans_ends')           # (num questions in dataset,)

    ctx_idxs = dataset_qtn_ctx_idxs[qtn_idxs]                   # (batch_size,)
    p_lens = dataset_ctx_lens[ctx_idxs]                         # (batch_size,)
    max_p_len = p_lens.max()
    p = dataset_ctxs[ctx_idxs][:,:max_p_len].T                  # (max_p_len, batch_size)
    p_mask = dataset_ctx_masks[ctx_idxs][:,:max_p_len].T        # (max_p_len, batch_size)
    float_p_mask = cast_floatX(p_mask)

    q_lens = dataset_qtn_lens[qtn_idxs]                         # (batch_size,)
    max_q_len = q_lens.max()
    q = dataset_qtns[qtn_idxs][:,:max_q_len].T                  # (max_q_len, batch_size)
    q_mask = dataset_qtn_masks[qtn_idxs][:,:max_q_len].T        # (max_q_len, batch_size)
    float_q_mask = cast_floatX(q_mask)

    a = dataset_anss[qtn_idxs]                                  # (batch_size,)
    a_stt = dataset_ans_stts[qtn_idxs]                          # (batch_size,)
    a_end = dataset_ans_ends[qtn_idxs]                          # (batch_size,)

    ###################################################
    # RaSoR
    ###################################################

    ff_dim = config.ff_dims[-1]

    p_emb = emb[p]        # (max_p_len, batch_size, emb_dim)
    q_emb = emb[q]        # (max_q_len, batch_size, emb_dim)

    p_star_parts = [p_emb]
    p_star_dim = config.emb_dim

    ############ q indep

    if config.ablation in [None, 'only_q_indep']:

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

      # (max_q_len, batch_size, ff_dim)     # contains junk where masked
      q_indep_ff = self.ff('q_indep_ff', q_indep_h, [2*config.hidden_dim] + config.ff_dims,
        'relu', config.ff_drop_x, bias_init=config.default_bias_init)
      if config.extra_drop_x:
        q_indep_ff = self.dropout(q_indep_ff, config.extra_drop_x)
      w_q = self.make_param('w_q', (ff_dim,), 'uniform')
      q_indep_scores = tt.dot(q_indep_ff, w_q)                                    # (max_q_len, batch_size)
      q_indep_weights = softmax_columns_with_mask(q_indep_scores, float_q_mask)   # (max_q_len, batch_size)
      q_indep = tt.sum(tt.shape_padright(q_indep_weights) * q_indep_h, axis=0)    # (batch_size, 2*hidden_dim)
      q_indep_repeated = tt.extra_ops.repeat(                                     # (max_p_len, batch_size, 2*hidden_dim)
        tt.shape_padleft(q_indep), max_p_len, axis=0)

      p_star_parts.append(q_indep_repeated)
      p_star_dim += 2 * config.hidden_dim
    
    ############ q aligned

    if config.ablation in [None, 'only_q_align']:

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

      p_mask_shuffled = float_p_mask.dimshuffle((1,0,'x'))                        # (batch_size, max_p_len, 1)
      q_mask_shuffled = float_q_mask.dimshuffle((1,'x',0))                        # (batch_size, 1, max_q_len)
      pq_mask = p_mask_shuffled * q_mask_shuffled                                 # (batch_size, max_p_len, max_q_len)

      q_align_weights = softmax_depths_with_mask(q_align_scores, pq_mask)         # (batch_size, max_p_len, max_q_len)
      q_emb_shuffled = q_emb.dimshuffle((1,0,2))                                  # (batch_size, max_q_len, emb_dim)
      q_align = batched_dot(q_align_weights, q_emb_shuffled)                      # (batch_size, max_p_len, emb_dim)
      q_align_shuffled = q_align.dimshuffle((1,0,2))                              # (max_p_len, batch_size, emb_dim)
    
      p_star_parts.append(q_align_shuffled)
      p_star_dim += config.emb_dim

    ############ passage-level bi-lstm

    p_star = tt.concatenate(p_star_parts, axis=2)     # (max_p_len, batch_size, p_star_dim)

    # (max_p_len, batch_size, 2*hidden_dim)
    p_level_h = self.stacked_bi_lstm('p_level_lstm', p_star, float_p_mask,
      config.num_bilstm_layers, p_star_dim, config.hidden_dim,
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

    if config.sep_stt_end_drop:
      p_level_h_for_stt = self.dropout(p_level_h, config.ff_drop_x)
      p_level_h_for_end = self.dropout(p_level_h, config.ff_drop_x)
    else:
      p_level_h_for_stt = p_level_h_for_end = self.dropout(p_level_h, config.ff_drop_x)

    # Having a single FF hidden layer allows to compute the FF over the concatenation
    # of span-start-hidden-state and span-end-hidden-state by operating the linear transformation
    # separately over each rather than over their concatenations.
    assert len(config.ff_dims) == 1

    if config.objective in ['span_multinomial', 'span_binary']:

      ############ scores

      p_stt_lin = self.linear(                              # (max_p_len, batch_size, ff_dim)
        'p_stt_lin', p_level_h_for_stt, 2*config.hidden_dim, ff_dim, bias_init=config.default_bias_init)
      p_end_lin = self.linear(                              # (max_p_len, batch_size, ff_dim)
        'p_end_lin', p_level_h_for_end, 2*config.hidden_dim, ff_dim, with_bias=False)

      # (batch_size, max_p_len*max_ans_len, ff_dim), (batch_size, max_p_len*max_ans_len)
      span_lin_reshaped, span_masks_reshaped = _span_sums(
        p_stt_lin, p_end_lin, p_lens, max_p_len, batch_size, ff_dim, config.max_ans_len)

      span_ff_reshaped = tt.nnet.relu(span_lin_reshaped)    # (batch_size, max_p_len*max_ans_len, ff_dim)
      w_a = self.make_param('w_a', (ff_dim,), 'uniform')
      span_scores_reshaped = tt.dot(span_ff_reshaped, w_a)  # (batch_size, max_p_len*max_ans_len)

      ############ classification

      classification_func = _span_multinomial_classification if config.objective == 'span_multinomial' else \
        _span_binary_classification
      # (batch_size,), (batch_size), (batch_size,)
      xents, accs, a_hats = classification_func(span_scores_reshaped, span_masks_reshaped, a)
      loss = xents.mean()
      acc = accs.mean()
      # (batch_size,), (batch_size)
      ans_hat_start_word_idxs, ans_hat_end_word_idxs = _tt_ans_idx_to_ans_word_idxs(a_hats, config.max_ans_len)

    elif config.objective == 'span_endpoints':

      ############ scores

      # note that dropout was already applied when assigning to p_level_h_for_stt/end
      p_stt_ff = self.ff(                                                 # (max_p_len, batch_size, ff_dim)
        'p_stt_ff', p_level_h_for_stt, [2*config.hidden_dim] + [ff_dim],
        'relu', dropout_ps=None, bias_init=config.default_bias_init)
      p_end_ff = self.ff(                                                 # (max_p_len, batch_size, ff_dim)
        'p_end_ff', p_level_h_for_end, [2*config.hidden_dim] + [ff_dim],
        'relu', dropout_ps=None, bias_init=config.default_bias_init)

      w_a_stt = self.make_param('w_a_stt', (ff_dim,), 'uniform')
      w_a_end = self.make_param('w_a_end', (ff_dim,), 'uniform')
      word_stt_scores = tt.dot(p_stt_ff, w_a_stt)                         # (max_p_len, batch_size)
      word_end_scores = tt.dot(p_end_ff, w_a_end)                         # (max_p_len, batch_size)

      ############ classification

      stt_log_probs, stt_xents = _word_multinomial_classification(        # (batch_size, max_p_len), (batch_size,)
        word_stt_scores.T, float_p_mask.T, a_stt)
      end_log_probs, end_xents = _word_multinomial_classification(        # (batch_size, max_p_len), (batch_size,)
        word_end_scores.T, float_p_mask.T, a_end)

      xents = stt_xents + end_xents                                       # (batch_size,)
      loss = xents.mean()

      ############ finding highest P(span) = P(span start) * P(span end)

      end_log_probs = end_log_probs.dimshuffle((1,0,'x'))                 # (max_p_len, batch_size, 1)
      stt_log_probs = stt_log_probs.dimshuffle((1,0,'x'))                 # (max_p_len, batch_size, 1)
      # (batch_size, max_p_len*max_ans_len, 1), (batch_size, max_p_len*max_ans_len)
      span_log_probs_reshaped, span_masks_reshaped = _span_sums(
        stt_log_probs, end_log_probs, p_lens, max_p_len, batch_size, 1, config.max_ans_len)

      span_log_probs_reshaped = span_log_probs_reshaped.reshape(          # (batch_size, max_p_len*max_ans_len)
        (batch_size, max_p_len*config.max_ans_len))
      a_hats = argmax_with_mask(                                          # (batch_size,)
        span_log_probs_reshaped, span_masks_reshaped)
      accs = cast_floatX(tt.eq(a_hats, a))                                # (batch_size,)

      acc = accs.mean()
      # (batch_size,), (batch_size)
      ans_hat_start_word_idxs, ans_hat_end_word_idxs = _tt_ans_idx_to_ans_word_idxs(a_hats, config.max_ans_len)

    else:
      raise AssertionError('unsupported objective')

    ############ optimization

    opt = AdamOptimizer(config, loss, self._params.values())
    updates = opt.get_updates()
    global_grad_norm = opt.get_global_grad_norm()
    self.get_lr_value = lambda : opt.get_lr_value()

    ############ interface

    trn_givens = {
      self._is_training : np.int32(1), 
      dataset_ctxs: trn_ctxs,
      dataset_ctx_masks: trn_ctx_masks,
      dataset_ctx_lens: trn_ctx_lens,
      dataset_qtns: trn_qtns,
      dataset_qtn_masks: trn_qtn_masks,
      dataset_qtn_lens: trn_qtn_lens,
      dataset_qtn_ctx_idxs: trn_qtn_ctx_idxs,
      dataset_anss: trn_anss,
      dataset_ans_stts: trn_ans_stts,
      dataset_ans_ends: trn_ans_ends}

    dev_givens = {
      self._is_training : np.int32(0), 
      dataset_ctxs: dev_ctxs,
      dataset_ctx_masks: dev_ctx_masks,
      dataset_ctx_lens: dev_ctx_lens,
      dataset_qtns: dev_qtns,
      dataset_qtn_masks: dev_qtn_masks,
      dataset_qtn_lens: dev_qtn_lens,
      dataset_qtn_ctx_idxs: dev_qtn_ctx_idxs,
      dataset_anss: dev_anss,
      dataset_ans_stts: dev_ans_stts,
      dataset_ans_ends: dev_ans_ends}

    tst_givens = {
      self._is_training : np.int32(0), 
      dataset_ctxs: tst_ctxs,
      dataset_ctx_masks: tst_ctx_masks,
      dataset_ctx_lens: tst_ctx_lens,
      dataset_qtns: tst_qtns,
      dataset_qtn_masks: tst_qtn_masks,
      dataset_qtn_lens: tst_qtn_lens,
      dataset_qtn_ctx_idxs: tst_qtn_ctx_idxs}
      #dataset_anss: tst_anss,
      #dataset_ans_stts: tst_ans_stts,
      #dataset_ans_ends: tst_ans_ends}

    self.train = theano.function(
      [qtn_idxs],
      [loss, acc, global_grad_norm],
      givens = trn_givens,
      updates = updates,
      on_unused_input = 'ignore')
      #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))

    self.eval_dev = theano.function(
      [qtn_idxs],
      [loss, acc, ans_hat_start_word_idxs, ans_hat_end_word_idxs],
      givens = dev_givens,
      updates = None,
      on_unused_input = 'ignore')

    self.eval_tst = theano.function(
      [qtn_idxs],
      [ans_hat_start_word_idxs, ans_hat_end_word_idxs],
      givens = tst_givens,
      updates = None,
      on_unused_input = 'ignore')

    # __init__ end
# Model end


def _span_sums(stt, end, p_lens, max_p_len, batch_size, dim, max_ans_len):
  # Sum of every start element and corresponding max_ans_len end elements.
  #
  # stt     (max_p_len, batch_size, dim)
  # end     (max_p_len, batch_size, dim)
  # p_lens  (batch_size,)
  max_ans_len_range = tt.shape_padleft(tt.arange(max_ans_len))          # (1, max_ans_len)
  offsets = tt.shape_padright(tt.arange(max_p_len))                     # (max_p_len, 1)
  end_idxs = max_ans_len_range + offsets                                # (max_p_len, max_ans_len)
  end_idxs_flat = end_idxs.flatten()                                    # (max_p_len*max_ans_len,)

  end_padded = tt.concatenate(                                          # (max_p_len+max_ans_len-1, batch_size, dim)
    [end, tt.zeros((max_ans_len-1, batch_size, dim))], axis=0)    
  end_structured = end_padded[end_idxs_flat]                            # (max_p_len*max_ans_len, batch_size, dim)
  end_structured = end_structured.reshape(                              # (max_p_len, max_ans_len, batch_size, dim)
    (max_p_len, max_ans_len, batch_size, dim))
  stt_shuffled = stt.dimshuffle((0,'x',1,2))                            # (max_p_len, 1, batch_size, dim)

  span_sums = stt_shuffled + end_structured                             # (max_p_len, max_ans_len, batch_size, dim)
  span_sums_reshaped = span_sums.dimshuffle((2,0,1,3)).reshape(         # (batch_size, max_p_len*max_ans_len, dim)
    (batch_size, max_p_len*max_ans_len, dim))

  p_lens_shuffled = tt.shape_padright(p_lens)                           # (batch_size, 1)
  end_idxs_flat_shuffled = tt.shape_padleft(end_idxs_flat)              # (1, max_p_len*max_ans_len)

  span_masks_reshaped = tt.lt(end_idxs_flat_shuffled, p_lens_shuffled)  # (batch_size, max_p_len*max_ans_len)
  span_masks_reshaped = cast_floatX(span_masks_reshaped)

  # (batch_size, max_p_len*max_ans_len, dim), (batch_size, max_p_len*max_ans_len)
  return span_sums_reshaped, span_masks_reshaped


###################################################
# Variable-length data to GPU matrices and masks
###################################################

def _gpu_dataset(name, dataset, config):
  if dataset:
    ds_vec = dataset.vectorized
    ctxs, ctx_masks, ctx_lens = _gpu_sequences(name + '_ctxs', ds_vec.ctxs, ds_vec.ctx_lens)
    qtns, qtn_masks, qtn_lens = _gpu_sequences(name + '_qtns', ds_vec.qtns, ds_vec.qtn_lens)
    qtn_ctx_idxs = gpu_int32(name + '_qtn_ctx_idxs', ds_vec.qtn_ctx_idxs)
    anss, ans_stts, ans_ends = _gpu_answers(name, ds_vec.anss, config.max_ans_len)
  else:
    ctxs = ctx_masks = qtns = qtn_masks = gpu_int32(name + '_empty_matrix', np.zeros((1,1), dtype=np.int32))
    ctx_lens = qtn_lens = qtn_ctx_idxs = anss = ans_stts = ans_ends = \
      gpu_int32(name + '_empty_vector', np.zeros(1, dtype=np.int32))
  return ctxs, ctx_masks, ctx_lens, qtns, qtn_masks, qtn_lens, qtn_ctx_idxs, anss, ans_stts, ans_ends


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


def _gpu_answers(name, anss, max_ans_len):
  assert anss.dtype == np.int32
  assert anss.shape[1] == 2
  anss_val = np.array([_np_ans_word_idxs_to_ans_idx(ans_stt, ans_end, max_ans_len) for \
    ans_stt, ans_end in anss], dtype=np.int32)
  ans_stts_val = anss[:,0]
  ans_ends_val = anss[:,1]

  gpu_anss = gpu_int32(name + '_anss', anss_val)
  gpu_ans_stts = gpu_int32(name + '_ans_stts', ans_stts_val)
  gpu_ans_ends = gpu_int32(name + '_ans_ends', ans_ends_val)
  return gpu_anss, gpu_ans_stts, gpu_ans_ends


###################################################
# Classification
###################################################

def _span_multinomial_classification(x, x_mask, y):
  # x       float32 (batch_size, num_classes)   scores i.e. logits
  # x_mask  int32   (batch_size, num_classes)   score masks (each sample has a variable number of classes)
  # y       int32   (batch_size,)               target classes i.e. ground truth answers (given as class indices)
  assert x.ndim == x_mask.ndim == 2
  assert y.ndim == 1

  # substracting min needed since all non masked-out elements of a row may be negative.
  x *= x_mask
  x -= x.min(axis=1, keepdims=True)             # (batch_size, num_classes)
  x *= x_mask                                   # (batch_size, num_classes)
  y_hats = tt.argmax(x, axis=1)                 # (batch_size,)
  accs = cast_floatX(tt.eq(y_hats, y))          # (batch_size,)

  x -= x.max(axis=1, keepdims=True)             # (batch_size, num_classes)
  x *= x_mask                                   # (batch_size, num_classes)
  exp_x = tt.exp(x)                             # (batch_size, num_classes)
  exp_x *= x_mask                               # (batch_size, num_classes)

  sum_exp_x = exp_x.sum(axis=1)                 # (batch_size,)
  log_sum_exp_x = tt.log(sum_exp_x)             # (batch_size,)

  x_star = x[tt.arange(x.shape[0]), y]          # (batch_size,)
  xents = log_sum_exp_x - x_star                # (batch_size,)

  return xents, accs, y_hats


def _span_binary_classification(x, x_mask, y):
  # x       float32 (batch_size, num_classes)   scores i.e. logits
  # x_mask  int32   (batch_size, num_classes)   score masks (each sample has a variable number of classes)
  # y       int32   (batch_size,)               target classes i.e. ground truth answers (given as class indices)
  assert x.ndim == x_mask.ndim == 2
  assert y.ndim == 1

  # placing min in masked-out elements needed since all non masked-out elements of a row may be negative.
  x_min = x.min(axis=1, keepdims=True)                  # (batch_size, 1)
  x = x_mask * x + (1 - x_mask) * x_min                 # (batch_size, num_classes)
  y_hats = tt.argmax(x, axis=1)                         # (batch_size,)
  accs = cast_floatX(tt.eq(y_hats, y))                  # (batch_size,)

  log_z = tt.log(1 + tt.exp(-x))                        # (batch_size, num_classes)
  xents_false = x + log_z                               # (batch_size, num_classes)
  xents_false *= x_mask                                 # (batch_size, num_classes)
  sum_xents_false = xents_false.sum(axis=1)             # (batch_size,)

  x_star = x[tt.arange(x.shape[0]), y]                  # (batch_size,)
  sum_xents = sum_xents_false - x_star                  # (batch_size,)
  #xents = sum_xents / x_mask.sum(axis=1, keepdims=True) # (batch_size,)
  xents = sum_xents

  return xents, accs, y_hats


def _word_multinomial_classification(x, x_mask, y):
  # x       float32 (batch_size, num_classes)   scores i.e. logits
  # x_mask  int32   (batch_size, num_classes)   score masks (each sample has a variable number of classes)
  # y       int32   (batch_size,)               target classes i.e. ground truth answers (given as class indices)
  assert x.ndim == x_mask.ndim == 2
  assert y.ndim == 1

  # substracting min needed since all non masked-out elements of a row may be negative.
  x *= x_mask
  x -= x.min(axis=1, keepdims=True)             # (batch_size, num_classes)
  x *= x_mask                                   # (batch_size, num_classes)
  x -= x.max(axis=1, keepdims=True)             # (batch_size, num_classes)
  x *= x_mask                                   # (batch_size, num_classes)
  exp_x = tt.exp(x)                             # (batch_size, num_classes)
  exp_x *= x_mask                               # (batch_size, num_classes)

  sum_exp_x = exp_x.sum(axis=1, keepdims=True)  # (batch_size, 1)
  log_sum_exp_x = tt.log(sum_exp_x)             # (batch_size, 1)

  log_probs = x - log_sum_exp_x                 # (batch_size, num_classes)
  log_probs *= x_mask

  x_star_log_probs = log_probs[tt.arange(x.shape[0]), y]  # (batch_size,)
  xents = -x_star_log_probs

  return log_probs, xents

