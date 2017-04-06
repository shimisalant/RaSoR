import sys
import logging
import time
import argparse

import numpy as np

from base.utils import set_up_logger
from utils import EpochResult, format_epoch_results, plot_epoch_results
from reader import get_data, construct_answer_hat, write_test_predictions
from evaluate11 import metric_max_over_ground_truths, exact_match_score, f1_score


class Config(object):
  def __init__(self, compared=[], **kwargs):
    self.name = None
    self.desc = None
    self.device = None                      # 'cpu' / 'gpu<index>'
    self.plot = False                       # whether to plot training graphs
    self.save_freq = None                   # how often to save model (in epochs); None for only after best EM/F1 epochs

    self.word_emb_data_path_prefix = None   # path of preprocessed word embedding data, produced by setup.py
    self.tokenized_trn_json_path = None     # path of tokenized training set JSON, produced by setup.py
    self.tokenized_dev_json_path = None     # path of tokenized dev set JSON, produced by setup.py
    self.test_json_path = None              # path of test set JSON
    self.pred_json_path = None              # path of test predictions JSON
    self.tst_load_model_path = None         # path of trained model data, used for producing test set predictions
    self.tst_split = True                   # whether to split hyphenated unknown words of test set, see setup.py

    self.seed = np.random.random_integers(1e6, 1e9)
    self.max_ans_len = 30                   # maximal answer length, answers of longer length are discarded
    self.emb_dim = 300                      # dimension of word embeddings
    self.learn_single_unk = False           # whether to have a single tunable word embedding for all unknown words
                                            # (or multiple fixed random ones)
    self.init_scale = 5e-3                  # uniformly random weights are initialized in [-init_scale, +init_scale]
    self.learning_rate = 1e-3
    self.lr_decay = 0.95
    self.lr_decay_freq = 5000               # frequency with which to decay learning rate, measured in updates
    self.max_grad_norm = 10                 # gradient clipping
    self.ff_dims = [100]                    # dimensions of hidden FF layers
    self.ff_drop_x = 0.2                    # dropout rate of FF layers
    self.batch_size = 40
    self.max_num_epochs = 150               # max number of epochs to train for

    self.num_bilstm_layers = 2              # number of BiLSTM layers, where BiLSTM is applied
    self.hidden_dim = 100                   # dimension of hidden state of each uni-directional LSTM
    self.lstm_drop_h = 0.1                  # dropout rate for recurrent hidden state of LSTM
    self.lstm_drop_x = 0.4                  # dropout rate for inputs of LSTM
    self.lstm_couple_i_and_f = True         # customizable LSTM configuration, see base/model.py
    self.lstm_learn_initial_state = False
    self.lstm_tie_x_dropout = True
    self.lstm_sep_x_dropout = False
    self.lstm_sep_h_dropout = False
    self.lstm_w_init = 'uniform'
    self.lstm_u_init = 'uniform'
    self.lstm_forget_bias_init = 'uniform'
    self.default_bias_init = 'uniform'

    self.extra_drop_x = 0                   # dropout rate at an extra possible place
    self.q_aln_ff_tie = True                # whether to tie the weights of the FF over question and the FF over passage
    self.sep_stt_end_drop = True            # whether to have separate dropout masks for span start and
                                            # span end representations

    self.adam_beta1 = 0.9                   # see base/optimizer.py
    self.adam_beta2 = 0.999
    self.adam_eps = 1e-8

    self.objective = 'span_multinomial'     # 'span_multinomial': multinomial distribution over all spans
                                            # 'span_binary':      logistic distribution per span
                                            # 'span_endpoints':   two multinomial distributions, over span start and end

    self.ablation = None                    # 'only_q_align':     question encoded only by passage-aligned representation
                                            # 'only_q_indep':     question encoded only by passage-independent representation
                                            # None:               question encoded by both

    assert all(k in self.__dict__ for k in kwargs)
    assert all(k in self.__dict__ for k in compared)
    self.__dict__.update(kwargs)
    self._compared = compared

  def __repr__(self):
    ks = sorted(k for k in self.__dict__ if k not in ['name', 'desc', '_compared'])
    return '\n'.join('{:<30s}{:<s}'.format(k, str(self.__dict__[k])) for k in ks)

  def format_compared(self):
    return '\n'.join([
      ''.join('{:12s} '.format(k[:12]) for k in sorted(self._compared)),
      ''.join('{:12s} '.format(str(self.__dict__[k])[:12]) for k in sorted(self._compared))])


def _trn_epoch(config, model, data, epoch, np_rng):
  logger = logging.getLogger()
  # indices of questions which have a valid answer
  valid_qtn_idxs = np.flatnonzero(data.trn.vectorized.qtn_ans_inds).astype(np.int32)
  np_rng.shuffle(valid_qtn_idxs)
  num_samples = valid_qtn_idxs.size
  batch_sizes = []
  losses = []
  accs = []
  samples_per_sec = []
  ss = range(0, num_samples, config.batch_size)
  for b, s in enumerate(ss, 1):
    batch_idxs = valid_qtn_idxs[s:min(s+config.batch_size, num_samples)]
    batch_sizes.append(len(batch_idxs))

    start_time = time.time()
    loss, acc, global_grad_norm = model.train(batch_idxs)
    samples_per_sec.append(len(batch_idxs) / (time.time() - start_time))

    losses.append(loss)
    accs.append(acc)
    if b % 100 == 0 or b == len(ss):
      logger.info(
        '{:<8s} {:<15s} lr={:<8.7f} : train loss={:<8.5f}\tacc={:<8.5f}\tgrad={:<8.5f}\tsamples/sec={:<.1f}'.format(
        config.device, 'e'+str(epoch)+'b'+str(b)+'\\'+str(len(ss)), float(model.get_lr_value()),
        float(loss), float(acc), float(global_grad_norm), float(samples_per_sec[b-1])))
  trn_loss = np.average(losses, weights=batch_sizes)
  trn_acc = np.average(accs, weights=batch_sizes)
  trn_samples_per_sec = np.average(samples_per_sec, weights=batch_sizes)
  return trn_loss, trn_acc, trn_samples_per_sec


def _dev_epoch(config, model, data):
  logger = logging.getLogger()
  num_all_samples = data.dev.vectorized.qtn_ans_inds.size
  ans_hat_starts = np.zeros(num_all_samples, dtype=np.int32)
  ans_hat_ends = np.zeros(num_all_samples, dtype=np.int32)

  # indices of questions which have a valid answer
  valid_qtn_idxs = np.flatnonzero(data.dev.vectorized.qtn_ans_inds).astype(np.int32)
  num_valid_samples = valid_qtn_idxs.size
  batch_sizes = []
  losses = []
  accs = []
  ss = range(0, num_valid_samples, config.batch_size)
  for b, s in enumerate(ss, 1):
    batch_idxs = valid_qtn_idxs[s:min(s+config.batch_size, num_valid_samples)]
    batch_sizes.append(len(batch_idxs))

    loss, acc, ans_hat_start_word_idxs, ans_hat_end_word_idxs = model.eval_dev(batch_idxs)

    losses.append(loss)
    accs.append(acc)
    ans_hat_starts[batch_idxs] = ans_hat_start_word_idxs
    ans_hat_ends[batch_idxs] = ans_hat_end_word_idxs
    if b % 100 == 0 or b == len(ss):
      logger.info('{:<8s} {:<15s} : dev valid'.format(config.device, 'b'+str(b)+'\\'+str(len(ss))))
  dev_loss = np.average(losses, weights=batch_sizes)
  dev_acc = np.average(accs, weights=batch_sizes)

  # indices of questions which have an invalid answer
  invalid_qtn_idxs = np.setdiff1d(np.arange(num_all_samples), valid_qtn_idxs).astype(np.int32)
  num_invalid_samples = invalid_qtn_idxs.size
  ss = range(0, num_invalid_samples, config.batch_size)
  for b, s in enumerate(ss, 1):
    batch_idxs = invalid_qtn_idxs[s:min(s+config.batch_size, num_invalid_samples)]

    _, _, ans_hat_start_word_idxs, ans_hat_end_word_idxs = model.eval_dev(batch_idxs)

    ans_hat_starts[batch_idxs] = ans_hat_start_word_idxs
    ans_hat_ends[batch_idxs] = ans_hat_end_word_idxs
    if b % 100 == 0 or b == len(ss):
      logger.info('{:<8s} {:<15s} : dev invalid'.format(config.device, 'b'+str(b)+'\\'+str(len(ss))))

  # calculate EM, F1
  ems = []
  f1s = []
  for qtn_idx, (ans_hat_start_word_idx, ans_hat_end_word_idx) in enumerate(zip(ans_hat_starts, ans_hat_ends)):
    qtn = data.dev.tabular.qtns[qtn_idx]
    ctx = data.dev.tabular.ctxs[qtn.ctx_idx]
    ans_hat_str = construct_answer_hat(ctx, ans_hat_start_word_idx, ans_hat_end_word_idx)
    ans_strs = qtn.ans_texts
    ems.append(metric_max_over_ground_truths(exact_match_score, ans_hat_str, ans_strs))
    f1s.append(metric_max_over_ground_truths(f1_score, ans_hat_str, ans_strs))
  dev_em = np.mean(ems)
  dev_f1 = np.mean(f1s)
  return dev_loss, dev_acc, dev_em, dev_f1


def _tst_epoch(config, model, data):
  logger = logging.getLogger()
  num_samples = data.tst.vectorized.qtns.shape[0]
  ans_hat_starts = np.zeros(num_samples, dtype=np.int32)
  ans_hat_ends = np.zeros(num_samples, dtype=np.int32)
  ss = range(0, num_samples, config.batch_size)
  for b, s in enumerate(ss, 1):
    e = min(s + config.batch_size, num_samples)
    batch_idxs = np.arange(s, e, dtype=np.int32)

    ans_hat_start_word_idxs, ans_hat_end_word_idxs = model.eval_tst(batch_idxs)

    ans_hat_starts[batch_idxs] = ans_hat_start_word_idxs
    ans_hat_ends[batch_idxs] = ans_hat_end_word_idxs
    if b % 100 == 0 or b == len(ss):
      logger.info('{:<8s} {:<15s} : test'.format(config.device, 'b'+str(b)+'\\'+str(len(ss))))
  ans_hats = {}
  for qtn_idx, (ans_hat_start_word_idx, ans_hat_end_word_idx) in enumerate(zip(ans_hat_starts, ans_hat_ends)):
    qtn = data.tst.tabular.qtns[qtn_idx]
    ctx = data.tst.tabular.ctxs[qtn.ctx_idx]
    ans_hat_str = construct_answer_hat(ctx, ans_hat_start_word_idx, ans_hat_end_word_idx)
    ans_hats[qtn.qtn_id] = ans_hat_str
  return ans_hats


def _get_configs():
  compared = [
    'device', 'objective', 'ablation', 'batch_size', 'ff_dims', 'ff_drop_x',
    'hidden_dim', 'lstm_drop_h', 'lstm_drop_x', 'lstm_drop_h', 'q_aln_ff_tie']
  common = {
    'name': 'RaSoR',
    'desc': 'Recurrent span representations',
    'word_emb_data_path_prefix': 'data/preprocessed_glove_with_unks.split',
    'tokenized_trn_json_path': 'data/train-v1.1.tokenized.split.json',
    'tokenized_dev_json_path': 'data/dev-v1.1.tokenized.split.json',
    'plot': True
  }
  configs = [

    # Objective comparison:

    Config(compared,
      objective = 'span_multinomial',
      tst_load_model_path = 'models/RaSoR_cfg0_best_em.pkl',
      **common),

    Config(compared,
      objective = 'span_binary',
      tst_load_model_path = 'models/RaSoR_cfg1_best_em.pkl',
      **common),

    Config(compared,
      objective = 'span_endpoints',
      tst_load_model_path = 'models/RaSoR_cfg2_best_em.pkl',
      **common),
      
    # Ablation study:

    Config(compared,
      objective = 'span_multinomial',
      ablation = 'only_q_align',
      tst_load_model_path = 'models/RaSoR_cfg3_best_em.pkl',
      **common),

    Config(compared,
      objective = 'span_multinomial',
      ablation = 'only_q_indep',
      tst_load_model_path = 'models/RaSoR_cfg4_best_em.pkl',
      **common),

    ]

  return configs

  
def _main(config, config_idx, train):
  base_filename = config.name + '_cfg' + str(config_idx)
  logger = set_up_logger('logs/' + base_filename + '.log')
  title = '{}: {} ({}) config index {}'.format(__file__, config.name, config.desc, config_idx)
  logger.info('START ' + title + '\n\n{}\n'.format(config))

  data = get_data(config, train)

  if config.device != 'cpu':
    assert 'theano' not in sys.modules 
    import theano.sandbox.cuda
    theano.sandbox.cuda.use(config.device)
  from model import get_model
  model = get_model(config, data)

  if not train:
    assert config.tst_load_model_path
    if not model.load(config.tst_load_model_path):
      raise AssertionError('Failed loading model weights from {}'.format(config.tst_load_model_path))
    ans_hats = _tst_epoch(config, model, data)
    write_test_predictions(ans_hats, config.pred_json_path)
    logger.info('END ' + title)
    return

  # Training loop
  epoch_results = []
  max_em = -np.inf
  max_f1 = -np.inf
  np_rng = np.random.RandomState(config.seed // 2)
  for epoch in range(1, config.max_num_epochs+1):
    trn_loss, trn_acc, trn_samples_per_sec = _trn_epoch(config, model, data, epoch, np_rng)
    dev_loss, dev_acc, dev_em, dev_f1 = _dev_epoch(config, model, data)
    if dev_em > max_em:
      model.save('models/' + base_filename + '_best_em.pkl')
      max_em = dev_em
    if dev_f1 > max_f1:
      model.save('models/' + base_filename + '_best_f1.pkl')
      max_f1 = dev_f1
    if config.save_freq and epoch % config.save_freq == 0:
      model.save('models/' + base_filename + '_e{:03d}.pkl'.format(epoch))
    epoch_results.append(
      EpochResult(trn_loss, trn_acc, dev_loss, dev_acc, dev_em, dev_f1))
    if config.plot:
      plot_epoch_results(epoch_results, 'logs/' + base_filename + '.png')
    logger.info('\n\nFinished epoch {} for: (config index {}) (samples/sec: {:<.1f})\n{}\n\nResults:\n{}\n\n'.format(
      epoch, config_idx, trn_samples_per_sec, config.format_compared(), format_epoch_results(epoch_results)))
  logger.info('END ' + title)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--device', help='device e.g. cpu, gpu0, gpu1, ...', default='cpu')
  parser.add_argument('--train', help='whether to train', action='store_true')
  parser.add_argument('--cfg_idx', help='configuration index', type=int, default=0)
  parser.add_argument('test_json_path', nargs='?', help='test JSON file for which answers should be predicted')
  parser.add_argument('pred_json_path', nargs='?', help='where to write test predictions to')
  args = parser.parse_args()
  if bool(args.test_json_path) != bool(args.pred_json_path) or bool(args.test_json_path) == args.train:
    parser.error('Specify both test_json_path and pred_json_path, or only --train')
  config = _get_configs()[args.cfg_idx]
  config.device = args.device
  config.test_json_path = args.test_json_path
  config.pred_json_path = args.pred_json_path
  _main(config, args.cfg_idx, args.train)

