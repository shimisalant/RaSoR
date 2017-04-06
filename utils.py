import numpy as np
from collections import namedtuple


EpochResult = namedtuple('EpochResult', [
  'trn_loss', 'trn_acc',
  'dev_loss', 'dev_acc', 'dev_em', 'dev_f1'])


def _format_epoch_result(epoch_title, epoch_idx, er):
  return ('{:<10s} {:<6s} :   '
    'trn: loss={:<5.3f} acc={:<6.3f}   '
    'dev: loss={:<5.3f} acc={:<6.3f}   '
    'em={:<6.3f} f1={:<6.3f}').format(
      epoch_title, '(e'+str(epoch_idx+1)+')',
      er.trn_loss, 100*er.trn_acc,
      er.dev_loss, 100*er.dev_acc,
      100*er.dev_em, 100*er.dev_f1)


def format_epoch_results(epoch_results):
  idx_last = len(epoch_results) - 1
  idx_best_em = np.argmax([er.dev_em for er in epoch_results])
  idx_best_f1 = np.argmax([er.dev_f1 for er in epoch_results])
  return '\n'.join([
    _format_epoch_result('Last epoch', idx_last, epoch_results[idx_last]),
    _format_epoch_result('Best EM', idx_best_em, epoch_results[idx_best_em]),
    _format_epoch_result('Best F1', idx_best_f1, epoch_results[idx_best_f1])])


def _plot_series(ax, x, y, color, label, is_argmin):
  ax.plot(x, y, color=color, label=label)
  idx = np.argmin(y) if is_argmin else np.argmax(y)
  ax.plot(x[idx], y[idx], color=color, marker='o', label=None)


def plot_epoch_results(epoch_results, filename):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  trn_losses = [er.trn_loss for er in epoch_results]
  trn_accs = [er.trn_acc for er in epoch_results]
  dev_losses = [er.dev_loss for er in epoch_results]
  dev_accs = [er.dev_acc for er in epoch_results]
  dev_ems = [er.dev_em for er in epoch_results]
  dev_f1s = [er.dev_f1 for er in epoch_results]
  epoch_nums = range(1, len(epoch_results)+1)

  fig, axarr = plt.subplots(2, sharex=True, figsize=(12, 9))
  axarr[1].set_xlabel('epoch')

  _plot_series(axarr[0], epoch_nums, trn_losses, 'blue', 'train loss', True)
  _plot_series(axarr[0], epoch_nums, dev_losses, 'red', 'dev loss', True)
  axarr[0].legend(loc='upper right', prop={'size':12})

  _plot_series(axarr[1], epoch_nums, trn_accs, 'blue', 'train acc', False)
  _plot_series(axarr[1], epoch_nums, dev_accs, 'red', 'dev acc', False)
  _plot_series(axarr[1], epoch_nums, dev_ems, 'pink', 'dev em', False)
  _plot_series(axarr[1], epoch_nums, dev_f1s, 'cyan', 'dev f1', False)
  axarr[1].legend(loc='lower right', prop={'size':12})

  plt.tight_layout()
  plt.savefig(filename)
  plt.close(fig)

