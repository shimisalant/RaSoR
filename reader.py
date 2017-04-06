# -*- coding: utf-8 -*-
import logging
import io
import json
import numpy as np

from collections import namedtuple, Counter

from setup import WordEmbData, read_word_emb_data, tokenize_tst_json


#######################################
# Types:
#######################################

SquadData = namedtuple('SquadData', [
  'word_emb_data',      # WordEmbData
  'trn',                # SquadDataset
  'dev',                # SquadDataset
  'tst'                 # SquadDataset
])

SquadDataset = namedtuple('SquadDataset', [
  'tabular',            # SquadDatasetTabular
  'vectorized'          # SquadDatasetVectorized
])

SquadDatasetVectorized = namedtuple('SquadDatasetVectorized', [
  'ctxs',               # int32 (num contexts, max context length)
  'ctx_lens',           # int32 (num contexts,)
  'qtns',               # int32 (num questions, max question length)
  'qtn_lens',           # int32 (num questions,)
  'qtn_ctx_idxs',       # int32 (num questions,)      index of context of question
  'qtn_ans_inds',       # int32 (num questions,)      indicator of whether question has a valid answer
  'anss'                # int32 (num questions, 2)    we keep only first valid answer as (answer start word idx,
                        #                             answer end word idx), undefined for all invalid
])

TokenizedText = namedtuple('TokenizedText', [
  'text',               # original text string
  'tokens',             # list of parsed tokens
  'originals',          # list of original tokens (may differ from parsed ones)
  'whitespace_afters',  # list of whitespace strings, each appears after corresponding original token in original text
])

SquadArticle = namedtuple('SquadArticle', [
  'art_title_str'
])

SquadContext = namedtuple('SquadContext', [
  'art_idx',
  'tokenized'           # TokenizedText of context's text
])

SquadQuestion = namedtuple('SquadQuestion', [
  'ctx_idx',
  'qtn_id',
  'tokenized',          # TokenizedText of question's text
  'ans_texts',          # list of (possibly multiple) answer text strings
  'ans_word_idxs'       # list where each entry is either a (answer start word index, answer end word index) tuple
                        # or None for answers that we failed to parse
])

class SquadDatasetTabular(object):
  def __init__(self):
    self.arts = []    # SquadArticle objects
    self.ctxs = []    # SquadContext objects
    self.qtns = []    # SquadQuestion objects
  def new_article(self, art_title_str):
    self.arts.append(SquadArticle(art_title_str))
    return len(self.arts) - 1
  def new_context(self, art_idx, ctx_tokenized):
    self.ctxs.append(SquadContext(art_idx, ctx_tokenized))
    return len(self.ctxs) - 1
  def new_question(self, ctx_idx, qtn_id, qtn_tokenized, ans_texts, ans_word_idxs):
    self.qtns.append(
      SquadQuestion(ctx_idx, qtn_id, qtn_tokenized, ans_texts, ans_word_idxs))


#######################################
# Functionality:
#######################################

def get_data(config, train):
  word_emb_data = read_word_emb_data(config.word_emb_data_path_prefix)
  word_strs = set()
  if train:
    trn_tab_ds = _make_tabular_dataset(
      config.tokenized_trn_json_path, word_strs, has_answers=True, max_ans_len=config.max_ans_len)
    dev_tab_ds = _make_tabular_dataset(
      config.tokenized_dev_json_path, word_strs, has_answers=True, max_ans_len=config.max_ans_len)

    word_emb_data = _contract_word_emb_data(word_emb_data, word_strs, config.learn_single_unk)
    trn_vec_ds = _make_vectorized_dataset('train', trn_tab_ds, word_emb_data)
    dev_vec_ds = _make_vectorized_dataset('dev', dev_tab_ds, word_emb_data)

    trn_ds = SquadDataset(trn_tab_ds, trn_vec_ds)
    dev_ds = SquadDataset(dev_tab_ds, dev_vec_ds)
    return SquadData(word_emb_data, trn_ds, dev_ds, None)
  else:
    tokenized_test_json_path = config.test_json_path + '.tokenized.tmp'
    tokenize_tst_json(config.test_json_path, tokenized_test_json_path, config.tst_split)
    tst_tab_ds = _make_tabular_dataset(
      tokenized_test_json_path, word_strs, has_answers=False, max_ans_len=config.max_ans_len)

    word_emb_data = _contract_word_emb_data(word_emb_data, word_strs, config.learn_single_unk)
    tst_vec_ds = _make_vectorized_dataset('test', tst_tab_ds, word_emb_data)

    tst_ds = SquadDataset(tst_tab_ds, tst_vec_ds)
    return SquadData(word_emb_data, None, None, tst_ds)


def construct_answer_hat(ctx, ans_hat_start_word_idx, ans_hat_end_word_idx):
  ctx_originals = ctx.tokenized.originals
  ctx_whitespace_afters = ctx.tokenized.whitespace_afters
  ans_hat_str = ''
  for word_idx in range(ans_hat_start_word_idx, ans_hat_end_word_idx+1):
    ans_hat_str += ctx_originals[word_idx]
    if word_idx < ans_hat_end_word_idx:
      ans_hat_str += ctx_whitespace_afters[word_idx]
  return ans_hat_str


def write_test_predictions(ans_hats, pred_json_path):
  logger = logging.getLogger()
  s = json.dumps(ans_hats, ensure_ascii=False)
  with io.open(pred_json_path, 'w', encoding='utf-8') as f:
    f.write(s)
  logger.info('Written test predictions to {}'.format(pred_json_path))


def _make_tabular_dataset(tokenized_json_path, word_strs, has_answers, max_ans_len=None):
  logger = logging.getLogger()
  tabular = SquadDatasetTabular()

  num_questions = 0
  num_answers = 0
  num_invalid_answers = 0
  num_long_answers = 0
  num_invalid_questions = 0

  answers_per_question_counter = Counter()
  with io.open(tokenized_json_path, 'r', encoding='utf-8') as f:
    j = json.load(f)
    #version = j['version']
    data = j['data']
    for article in data:
      art_title_str = article['title']
      art_idx = tabular.new_article(art_title_str)

      paragraphs = article['paragraphs']
      for paragraph in paragraphs:
        ctx_str = paragraph['context']
        ctx_tokens = paragraph['tokens']
        word_strs.update(ctx_tokens)
        ctx_originals = paragraph['originals']
        ctx_whitespace_afters = paragraph['whitespace_afters']
        ctx_tokenized = TokenizedText(ctx_str, ctx_tokens, ctx_originals, ctx_whitespace_afters)
        ctx_idx = tabular.new_context(art_idx, ctx_tokenized)

        qas = paragraph['qas']
        for qa in qas:
          num_questions += 1
          qtn_id = qa['id']

          qtn_str = qa['question']
          qtn_tokens = qa['tokens']
          word_strs.update(qtn_tokens)
          qtn_originals = qa['originals']
          qtn_whitespace_afters = qa['whitespace_afters']
          qtn_tokenized = TokenizedText(qtn_str, qtn_tokens, qtn_originals, qtn_whitespace_afters)

          ans_texts = []
          ans_word_idxs = []
          if has_answers:
            answers = qa['answers']
            assert answers
            for answer in answers:
              num_answers += 1
              ans_text = answer['text']
              assert ans_text
              ans_texts.append(ans_text)
              if not answer['valid']:
                ans_word_idxs.append(None)
                num_invalid_answers += 1
                continue
              ans_start_word_idx = answer['start_token_idx']
              ans_end_word_idx = answer['end_token_idx']
              if max_ans_len and ans_end_word_idx - ans_start_word_idx + 1 > max_ans_len:
                ans_word_idxs.append(None)
                num_long_answers += 1
              else:
                ans_word_idxs.append((ans_start_word_idx, ans_end_word_idx))
            answers_per_question_counter[len(ans_texts)] += 1   # this counts also invalid answers
            num_invalid_questions += 1 if all(ans is None for ans in ans_word_idxs) else 0

          tabular.new_question(ctx_idx, qtn_id, qtn_tokenized, ans_texts, ans_word_idxs)

  logger.info('Processed {:s}:\n'
    '\ttotal {:d} questions, {:d} invalid questions, '
    'total {:d} answers, {:d} invalid answers, {:d} too long answers\n'
    '\t{{x: num of questions having x answers}}: {{{:s}}}'.format(
    tokenized_json_path,
    num_questions, num_invalid_questions,
    num_answers, num_invalid_answers, num_long_answers,
    ', '.join('{:d}: {:d}'.format(x, num_x) for x, num_x in sorted(answers_per_question_counter.iteritems()))))
  return tabular


def _contract_word_emb_data(old_word_emb_data, word_strs, is_single_unk):
  logger = logging.getLogger()
  old_word_emb, old_str_to_word, old_first_known_word, old_first_unknown_word, old_first_unallocated_word = \
    old_word_emb_data

  known_word_strs = []
  unknown_word_strs = []
  for word_str in word_strs:
    if word_str in old_str_to_word and old_str_to_word[word_str] < old_first_unknown_word:
      known_word_strs.append(word_str)
    else:
      unknown_word_strs.append(word_str)

  str_to_word = {}
  emb_size = old_first_known_word + (len(known_word_strs)+1 if is_single_unk else len(word_strs))
  word_emb = np.zeros((emb_size, old_word_emb.shape[1]), dtype=np.float32)

  for i, word_str in enumerate(known_word_strs):
    word = old_first_known_word + i
    str_to_word[word_str] = word
    word_emb[word, :] = old_word_emb[old_str_to_word[word_str]]

  first_unknown_word = old_first_known_word + len(known_word_strs)

  if is_single_unk:
    for word_str in unknown_word_strs:
      str_to_word[word_str] = first_unknown_word
    logger.info('Contracted word embeddings (single embedding for unknown word-types): '
      '{} known word-types, {} unknown word-types'.format(len(known_word_strs), len(unknown_word_strs)))
  else:
    num_new_unks = 0
    for i, word_str in enumerate(unknown_word_strs):
      word = first_unknown_word + i
      str_to_word[word_str] = word
      if word_str in old_str_to_word:
        word_emb[word, :] = old_word_emb[old_str_to_word[word_str]]
      else:
        if old_first_unallocated_word + num_new_unks >= len(old_word_emb):
          logger.info('Error: too many unknown words, can increase number of alloted random embeddings in setup.py')
          sys.exit(1)
        word_emb[word, :] = old_word_emb[old_first_unallocated_word + num_new_unks]
        num_new_unks += 1
    logger.info('Contracted word embeddings (multiple embeddings for unknown word-types):\n'
      '\t{} known word-types, {} pre-existing unknown word-types, {} new unknown word-types'.format(
        len(known_word_strs), len(unknown_word_strs) - num_new_unks, num_new_unks))
  return WordEmbData(
    word_emb, str_to_word, old_first_known_word, first_unknown_word, None)


def _make_vectorized_dataset(name, tabular, word_emb_data):
  num_ctxs = len(tabular.ctxs)
  num_qtns = len(tabular.qtns)
  max_ctx_len = max(len(ctx.tokenized.tokens) for ctx in tabular.ctxs)
  max_qtn_len = max(len(qtn.tokenized.tokens) for qtn in tabular.qtns)

  ctxs = np.zeros((num_ctxs, max_ctx_len), dtype=np.int32)
  ctx_lens = np.zeros(num_ctxs, dtype=np.int32)
  qtns = np.zeros((num_qtns, max_qtn_len), dtype=np.int32)
  qtn_lens = np.zeros(num_qtns, dtype=np.int32)
  qtn_ctx_idxs = np.zeros(num_qtns, dtype=np.int32)
  qtn_ans_inds = np.zeros(num_qtns, dtype=np.int32)
  anss = np.zeros((num_qtns, 2), dtype=np.int32)

  for ctx_idx, ctx in enumerate(tabular.ctxs):
    ctx_words = [word_emb_data.str_to_word[word_str] for word_str in ctx.tokenized.tokens]
    ctxs[ctx_idx, :len(ctx_words)] = ctx_words
    ctx_lens[ctx_idx] = len(ctx_words)
    
  for qtn_idx, qtn in enumerate(tabular.qtns):
    qtn_words = [word_emb_data.str_to_word[word_str] for word_str in qtn.tokenized.tokens]
    qtns[qtn_idx, :len(qtn_words)] = qtn_words
    qtn_lens[qtn_idx] = len(qtn_words)
    qtn_ctx_idxs[qtn_idx] = qtn.ctx_idx
    ans = next((ans for ans in qtn.ans_word_idxs if ans), None) if qtn.ans_word_idxs else None
    if ans:
      ans_start_word_idx, ans_end_word_idx = ans
      anss[qtn_idx] = [ans_start_word_idx, ans_end_word_idx]
      qtn_ans_inds[qtn_idx] = 1
    else:
      qtn_ans_inds[qtn_idx] = 0

  qs = [1., 2., 5.] + list(np.arange(10., 91., 10.)) + [95., 99., 100.]
  msg = 'Vectorized {} samples. Lengths:\n'.format(name) + '\n'.join([
    '\t{:<15s}{:s}'.format('percentile:', ''.join(['%-5d' % q for q in qs])),
    '\t{:<15s}{:s}'.format('ctx length:', ''.join(['%-5d' % ctx_p for ctx_p in np.percentile(ctx_lens, qs)])),
    '\t{:<15s}{:s}'.format('qtn length:', ''.join(['%-5d' % qtn_p for qtn_p in np.percentile(qtn_lens, qs)]))])
  logging.getLogger().info(msg)
  return SquadDatasetVectorized(ctxs, ctx_lens, qtns, qtn_lens, qtn_ctx_idxs, qtn_ans_inds, anss)

