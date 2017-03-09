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
  'ctxs',               # float32 (num samples, max context length)
  'qtns',               # float32 (num samples, max question length)
  'anss',               # int32 (num samples, max number of answers, 2)
  'ctx_lens',           # int32 (num samples,)
  'qtn_lens',           # int32 (num samples,)
  'num_anss'            # int32 (num samples,)
])

TokenizedText = namedtuple('TokenizedText', [
  'text',               # original text string
  'tokens',             # list of parsed tokens
  'originals',          # list of original tokens (may differ from parsed ones)
  'whitespace_afters',  # list of whitespace strings, each appears after corresponding original token in original text
])

SquadArticle = namedtuple('SquadArticle', [
  'art_id',
  'art_title_str'
])

SquadContext = namedtuple('SquadContext', [
  'art_id',
  'ctx_id',
  'tokenized'           # TokenizedText of context's text
])

SquadQuestion = namedtuple('SquadQuestion', [
  'ctx_id',
  'qtn_id',
  'tokenized',          # TokenizedText of question's text
  'ans_texts',          # list of (possibly multiple) answer text strings
  'ans_word_idxs'       # list of (answer start word index, answer end word index) tuples
])

class SquadDatasetTabular(object):
  def __init__(self):
    self.arts = {}    # art_id to SquadArticle
    self.ctxs = {}    # ctx_id to SquadContext
    self.qtns = []    # list of SquadQuestion objects
  def new_article(self, art_idx, art_title_str):
    art_id = str(art_idx)
    assert art_id not in self.arts
    self.arts[art_id] = SquadArticle(art_id, art_title_str)
    return art_id
  def new_context(self, art_id, ctx_idx, ctx_tokenized):
    ctx_id = art_id + '_' + str(ctx_idx)
    assert ctx_id not in self.ctxs
    self.ctxs[ctx_id] = SquadContext(art_id, ctx_id, ctx_tokenized)
    return ctx_id
  def new_question(self, ctx_id, qtn_id, qtn_tokenized, ans_texts, ans_word_idxs):
    self.qtns.append(
      SquadQuestion(ctx_id, qtn_id, qtn_tokenized, ans_texts, ans_word_idxs))


#######################################
# Functionality:
#######################################

def get_data(config):
  word_emb_data = read_word_emb_data(config.word_emb_data_path_prefix)
  word_strs = set()
  if config.is_train:
    trn_tab_ds = _make_tabular_dataset(
      config.tokenized_trn_json_path, word_strs, has_answers=True, max_ans_len=config.max_ans_len)
    dev_tab_ds = _make_tabular_dataset(
      config.tokenized_dev_json_path, word_strs, has_answers=True, max_ans_len=config.max_ans_len)

    word_emb_data = _contract_word_emb_data(word_emb_data, word_strs, config.learn_single_unk)
    trn_vec_ds = _make_vectorized_dataset('train', trn_tab_ds, word_emb_data, has_answers=True)
    dev_vec_ds = _make_vectorized_dataset('dev', dev_tab_ds, word_emb_data, has_answers=True)

    trn_ds = SquadDataset(trn_tab_ds, trn_vec_ds)
    dev_ds = SquadDataset(dev_tab_ds, dev_vec_ds)
    return SquadData(word_emb_data, trn_ds, dev_ds, None)
  else:
    tokenized_tst_json_path = config.tst_json_path + '.tokenized.tmp'
    tokenize_tst_json(config.tst_json_path, tokenized_tst_json_path, config.tst_split)
    tst_tab_ds = _make_tabular_dataset(
      tokenized_tst_json_path, word_strs, has_answers=False, max_ans_len=config.max_ans_len)

    word_emb_data = _contract_word_emb_data(word_emb_data, word_strs, config.learn_single_unk)
    tst_vec_ds = _make_vectorized_dataset('test', tst_tab_ds, word_emb_data, has_answers=False)

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


def write_test_predictions(ans_hats, tst_prd_json_path):
  logger = logging.getLogger()
  s = json.dumps(ans_hats, ensure_ascii=False)
  with io.open(tst_prd_json_path, 'w', encoding='utf-8') as f:
    f.write(s)
  logger.info('Written test predictions to {}'.format(tst_prd_json_path))


def _make_tabular_dataset(tokenized_json_path, word_strs, has_answers, max_ans_len=None):
  logger = logging.getLogger()
  tabular = SquadDatasetTabular()

  num_processed_samples = 0
  num_discarded_samples = 0
  answers_per_question_counter = Counter()
  with io.open(tokenized_json_path, 'r', encoding='utf-8') as f:
    j = json.load(f)
    #version = j['version']
    data = j['data']
    for article_idx, article in enumerate(data):
      art_title_str = article['title']
      art_id = tabular.new_article(article_idx, art_title_str)

      paragraphs = article['paragraphs']
      for paragraph_idx, paragraph in enumerate(paragraphs):
        ctx_str = paragraph['context']
        ctx_tokens = paragraph['tokens']
        word_strs.update(ctx_tokens)
        ctx_originals = paragraph['originals']
        ctx_whitespace_afters = paragraph['whitespace_afters']
        ctx_tokenized = TokenizedText(ctx_str, ctx_tokens, ctx_originals, ctx_whitespace_afters)
        ctx_id = tabular.new_context(art_id, paragraph_idx, ctx_tokenized)

        qas = paragraph['qas']
        for qa in qas:
          num_processed_samples += 1
          qtn_id = qa['id']

          qtn_str = qa['question']
          qtn_tokens = qa['tokens']
          word_strs.update(qtn_tokens)
          qtn_originals = qa['originals']
          qtn_whitespace_afters = qa['whitespace_afters']
          qtn_tokenized = TokenizedText(qtn_str, qtn_tokens, qtn_originals, qtn_whitespace_afters)

          ans_texts = None
          ans_word_idxs = None
          if has_answers:
            answers = qa['answers']
            assert answers
            ans_word_idxs = []
            ans_texts = []
            for answer in answers:
              ans_text = answer['text']
              ans_start_word_idx = answer['start_token_idx']
              ans_end_word_idx = answer['end_token_idx']
              if max_ans_len and ans_end_word_idx - ans_start_word_idx + 1 > max_ans_len:
                # discard answer
                continue
              ans_word_idxs.append((ans_start_word_idx, ans_end_word_idx))
              ans_texts.append(ans_text)
            if not ans_texts:
              # discard question
              num_discarded_samples += 1
              continue
            answers_per_question_counter[len(ans_texts)] += 1

          tabular.new_question(ctx_id, qtn_id, qtn_tokenized, ans_texts, ans_word_idxs)

  logger.info('Processed {:s}:\n'
    '\ttotal {:<6d} samples, discarded {:<6d}, kept {:<6d}\n'
    '\t{{x: num of questions having x answers}}: {:s}'.format(
    tokenized_json_path,
    num_processed_samples, num_discarded_samples, len(tabular.qtns),
    str(answers_per_question_counter)))

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


def _make_vectorized_dataset(name, tabular, word_emb_data, has_answers):
  num_samples = len(tabular.qtns)
  max_ctx_len = max([len(ctx.tokenized.tokens) for ctx in tabular.ctxs.values()])   # context length
  max_qtn_len = max([len(qtn.tokenized.tokens) for qtn in tabular.qtns])            # question length
  ctxs = np.zeros((num_samples, max_ctx_len), dtype=np.int32)
  qtns = np.zeros((num_samples, max_qtn_len), dtype=np.int32)
  ctx_lens = np.zeros(num_samples, dtype=np.int32)
  qtn_lens = np.zeros(num_samples, dtype=np.int32)

  if has_answers:
    # each answer made of index of first answer word, and index of last answer word
    max_num_ans = max([len(qtn.ans_word_idxs) for qtn in tabular.qtns])
    anss = np.zeros((num_samples, max_num_ans, 2), dtype=np.int32)
    num_anss = np.zeros(num_samples, dtype=np.int32)
    ans_lens = []
  else:
    anss = None
    num_anss = None

  ctx_id_to_sample_idx = {}
  for sample_idx, qtn in enumerate(tabular.qtns):
    ctx_id = qtn.ctx_id
    ctx = tabular.ctxs[ctx_id]

    if ctx_id in ctx_id_to_sample_idx:
      ctx_len = ctx_lens[ctx_id_to_sample_idx[ctx_id]]
      ctx_words = ctxs[ctx_id_to_sample_idx[ctx_id]][:ctx_len]
    else:
      ctx_words = [word_emb_data.str_to_word[word_str] for word_str in ctx.tokenized.tokens]
      ctx_len = len(ctx_words)
      ctx_id_to_sample_idx[ctx_id] = sample_idx

    ctxs[sample_idx, :ctx_len] = ctx_words
    ctx_lens[sample_idx] = ctx_len
    
    qtn_words = [word_emb_data.str_to_word[word_str] for word_str in qtn.tokenized.tokens]
    qtns[sample_idx, :len(qtn_words)] = qtn_words
    qtn_lens[sample_idx] = len(qtn_words)

    if has_answers:
      for a, (ans_start_word_idx, ans_end_word_idx) in enumerate(qtn.ans_word_idxs):
        anss[sample_idx, a, :] = [ans_start_word_idx, ans_end_word_idx]
        ans_lens.append(ans_end_word_idx - ans_start_word_idx + 1)
      num_anss[sample_idx] = len(qtn.ans_word_idxs)

  qs = [1., 2., 5.] + list(np.arange(10., 91., 10.)) + [95., 99., 100.]
  msg = 'Vectorized {} samples. Lengths:\n'.format(name) + '\n'.join([
    '\t{:<15s}{:s}'.format('percentile:', ''.join(['%-5d' % q for q in qs])),
    '\t{:<15s}{:s}'.format('ctx length:', ''.join(['%-5d' % ctx_p for ctx_p in np.percentile(ctx_lens, qs)])),
    '\t{:<15s}{:s}'.format('qtn length:', ''.join(['%-5d' % qtn_p for qtn_p in np.percentile(qtn_lens, qs)]))])
  if has_answers:
    msg += '\n\t{:<15s}{:s}'.format('ans length:', ''.join(['%-5d' % ans_p for ans_p in np.percentile(ans_lens, qs)]))
  logging.getLogger().info(msg)

  return SquadDatasetVectorized(ctxs, qtns, anss, ctx_lens, qtn_lens, num_anss)

