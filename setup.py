import argparse
import os
import sys
import io
import logging
import json
import unicodedata
import cPickle

import numpy as np

from collections import Counter, namedtuple

from base.utils import set_up_logger


#######################################
# Word embedding:
#######################################

WordEmbData = namedtuple('WordEmbData', [
  'word_emb',                 # float32 (num words, emb dim)
  'str_to_word',              # map word string to word index
  'first_known_word',         # words found in GloVe are at positions [first_known_word, first_unknown_word)
  'first_unknown_word',       # words not found in GloVe are at positions [first_unknown_word, first_unallocated_word)
  'first_unallocated_word'    # extra random embeddings
])


def _write_word_emb_data(path_prefix, word_emb_data):
  metadata_path, emb_path = _get_word_emb_data_paths(path_prefix)
  with open(metadata_path, 'wb') as f:
    cPickle.dump((word_emb_data.str_to_word, word_emb_data.first_known_word,
      word_emb_data.first_unknown_word, word_emb_data.first_unallocated_word),
      f, protocol=cPickle.HIGHEST_PROTOCOL)
  with open(emb_path, 'wb') as f:
    np.save(f, word_emb_data.word_emb)
  logging.getLogger().info('Written word embedding data:\n\t{}\n\t{}'.format(metadata_path, emb_path))


def read_word_emb_data(path_prefix):
  metadata_path, emb_path = _get_word_emb_data_paths(path_prefix)
  with open(metadata_path, 'rb') as f:
    str_to_word, first_known_word, first_unknown_word, first_unallocated_word = cPickle.load(f)
  with open(emb_path, 'rb') as f:
    word_emb = np.load(f)
  word_emb_data = WordEmbData(
    word_emb, str_to_word, first_known_word, first_unknown_word, first_unallocated_word)
  logging.getLogger().info('Read word embedding data from:\n\t{}\n\t{}'.format(metadata_path, emb_path))
  return word_emb_data
  

def _get_word_emb_data_paths(path_prefix):
  return path_prefix + '.metadata.pkl', path_prefix + '.emb.npy'


#######################################
# Preprocess GloVe:
#######################################

def _download_and_unzip_glove(glove_zip_url, glove_zip_path, glove_txt_path):
  logger = logging.getLogger()
  if os.path.isfile(glove_txt_path):
    logger.info('GloVe raw text found at {}'.format(glove_txt_path))
    return
  if not os.path.isfile(glove_zip_path):
    logger.info('Downloading GloVe')
    wget_cmd = 'wget {} -O {}'.format(glove_zip_url, glove_zip_path)
    if os.system(wget_cmd) != 0:
      logger.info('Failure executing "{}"'.format(wget_cmd))
      sys.exit(1)
  logger.info('Unzipping GloVe')
  unzip_cmd = 'unzip {} -d {}'.format(glove_zip_path, os.path.dirname(glove_txt_path))
  if os.system(unzip_cmd) != 0:
    logger.info('Failure executing "{}"'.format(unzip_cmd))
    sys.exit(1)


def _write_glove_data(glove_txt_path, glove_max_words, glove_emb_dim,
  glove_first_known_word_idx, glove_strs_path, glove_preproc_path_prefix):
  logger = logging.getLogger()

  glove_str_to_word = {}
  glove_word_emb = np.zeros((glove_max_words, glove_emb_dim), dtype=np.float32)
  dups = Counter()
  logger.info('Processing raw GloVe...')
  with io.open(glove_txt_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
      strs = line.rstrip().split(" ")
      word_str = strs[0]
      if word_str in glove_str_to_word:
        dups[word_str] += 1
      else:
        word_emb = map(float, strs[1:])
        if len(word_emb) != glove_emb_dim:
          raise AssertionError('Error. Line {}'.format(i))
        word_idx = glove_first_known_word_idx + len(glove_str_to_word)
        glove_str_to_word[word_str] = word_idx
        glove_word_emb[word_idx, :] = word_emb
      if i % 200000 == 0:
        logger.info('\tprocessed {:d} lines'.format(i))
  logger.info('\tprocessed {:d} lines in total'.format(i))
  num_glove_words = len(glove_str_to_word)
  glove_first_unknown_word_idx = glove_first_known_word_idx + num_glove_words
  glove_word_emb = glove_word_emb[:glove_first_unknown_word_idx]

  logger.info('Done. Number of GloVe words: {}'.format(num_glove_words))
  # for dup_str, dup_count in dups.most_common():
  #   print 'Duplicate word-type: [', dup_str, '] number of appearances:', dup_count
  #   for idx, c in enumerate(dup_str):
  #     print '\t', idx, hex(ord(c)), unicodedata.name(c)
  logger.info('Total number of duplicate word-types: {}'.format(len(dups)))

  glove_word_emb_data = WordEmbData(
    glove_word_emb, glove_str_to_word, glove_first_known_word_idx, glove_first_unknown_word_idx, None)
  _write_word_emb_data(glove_preproc_path_prefix, glove_word_emb_data)

  # Write a txt file listing GloVe words, read by Java program which parses
  # the train / dev / test JSON files.
  with io.open(glove_strs_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(glove_str_to_word.keys()))

  return glove_word_emb_data


#######################################
# Preprocess datasets:
#######################################

def _tokenize_json(json_path, tokenized_json_path, has_answers, glove_strs_path, split):
  split_flag = ' --split' if split else ''
  has_answers_flag = ' --has_answers' if has_answers else ''
  cmd_pattern = 'java -cp "tokenizer/:tokenizer/*" SquadTokenizer {} {} --words_txt={}{}{}'
  cmd = cmd_pattern.format(json_path, tokenized_json_path, glove_strs_path, has_answers_flag, split_flag)
  _os_exec(cmd)


def tokenize_tst_json(test_json_path, tokenized_test_json_path, split):
  _tokenize_json(test_json_path, tokenized_test_json_path, False, GLOVE_STRS_PATH, split)


def _os_exec(cmd):
  if os.system(cmd) != 0:
    logging.getLogger().info('Failure executing "{}"'.format(cmd))
    sys.exit(1)


def _add_extra_embeddings(tokenized_json_paths, old_word_emb_data, num_unallocated):
  unknown_words = set()
  for tokenized_json_path in tokenized_json_paths:
    with io.open(tokenized_json_path, 'r', encoding='utf-8') as f:
      j = json.load(f)
      j_unknown_words_array = j['unknown_words']
      # tokenizer/SquadTokenizer.java produces a "?" token for untokenizable characters, local work around:
      dups = [dup_word for dup_word, dup_count in Counter(j_unknown_words_array).iteritems() if dup_count > 1]
      if dups:
        assert len(dups) == 1 and dups[0] == '?'
      j_unknown_words = set(j_unknown_words_array)
      unknown_words.update(j_unknown_words)

  old_word_emb, old_str_to_word, old_first_known_word, _, _ = old_word_emb_data

  unknown_words_to_add = unknown_words.difference(old_str_to_word.keys())
  extra_word_embs = _get_random_embeddings(len(unknown_words_to_add) + num_unallocated, old_word_emb_data)
  word_emb = np.concatenate([old_word_emb, extra_word_embs], axis=0)

  first_unknown_word = len(old_word_emb)
  str_to_word_to_add = {s: w for w, s in enumerate(unknown_words_to_add, first_unknown_word)}
  old_str_to_word.update(str_to_word_to_add)    # inplace

  first_unallocated_word = first_unknown_word + len(unknown_words_to_add)
  return WordEmbData(word_emb, old_str_to_word, old_first_known_word, first_unknown_word, first_unallocated_word)
  

def _get_random_embeddings(num_embeddings, word_emb_data):
  known_word_emb = word_emb_data.word_emb[word_emb_data.first_known_word:word_emb_data.first_unknown_word]
  np_rng = np.random.RandomState(123)
  rnd_idxs = np_rng.permutation(len(known_word_emb))[:100000]
  known_subset = known_word_emb[rnd_idxs]
  known_mean, known_cov = np.mean(known_subset, axis=0), np.cov(known_subset, rowvar=0)
  unknown_word_embs = np_rng.multivariate_normal(mean=known_mean, cov=known_cov, size=num_embeddings).astype(np.float32)
  return unknown_word_embs


#######################################
# Misc:
#######################################

def _write_dummy_tst_json(dev_json_path, dummy_test_json_path):
  change_char_ps = [1 - 1e-5, 1e-5]
  with io.open(dev_json_path, 'r', encoding='utf-8') as f:
    j = json.load(f)
    for article in j['data']:
      for paragraph in article['paragraphs']:
        paragraph['context'] = ''.join(
          [(c if np.random.choice([True, False], p=change_char_ps) else '?') for c in paragraph['context']])
        for qa in paragraph['qas']:
          qa['question'] = ''.join(
            [(c if np.random.choice([True, False], p=change_char_ps) else '?') for c in qa['question']])
          del qa['answers']
  with io.open(dummy_test_json_path, 'w', encoding='utf-8') as f:
    f.write(unicode(json.dumps(j, ensure_ascii=False)))
  logging.getLogger().info('Written dummy test JSON to {}'.format(dummy_test_json_path))


#######################################
# Program:
#######################################

def _print_title(s):
  logging.getLogger().info('\n' + 30*'#' + '\n# {}\n'.format(s) + 30*'#' + '\n')


GLOVE_ZIP_URL = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
GLOVE_ZIP_PATH = 'data/glove.840B.300d.zip'
GLOVE_TXT_PATH = 'data/glove.840B.300d.txt'
GLOVE_MAX_WORDS = 2200000
GLOVE_EMB_DIM = 300
GLOVE_FIRST_KNOWN_WORD_IDX = 2     # reserve two first entries of word embedding matrix
GLOVE_STRS_PATH = 'data/glove.840B.300d_strs.txt'
GLOVE_PREPROC_PATH_PREFIX = 'data/preprocessed_glove'
TRN_JSON_PATH = 'data/train-v1.1.json'
DEV_JSON_PATH = 'data/dev-v1.1.json'
DUMMY_TST_JSON_PATH = 'data/dummy-test-v1.1.json'
GLOVE_PREPROC_WITH_UNKS_PATH_PREFIX = 'data/preprocessed_glove_with_unks'


def _main(split, num_extra_embeddings, write_dummy_test):
  _print_title('Preprocessing GloVe')

  _download_and_unzip_glove(GLOVE_ZIP_URL, GLOVE_ZIP_PATH, GLOVE_TXT_PATH)

  glove_word_emb_data = _write_glove_data(
    GLOVE_TXT_PATH, GLOVE_MAX_WORDS, GLOVE_EMB_DIM,
    GLOVE_FIRST_KNOWN_WORD_IDX, GLOVE_STRS_PATH, GLOVE_PREPROC_PATH_PREFIX)

  _print_title('Tokenizing JSON')

  filename_suffix = '.tokenized.split.json' if split else '.tokenized.json'
  tokenized_trn_json_path = TRN_JSON_PATH.replace('.json', filename_suffix)
  tokenized_dev_json_path = DEV_JSON_PATH.replace('.json', filename_suffix)
  _tokenize_json(TRN_JSON_PATH, tokenized_trn_json_path, True, GLOVE_STRS_PATH, split)
  _tokenize_json(DEV_JSON_PATH, tokenized_dev_json_path, True, GLOVE_STRS_PATH, split)

  _print_title('Adding random embeddings for unknown words')

  glove_with_unks_word_emb_data = _add_extra_embeddings(
    [tokenized_trn_json_path, tokenized_dev_json_path], glove_word_emb_data, num_extra_embeddings)
  _write_word_emb_data(
    GLOVE_PREPROC_WITH_UNKS_PATH_PREFIX + ('.split' if split else ''), glove_with_unks_word_emb_data)

  if write_dummy_test:
    _print_title('Writing dummy test JSON')
    _write_dummy_tst_json(DEV_JSON_PATH, DUMMY_TST_JSON_PATH)


if __name__ == '__main__':
  logger = set_up_logger(log_filename=None, datetime=False)
  parser = argparse.ArgumentParser()
  parser.add_argument('--split',
    help='whether to split unknown hyphenated words which have a known constituent token', action='store_false')
  parser.add_argument('--num_extra_embeddings',
    help='number of extra random embeddings to produce', type=int, default=100000)
  parser.add_argument('--write_dummy_test',
    help='whether to write a dummy test JSON file', action='store_true')
  args = parser.parse_args()
  logger.info('\n' + str(args))
  _main(args.split, args.num_extra_embeddings, args.write_dummy_test)

