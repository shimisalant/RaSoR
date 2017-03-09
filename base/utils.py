
import logging
import sys
import os


def namer(prefix):
  return lambda suffix: '%s_%s' % (prefix, suffix)


def set_up_logger(log_filename=None, datetime=True):
  LOG_LEVEL = logging.INFO
  log_format = '%(asctime)s    %(message)s' if datetime else '%(message)s' 
  logger = logging.getLogger()
  logger.setLevel(LOG_LEVEL)
  console_handler = logging.StreamHandler(sys.stdout)
  console_handler.setFormatter(logging.Formatter(log_format))
  console_handler.setLevel(LOG_LEVEL)
  logger.addHandler(console_handler)
  if log_filename:
    verify_dir_exists(log_filename)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.setLevel(LOG_LEVEL)
    logger.addHandler(file_handler)
  return logger


def verify_dir_exists(filename):
  if not os.path.exists(os.path.dirname(filename)):
    try:
      os.makedirs(os.path.dirname(filename))
    except OSError as exc: # Guard against race condition
      if exc.errno != errno.EEXIST:
        raise

