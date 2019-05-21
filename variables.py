import re

DATA_SEQUENCE = ['token', 'category', 'flag', 'interval']

TOKEN_PAD = '<PAD>'  # Token for padding
TOKEN_UNK = '<UNK>'  # Token for unknown words
TOKEN_MASK = '<MASK>'  # Token for masking

VALUE_PAD = 0
VALUE_UNK = 1
VALUE_MASK = 2

MODEL_FILE_FORMAT = 'weights.{epoch:02d}-{val_loss:.2f}.h5'
MODEL_REGEX_PATTERN = re.compile(r'^.*weights\.(\d+)\-\d+\.\d+\.h5$')
LAST_MODEL_FILE_FORMAT = 'last.h5'
