# coding:utf8
import warnings

class DefaultConfig(object):
    raw_data_root = './data/raw_data/'
    processed_data_root = './data/processed_data/'
    ppd_data = './data/ppd_data/'
    model = './model'

    MAX_NB_WORDS = 300000
    MAX_NB_CHARS = 300000

    EMBEDDING_DIM = 300

    MAX_WORD_SEQUENCE_LENGTH = 20
    MAX_CHAR_SEQUENCE_LENGTH = 35

    NB_CUDA_VISIBLE_DEVICES = '0'

    is_use_word = True

opt = DefaultConfig()