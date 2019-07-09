import os
from os.path import join
from gensim.models import Word2Vec, FastText
import re
import pickle
import numpy as np
import config as c


def get_models_names(path, m_type):
    """

    :param path: string. path of the files.
    :param m_type: string. model type ('2.01', '2.02', '2.03')
    :return: List. models of selected type.
    """
    return [f for f in os.listdir(path) if re.match(r".*_model_" + m_type + r"\.model$", f)]


def load_model(path, m_type, name):
    full_name = name + '_model_' + m_type + '.model'
    if m_type in ['2.01', '2.02', '2.03']:
        return Word2Vec.load(join(path, full_name))
    return FastText.load(join(path, full_name))


def load_tfidf(path, name):
    full_name = 'tf_idf_' + name
    if c.MODEL_TYPE == '2.02':
        full_name = full_name + '_model_' + '2.02' + '.model'
    with open(join(path, full_name + '.pickle'), 'rb') as handle:
        tfidf = pickle.load(handle)
    return tfidf


def filter_pairs(lst):
    print('filter pairs')
    # lst_2 = [(x[1], x[2]) for x in lst]
    with open(join(c.combinations_path, 'lst_2' + '.pickle'), 'rb') as handle:
        lst_2 = pickle.load(handle)
    to_filter = []
    with open(join(c.vocab_distr_path, 'pairs_found' + '.txt')) as afile:
        for s in afile:
            s = s.split("\'")
            to_filter.append((s[1], s[3]))
            to_filter.append((s[3], s[1]))
    with open(join(c.combinations_path, 'to_filter_' + str(len(lst)) + '_' + c.MODEL_TYPE + '.pickle'), 'wb') as handle:
        pickle.dump(to_filter, handle, protocol=pickle.HIGHEST_PROTOCOL)
    to_keep = set(lst_2) - set(to_filter)
    n_lst = []
    for i, (m1, m2) in enumerate(to_keep):
        n_lst = n_lst + [(i + 1, m1, m2)]
    print(f"original: {len(lst)}, to filter: {len(to_filter)/2}, filtered: {len(n_lst)}")
    return n_lst







