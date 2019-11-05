import os
from os.path import join
from gensim.models import Word2Vec, FastText
import re
import pickle
import numpy as np


def get_models_names(config_dict):
    """

    :param path: string. path of the files.
    :return: List. models of selected type.
    """
    model_type = config_dict['general_config']['model_type']
    path = config_dict['data_path'][config_dict['machine']]
    lst = [f for f in os.listdir(path) if re.match(r".*_model_" + model_type + r"\.model$", f)]
    return [x.replace('_model_' + model_type + '.model', '') for x in lst]


def load_model(path, name, config_dict):
    """

    :param path:
    :param m_type:
    :param name:
    :return:
    """
    full_name = name + '_model_' + config_dict['general_config']['model_type'] + '.model'
    if config_dict['general_config']['embedding_model_name'] == 'w2v':
        return Word2Vec.load(join(path, full_name))
    elif config_dict['general_config']['embedding_model_name'] == 'fastext':
        return FastText.load(join(path, full_name))
    else:
        raise IOError("'embedding_model_name' (in the config file) is illegal - must be 'w2v' or 'fastext'")


def load_tfidf(path, name):
    """

    :param path:
    :param name:
    :return:
    """
    full_name = 'tf_idf_' + name
    with open(join(path, full_name + '.pickle'), 'rb') as handle:
        tfidf = pickle.load(handle)
    return tfidf


def filter_pairs(lst, config_dict):
    """
    filter out part of the pairs in lst (filters the ones which are in the 'pairs_found.txt' file
    :param lst:
    :return:
    """
    print('filter pairs')
    lst_2 = [(x[1], x[2]) for x in lst]
    # with open(join(config_dict['combinations_path'][config_dict['machine']], 'lst_2' + '.pickle'), 'rb') as handle:
    #     lst_2 = pickle.load(handle)
    to_filter = []
    with open(join(config_dict['vocab_distr_path'][config_dict['machine']], 'pairs_found' + '.txt')) as afile:
        for s in afile:
            s = s.split("\'")
            to_filter.append((s[1], s[3]))
            to_filter.append((s[3], s[1]))
    with open(join(config_dict['combinations_path'][config_dict['machine']], 'to_filter_' + str(len(lst)) + '_' + config_dict['general_config']['model_type'] + '.pickle'), 'wb') as handle:
        pickle.dump(to_filter, handle, protocol=pickle.HIGHEST_PROTOCOL)
    to_keep = set(lst_2) - set(to_filter)
    n_lst = []
    for i, (m1, m2) in enumerate(to_keep):
        n_lst = n_lst + [(i + 1, m1, m2)]
    print(f"original: {len(lst)}, to filter: {len(to_filter)/2}, filtered: {len(n_lst)}")
    return n_lst
