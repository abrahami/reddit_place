import os
from os.path import join
from gensim.models import Word2Vec, FastText
import re
import pickle


def get_models_names(path, m_type):
    """

    :param path: string. path of the files.
    :param m_type: string. model type ('2.01', '2.02', '2.03')
    :return: List. models of selected type.
    """
    return [f for f in os.listdir(path) if re.match(r".*_model_" + m_type + r"\.model$", f)]


def load_model(path, m_type, name):
    if m_type in ['2.01', '2.02', '2.03']:
        return Word2Vec.load(join(path, name))
    return FastText.load(join(path, name))


def load_tfidf(path, name):
    f_name = 'tf_idf_' + name.replace('_1', '_2')
    with open(join(path, f_name + '.pickle'), 'rb') as handle:
        tfidf = pickle.load(handle)
    return tfidf


