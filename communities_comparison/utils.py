import os
from os.path import join
from gensim.models import Word2Vec, FastText
import re


def get_models_names(path, m_type):
    """

    :param path: string. path of the files.
    :param m_type: string. model type ('1', '2')
    :return: List. models of selected type.
    """
    return [f for f in os.listdir(path) if re.match(r".*_model_" + m_type + r".*\.model$", f)]


def load_model(path, m_type, name):
    if m_type == '1':
        return Word2Vec.load(join(path, name))
    elif m_type == '2':
        return FastText.load(join(path, name))