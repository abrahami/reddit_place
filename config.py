from os.path import join, sep
from os import getcwd
import os
import platform as p


LABELS_VERSION = 'V3'
# MODEL_TYPE = '2.02'
MODEL_TYPE = '2.03'
N = 1565
# N = 200
VOCAB_PERC_THRES = 20
# Word2Vec
METHOD = 'intersection'
# FastText
# METHOD = 'union'
CPU_COUNT = 50  # mp.cpu_count()
SEED = 7

# region Params PATH
if p.system() == 'Windows':
    if 'ssarusi' in getcwd():
        data_path = join('C:', sep, 'Users', 'ssarusi', 'Desktop', 'second_degree', 'semester_d', 'NLP', 'data')
        users_data_path = join('C:', sep, 'Users', 'ssarusi', 'Desktop', 'second_degree', 'semester_d', 'NLP', 'data')
        vocab_distr_path = join('C:', sep, 'Users', 'ssarusi', 'Desktop', 'second_degree', 'semester_d', 'NLP',
                                'vocab_distribution')
        tf_idf_path = join('C:', sep, 'Users', 'ssarusi', 'Desktop', 'second_degree', 'semester_d', 'NLP', 'tf_idf')
        scores_path = join('C:', sep, 'Users', 'ssarusi', 'Desktop', 'second_degree', 'semester_d', 'NLP', 'scores')
        dis_path = join('C:', sep, 'Users', 'ssarusi', 'Desktop', 'second_degree', 'semester_d', 'NLP', 'distances')
        combinations_path = join('C:', sep, 'Users', 'ssarusi', 'Desktop', 'second_degree', 'semester_d', 'NLP', 'combinations')
else:
    data_path = join('C:', sep, 'data', 'work', 'data', 'reddit_place', 'embedding', 'embedding_per_sr', MODEL_TYPE)
    users_data_path = join('C:', sep, 'data', 'work', 'data', 'reddit_place')
    vocab_distr_path = join('C:', sep, 'data', 'home', 'shanisa', 'project_data', 'vocab_distribution', MODEL_TYPE)
    tf_idf_path = join('C:', sep, 'data', 'home', 'shanisa', 'project_data', 'tf_idf', MODEL_TYPE, 'N_' + str(N))
    scores_path = join('C:', sep, 'data', 'home', 'shanisa', 'project_data', 'scores', MODEL_TYPE, 'N_' + str(N))
    dis_path = join('C:', sep, 'data', 'home', 'shanisa', 'project_data', 'distances', MODEL_TYPE, 'N_' + str(N))
    combinations_path = join('C:', sep, 'data', 'home', 'shanisa', 'project_data', 'combinations')
    labels_path = join('C:', sep, 'data', 'home', 'shanisa', 'project_data', 'labels')
    for p in [vocab_distr_path, tf_idf_path, scores_path, dis_path, combinations_path, labels_path]:
        if not os.path.exists(p):
            os.makedirs(p)
# endregion

# AD_HOC_NAMES = ['patriots', 'greenbaypackers', 'womenfortrump']
# AD_HOC_NAMES = ['guitar', 'music', 'spongebob']
AD_HOC_NAMES = None

APPLY_VOCAB_THRES, CALC_VOCAB_DISTR = False, False
LOAD_VALID_NAMES = True
CALC_COMBINATIONS, FILTER_PAIRS = False, True
CALC_TF_IDF, CALC_IDF = False, False
CALC_SCORES, CALC_DIS, SAVE_DIS_MATRIX = True, True, False
# CALC_SCORES, CALC_DIS, SAVE_DIS_MATRIX = False, False, False


# import pickle
# import numpy as np
#
# with open(join(vocab_distr_path, 'vocab_length_distribution' + '.pickle'), 'rb') as handle:
#     vocab_d = pickle.load(handle)
# min_vocab_length = np.percentile(a=vocab_d['vocab_length'], q=VOCAB_PERC_THRES)
# print()
