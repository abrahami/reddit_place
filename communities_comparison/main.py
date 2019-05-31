# Authors: Shani Cohen (shanisa)
# Python version: 3.7
# Last update: 29.5.2019

import sys
if sys.platform == 'linux':
    sys.path.append('/data/home/shanisa/reddit_place')
from communities_comparison.utils import get_models_names
from communities_comparison.tf_idf import calc_tf_idf_all_models
from communities_comparison.compare import calc_scores_all_models
import config as c
import time
import os
from os.path import join
import numpy as np
import pandas as pd
import commentjson
import datetime
import sys
import pickle
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, FastText
import re


# ###################################################### Configurations ##################################################
# config_dict = commentjson.load(open(os.path.join(os.getcwd(), 'srs_words_emnedding_config.json')))
# machine = 'yalla' if sys.platform == 'linux' else os.environ['COMPUTERNAME']
# data_path = config_dict['data_dir'][machine]
# ########################################################################################################################

# # tsne_plot(model_gameswap)
# # tsne_plot(model_babymetal)


if __name__ == "__main__":
    print(f"MODEL_TYPE = {c.MODEL_TYPE}")
    print(f"COMMUNITIES TO COMPARE = {c.TOP}")
    m_names = get_models_names(path=c.data_path, m_type=c.MODEL_TYPE)
    m_names = sorted(m_names)[:c.TOP]

    print(f"\n1 - CALC_TF_IDF -> {c.CALC_TF_IDF}")
    start_1 = time.time()
    if c.CALC_TF_IDF:
        calc_tf_idf_all_models(m_names=m_names, m_type=c.MODEL_TYPE)
    print(f"CALC_TF_IDF - elapsed time (min): {(time.time()-start_1)/60}")

    print(f"\n2 - CALC_SCORES -> {c.CALC_SCORES}")
    start_2 = time.time()
    if c.CALC_SCORES:
        calc_scores_all_models(m_names=m_names, m_type=c.MODEL_TYPE)
    print(f"CALC_SCORES - elapsed time (min): {(time.time()-start_2)/60}")

    print(f"\n3 - VISUALIZATION -> {c.VISUALIZATION}")
    if c.VISUALIZATION:
        calc_scores_all_models(m_names=m_names, m_type=c.MODEL_TYPE)

