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
import multiprocessing as mp
import os
from os.path import join
import numpy as np
import pandas as pd
import commentjson
import datetime
import sys
import pickle
import datetime as dt
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, FastText
import re
import random as rn

np.random.seed(c.SEED)
rn.seed(c.SEED)


if __name__ == "__main__":
    print(f"start time {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"MODEL_TYPE = {c.MODEL_TYPE}")
    print(f"COMMUNITIES TO COMPARE = {c.N}")
    m_names = get_models_names(path=c.data_path, m_type=c.MODEL_TYPE)
    # m_names = sorted(m_names)[:c.N]
    m_names = rn.sample(m_names, c.N)

    print(f"\n1 - CALC_TF_IDF -> {c.CALC_TF_IDF}")
    start_1 = time.time()
    if c.CALC_TF_IDF:
        calc_tf_idf_all_models(m_names=m_names, m_type=c.MODEL_TYPE)
    print(f"CALC_TF_IDF - elapsed time (min): {(time.time()-start_1)/60}")

    print(f"\n2 - CALC_SCORES -> {c.CALC_SCORES}")
    start_2 = time.time()
    if c.CALC_SCORES:
        calc_scores_all_models(m_names=m_names, m_type=c.MODEL_TYPE)
    else:
        metrics_f_name = 'metrics_m_type_' + c.MODEL_TYPE
        with open(join(c.scores_path, metrics_f_name + '.pickle'), 'rb') as handle:
            metrics = pickle.load(handle)
        metrics.to_csv(join(c.scores_path, metrics_f_name + '.csv'))
    print(f"CALC_SCORES - elapsed time (min): {(time.time()-start_2)/60}")

