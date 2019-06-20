# Authors: Shani Cohen (ssarusi)
# Python version: 3.7
# Last update: 14.6.2019

import sys
if sys.platform == 'linux':
    sys.path.append('/data/home/shanisa/reddit_place')
print(f"\nCurrent platform: {sys.platform}")
from communities_comparison.utils import get_models_names
from communities_comparison.tf_idf import calc_tf_idf_all_models, calc_vocab_distribution
from communities_comparison.compare import calc_scores_all_models
from communities_comparison.clustering import cluster_communities, enrich_data
import config as c
import time
from os.path import join
import numpy as np
import pickle
import datetime as dt
import random as rn

np.random.seed(c.SEED)
rn.seed(c.SEED)


if __name__ == "__main__":
    print(f"start time {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"MODEL_TYPE = {c.MODEL_TYPE}")
    print(f"VOCAB_PERC_THRES = {c.VOCAB_PERC_THRES}")
    print(f"COMMUNITIES TO COMPARE = {c.N}")
    print(f"METHOD = {c.METHOD}")
    if c.AD_HOC_NAMES:
        print(f"AD_HOC_NAMES = {c.AD_HOC_NAMES}")
        valid_m_names = c.AD_HOC_NAMES
    else:
        m_names = get_models_names(path=c.data_path, m_type=c.MODEL_TYPE)
        # m_names = sorted(m_names)[:c.N]
        if c.APPLY_VOCAB_THRES:
            valid_m_names = calc_vocab_distribution(m_names=m_names)
        # valid_m_names = rn.sample(list(valid_m_names), c.N)

    print(f"\n1 - CALC_TF_IDF -> {c.CALC_TF_IDF}")
    start_1 = time.time()
    if c.CALC_TF_IDF:
        calc_tf_idf_all_models(m_names=valid_m_names, m_type=c.MODEL_TYPE)
    print(f"CALC_TF_IDF - elapsed time (min): {(time.time()-start_1)/60}")

    print(f"\n2 - CALC_SCORES -> {c.CALC_SCORES}")
    print(f"2.1 - SAVE_DIS_MATRIX -> {c.SAVE_DIS_MATRIX}")
    start_2 = time.time()
    if c.CALC_SCORES:
        calc_scores_all_models(m_names=valid_m_names, m_type=c.MODEL_TYPE)
    else:
        with open(join(c.scores_path, c.metrics_f_name + '.pickle'), 'rb') as handle:
            metrics = pickle.load(handle)
        metrics.to_csv(join(c.scores_path, c.metrics_f_name + '.csv'))
    print(f"CALC_SCORES - elapsed time (min): {(time.time()-start_2)/60}")

    print(f"\n3 - ENRICH_DATA -> {c.ENRICH_DATA}")
    start_3 = time.time()
    if c.ENRICH_DATA:
        metrics = enrich_data(df=metrics, comms=valid_m_names)

    print(f"ENRICH_DATA - elapsed time (min): {(time.time() - start_3) / 60}")

    print(f"\n4 - CLUSTERING -> {c.CLUSTERING}")
    start_4 = time.time()
    if c.CLUSTERING:
        cluster_communities(df=metrics)

    print(f"CLUSTERING - elapsed time (min): {(time.time() - start_4) / 60}")


