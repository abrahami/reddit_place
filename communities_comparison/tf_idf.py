from os.path import join
import sys
import platform as p
from communities_comparison.utils import get_models_names, load_model
import config as c
import pickle
import math
import multiprocessing as mp
import numpy as np
import pandas as pd
from IPython.display import display


def add_w_count(model, all_wc):
    """

    :param model: Gensim model- to count its words' frequency
    :param all_wc: Dict- the cumulative frequency of words of all models
    :return: Dict- the updated cumulative frequency of words of all models
    """
    wc = dict()
    for (k, v) in model.wv.vocab.items():
        wc[k] = v.count
    all_wc = {k: all_wc.get(k, 0) + wc.get(k, 0) for k in set(all_wc) | set(wc)}
    return all_wc


def add_wd_count(model, all_wdc, n_wc):
    """

    :param model: Gensim model- to retrieve its words
    :param all_wdc: Dict- the cumulative count of documents containing each word (including words from all models)
    :param n_wc: file name for wc
    :return: Dict- the updated cumulative count of documents containing each word
    """
    dc = dict()
    wc = dict()
    for (k, v) in model.wv.vocab.items():
        dc[k] = 1  # add 1 document count to the words appear in the current document (community)
        wc[k] = v.count  # add 1 document count to the words appear in the current document (community)
    with open(join(c.tf_idf_path, n_wc + '.pickle'), 'wb') as handle:
        pickle.dump(wc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    all_wdc = {k: all_wdc.get(k, 0) + dc.get(k, 0) for k in set(all_wdc) | set(dc)}
    return all_wdc


def calc_idf(m_names, m_type):
    """

    :param m_names: List. names of the models to load.
    :param m_type: string. model type ('2.01', '2.02', '2.03').
    :return: Dict. idf score for each word in the corpus (words from all communities).
    """
    idf_name = 'idf_dict_m_type_' + m_type
    if c.CALC_IDF:
        N = len(m_names)
        total_wdc = dict()  # document-word count in all documents (communities)
        for i, m_name in enumerate(m_names):
            if i % 50 == 0:
                print(f" i: {i}")
            curr_model = load_model(path=c.data_path, m_type=m_type, name=m_name)
            n_wc = 'wc_' + m_name
            total_wdc = add_wd_count(model=curr_model, all_wdc=total_wdc, n_wc=n_wc)  # save dict of wc

        # idf (inverse document frequency):  idf(t, D) = log(N / |{d in D : t in T}|)
        #      dividing the total number of documents by the number of documents containing the term,
        #      and then taking the logarithm of that quotient
        idf = {k: math.log(N/v) for (k, v) in total_wdc.items()}
        with open(join(c.tf_idf_path, idf_name + '.pickle'), 'wb') as handle:
            pickle.dump(idf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(join(c.tf_idf_path, idf_name + '.pickle'), 'rb') as handle:
            idf = pickle.load(handle)
    return idf


# def calc_tf_idf(wc, m_f_name, idf):
#     """
#
#     :param wc: Dict. words count of the model to calc its tf-idf.
#     :param m_f_name: String. name for saving model's tf-idf file.
#     :param idf: Dict. idf score for each word in the corpus (words from all communities).
#     :return: None.  save 'tf-idf' vector per community.
#     """
#     # tf (term frequency):  tf(t,d)= f[t,d] (the raw count of term t in document d)
#     # tf_idf(t,d,D) = tf(t,d) * idf(t,D)
#
#     tf_idf = {k: v * idf.get(k) for (k, v) in wc.items()}
#     with open(join(c.tf_idf_path, m_f_name + '.pickle'), 'wb') as handle:
#         pickle.dump(tf_idf, handle, protocol=pickle.HIGHEST_PROTOCOL)


def calc_tf_idf(m_name, idf):
    """

    :param m_name: String. model's name.
    :param idf: Dict. idf score for each word in the corpus (words from all communities).
    :return: None.  save 'tf-idf' vector per community.
    """
    # tf (term frequency):  tf(t,d)= f[t,d] (the raw count of term t in document d)
    # tf_idf(t,d,D) = tf(t,d) * idf(t,D)

    wc_f_name = 'wc_' + m_name
    with open(join(c.tf_idf_path, wc_f_name + '.pickle'), 'rb') as handle:
        wc = pickle.load(handle)

    tf_idf = {k: v * idf.get(k) for (k, v) in wc.items()}
    m_f_name = 'tf_idf_' + m_name
    with open(join(c.tf_idf_path, m_f_name + '.pickle'), 'wb') as handle:
        pickle.dump(tf_idf, handle, protocol=pickle.HIGHEST_PROTOCOL)


def calc_tf_idf_all_models(m_names, m_type):
    print("calc idf")
    idf = calc_idf(m_names=m_names, m_type=m_type)
    lst = []
    for m in m_names:
        lst = lst + [(m, idf)]
    print("calc tf-idf")
    pool = mp.Pool(processes=c.CPU_COUNT)
    with pool as pool:
        pool.starmap(calc_tf_idf, lst)

    # for i, m_name in enumerate(m_names):
    #     if i % 100 == 0:
    #         print(f" i: {i}, model name: {m_name}")
    #     wc_f_name = 'wc_' + m_name
    #     with open(join(c.tf_idf_path, wc_f_name + '.pickle'), 'rb') as handle:
    #         wc = pickle.load(handle)
    #     m_f_name = 'tf_idf_' + m_name
    #     calc_tf_idf(wc=wc, m_f_name=m_f_name, idf=idf)
    return


def get_vocab_length(m_name, m_type):
    model = load_model(path=c.data_path, m_type=m_type, name=m_name)
    res = {'m_name': m_name, 'vocab_length': len(model.wv.index2entity)}
    return res


def calc_vocab_distribution(m_names):
    vd_name = 'vocab_length_distribution'

    if c.CALC_VOCAB_DISTR:
        vocab_d = pd.DataFrame(columns=['m_name', 'vocab_length'])
        lst = []
        for m in m_names:
            lst = lst + [(m, c.MODEL_TYPE)]
        print("calc vocab distribution")
        pool = mp.Pool(processes=c.CPU_COUNT)
        with pool as pool:
            results = pool.starmap(get_vocab_length, lst)

        for res in results:
            vocab_d = vocab_d.append(res, ignore_index=True)

        p_lst = list(range(0, 100, 5))
        df = pd.DataFrame(data={'percentile': p_lst, 'vocab_length': np.percentile(a=vocab_d['vocab_length'], q=p_lst)})
        display(df)

        with open(join(c.vocab_distr_path, vd_name + '.pickle'), 'wb') as handle:
            pickle.dump(vocab_d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(join(c.vocab_distr_path, vd_name + '.pickle'), 'rb') as handle:
            vocab_d = pickle.load(handle)

    # keep only models with vocab length above threshold (a certain percentile from distribution)
    min_vocab_length = np.percentile(a=vocab_d['vocab_length'], q=c.VOCAB_PERC_THRES)
    valid_vocab_d = vocab_d[vocab_d['vocab_length'] > min_vocab_length]
    return valid_vocab_d['m_name']






