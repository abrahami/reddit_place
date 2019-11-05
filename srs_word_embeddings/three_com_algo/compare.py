import platform as p
# if p.system() == 'Windows':
from three_com_algo.utils import load_model, load_tfidf, filter_pairs
from os.path import join
import numpy as np
from itertools import combinations
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models.keyedvectors import WordEmbeddingSimilarityIndex
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from operator import itemgetter
import pickle
import time
import multiprocessing as mp
import datetime as dt
import traceback


def generate_weights(model):
    """

    :param model:
    :return:
    """
    words = model.wv.index2entity
    r = np.random.random(len(words))
    weights = r/r.sum()
    return dict(zip(words, weights))


def calc_vec_distances(name, vectors_matrix, metric, config_dict):
    """
    calc the distance between vectors representing words (for each SR seperatly
    :param name: String. file name for distances matrix
    :param vectors_matrix: Numpy array. he vectors to calc distances between them
    :param metric: String. distance metric.
    :return: Numpy array - distances matrix.
    """
    if eval(config_dict['current_run_flags']['calc_dis']):
        dis_matrix = pdist(X=vectors_matrix, metric=metric)
        if eval(config_dict['current_run_flags']['save_dis_matrix']):
            with open(join(config_dict['dis_path'][config_dict['machine']], name + '.pickle'), 'wb') as handle:
                pickle.dump(dis_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(join(config_dict['dis_path'][config_dict['machine']], name + '.pickle'), 'rb') as handle:
            dis_matrix = pickle.load(handle)
    return dis_matrix


def get_selected_weights(w_dict, keys_lst, normalize, config_dict):
    """

    :param w_dict: dict
        the full dictionary of the SR (including words we don't really care about in a specific comparison
    :param keys_lst: list
        includes the desired words to take into account (either the intersected words between SRs or unions between them)
    :param normalize:
    :return:
    """
    # case we have to add to the dictionary some new words which belong the compared SR - hence zero value words are added
    if config_dict['algo_parmas']['method'] == 'union':
        new_w = np.setdiff1d(keys_lst, np.array(list(w_dict.keys())), assume_unique=True)
        w_dict.update(dict(zip(new_w, np.zeros(len(new_w)))))  # add new words to dict with weight=0
    v = np.array(itemgetter(*keys_lst)(w_dict))
    if normalize:
        return v/v.sum()
    return v


def calc_pairwise_weights(arr, f_name, normalize, calc=False):
    """

    :param arr: Numpy array. array of positive weights.
    :param f_name: String. file name for the pairwise_weights (save/load).
    :param normalize: whether to normalize the weights vector.
    :param calc: Boolean.
    :return: condensed matrix of max weight for each two elements in arr.
    """
    if calc:
        n = len(arr)
        N = n * (n - 1) // 2
        idx = np.concatenate(([0], np.arange(n - 1, 0, -1).cumsum()))
        start, stop = idx[:-1], idx[1:]
        pairs_w = np.empty(N, dtype=float)
        for j, i in enumerate(range(n - 1)):
            s0, s1 = start[j], stop[j]
            curr = np.array([np.repeat(arr[i], len(arr[i + 1:])), np.array(arr[i + 1:])])
            # taking the maximum out of the two
            pairs_w[s0:s1] = np.amax(curr, axis=0)
    #     with open(join(c.tf_idf_path, f_name + '.pickle'), 'wb') as handle:
    #         pickle.dump(pairs_w, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # else:
    #     with open(join(c.tf_idf_path, f_name + '.pickle'), 'rb') as handle:
    #         pairs_w = pickle.load(handle)

    if normalize:
        return pairs_w/pairs_w.sum()
    return pairs_w


def keep_top_weights(arr, top_perc):
    """

    :param arr: Numpy array- original weights.
    :param top_perc: Int - the percentage of top (highest) weights to keep.
    :return: Numpy array- new weights after selecting top.
    """
    min_thres = np.percentile(arr, 100-top_perc)
    arr_t = np.where(arr < min_thres, 0, arr)
    # normalizing the vector (those which are at the top 'top_perc')
    return arr_t/arr_t.sum()


def calc_distance_between_comm(d1, d2, w):
    """

    :param d1: condensed matrix of distances between intersection words - community 1
    :param d2: condensed matrix of distances between intersection words - community 2
    :param w: condensed matrix of weights for pairs of intersection words
    Note: words' order should be identical in d1, d2, w

    :return: float [0, 2] - distance between community 1 and 2 = weighted average of absolute difference between intersection words
    """
    abs_d = abs(d1 - d2)
    return np.matmul(abs_d, w.T)


def compare(i, n_model_1, n_model_2, config_dict):
    """

    :param i: Int. current iteration.
    :param n_model_1: String. name of model_1.
    :param n_model_2: String. name of model_2.
    :return:
    """
    try:
        start = time.time()
        tf_idf_path = config_dict['tf_idf_path'][config_dict['machine']]
        data_path = config_dict['data_path'][config_dict['machine']]
        model_type = config_dict['general_config']['model_type']
        weights_1 = load_tfidf(path=tf_idf_path, name=n_model_1)
        weights_2 = load_tfidf(path=tf_idf_path, name=n_model_2)
        model_1 = load_model(path=data_path, name=n_model_1, config_dict=config_dict)
        model_2 = load_model(path=data_path, name=n_model_2, config_dict=config_dict)

        # 1- word vectors
        wv_1, wv_2 = model_1.wv, model_2.wv

        # 2- intersection and union
        intersec = np.intersect1d(wv_1.index2entity, wv_2.index2entity)
        wc1, wc2, wc_inter = len(wv_1.index2entity), len(wv_2.index2entity), len(intersec)
        method = config_dict['algo_parmas']['method']
        if method == 'intersection':
            w_lst = list(intersec)
        elif method == 'union':
            union = np.union1d(wv_1.index2entity, wv_2.index2entity)
            # inter_idx = np.in1d(union, intersec)
            w_lst = list(union)
        else:
            raise IOError("method must be either 'intersection' or 'union'")
        wv_1_selected, wv_2_selected = wv_1[w_lst], wv_2[w_lst]

        # 3- calc vectors distances (within each SR separately)
        dis_metric = 'cosine'
        name_1 = n_model_1 + '_' + method + '_' + n_model_2
        name_2 = n_model_2 + '_' + method + '_' + n_model_1
        dis_1 = calc_vec_distances(name=name_1, vectors_matrix=wv_1_selected, metric=dis_metric,
                                   config_dict=config_dict)
        dis_2 = calc_vec_distances(name=name_2, vectors_matrix=wv_2_selected, metric=dis_metric,
                                   config_dict=config_dict)

        # 3.1 - find indexes of intersection
        # union = np.union1d(wv_1.index2entity, wv_2.index2entity)
        # inter_idx = np.in1d(union, intersec)
        # inter_dis_idx = pdist(X=inter_idx.reshape(-1, 1), metric=lambda u, v: np.logical_and(u, v))
        # dis_1, dis_2 = dis_1[inter_dis_idx], dis_2[inter_dis_idx]

        # 4- calc weights for intersection words (tf-idf based)
        # get weights of selected words per community
        w_1 = get_selected_weights(w_dict=weights_1, keys_lst=w_lst, normalize=False, config_dict=config_dict)
        w_2 = get_selected_weights(w_dict=weights_2, keys_lst=w_lst, normalize=False, config_dict=config_dict)
        # weight per word (max)
        w_max = np.amax(np.array([w_1, w_2]), axis=0)
        # weight per pair of words (max of the 2 elements in pair)
        f_name = 'pair_w_' + n_model_1 + '_' + n_model_2
        w_pairs = calc_pairwise_weights(arr=w_max, f_name=f_name, normalize=False, calc=True)

        # 5- keep top weights + normalize. Currently taking only top 25% out of the selected words (those with the highest tf-idf)
        top_perc = int(config_dict['algo_parmas']['top_tf_idf_weights_perc'] * 100)
        w_pairs = keep_top_weights(arr=w_pairs, top_perc=top_perc)

        # 6- compare communities
        score = calc_distance_between_comm(d1=dis_1, d2=dis_2, w=w_pairs)  # w=np.ones(len(dis_1))
        res = {'name_m1': n_model_1, 'name_m2': n_model_2, 'score': score, 'wc_m1': wc1, 'wc_m2': wc2, 'wc_inter': wc_inter}
        print(f"iteration:{i}, {res}, elapsed time (min): {(time.time() - start) / 60}")
        return res

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        res = {'name_m1': n_model_1, 'name_m2': n_model_2, 'score': None, 'wc_m1': None, 'wc_m2': None,
               'wc_inter': None}
        return res


def calc_scores_all_models(m_names, m_type, config_dict):
    """

    :param m_names: list
        list of names of all communities
    :param m_type:
    :return:
    """
    metrics = pd.DataFrame(columns=['name_m1', 'name_m2', 'score', 'wc_m1', 'wc_m2', 'wc_inter'])
    print(f"tot_i = {int(len(m_names)*(len(m_names)-1)/2)}")
    if eval(config_dict['current_run_flags']['calc_combinations']):
        lst = []
        for i, (m1, m2) in enumerate(combinations(iterable=m_names, r=2)):
            lst = lst + [(i+1, m1, m2, config_dict)]
        with open(join(config_dict['combinations_path'][config_dict['machine']], 'combinations__' + str(len(m_names)) + '.pickle'), 'wb') as handle:
            pickle.dump(lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # case some of the combinations were already calc
    else:
        with open(join(config_dict['combinations_path'][config_dict['machine']], 'combinations__' + str(len(m_names)) + '.pickle'), 'rb') as handle:
            lst = pickle.load(handle)
    if eval(config_dict['current_run_flags']['filter_pairs']):
        lst = filter_pairs(lst=lst, config_dict=config_dict)
        with open(join(config_dict['combinations_path'][config_dict['machine']],
                       'combinations__after_filter_' + str(len(m_names)) + '_' +
                       config_dict['general_config']['model_type'] + '.pickle'), 'wb') as handle:
            pickle.dump(lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(join(c.combinations_path, 'combinations__after_filter_' + str(len(m_names)) + '.pickle'), 'rb') as handle:
        #     lst = pickle.load(handle)
    # starting a multi=process job
    pool = mp.Pool(processes=config_dict['general_config']['cpu_count'])
    print('start compare')
    with pool as pool:
        results = pool.starmap(compare, lst)

    for res in results:
        metrics = metrics.append(res, ignore_index=True)
    for col in ['name_m1', 'name_m2']:
        metrics[col] = metrics[col].apply(lambda x: x.replace('_model_' +
                                                              config_dict['general_config']['model_type'] + '.model', ''))
    metrics_f_name = 'pairs_similarity_results_' + m_type
    with open(join(config_dict['scores_path'][config_dict['machine']], metrics_f_name + '.pickle'), 'wb') as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metrics.to_csv(join(config_dict['scores_path'][config_dict['machine']], metrics_f_name + '.csv'), index=False)
