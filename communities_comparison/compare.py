import platform as p
print(f"\nCurrent platform: {p.platform()}")
# if p.system() == 'Windows':
import config as c
from communities_comparison.utils import load_model, load_tfidf
from communities_comparison.visualization import tsne_plot
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


def generate_weights(model):
    words = model.wv.index2entity
    r = np.random.random(len(words))
    weights = r/r.sum()
    return dict(zip(words, weights))


def calc_vec_distances(vectors_matrix, metric, square_form=False):
    if square_form:
        return squareform(X=pdist(X=vectors_matrix, metric=metric))
    return pdist(X=vectors_matrix, metric=metric)


def get_inter_weights(w_dict, keys_lst, normalize):
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
            pairs_w[s0:s1] = np.amax(curr, axis=0)
        with open(join(c.tf_idf_path, f_name + '.pickle'), 'wb') as handle:
            pickle.dump(pairs_w, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(join(c.tf_idf_path, f_name + '.pickle'), 'rb') as handle:
            pairs_w = pickle.load(handle)

    if normalize:
        return pairs_w/pairs_w.sum()
    return pairs_w


def select_top_weights(arr, top_perc):
    """

    :param arr: Numpy array- original weights.
    :param top_perc: Int - the percentage of top (highest) weights to keep.
    :return: Numpy array- new weights after selecting top.
    """
    min_thres = np.percentile(arr, 100-top_perc)
    arr_t = np.where(arr < min_thres, 0, arr)
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


def compare(n_model_1, n_model_2, weights_1, weights_2):
    """

    :param n_model_1: String. name of model_1.
    :param n_model_2: String. name of model_1.
    :param weights_1: Dict. weight for each word in model 1 vocab.
    :param weights_2: Dict. weight for each word in model 2 vocab.
    :return:
    """
    model_1 = load_model(path=c.data_path, m_type=c.MODEL_TYPE, name=n_model_1)
    model_2 = load_model(path=c.data_path, m_type=c.MODEL_TYPE, name=n_model_2)

    # 1- word vectors
    wv_1, wv_2 = model_1.wv, model_2.wv

    # 2- intersection
    intersec = np.intersect1d(wv_1.index2entity, wv_2.index2entity)
    wc1, wc2, wc_inter = len(wv_1.index2entity), len(wv_2.index2entity), len(intersec)
    print(f"\n words count- model_1: {wc1}, model_2: {wc2}, "
          f"intersection: {wc_inter}")
    wv_1_inter, wv_2_inter = wv_1[list(intersec)], wv_2[list(intersec)]

    # 3- calc vectors distances
    dis_metric = 'cosine'
    dis_1 = calc_vec_distances(vectors_matrix=wv_1_inter, metric=dis_metric)
    dis_2 = calc_vec_distances(vectors_matrix=wv_2_inter, metric=dis_metric)

    # 4- calc weights for intersection words
    print("\ncalc weights")
    # get weights of intersection words per community
    w_1 = get_inter_weights(w_dict=weights_1, keys_lst=list(intersec), normalize=False)
    w_2 = get_inter_weights(w_dict=weights_2, keys_lst=list(intersec), normalize=False)
    # weight per word in intersection (max)
    w_intersec = np.amax(np.array([w_1, w_2]), axis=0)
    # weight per pair of words in intersection (max of the 2 elements in pair)
    f_name = 'pair_w_' + n_model_1 + '_' + n_model_2
    w_pairs = calc_pairwise_weights(arr=w_intersec, normalize=False, calc=True)
    print(f"sum of w_pairs: {sum(w_pairs)}")

    # 5- select top weights + normalize
    w_pairs = select_top_weights(arr=w_pairs, top_perc=25)
    print(f"sum of w_pairs_t: {sum(w_pairs)}")

    # 6- compare communities
    print("\ncompare communities")
    score = calc_distance_between_comm(d1=dis_1, d2=dis_2, w=w_pairs)  # w=np.ones(len(dis_1))
    print(f"score: {score}")

    return score, wc1, wc2, wc_inter


def calc_scores_all_models(m_names, m_type):
    metrics = pd.DataFrame(columns=['name_m1', 'name_m2', 'score', 'wc_m1', 'wc_m2', 'wc_inter'])
    for (m1, m2) in combinations(iterable=m_names, r=2):
        tfidf_1 = load_tfidf(path=c.tf_idf_path, name=m1)
        tfidf_2 = load_tfidf(path=c.tf_idf_path, name=m2)
        score, wc1, wc2, wc_inter = compare(n_model_1=m1, n_model_2=m2, weights_1=tfidf_1, weights_2=tfidf_2)
        res = {'name_m1': m1, 'name_m2': m2, 'score': score, 'wc_m1': wc1, 'wc_m2': wc2, 'wc_inter': wc_inter}
        metrics = metrics.append(res, ignore_index=True)

    metrics_f_name = 'metrics_m_type_' + m_type
    with open(join(c.scores_path, metrics_f_name + '.pickle'), 'wb') as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)


# # 6- visualization
# df_1 = pd.DataFrame()
# df_1['weight'] = w_1
# df_1['word'] = intersec
# df_1 = df_1.sort_values(by=['weight'], ascending=False).reset_index(drop=True)
# # tsnescatterplot(model_1, df_1['word'][0], list(df_1['word'][1:10]))

# todo- should consider also the size of intersection.
#  possible formula: len(intersection)/len(union) * ans (?)


# weights_gameswap = generate_weights(model=model_gameswap)
