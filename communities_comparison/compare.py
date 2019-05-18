import platform as p
print(f"\nCurrent platform: {p.platform()}")
import config as c
from os.path import join
import numpy as np
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models.keyedvectors import WordEmbeddingSimilarityIndex
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from operator import itemgetter
np.random.seed(c.SEED)


def generate_weights(model):
    words = model.wv.index2entity
    r = np.random.random(len(words))
    weights = r/r.sum()
    return dict(zip(words, weights))


def calc_vec_distances(vectors_matrix, metric, square_form=False):
    if square_form:
        return squareform(X=pdist(X=vectors_matrix, metric=metric))
    return pdist(X=vectors_matrix, metric=metric)


def get_normalized_weights(w_dict, keys_lst):
    v = np.array(itemgetter(*keys_lst)(w_dict))
    # todo- consider to remove normalization and use it later
    return v/v.sum()


def get_pairs_weights(arr):
    """

    :param arr: Numpy array. array of positive weights, sum(arr) = 1.
    :return: condensed matrix of mean weight for each two elements in arr
    """
    # todo- should be implemented more efficiently
    # n(n-1)/2
    n = len(arr)
    con_w = np.empty(int(n*(n-1)/2))
    idx = 0
    for i in range(0, len(arr)):
        for j in range(i+1, len(arr)):
            con_w[idx] = np.mean((arr[i], arr[j]))
            idx += 1
    # todo- consider to add normalization/ used other method for weighting
    return con_w/con_w.sum()
    # return con_w


def calc_distance_between_comm(d1, d2, w):
    """

    :param d1: condensed matrix of distances between intersection words - community 1
    :param d2: condensed matrix of distances between intersection words - community 2
    :param w: condensed matrix of weights for pairs of intersection words
    Note: words' order should be identical in d1, d2, w

    :return: int - distance between community 1 and 2 = weighted average of absolute difference between intersection words
    """
    abs_d = abs(d1 - d2)
    return np.matmul(abs_d, w.T)


def compare(model_1, model_2, weights_1, weights_2):
    # 1- word vectors
    wv_1, wv_2 = model_1.wv, model_2.wv

    # 2- intersection
    intersec = np.intersect1d(wv_1.index2entity, wv_2.index2entity)
    print(f"\n words count- model_1: {len(wv_1.index2entity)}, model_2: {len(wv_2.index2entity)}, "
          f"intersection: {len(intersec)}")
    wv_1_inter, wv_2_inter = wv_1[list(intersec)], wv_2[list(intersec)]

    # 3- calc vectors distances
    dis_metric = 'cosine'
    dis_1 = calc_vec_distances(vectors_matrix=wv_1_inter, metric=dis_metric)
    dis_2 = calc_vec_distances(vectors_matrix=wv_2_inter, metric=dis_metric)
    # dis_m_1 = calc_distances(vectors_matrix=wv_1_inter, metric=dis_metric, square_form=True)
    # dis_m_2 = calc_distances(vectors_matrix=wv_2_inter, metric=dis_metric, square_form=True)

    # 4- calc weights for intersection words
    print("\ncalc weights")
    # get normalized weights of intersection words per community
    w_1 = get_normalized_weights(w_dict=weights_1, keys_lst=list(intersec))  # scale (0,1), sum=1
    w_2 = get_normalized_weights(w_dict=weights_2, keys_lst=list(intersec))  # scale (0,1), sum=1
    # weight per word in intersection (mean)
    w_intersec = (w_1 + w_2)/2  # scale (0,1), sum=1
    # weight per pair of words in intersection
    w_pairs = get_pairs_weights(arr=w_intersec)
    print(f"sum of w_pairs: {sum(w_pairs)}")

    # 5- compare communities
    print("\ncompare communities")
    # ans = calc_distance_between_comm(d1=dis_1, d2=dis_2, w=np.ones(len(dis_1)))
    ans = calc_distance_between_comm(d1=dis_1, d2=dis_2, w=w_pairs)
    print(f"score: {ans}")

    # todo- should consider also the size of intersection.
    #  possible formula: len(intersection)/len(union) * ans (?)


model_gameswap = Word2Vec.load(join(c.home_path, 'gameswap_model_1.01' + '.model'))
model_babymetal = Word2Vec.load(join(c.home_path, 'babymetal_model_1.01' + '.model'))
# temp - should be provided
weights_gameswap = generate_weights(model=model_gameswap)
weights_babymetal = generate_weights(model=model_babymetal)

# wv_babymetal = model_babymetal.wv
# #
# idx = [
#     7, 8, 12, 17, 52, 86, 98, 111, 117,
#     # numbers
#     140, 160, 179, 182, 217, 251, 261, 287, 386, 391, 459, 484, 511, 519, 523, 528, 609, 619,
#     # smilies
#     304, 320, 336, 526, 577,
#     # other symbols
#     390, 395, 712, 646, 948,
#     # a single character/ suffix
#     479, 504, 535, 545, 744, 587, 506, 517]
#
# wrong_strings = [wv_babymetal.index2entity[i] for i in idx]

# dis_gameswap = wv_gameswap.distances('the', other_words=list(intersection))

# todo- two elements to consider when comparing communities:
#  1 - size of intersection
#  2 - similarity based on intersection
compare(model_1=model_gameswap, model_2=model_babymetal,
        weights_1=weights_gameswap, weights_2=weights_babymetal)
