import config as c
import numpy as np
from os.path import join
import pandas as pd
import pickle
import datetime as dt
from scipy.spatial.distance import pdist


def get_map(df):
    n = len(df)
    N = n * (n - 1) // 2
    idx = np.concatenate(([0], np.arange(n - 1, 0, -1).cumsum()))
    start, stop = idx[:-1], idx[1:]
    map_ = np.empty((N, 2), dtype=int)  # save original indexes
    for j, i in enumerate(range(n - 1)):
        s0, s1 = start[j], stop[j]
        map_[s0:s1, 0] = j
        map_[s0:s1, 1] = np.arange(j + 1, n)
    map_ = pd.DataFrame(data=map_, columns=['i', 'j'])
    map_ = pd.merge(map_, df, left_on=['i'], right_on=['index'], how='left')
    map_ = pd.merge(map_, df, left_on=['j'], right_on=['index'], how='left')
    map_ = map_.loc[:, ['com_x', 'com_y']].rename(columns={'com_x': 'name_m1', 'com_y': 'name_m2'})
    return map_


# comms = [x.replace('_model_' + c.MODEL_TYPE + '.model', '') for x in comms]
# # users based representation
# with open(join(c.users_data_path, 'communities_overlap_model_13_4_2019_dict' + '.p'), 'rb') as handle:
#     users_rep = pickle.load(handle)
# ur = pd.DataFrame.from_dict(data=users_rep, orient='index').reset_index().rename(columns={'index': 'com'})
# ur = ur[ur.com.isin(comms)].reset_index(drop=True).reset_index()
# with open(join(c.users_data_path, 'users_rep_N_' + str(c.N) + '.pickle'), 'wb') as handle:
#     pickle.dump(ur, handle, protocol=pickle.HIGHEST_PROTOCOL)
def calc_users_similarity(df, metric):
    cols = list(df.columns.values)
    with open(join(c.users_data_path, 'users_rep_N_' + str(c.N) + '.pickle'), 'rb') as handle:
        ur = pickle.load(handle)
    dis_matrix = pdist(X=np.array(ur.iloc[:, 2:]), metric=metric)
    ur = ur.loc[:, ['index', 'com']]
    map_ = get_map(df=ur)
    map_['users_rep_distance'] = dis_matrix
    df_ = pd.merge(df, map_, on=['name_m1', 'name_m2'], how='left')
    df_ = pd.merge(df_, map_, left_on=['name_m1', 'name_m2'], right_on=['name_m2', 'name_m1'], how='left')
    df_['users_rep_distance'] = np.nanmax(df_[['users_rep_distance_x', 'users_rep_distance_y']].values, axis=1)
    df_ = df_.rename(columns={'name_m1_x': 'name_m1', 'name_m2_x': 'name_m2'})
    df_ = df_.loc[:, cols + ['users_rep_distance']]
    return df_


def add_rplace_participation(df):
    cols = list(df.columns.values)
    rplace = pd.read_csv(join(c.users_data_path, 'subreddits_rplace' + '.csv'))
    print(len(rplace))
    df_ = pd.merge(df, rplace, left_on=['name_m1'], right_on=['name'], how='left')
    df_ = df_.rename(columns={'rplace': 'rplace_name_m1'})
    df_ = pd.merge(df_, rplace, left_on=['name_m2'], right_on=['name'], how='left')
    df_ = df_.rename(columns={'rplace': 'rplace_name_m2'})
    df_['rplace_match'] = np.where(df_['rplace_name_m1'] == df_['rplace_name_m2'], 1, 0)
    df_ = df_.loc[:, cols + ['rplace_name_m1', 'rplace_name_m2', 'rplace_match']]
    return df_


def enrich_data(df):
    for col in ['name_m1', 'name_m2']:
        df[col] = df[col].apply(lambda x: x.replace('_model_' + c.MODEL_TYPE + '.model', ''))
    df['intersection/union'] = df['wc_inter']/(df['wc_m1']+df['wc_m2']-df['wc_inter'])
    df['intersection/union'] = df['intersection/union'].apply(lambda x: round(x, 5))
    sim_metric = 'cosine'
    df = calc_users_similarity(df=df, metric=sim_metric)
    df = add_rplace_participation(df=df)
    return df


def get_cluster_members(df, sub_category, labels):
    df['is_member'] = df['sub_category_' + sub_category].apply(lambda x: x in labels)
    members = set(df['subreddit'][df['is_member']])

    return members


def get_cluster_data(df, dis_col, sub_category, labels, general_category, only_clus_members):
    '''
    
    :param df: DataFrame. final results for all subreddits.
    :param dis_col: String. the name of the distance column for sorting.
    :param sub_category: String. '1' or '2'.
    :param labels: List of strings. the labels representing the cluster.
    :param general_category: The name of sub category '1' that includes the selected labels.
    :param only_clus_members: Boolean. whether to select pairs that are both belong to the cluster (AND).
                              otherwise select pairs where ONLY ONE element belongs to the cluster (OR).
    :return: DataFrame. results for selected cluster elements.
    '''

    cols = df.columns.values
    df['m1_in'] = df['sub_category_' + sub_category + '_m1'].apply(lambda x: x in labels)
    df['m2_in'] = df['sub_category_' + sub_category + '_m2'].apply(lambda x: x in labels)
    df['m1_in_general'] = df['sub_category_1_m1'].apply(lambda x: x in general_category)
    df['m2_in_general'] = df['sub_category_1_m2'].apply(lambda x: x in general_category)
    df['select'] = df.m1_in & df.m2_in if only_clus_members \
        else (df.m1_in | df.m2_in) & np.logical_not(df.m1_in_general & df.m2_in_general)
    ans = df.loc[df['select'], cols].sort_values(by=[dis_col]).reset_index(drop=True)

    return ans


def get_metrics(df, col):
    '''

    :param df: DataFrame. final results for selected subreddits.
    :param col: String. column name to for metrics calculations.
    :return: (numeric, numeric, numeric, Numpy array)  (max, mean, median, distances)
    '''
    digits = 3
    max_ = round(np.max(df[col]), digits)
    mean_ = round(np.mean(df[col]), digits)
    median_ = round(np.median(df[col]), digits)
    distances_ = np.array(df[col])

    return max_, mean_, median_, distances_


# def cluster_communities(df):
#     clus_res = pd.DataFrame(columns=['name', 'avg_score', 'median_score'])
#
#     name = 'test_cluster'
#     lst = ['arma_model_2.02.model', 'iamverysmart_model_2.02.model', 'colombia_model_2.02.model']
#
#     res = get_metric(df=df, comms=lst, name=name)
#     clus_res = clus_res.append(res, ignore_index=True)
#
#     clus_res_name = 'clus_res_m_type_' + c.MODEL_TYPE + '_' + dt.datetime.now().strftime("%Y_%m_%d")
#     with open(join(c.scores_path, clus_res_name + '.pickle'), 'wb') as handle:
#         pickle.dump(clus_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     clus_res.to_csv(join(c.scores_path, clus_res_name + '.csv'), index=False)


