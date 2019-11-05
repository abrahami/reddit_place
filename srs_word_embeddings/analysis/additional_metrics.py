import numpy as np
from os.path import join
import pandas as pd
import pickle
import datetime as dt
from scipy.spatial.distance import pdist


def get_map(df):
    """

    :param df:
    :return:
    """
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
def calc_similarity_based_embeddings(df, embedding_file_path, metric, column_name_in_df):
    """
    adding information to df, using embeddings similarity
    :param df:
    :param metric:
    :return:
    """
    cols = list(df.columns.values)
    relevant_communities = set(list(df.iloc[:, 0]) + list(df.iloc[:, 1]))
    # case the embedding_file_path directs us to a csv file
    with open(embedding_file_path, 'rb') as handle:
        embedding_dict = pickle.load(handle)
    # filtering embedding_dict only to the needed SRs in df
    shrinked_embedding_dict = {key: value for key, value in embedding_dict.items() if key in relevant_communities}
    # converting the dict to a pandas array
    ur = pd.DataFrame.from_dict(shrinked_embedding_dict, orient='index').reset_index(drop=False).reset_index(drop=False)
    ur = ur.rename(columns={"level_0": 'index', "index": 'com'})
    dis_matrix = pdist(X=np.array(ur.iloc[:, 2:]), metric=metric)
    ur = ur.loc[:, ['index', 'com']]
    map_ = get_map(df=ur)
    map_[column_name_in_df] = dis_matrix
    df_ = pd.merge(df, map_, on=['name_m1', 'name_m2'], how='left')
    df_ = pd.merge(df_, map_, left_on=['name_m1', 'name_m2'], right_on=['name_m2', 'name_m1'], how='left')
    df_[column_name_in_df] = np.nanmax(df_[[column_name_in_df + '_x', column_name_in_df + '_y']].values, axis=1)
    df_ = df_.rename(columns={'name_m1_x': 'name_m1', 'name_m2_x': 'name_m2'})
    df_ = df_.loc[:, cols + [column_name_in_df]]
    return df_


def add_rplace_participation(df, config_dict):
    """

    :param df:
    :return:
    """
    cols = list(df.columns.values)
    rplace = pd.read_csv(join(config_dict['users_data_path'][config_dict['machine']], 'subreddits_rplace' + '.csv'))
    print(len(rplace))
    df_ = pd.merge(df, rplace, left_on=['name_m1'], right_on=['name'], how='left')
    df_ = df_.rename(columns={'rplace': 'rplace_name_m1'})
    df_ = pd.merge(df_, rplace, left_on=['name_m2'], right_on=['name'], how='left')
    df_ = df_.rename(columns={'rplace': 'rplace_name_m2'})
    df_['rplace_match'] = np.where(df_['rplace_name_m1'] == df_['rplace_name_m2'], 1, 0)
    df_ = df_.loc[:, cols + ['rplace_name_m1', 'rplace_name_m2', 'rplace_match']]
    return df_


def enrich_data(df, config_dict):
    """
    adds useful information to a data-frame (e.g., user-based embeddings similarity)
    :param df:
    :return:
    """
    for col in ['name_m1', 'name_m2']:
        df[col] = df[col].apply(lambda x: x.replace('_model_' + config_dict['general_config']['model_type'] + '.model', ''))
    df['intersection/union'] = df['wc_inter']/(df['wc_m1']+df['wc_m2']-df['wc_inter'])
    df['intersection/union'] = df['intersection/union'].apply(lambda x: round(x, 5))
    sim_metric = 'cosine'
    df = calc_similarity_based_embeddings(df=df, embedding_file_path="/data/work/data/reddit_place/communities_overlap/communities_overlap_model_13_4_2019_dict.p",
                                          metric=sim_metric, column_name_in_df='users_rep_distance')
    df = calc_similarity_based_embeddings(df=df, embedding_file_path="/data/work/data/reddit_place/embedding/embedding_per_sr/3.00/communities_doc2vec_model_5_11_2019_dict.p",
                                          metric=sim_metric, column_name_in_df='doc2vec_rep_distance')
    df = add_rplace_participation(df=df, config_dict=config_dict)
    return df
