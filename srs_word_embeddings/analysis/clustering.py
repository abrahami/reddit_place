import numpy as np


def get_cluster_members(df, sub_category, labels):
    """

    :param df:
    :param sub_category:
    :param labels:
    :return:
    """

    df['is_member'] = df['sub_category_' + sub_category].apply(lambda x: x in labels)
    members = set(df['subreddit'][df['is_member']])

    return members


def get_cluster_data(df, dis_col, sub_category, labels, general_category, only_clus_members):
    """

    :param df: DataFrame. final results for all subreddits.
    :param dis_col: String. the name of the distance column for sorting.
    :param sub_category: String. '1' or '2'.
    :param labels: List of strings. the labels representing the cluster.
    :param general_category: The name of sub category '1' that includes the selected labels.
    :param only_clus_members: Boolean. whether to select pairs that are both belong to the cluster (AND).
                              otherwise select pairs where ONLY ONE element belongs to the cluster (OR).
    :return: DataFrame. results for selected cluster elements.
    """

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
    """

    :param df: DataFrame. final results for selected subreddits.
    :param col: String. column name to for metrics calculations.
    :return: (numeric, numeric, numeric, Numpy array)  (max, mean, median, distances)
    """
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


