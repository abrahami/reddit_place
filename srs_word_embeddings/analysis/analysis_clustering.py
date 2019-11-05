import sys
if sys.platform == 'linux':
    sys.path.append('/data/home/shanisa/reddit_place')
print(f"\nCurrent platform: {sys.platform}")
from srs_word_embeddings.analysis.clustering import get_cluster_members, get_cluster_data, get_metrics
import pandas as pd
from os.path import join
import numpy as np
from scipy.stats import mannwhitneyu
import commentjson
import os


def get_sub_category_2(df, sub_cate_1):
    '''

    :param df:
    :param sub_cate_1:
    :return:
    '''
    ans = set()
    for s in sub_cate_1:
        df_ = df.loc[df['sub_category_1'] == s, ['sub_category_2']].dropna()
        ans.update(set(list(df_.sub_category_2)))
    return sorted(ans)


def print_metrics(res, dis_col, general_cate, sub_cate, clus_labels, only_clus_members):
    '''

    :param res:
    :param dis_col:
    :param general_cate:
    :param sub_cate:
    :param clus_labels:
    :param only_clus_members:
    :return:
    '''
    df = get_cluster_data(df=res, dis_col=dis_col, sub_category=sub_cate, labels=clus_labels,
                          general_category=general_cate, only_clus_members=only_clus_members)
    if df.empty:
        # print("single member  OR  we still don't have the results of this sub category")
        return None, None, None, None, None
    else:
        max_, mean_, median_, distances_ = get_metrics(df=df, col=dis_col)
        # print(f"max: {max_}, mean: {mean_}, median: {median_}, current_pairs: {len(distances_)}")

        return max_, mean_, median_, distances_, df


def print_analysis(res, dis_col, labels, general_cate, sub_cate, clus_labels, mv_test):
    """
    :param res: df
        full results as a df (all combinations of pairs of communities)
    :param dis_col: str
        column name of the distance matrix (e.g. users_rep_distance)
    :param labels: df
        data frame of the labels
    :param general_cate: str
        the category (NOT the subcategory) of the analysed topic
        If we analyse 'sports' it will be 'sports'. If we analyse 'basketball' it will be 'sports'
    :param sub_cate: str
        '1' or '2'. If '1' - 1 it will take the higher level of category (e.g. sports)
        2 will take the "lower" level (e.g. 'football')
    :param clus_labels: list (strings)
        the categories to analyse (it can be a list, only for complex categories such as 'Internet Apps + Tech Related')
        In the standard way, it will be a single string
    :param mv_test: str
        the 'alternative' parameter which the mannwhitneyu gets as parameter

    :return: dict
        dictionary with all results
    """
    print(f"\nGENERAL_CATEGORY: {general_cate}")
    print(f"SUB_CATEGORY: {clus_labels} level: {sub_cate}")

    # cluster members
    members = get_cluster_members(df=labels, sub_category=sub_cate, labels=clus_labels)

    # within cluster metrics
    max_w, mean_w, median_w, distances_w, df_w = print_metrics(res=res, dis_col=dis_col, general_cate=general_cate,
                                                               sub_cate=sub_cate, clus_labels=clus_labels,
                                                               only_clus_members=True)
    if df_w is None:
        rec = {'distance_score': dis_col, 'general_category': general_cate, 'cluster_labels': clus_labels,
               'members': len(members), 'mann_whitney_u_statistic': None, 'mann_whitney_u_pvalue': None,
               'max_w': None, 'max_o': None, 'mean_w': None, 'mean_o': None, 'median_w': None,
               'median_o': None, 'potentially_in_cluster': None,
               'threshold (percentile)': None}
        return rec

    # out of cluster metrics
    max_o, mean_o, median_o, distances_o, df_o = print_metrics(res=res, dis_col=dis_col, general_cate=general_cate,
                                                               sub_cate=sub_cate, clus_labels=clus_labels,
                                                               only_clus_members=False)
    # mann whitney u
    mw_statistic, mw_pvalue = mannwhitneyu(x=distances_w, y=distances_o, alternative=mv_test)

    # potentially in cluster
    percentile = 90
    thresh = np.percentile(distances_w, percentile)
    df_ = df_o[df_o[dis_col] <= thresh]
    potentially_in_cluster = set(list(df_.name_m1) + list(df_.name_m2)) - members
    rec = {'distance_score': dis_col, 'general_category': general_cate, 'cluster_labels': clus_labels,
           'members': len(members), 'mann_whitney_u_statistic': mw_statistic, 'mann_whitney_u_pvalue': mw_pvalue,
           'max_w': max_w, 'max_o': max_o, 'mean_w': mean_w, 'mean_o': mean_o, 'median_w': median_w,
           'median_o': median_o, 'potentially_in_cluster': potentially_in_cluster, 'threshold (percentile)': percentile}

    return rec


###################################################### Configurations ##################################################
config_dict = commentjson.load(open(os.path.join(os.path.dirname(os.getcwd()), 'configurations',
                                                 'analysis_config.json')))
machine = 'yalla' if sys.platform == 'linux' else os.environ['COMPUTERNAME']
config_dict['machine'] = machine
# adjusting the condif dict to current run (e.g., model_type to paths)
config_dict['data_path'][machine] = join(config_dict['data_path'][machine], config_dict['general_config']['model_type'])
config_dict['vocab_distr_path'][machine] = join(config_dict['vocab_distr_path'][machine],
                                                config_dict['general_config']['model_type'])
config_dict['tf_idf_path'][machine] = join(config_dict['tf_idf_path'][machine],
                                           config_dict['general_config']['model_type'])
config_dict['scores_path'][machine] = join(config_dict['scores_path'][machine],
                                           config_dict['general_config']['model_type'])
config_dict['dis_path'][machine] = join(config_dict['dis_path'][machine], config_dict['general_config']['model_type'])
########################################################################################################################

if __name__ == "__main__":
    results = pd.DataFrame()
    labels_version = config_dict['general_config']['labels_version']
    model_type = config_dict['general_config']['model_type']
    print(f"MODEL_TYPE: {model_type}")
    print(f"LABELS_VERSION: {labels_version}")
    res_name = 'pairs_similarity_general_' + model_type + '_' + labels_version
    res_df = pd.read_csv(join(config_dict['scores_path'][config_dict['machine']], res_name + '.csv'))
    labels_df = pd.read_csv(join(config_dict['labels_path'][config_dict['machine']], 'labeled_subreddits_' +
                                 labels_version + '.csv'))

    # distance_col = 'score'
    distance_col = 'users_rep_distance'

    # mvu_test = 'greater'
    # mvu_test = 'two-sided'
    mvu_test = 'less'

    print(f"DISTANCE COLUMN: {distance_col}")

    categories = [['Sports'], ['Internet/Apps', 'Tech Related'], ['Music'], ['News/Politics'], ['Food']]
    # categories = [['Sports']]
    for cate in categories:
        sub_category = '1'
        re = print_analysis(res=res_df, dis_col=distance_col, labels=labels_df, general_cate=cate, sub_cate=sub_category,
                            clus_labels=cate, mv_test=mvu_test)
        if re is not None:
            results = results.append(re, ignore_index=True)

        sub_category = '2'
        cluster_labels = get_sub_category_2(df=labels_df, sub_cate_1=cate)
        for (i, l) in enumerate(cluster_labels):
            re = print_analysis(res=res_df, dis_col=distance_col, labels=labels_df, general_cate=cate,
                                sub_cate=sub_category, clus_labels=[l], mv_test=mvu_test)
            if re is not None:
                results = results.append(re, ignore_index=True)

    results_f_name = 'analysis_results_by_' + distance_col + '_' + model_type + '_labels_' + labels_version + '_' \
                     + mvu_test
    results.to_csv(join(config_dict['scores_path'][config_dict['machine']], results_f_name + '.csv'), index=False)
