import sys
if sys.platform == 'linux':
    sys.path.append('/data/home/shanisa/reddit_place')
print(f"\nCurrent platform: {sys.platform}")
from communities_comparison.clustering import get_cluster_members, get_cluster_data, get_metrics
import config as c
import pandas as pd
from os.path import join
import numpy as np
from scipy.stats import mannwhitneyu


def get_sub_category_2(df, sub_cate_1):
    ans = set()
    for s in sub_cate_1:
        df_ = df.loc[df['sub_category_1'] == s, ['sub_category_2']].dropna()
        ans.update(set(list(df_.sub_category_2)))
    return sorted(ans)


def print_metrics(res, dis_col, general_cate, sub_cate, clus_labels, only_clus_members):
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


results = pd.DataFrame()
print(f"MODEL_TYPE: {c.MODEL_TYPE}")
print(f"LABELS_VERSION: {c.LABELS_VERSION}")
res_name = 'pairs_similarity_general_' + c.MODEL_TYPE + '_' + c.LABELS_VERSION
res_df = pd.read_csv(join(c.scores_path, res_name + '.csv'))
labels_df = pd.read_csv(join(c.labels_path, 'labeled_subreddits_' + c.LABELS_VERSION + '.csv'))

# distance_col = 'score'
distance_col = 'users_rep_distance'
# mvu_test = 'greater'
mvu_test = 'less'
# mvu_test = 'two-sided'
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

results_f_name = 'analysis_results_by_' + distance_col + '_' + c.MODEL_TYPE + '_labels_' + c.LABELS_VERSION + '_' \
                 + mvu_test
results.to_csv(join(c.scores_path, results_f_name + '.csv'), index=False)

