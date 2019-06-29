from communities_comparison.clustering import get_cluster_members, get_cluster_data, get_metrics
import config as c
import pandas as pd
from os.path import join
import numpy as np


def get_sub_category_2(df, sub_cate_1):
    ans = set()
    for s in sub_cate_1:
        df_ = df.loc[df['sub_category_1'] == s, ['sub_category_2']].dropna()
        ans.update(set(list(df_.sub_category_2)))
    return sorted(ans)


def print_metrics(res, dis_col, sub_cate, clus_labels, only_clus_members):
    df = get_cluster_data(df=res, dis_col=dis_col, sub_category=sub_cate, labels=clus_labels,
                          only_clus_members=only_clus_members)
    if df.empty:
        print("single member  OR  we still don't have the results of this sub category")
    else:
        max_, mean_, median_, distances_ = get_metrics(df=df, col=dis_col)
        print(f"max: {max_}, mean: {mean_}, median: {median_}, current_pairs: {len(distances_)}")

        return max_, mean_, median_, distances_


def print_analysis(res, dis_col, labels, sub_cate, clus_labels):

    print(f"SUB_CATEGORY: {sub_cate}")
    print(f"LABELS: {clus_labels}")
    print(f"DISTANCE COLUMN: {dis_col}")

    # cluster members
    members = get_cluster_members(df=labels, sub_category=sub_cate, labels=clus_labels)
    print(f"---->   {len(members)} members")

    # within cluster metrics
    print(f"\nwithin cluster distances")
    max_w, mean_w, median_w, distances_w = print_metrics(res=res, dis_col=dis_col, sub_cate=sub_cate,
                                                         clus_labels=clus_labels, only_clus_members=True)

    # out of cluster metrics
    print(f"out of cluster distances")
    df = get_cluster_data(df=res, dis_col=dis_col, sub_category=sub_cate, labels=clus_labels, only_clus_members=False)
    max_o, mean_o, median_o, distances_o = get_metrics(df=df, col=dis_col)
    print(f"max: {max_o}, mean: {mean_o}, median: {median_o}, current_pairs: {len(distances_o)}")

    # potentially in cluster
    percentile = 100
    thresh = np.percentile(distances_w, percentile)
    df_ = df[df[dis_col] <= thresh]
    potentially_in_cluster = set(list(df_.name_m1) + list(df_.name_m2)) - members
    print(f"potentially_in_cluster: {len(potentially_in_cluster)} subreddits")
    # for x in sorted(potentially_in_cluster):
    #     print(x)


print(f"MODEL_TYPE: {c.MODEL_TYPE}")
print(f"LABELS_VERSION: {c.LABELS_VERSION}")
res_df = pd.read_csv(join(c.scores_path, 'pairs_similarity_general_' + c.LABELS_VERSION + '.csv'))
labels_df = pd.read_csv(join(c.labels_path, 'labeled_subreddits_' + c.LABELS_VERSION + '.csv'))

distance_col = 'score'
categories = [['Sports'], ['Internet/Apps', 'Tech Related'], ['Music'], ['News/Politics'], ['Food']]
# categories = [['Sports']]

for cate in categories:
    sub_category = '1'
    print_analysis(res=res_df, dis_col=distance_col, labels=labels_df, sub_cate=sub_category, clus_labels=cate)
    print("\n\n")

    sub_category = '2'
    cluster_labels = get_sub_category_2(df=labels_df, sub_cate_1=cate)
    print(f"SUB_CATEGORY: {sub_category}")
    print(f"LABELS: {cate}")
    print(f"DISTANCE COLUMN: {distance_col}")
    for (i, l) in enumerate(cluster_labels):
        members = get_cluster_members(df=labels_df, sub_category=sub_category, labels=[l])
        print(f"\nLABEL {i}: {l}   ---->   {len(members)} members")
        print_metrics(res=res_df, dis_col=distance_col, sub_cate=sub_category, clus_labels=[l], only_clus_members=True)

    print("\n")
    print("###############################################################################################")
    print("\n\n")
