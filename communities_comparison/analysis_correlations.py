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
import matplotlib.pyplot as plt


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


def get_ticks(measure):
    if measure == 'score':
        return np.array([0, 1, 2])
    elif measure == 'intersection/union':
        return np.array([0, 0.5, 1])
    elif measure == 'users_rep_distance':
        return np.array([0, 0.5, 1])


def get_title(pearson_df, spearman_df, m1, m2):
    pearson_corr = round(pearson_df.loc[m1, m2], 2)
    spearman_corr = round(spearman_df.loc[m1, m2], 2)
    t = f"Pearson = {pearson_corr}\nSpearman = {spearman_corr}"
    return t


def get_label(measure):
    if measure == 'score':
        return 'ccc distance'
    elif measure == 'intersection/union':
        return measure
    elif measure == 'users_rep_distance':
        return 'users-based distance'


def correlations_and_scatter(df, title):
    DPI = 100
    graph_color = "#3F5D7D"

    df4corr = df.loc[:, ['score', 'intersection/union', 'users_rep_distance']]
    print(f"\n{title}")
    print("calc correlations")
    pearson = df4corr.corr(method='pearson')
    spearman = df4corr.corr(method='spearman')

    tup = [('score', 'users_rep_distance'), ('score', 'intersection/union'), ('users_rep_distance', 'intersection/union')]
    cols = 3
    fig, ax = plt.subplots(nrows=1, ncols=cols, figsize=(65, 20))
    main_title_size = 100
    fig.suptitle(title, fontsize=main_title_size)

    print('start plot')
    for j, (x, y) in enumerate(tup):
        print(f"j = {j}, x= {x}, y= {y}")
        ax[j].spines["top"].set_visible(False)
        ax[j].spines["right"].set_visible(False)

        ax[j].get_xaxis().tick_bottom()
        ax[j].get_yaxis().tick_left()

        titles_size = 65  # 60 #55
        sub_title = get_title(pearson_df=pearson, spearman_df=spearman, m1=x, m2=y)
        ax[j].set_title(sub_title, fontsize=titles_size)

        x_pad = 13
        y_pad = 13
        x_lable_fs = 60  # 55 #50
        y_lable_fs = 60  # 55 #50
        x_tick_labels_fs = 65  # 60 #55
        y_tick_labels_fs = 65  # 60 #55

        ax[j].set_xlabel(get_label(x), fontsize=x_lable_fs, labelpad=x_pad)
        ax[j].set_ylabel(get_label(y), fontsize=y_lable_fs, labelpad=y_pad)
        ax[j].xaxis.set_tick_params(labelsize=x_tick_labels_fs)
        ax[j].yaxis.set_tick_params(labelsize=y_tick_labels_fs)

        x_ticks = get_ticks(measure=x)
        y_ticks = get_ticks(measure=y)

        fac = 1.1
        fac_0 = -0.1
        ax[j].set_yticks(y_ticks)
        ax[j].set_ylim(min(y_ticks), fac * max(y_ticks))
        ax[j].set_xticks(x_ticks)
        ax[j].set_xlim((fac_0 * max(x_ticks), fac * max(x_ticks)))

        # add space from axis
        pad_ticks = 10
        axi = 'both'
        ax[j].tick_params(axis=axi, which='major', pad=pad_ticks)

        dots_size = 100  #400
        x_data = df4corr[x]
        y_data = df4corr[y]
        ax[j].scatter(x_data, y_data, color=graph_color, s=dots_size)

        # set line_width
        line_width = 3
        plt.setp(ax[j].spines.values(), linewidth=line_width)

    for j in range(0, len(tup)):
        # set line_width
        line_width = 3
        plt.setp(ax[j].spines.values(), linewidth=line_width)

    left = 0.18  # the left side of the subplots of the figure
    right = 0.82  # the right side of the subplots of the figure
    bottom = 0.1  # the bottom of the subplots of the figure
    top = 0.77 #0.9  # the top of the subplots of the figure
    wspace = 0.50 #0.48  # 0.62  # the amount of width reserved for blank space between subplots
    hspace = 0.58  # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    f_name = title + '_' + c.MODEL_TYPE
    fig.savefig(join(c.scores_path, f_name + '.png'), bbox_inches="tight", dpi=DPI)
    # plt.show()
    print("done")


results = pd.DataFrame()
print(f"MODEL_TYPE: {c.MODEL_TYPE}")
print(f"LABELS_VERSION: {c.LABELS_VERSION}")
res_name = 'pairs_similarity_general_' + c.MODEL_TYPE + '_' + c.LABELS_VERSION
res_df = pd.read_csv(join(c.scores_path, res_name + '.csv'))
labels_df = pd.read_csv(join(c.labels_path, 'labeled_subreddits_' + c.LABELS_VERSION + '.csv'))

# ti = 'Correlations - all communities'
ti = 'All communities'
correlations_and_scatter(df=res_df, title=ti)

categories = [['Sports'], ['Internet/Apps', 'Tech Related'], ['Music'], ['News/Politics'], ['Food']]
# categories = [['Sports']]
for cate in categories:
    sub_category = '1'
    res_sub = get_cluster_data(df=res_df, dis_col='score', sub_category=sub_category, labels=cate,
                               general_category=cate, only_clus_members=True)
    # ti = 'Correlations - ' + cate[0] + ' communities'
    ti = cate[0] + ' communities'
    ti = ti.replace("/", " and ")
    correlations_and_scatter(df=res_sub, title=ti)

print("DONE all")



