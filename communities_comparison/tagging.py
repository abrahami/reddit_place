import config as c
import pandas as pd
from os.path import join
from communities_comparison.clustering import enrich_data
import numpy as np
import pickle


print(f"MODEL_TYPE: {c.MODEL_TYPE}")
print(f"LABELS_VERSION: {c.LABELS_VERSION}")
labels = pd.read_csv(join(c.labels_path, 'labeled_subreddits_' + c.LABELS_VERSION + '.csv'))
labels = labels[pd.notnull(labels.subreddit)]
scores = pd.read_csv(join(c.scores_path, 'pairs_similarity_results' + '.csv'))
# with open(join(c.scores_path, 'pairs_similarity_results' + '.pickle'), 'rb') as handle:
#     scores = pickle.load(handle)
scores = enrich_data(df=scores)
df = pd.merge(scores, labels, left_on=['name_m1'], right_on=['subreddit'], how='left')
df = df.rename(columns={'main_category': 'main_category_m1', 'sub_category_1': 'sub_category_1_m1',
                        'sub_category_2': 'sub_category_2_m1', 'sub_category_3': 'sub_category_3_m1'})
df = pd.merge(df, labels, left_on=['name_m2'], right_on=['subreddit'], how='left')
df = df.rename(columns={'main_category': 'main_category_m2', 'sub_category_1': 'sub_category_1_m2',
                        'sub_category_2': 'sub_category_2_m2', 'sub_category_3': 'sub_category_3_m2'})
cols = ['name_m1', 'name_m2', 'score', 'intersection/union', 'users_rep_distance', 'rplace_match',
        'sub_category_1_m1', 'sub_category_1_m2',
        'sub_category_2_m1', 'sub_category_2_m2']
# cols = cols + ['sub_category_3_m1', 'sub_category_3_m2']
res = df.loc[:, cols]
# res = df
name = 'pairs_similarity_general_' + c.LABELS_VERSION
res.to_csv(join(c.scores_path, name + '.csv'), index=False)

# name = 'pairs_similarity_with_labels'
# res = res[pd.notnull(res['sub_category_1_m1'])]
# res = res[pd.notnull(res['sub_category_1_m2'])]
# res.to_csv(join(c.scores_path, name + '.csv'), index=False)

