import sys
if sys.platform == 'linux':
    sys.path.append('/data/home/shanisa/reddit_place')
print(f"\nCurrent platform: {sys.platform}")
import pandas as pd
from os.path import join
from srs_word_embeddings.analysis.additional_metrics import enrich_data
import commentjson
import os

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
# purpose of this code is to generate a final file, including all pairs of SRs with all needed information
# we add to the existing 3com results the following information:
# 1. user-based score (paper from ACL 2017) +
# 2. participation in r/place
# 3. intersection/union value
# 4. lables to each SR (which we know)

# data is saved into (for example): /scores/2.05/pairs_similarity_general_2.05_V3.csv
if __name__ == "__main__":
    model_type = config_dict['general_config']['model_type']
    labels_version = config_dict['general_config']['labels_version']
    scores_path = config_dict['scores_path'][config_dict['machine']]
    print(f"MODEL_TYPE: {model_type}")
    print(f"LABELS_VERSION: {labels_version}")
    labels = pd.read_csv(join(config_dict['labels_path'][config_dict['machine']], 'labeled_subreddits_' + labels_version + '.csv'))
    labels = labels[pd.notnull(labels.subreddit)]
    scores = pd.read_csv(join(scores_path, 'pairs_similarity_results_' + model_type + '.csv'))
    # with open(join(c.scores_path, 'pairs_similarity_results' + '.pickle'), 'rb') as handle:
    #     scores = pickle.load(handle)
    scores = enrich_data(df=scores, config_dict=config_dict)
    df = pd.merge(scores, labels, left_on=['name_m1'], right_on=['subreddit'], how='left')
    df = df.rename(columns={'main_category': 'main_category_m1', 'sub_category_1': 'sub_category_1_m1',
                            'sub_category_2': 'sub_category_2_m1', 'sub_category_3': 'sub_category_3_m1'})
    df = pd.merge(df, labels, left_on=['name_m2'], right_on=['subreddit'], how='left')
    df = df.rename(columns={'main_category': 'main_category_m2', 'sub_category_1': 'sub_category_1_m2',
                            'sub_category_2': 'sub_category_2_m2', 'sub_category_3': 'sub_category_3_m2'})
    cols = ['name_m1', 'name_m2', 'score', 'intersection/union', 'users_rep_distance', 'doc2vec_rep_distance',
            'rplace_match', 'sub_category_1_m1', 'sub_category_1_m2', 'sub_category_2_m1', 'sub_category_2_m2']
    # cols = cols + ['sub_category_3_m1', 'sub_category_3_m2']
    res = df.loc[:, cols]
    # res = df
    name = 'pairs_similarity_general_' + model_type + '_' + labels_version
    res.to_csv(join(scores_path, name + '.csv'), index=False)
