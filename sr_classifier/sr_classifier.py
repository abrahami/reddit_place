# Authors: Avrahami Israeli (isabrah)
# Python version: 3.6
# Last update: 9.10.2018

"""
Main running file of SR classification (drawing/not drawing) task
Files needed in order to run this code are the following:
    1. clean_text_transformer.py
    2. clean_text_transformer.py
    3. reddit_data_preprocessing
    4. sub_reddit.py
    5. utils.py
    6. submission_shrinked.hdf (located under 'data_path' path)
    7. comments_shrinked.hdf (located under 'data_path' path)
    8. all_subredditts_based_atlas_and_submissions.xlsx (located under 'data_path' path)

"""

from reddit_data_preprocessing import RedditDataPrep
from sub_reddit import SubReddit
from utils import *
import sys
import pandas as pd
import numpy as np
import datetime
import collections
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

###################################################### Configurations ##################################################
data_path = '/home/isabrah/reddit_data/' if sys.platform == 'linux' \
    else 'C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\PhD\\reddit canvas\\data\\sr_relations\\'
use_comments_data = True
test_data_logic = 'submission_based' # either 'atlas_based' or 'submission_based' or None
########################################################################################################################

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    # Data Loading
    if sys.platform == 'linux':
        submission_shrinked = pd.read_hdf(data_path + 'submission_shrinked.hdf')
        comments_shrinked = pd.read_hdf(data_path + 'comments_shrinked.hdf') if use_comments_data else None
        subreddits_df = pd.read_excel(data_path + 'all_subredditts_based_atlas_and_submissions.xlsx',
                                      sheet_name='Detailed list')

    else:
        submission_shrinked = pd.read_hdf(data_path + 'submission_shrinked.hdf')
        comments_shrinked = pd.read_hdf(data_path + 'comments_shrinked.hdf') if use_comments_data else None
        subreddits_df = pd.read_excel(data_path + 'all_subredditts_based_atlas_and_submissions.xlsx',
                                      sheet_name='Detailed list')
    # Train data - taking only sr which are labeled and coming from the submission dataset (not the atlas one)
    train_subreddits_df = subreddits_df[(subreddits_df["Source"] != 'atlas_main') &
                                        (subreddits_df["trying_to_draw"] != '?') &
                                        (~(pd.isna(subreddits_df["trying_to_draw"])) |
                                         ~(pd.isna(subreddits_df["models_prediction"])))]

    train_subreddits_list = list(train_subreddits_df['SR'])
    train_submission = submission_shrinked[submission_shrinked['subreddit'].str.lower().isin(train_subreddits_list)]
    if use_comments_data:
        train_comments = comments_shrinked[comments_shrinked['subreddit'].str.lower().isin(train_subreddits_list)]
    else:
        train_comments = None

    # Test data - taking only sr which according to the test_data_logic variable. If it is 'atlas_based', we will take
    # SRs which come from the atlas data only, if it 'submission_based' we will not include SRs coming from the atlas
    if test_data_logic == 'atlas_based':
        test_subreddits_df = subreddits_df[(subreddits_df["Source"] == 'atlas_main') &
                                           (pd.isna(subreddits_df["trying_to_draw"])) &
                                           (subreddits_df["trying_to_draw"] != '?')]
        test_subreddits_list = list(test_subreddits_df['SR'])
        test_submission = submission_shrinked[submission_shrinked['subreddit'].str.lower().isin(test_subreddits_list)]
        if use_comments_data:
            test_comments = comments_shrinked[comments_shrinked['subreddit'].str.lower().isin(test_subreddits_list)]

    elif test_data_logic == 'submission_based':
        test_subreddits_df = subreddits_df[(subreddits_df["Source"] != 'atlas_main') &
                                           (pd.isna(subreddits_df["trying_to_draw"])) &
                                           (subreddits_df["trying_to_draw"] != '?') &
                                           (~subreddits_df['SR'].isin(train_subreddits_list))]
        test_subreddits_list = list(test_subreddits_df['SR'])
        test_submission = submission_shrinked[submission_shrinked['subreddit'].str.lower().isin(test_subreddits_list)]
        if use_comments_data:
            test_comments = comments_shrinked[comments_shrinked['subreddit'].str.lower().isin(test_subreddits_list)]

    elif test_data_logic is None:
        test_subreddits_df = None
        test_submission = None
        test_comments = None
    else:
        print('test_data_logic must be either ''atlas_based'' or ''submission_based'' or None. Got something else')
        sys.exit(1)

    duration = (datetime.datetime.now() - start_time).seconds
    print("End of loading phase. Took us up to now: {} seconds".format(duration))

    # Data prep - train
    submission_dp_obj = RedditDataPrep(is_submission_data=True, remove_stop_words=False)
    comments_dp_obj = RedditDataPrep(most_have_regex=None, is_submission_data=False, remove_stop_words=False)
    train_sr_obj = []
    for idx, sr in enumerate(train_subreddits_list):
        cur_sr_submission = train_submission[train_submission['subreddit'].str.lower() == sr]
        # pulling out the meta data about the SR (the iloc[0] is used only to convert it into Series)
        sr_meta_data = train_subreddits_df[train_subreddits_df["SR"] == sr].iloc[0]
        train_sr_obj += [SubReddit(meta_data=sr_meta_data.to_dict())]
        cur_sr_submission_after_dp, submission_text = submission_dp_obj.data_pre_process(reddit_df=cur_sr_submission)
        train_sr_obj[idx].submissions_as_list = submission_text
        # pulling out the comments data - case we want to use it. If not, we'll skip to create the expl. features phase
        if use_comments_data:
            # first, we filter the comments, so only ones in the current sr we work with will appear
            cur_sr_comments = train_comments[train_comments['subreddit'].str.lower() == sr]
            submission_ids = set(['t3_'+sub_id for sub_id in cur_sr_submission_after_dp['id']])
            # second, we filter the comments, so only ones which are releant to the submissions dataset will appear
            # (this is due to the fact that we have already filtered a lot of submissions in pre step)
            cur_sr_comments = cur_sr_comments[cur_sr_comments['link_id'].isin(submission_ids)]
            cur_sr_comments_after_dp, comments_text = comments_dp_obj.data_pre_process(reddit_df=cur_sr_comments)
            train_sr_obj[idx].comments_as_list = comments_text
            train_sr_obj[idx].create_explanatory_features(submission_data=cur_sr_submission_after_dp,
                                                          comments_data=cur_sr_comments_after_dp)
        else:
            train_sr_obj[idx].create_explanatory_features(submission_data=cur_sr_submission_after_dp,
                                                          comments_data=None)

    # filtering out sr_objects which have zero submissions - these will probably have a label of -1 (didn't try to draw)
    train_sr_obj_shrinked = [sr_obj for sr_obj in train_sr_obj if sr_obj.explanatory_features['submission_amount'] > 0]
    # shuffle the data (the sr objects) so it will have different order
    idx = np.random.RandomState(seed=SEED).permutation(len(train_sr_obj_shrinked))
    train_sr_obj_shrinked = [train_sr_obj_shrinked[i] for i in idx]

    # creating the y vector feature and printing status
    y_train_data = []
    for cur_sr_obj in train_sr_obj_shrinked:
            y_train_data += [cur_sr_obj.trying_to_draw]

    print("\nTrain Data shape after all cleaning is as follow: {} SRs are going to be used for training,"
          "{} SRs were removed and will be labeled as -1, since no relevant posts were found in them. Target feature"
          " distribution is: {}".format(len(train_sr_obj_shrinked), len(train_sr_obj) - len(train_sr_obj_shrinked),
                                        dict(collections.Counter(y_train_data))))
    # Data prep - test
    if test_data_logic is not None:
        test_sr_obj = []
        for idx, sr in enumerate(test_subreddits_list):
            cur_sr_submission = test_submission[test_submission['subreddit'].str.lower() == sr]
            # pulling out the meta data about the SR (the iloc[0] is used only to convert it into Series)
            sr_meta_data = test_subreddits_df[test_subreddits_df["SR"] == sr].iloc[0]
            test_sr_obj += [SubReddit(meta_data=sr_meta_data.to_dict())]
            cur_sr_submission_after_dp, submission_text = submission_dp_obj.data_pre_process(reddit_df=cur_sr_submission)
            test_sr_obj[idx].submissions_as_list = submission_text
            # pulling out the comments data - case we want to use it.
            # If not, we'll skip to create the expl. features phase
            if use_comments_data:
                # first, we filter the comments, so only ones in the current sr we work with will appear
                cur_sr_comments = test_comments[test_comments['subreddit'].str.lower() == sr]
                submission_ids = set(['t3_'+sub_id for sub_id in cur_sr_submission_after_dp['id']])
                # second, we filter the comments, so only ones which are releant to the submissions dataset will appear
                # (this is due to the fact that we have already filtered a lot of submissions in pre step)
                cur_sr_comments = cur_sr_comments[cur_sr_comments['link_id'].isin(submission_ids)]
                cur_sr_comments_after_dp, comments_text = comments_dp_obj.data_pre_process(reddit_df=cur_sr_comments)
                test_sr_obj[idx].comments_as_list = comments_text
                test_sr_obj[idx].create_explanatory_features(submission_data=cur_sr_submission_after_dp,
                                                             comments_data=cur_sr_comments_after_dp)
            else:
                test_sr_obj[idx].create_explanatory_features(submission_data=cur_sr_submission_after_dp,
                                                             comments_data=None)

        # filtering out sr_objects which have zero submissions -
        # these will probably have a label of -1 (didn't try to draw)
        test_sr_obj_shrinked = [sr_obj for sr_obj in test_sr_obj if sr_obj.explanatory_features['submission_amount'] > 0]

        print("\nTest Data shape after all cleaning is as follow: {} SRs are going to be used as test-set, "
              "{} SRs were removed and will be labeled as -1, since no relevant posts "
              "were found in them.".format(len(test_sr_obj_shrinked), len(test_sr_obj) - len(test_sr_obj_shrinked)))

    # Modeling (learning phase)
    reddit_tokenizer = submission_dp_obj.tokenize_text
    cv_res, pipeline = fit_model(sr_objects=train_sr_obj_shrinked, y_vector=y_train_data, tokenizer=reddit_tokenizer,
                                 use_two_vectorizers=False, clf_model=GradientBoostingClassifier, stop_words=STOPLIST,
                                 ngram_size=2, vectorizers_general_params={'max_df': 0.8, 'min_df': 3},
                                 clf_parmas={'random_state': SEED, 'max_depth': 3,  'n_estimators': 100})
                                 #clf_parmas={'C': 0.2})

    print("Mean accuracy of current CV run is: {}. Here are the full results:\n".format(np.mean(cv_res['test_acc'])))
    for key, res in cv_res.items():
        print(key, res)
    # Analysis phase
    # pulling out the most dominant features, we need to train again based on the whole data-set
    # CURRENTLY WORKS ONLY WHEN use_two_vectorizers=false!! IF WANTS TO BE FIXED - WE CAN ADD ANOTHER PARAMETER TO
    # 'vectorizer' PARAMETER
    pipeline.fit(train_sr_obj_shrinked, y_train_data)
    clf = pipeline.steps[1][1]
    print_n_most_informative(vectorizer=[pipeline.named_steps['union'].get_params()[
                                             'ngram_features'].get_params()['steps'][1][1],
                                         pipeline.named_steps['union'].get_params()[
                                             'numeric_meta_features'].get_params()['steps'][1][1]], clf=clf, N=10)

    # Prediction phase (new and unseen data), saving results to csv
    if test_data_logic is not None:
        test_prediction = predict_model(pipeline=pipeline, sr_objects=test_sr_obj_shrinked)
        test_prediction = sorted(test_prediction, key=lambda t: t[1], reverse=True)
        with open('sr_classifier_predictions.csv', 'w', newline='') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(['SR', 'drawing_probability'])
            for row in test_prediction:
                csv_out.writerow(row)
    # summary
    duration = (datetime.datetime.now() - start_time).seconds
    print("End of full code. Took us up to now: {} seconds".format(duration))
