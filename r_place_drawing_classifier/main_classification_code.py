# Authors: Avrahami Israeli (isabrah)
# Python version: 3.6
# Last update: 6.11.2018

import sys
if sys.platform == 'linux':
    sys.path.append('/home/reddit_pycharm_proj_with_own_pc')
import os
import numpy as np
import collections
import datetime
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sr_classifier.utils import fit_model, print_n_most_informative
from r_place_drawing_classifier.utils import get_submissions_subset, get_comments_subset, save_results_to_csv, examine_word
from data_loaders.general_loader import sr_sample
from sr_classifier.reddit_data_preprocessing import RedditDataPrep
from sr_classifier.sub_reddit import SubReddit

STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))

###################################################### Configurations ##################################################
data_path = '/home/reddit_data/' if sys.platform == 'linux' \
    else 'C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\PhD\\reddit canvas\\data\\'
comments_data_usage = {'meta_data': True, 'corpus': False}
SEED = 1984
build_sr_objects = False
########################################################################################################################
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    if build_sr_objects:
        # first step will be to sample the data, then we will represent the result as a dictionary
        srs_mapping = sr_sample(data_path=data_path, sample_size='1:1', threshold_to_define_as_drawing=0.7,
                                balanced_sampling_based_sr_size=True, internal_sr_metadata=True,
                                sr_statistics_usage=True)
        srs_mapping = {sr[0]: (sr[1], sr[2], sr[3]) for sr in srs_mapping}
        print("We are going to handle {} srs. This is the drawing/not-drawing "
              "distribution: {}".format(len(srs_mapping),
                                        collections.Counter([value[2] for key, value in srs_mapping.items()])))
        srs_names = list(srs_mapping.keys())
        # pulling the submission data, based on the subset of SRs we decided on
        submission_data = get_submissions_subset(files_path=data_path + 'place_classifier_csvs/' if sys.platform == 'linux' else data_path + 'place_classifier_csvs\\',
                                                 srs_to_include=srs_names)

        # same thing for the comments data
        if comments_data_usage['meta_data'] or comments_data_usage['corpus']:
            comments_data = get_comments_subset(files_path=data_path + 'place_classifier_csvs/' if sys.platform == 'linux' else data_path + 'place_classifier_csvs\\',
                                                srs_to_include=srs_names)
        else:
            comments_data = None
        # Data prep - train
        submission_dp_obj = RedditDataPrep(is_submission_data=True, remove_stop_words=False, most_have_regex=None)
        comments_dp_obj = RedditDataPrep(is_submission_data=False, remove_stop_words=False, most_have_regex=None)

        sr_objects = []
        empty_srs = 0
        # iterating over the SRs and adding some useful information for them (i.e., extra meta-features)
        for idx, sr in enumerate(srs_names):
            cur_sr_submission = submission_data[submission_data['subreddit'].str.lower() == sr]
            # case there are no relevant submissions to this sr
            if cur_sr_submission.shape[0] == 0:
                empty_srs += 1
                continue
            # pulling out the meta data about the SR (the iloc[0] is used only to convert it into Series)
            sr_meta_data = {'SR': sr, 'creation_utc': srs_mapping[sr][1], 'end_state': None,
                            'num_of_users': srs_mapping[sr][0],
                            'trying_to_draw': 'Yes' if srs_mapping[sr][2] == 'drawing' else 'No', 'models_prediction': -1}
            cur_sr_obj = SubReddit(meta_data=sr_meta_data)
            cur_sr_submission_after_dp, submission_text = submission_dp_obj.data_pre_process(reddit_df=cur_sr_submission)
            cur_sr_obj.submissions_as_list = submission_text

            # pulling out the comments data - case we want to use it. There are a few option of comments usage
            # case we want to use comments data for either creation of meta-data and as part of the corpus (or both)
            if comments_data_usage['meta_data'] or comments_data_usage['corpus']:
                # first, we filter the comments, so only ones in the current sr we work with will appear
                cur_sr_comments = comments_data[comments_data['subreddit'].str.lower() == sr]
                submission_ids = set(['t3_'+sub_id for sub_id in cur_sr_submission_after_dp['id']])
                # second, we filter the comments, so only ones which are relevant to the submissions dataset will appear
                # (this is due to the fact that we have already filtered a lot of submissions in pre step)
                cur_sr_comments = cur_sr_comments[cur_sr_comments['link_id'].isin(submission_ids)]
                cur_sr_comments_after_dp, comments_text = comments_dp_obj.data_pre_process(reddit_df=cur_sr_comments)
            # case we want to use comments data for meta-features creation (very logical to be used)
            if comments_data_usage['meta_data']:
                cur_sr_obj.create_explanatory_features(submission_data=cur_sr_submission_after_dp,
                                                       comments_data=cur_sr_comments_after_dp)
            # case we want to use only submission data for meta-data creation
            else:
                cur_sr_obj.create_explanatory_features(submission_data=cur_sr_submission_after_dp, comments_data=None)
            # case we want to use comments data as part of the corpus creation (most of the times not the case)
            if comments_data_usage['corpus']:
                cur_sr_obj.comments_as_list = comments_text
            sr_objects += [cur_sr_obj]
            # printing progress
            duration = (datetime.datetime.now() - start_time).seconds
            if idx % 100 == 0:
                print("Finished passing over {} SRs. Took us up to now {} seconds and we created {} SRs objects "
                      "(some where empty, so weren't created)".format(idx, duration, len(sr_objects)))

        # shuffle the data (the sr objects) so it will have different order
        idx = np.random.RandomState(seed=SEED).permutation(len(sr_objects))
        sr_objects = [sr_objects[i] for i in idx]
        # saving the data up to now into a pickle file
        pickle.dump(sr_objects, open("sr_objects_102016_to_032017_balanced.p", "wb"))

        duration = (datetime.datetime.now() - start_time).seconds
        print("\nData shape after all cleaning is as follow: {} SRs are going to be used for training,"
              "{} SRs were removed, since no relevant posts were found in them."
              "Total run time is: {}".format(len(sr_objects), empty_srs, duration))
    # case we do not build SR objects, but rather using existing pickle file holding these objects
    else:
        sr_objects = pickle.load(open(data_path + "sr_objects_012017_to_032017_balanced.p", "rb"))
        sr_objects = sr_objects[0:1000]
        # creating the y vector feature and printing status
        y_data = []
        for idx, cur_sr_obj in enumerate(sr_objects):
            y_data += [cur_sr_obj.trying_to_draw]
            # this was used in order to create random y vector, and see results really get totally random
            # y_data += [int(np.random.choice(a=[-1, 1], size=1))]
            # fixing the 'days_pazam' feature if needed
            cur_sr_obj.explanatory_features.pop('submission_amount_normalized')
            if type(cur_sr_obj.explanatory_features['days_pazam']) is datetime.timedelta:
                cur_sr_obj.explanatory_features['days_pazam'] = cur_sr_obj.explanatory_features['days_pazam'].days
        print("Target feature distribution is: {}".format(collections.Counter(y_data)))
        # Modeling (learning phase)
        submission_dp_obj = RedditDataPrep(is_submission_data=True, remove_stop_words=False, most_have_regex=None)
        reddit_tokenizer = submission_dp_obj.tokenize_text
        '''
        for sr in sr_objects[80:100]:
            examine_word(sr_object=sr, regex_required='rank', tokenizer=reddit_tokenizer,
                         saving_file=os.getcwd())
        print("Finished analysis phase, moving to modeling")
        '''
        cv_res, pipeline = fit_model(sr_objects=sr_objects, y_vector=y_data, tokenizer=reddit_tokenizer,
                                     use_two_vectorizers=False, clf_model=GradientBoostingClassifier, stop_words=STOPLIST,
                                     ngram_size=2,
                                     vectorizers_general_params={'max_df': 0.8, 'min_df': 3, 'max_features': 300},
                                     clf_parmas={'random_state': SEED, 'max_depth': 3,  'n_estimators': 50})
                                     #clf_parmas={'C': 1.0})
        save_results_to_csv(start_time=start_time, SRs_amount=len(sr_objects),
                            models_params=pipeline.steps, results=cv_res, saving_path=os.getcwd())
        print("Full modeling code has ended. Results are as follow: {}."
              "The process started at {} and finished at {}".format(cv_res, start_time, datetime.datetime.now()))

        # Analysis phase
        # pulling out the most dominant features, we need to train again based on the whole data-set
        # CURRENTLY WORKS ONLY WHEN use_two_vectorizers=false!! IF WANTS TO BE FIXED - WE CAN ADD ANOTHER PARAMETER TO
        # 'vectorizer' PARAMETER
        pipeline.fit(sr_objects, y_data)
        clf = pipeline.steps[1][1]
        print_n_most_informative(vectorizer=[pipeline.named_steps['union'].get_params()[
                                                 'ngram_features'].get_params()['steps'][1][1],
                                             pipeline.named_steps['union'].get_params()[
                                                 'numeric_meta_features'].get_params()['steps'][1][1]], clf=clf, N=20)
