# Authors: Avrahami Israeli (isabrah)
# Python version: 3.6
# Last update: 6.11.2018

import warnings
warnings.simplefilter("ignore")
import gc
import sys
if sys.platform == 'linux':
    sys.path.append('/data/home/isabrah/reddit_project_sync_with_git_and_own_pc')
import os
import numpy as np
import collections
import datetime
import pickle
import multiprocessing as mp
import pandas as pd
from itertools import chain
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sr_classifier.utils import fit_model, print_n_most_informative
from r_place_drawing_classifier.utils import get_submissions_subset, get_comments_subset, \
    save_results_to_csv, examine_word, remove_huge_srs
from data_loaders.general_loader import sr_sample_based_subscribers, sr_sample_based_submissions
from sr_classifier.reddit_data_preprocessing import RedditDataPrep
from sr_classifier.sub_reddit import SubReddit
from nn_classifier import NNClassifier
from multiscorer import MultiScorer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from pandas.io.json import json_normalize
#import xgboost as xgb

STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))

###################################################### Configurations ##################################################
data_path = '/data/home/orentsur/data/reddit_place/' if sys.platform == 'linux' \
    else 'C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\PhD\\reddit canvas\\data\\'
comments_data_usage = {'meta_data': True, 'corpus': False}
SEED = 1984
build_sr_objects = False
use_networks_meta_data = True
save_results = False
model_authors_seq = False
use_external_embedding = True
remove_biggest_srs = False
subsmaple_submissions = True
classifier_type = 'BOW'  # should be 'DL' / 'BOW'
batch_number = 19
########################################################################################################################


def _sr_creation(srs_mapping, submission_dp_obj, comments_dp_obj, srs_to_create, pool_number):
    print("Pool # {} has started running the _sr_creation function".format(pool_number))
    start_time = datetime.datetime.now()
    empty_srs = 0
    sr_objects = []
    submission_data = get_submissions_subset(
        files_path=data_path + 'place_classifier_csvs/' if sys.platform == 'linux' else data_path + 'place_classifier_csvs\\',
        srs_to_include=srs_to_create)

    # same thing for the comments data
    if comments_data_usage['meta_data'] or comments_data_usage['corpus']:
        comments_data = get_comments_subset(
            files_path=data_path + 'place_classifier_csvs/' if sys.platform == 'linux' else data_path + 'place_classifier_csvs\\',
            srs_to_include=srs_to_create)
    else:
        comments_data = None

    for idx, sr in enumerate(srs_to_create):
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
        del cur_sr_submission
        cur_sr_obj.submissions_as_list = submission_text

        # pulling out the comments data - case we want to use it. There are a few option of comments usage
        # case we want to use comments data for either creation of meta-data and as part of the corpus (or both)
        if comments_data_usage['meta_data'] or comments_data_usage['corpus']:
            # first, we filter the comments, so only ones in the current sr we work with will appear
            cur_sr_comments = comments_data[comments_data['subreddit'].str.lower() == sr]
            submission_ids = set(['t3_' + sub_id for sub_id in cur_sr_submission_after_dp['id']])
            # second, we filter the comments, so only ones which are relevant to the submissions dataset will appear
            # (this is due to the fact that we have already filtered a lot of submissions in pre step)
            cur_sr_comments = cur_sr_comments[cur_sr_comments['link_id'].isin(submission_ids)]
            cur_sr_comments_after_dp, comments_text = comments_dp_obj.data_pre_process(reddit_df=cur_sr_comments)
            del cur_sr_comments
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
        gc.collect()
        # printing progress
        if idx % 50 == 0 and idx != 0:
            duration = (datetime.datetime.now() - start_time).seconds
            print("Pool # {} reporting finish of the {} iteration. Took this pool {} seconds, "
                  "moving forward".format(pool_number, idx, duration))
    duration = (datetime.datetime.now() - start_time).seconds
    print("Pool # {} reporting finished passing over all SRs. Took him up to now {} seconds and he created {} SRs "
          "objects (some where empty, so weren't created)".format(pool_number, duration, len(sr_objects)))
    del submission_data
    del comments_data
    gc.collect()
    return sr_objects


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    if build_sr_objects:
        # first step will be to sample the data, then we will represent the result as a dictionary here
        '''
        srs_mapping = sr_sample_based_subscribers(data_path=data_path, sample_size='1:1',
                                                  threshold_to_define_as_drawing=0.7,
                                                  balanced_sampling_based_sr_size=True, internal_sr_metadata=True,
                                                  sr_statistics_usage=True)
        '''
        # sampling SRs based on number of submissions
        srs_mapping = sr_sample_based_submissions(data_path=data_path, sample_size='1.1:1', start_peoriod='2017-01',
                                                  end_period='2017-03', threshold_to_define_as_drawing=0.7)
        srs_mapping = {sr[0]: (sr[1], sr[2], sr[3]) for sr in srs_mapping}
        print("We are going to handle {} srs. This is the drawing/not-drawing "
              "distribution: {}".format(len(srs_mapping),
                                        collections.Counter([value[2] for key, value in srs_mapping.items()])))
        # here we sample only 1000 SRs, we will do it 5 times to cover all SRs sampled
        srs_names = list(srs_mapping.keys())
        # sorting the list of names, so we can transfer it to couple of slaves on the cluster without worrying about the
        # order of them
        srs_names.sort()
        srs_names = srs_names[(400+batch_number*100):(400+(batch_number+2)*100)]
        # pulling the submission data, based on the subset of SRs we decided on
        # Data prep - train
        submission_dp_obj = RedditDataPrep(is_submission_data=True, remove_stop_words=False, most_have_regex=None)
        comments_dp_obj = RedditDataPrep(is_submission_data=False, remove_stop_words=False, most_have_regex=None)
        processes_amount = 1
        chunk_size = int(len(srs_names) * 1.0 / processes_amount)
        srs_names_in_chunks = [srs_names[i * chunk_size: i * chunk_size + chunk_size] for i in
                               range(processes_amount - 1)]
        # last one can be tricky, so handling it more carefully
        srs_names_in_chunks.append(srs_names[(processes_amount - 1) * chunk_size:])
        input_for_pool = [(srs_mapping, submission_dp_obj, comments_dp_obj, srs_names_in_chunks[i], i) for i in
                          range(processes_amount)]
        pool = mp.Pool(processes=processes_amount)
        with pool as pool:
            results = pool.starmap(_sr_creation, input_for_pool)
        sr_objects = list(chain.from_iterable(results))

        # shuffle the data (the sr objects) so it will have different order
        idx = np.random.RandomState(seed=SEED).permutation(len(sr_objects))
        sr_objects = [sr_objects[i] for i in idx]
        # saving the data up to now into a pickle file
        pickle.dump(sr_objects, open(data_path + "sr_objects_102016_to_032017_sampling_based_submission_batch_"+str(batch_number)+".p", "wb"))

        duration = (datetime.datetime.now() - start_time).seconds
        print("\nData shape after all cleaning is as follow: {} SRs are going to be used for training,"
              "Total run time is: {}".format(len(sr_objects), duration))
    # case we do not build SR objects, but rather using existing pickle file holding these objects
    else:
        #sr_objects = pickle.load(open(data_path + "sr_objects_102016_to_032017_sample.p", "rb"))
        #sr_objects = pickle.load(open(data_path + "sr_objects_6_months_sampling_based_subscibers_5_12_2018.p", "rb"))
        sr_objects = pickle.load(open(data_path + "sr_objects_6_months_sampling_based_submission_30_12_2018.p", "rb"))
        sr_objects = sr_objects
        # function to remove huge SRs, so parallelism can be applied
        if remove_biggest_srs:
            sr_objects = remove_huge_srs(sr_objects=sr_objects, quantile=0.2)

        # adding meta features created by Alex for each SR network and data-prep to the meta-features
        missing_srs_due_to_meta_features = []
        missing_srs_due_to_authors_seq = []
        if model_authors_seq:
            with open(data_path + "sequence_dict.pkl", 'rb') as f:
                conversations_seq = pickle.load(f)
        else:
            conversations_seq = None
        network_features = data_path + 'graph_dict.pickle' if use_networks_meta_data else None
        for idx, cur_sr_obj in enumerate(sr_objects):

            res = cur_sr_obj.meta_features_handler(smooth_zero_features=True,
                                                   network_features=network_features)
                                                   #features_to_exclude=set(sr_objects[idx].explanatory_features) - {'submission_amount'})
            # case there was a problem with the function, we will remove the sr from the data
            if res != 0:
                missing_srs_due_to_meta_features.append(cur_sr_obj.name)
            # case we want to model authors sequence instead of sequence of words in a submission
            if model_authors_seq:
                try:
                    cur_sr_obj.replace_sentences_with_authors_seq(conversations=conversations_seq[cur_sr_obj.name])
                # case the SR is not in the dict Alex created
                except KeyError:
                    missing_srs_due_to_authors_seq.append(cur_sr_obj.name)
            # submission data under sampling
            if subsmaple_submissions:
                cur_sr_obj.subsample_submissions_data(subsample_logic='score', percentage=0.2,
                                                      maximum_submissions=10, seed=1984)
        duration = (datetime.datetime.now() - start_time).seconds
        combined_missing_srs = set(missing_srs_due_to_meta_features + missing_srs_due_to_authors_seq)
        print("Ended the process of adding network meta features and converting sentences into authors sequence "
              "(if it was required).\n Network features were not found for {} srs, authors sequance was not found"
              "for {} SRs. Bottom line, {} will be removed due to these 2 processes."
              "Up to now we ran for {} sec.".format(len(missing_srs_due_to_meta_features),
                                                    len(missing_srs_due_to_authors_seq),
                                                    len(combined_missing_srs),
                                                    duration))
        # deleting the terrible object :)
        del conversations_seq
        gc.collect()
        sr_objects = [sr for sr in sr_objects if sr.name not in set(combined_missing_srs)]
        # creating the y vector feature and printing status
        y_data = []
        for idx, cur_sr_obj in enumerate(sr_objects):
            y_data += [cur_sr_obj.trying_to_draw]
            # this was used in order to create random y vector, and see results really get totally random
            # y_data += [int(np.random.choice(a=[-1, 1], size=1))]
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
        if classifier_type == 'BOW':
            cv_res, pipeline, predictions =\
                fit_model(sr_objects=sr_objects, y_vector=y_data, tokenizer=reddit_tokenizer, ngram_size=2,
                          use_two_vectorizers=False,
                          #clf_model=xgb.XGBClassifier,
                          clf_model=GradientBoostingClassifier,
                          stop_words=STOPLIST,
                          vectorizers_general_params={'max_df': 0.8, 'min_df': 3, 'max_features': 300},
                          #clf_parmas={'hidden_layer_sizes': (100, 50, 10)})
                          clf_parmas={'random_state': SEED, 'max_depth': 3,  'n_estimators': 100})
                          #clf_parmas={'C': 1.0})
            if save_results:
                save_results_to_csv(start_time=start_time, SRs_amount=len(sr_objects),
                                    models_params=pipeline.steps, results=cv_res, saving_path=os.getcwd())

                # Analysis phase
                res_summary = [(y_data[i], predictions[i, 1], sr_objects[i].name) for i in range(len(y_data))]
                res_summary_df = pd.DataFrame(res_summary, columns=['true_y', 'prediction_to_draw', 'sr_name'])
                res_summary_df.to_csv(data_path + 'results_summary_submission_based_sampling_bow_plus_meta.csv')
            print("Full modeling code has ended. Results are as follow: {}."
                  "The process started at {} and finished at {}".format(cv_res, start_time, datetime.datetime.now()))

            # pulling out the most dominant features, we need to train again based on the whole data-set
            # CURRENTLY WORKS ONLY WHEN use_two_vectorizers=false!! IF WANTS TO BE FIXED - WE CAN ADD ANOTHER PARAMETER TO
            # 'vectorizer' PARAMETER
            pipeline.fit(sr_objects, y_data)
            clf = pipeline.steps[1][1]
            print_n_most_informative(vectorizer=[pipeline.named_steps['union'].get_params()[
                                                     'ngram_features'].get_params()['steps'][1][1],
                                                 pipeline.named_steps['union'].get_params()[
                                                     'numeric_meta_features'].get_params()['steps'][1][1]], clf=clf, N=20)
        elif classifier_type == 'DL':
            '''
            handling the embedding file (if we want to use an external one). This can be applied only in case we model
            the actual words in each SR, since otherwise we use the authors names, and it doesn't make sense to use
            pre defined embedding in such cases
            '''
            embedding_file = data_path + 'embedding/' + 'glove.42B.300d.txt' \
                if use_external_embedding and not model_authors_seq else None
            eval_measures_dict = {'accuracy': accuracy_score, 'precision': precision_score, 'recall': recall_score}
            dl_obj = NNClassifier(tokenizer=reddit_tokenizer, eval_measures=eval_measures_dict,
                                  model_type='lstm', emb_size=300, hid_size=500, early_stopping=True,
                                  epochs=20, use_bilstm=False, use_meta_features=True, model_sentences=False,
                                  maximum_sent_per_sr=None, seed=1984)
            cv_obj = StratifiedKFold(n_splits=5, random_state=SEED)
            cv_obj.get_n_splits(sr_objects, y_data)
            all_test_data_pred = []
            for cv_idx, (train_index, test_index) in enumerate(cv_obj.split(sr_objects, y_data)):
                print("Fold {} starts".format(cv_idx))
                cur_train_sr_objects = [sr_objects[i] for i in train_index]
                cur_test_sr_objects = [sr_objects[i] for i in test_index]
                cur_y_train = [y_data[i] for i in train_index]
                cur_y_test = [y_data[i] for i in test_index]
                #cur_results, cur_model, cur_test_predictions = \
                #    dl_obj.fit_two_layers_lstm(train_data=cur_train_sr_objects, test_data=cur_test_sr_objects,
                #                               embedding_file=embedding_file, second_layer_as_lstm=False)

                cur_results, cur_model, cur_test_predictions = \
                    dl_obj.fit_simple_mlp(train_data=cur_train_sr_objects, test_data=cur_test_sr_objects)

                print("Fold # {} has ended, updated results list is: {}".format(cv_idx, cur_results))
                cur_test_sr_names = [sr_obj.name for sr_obj in cur_test_sr_objects]
                all_test_data_pred.extend([(y, pred, name) for name, y, pred in
                                           zip(cur_test_sr_names, cur_y_test, cur_test_predictions)])

            # saving results to file if needed
            if save_results:
                eval_results = dl_obj.eval_results
                dl_params_tp_save = dl_obj.__dict__
                dl_params_tp_save.pop('w2i')
                dl_params_tp_save.pop('t2i')
                dl_params_tp_save.pop('eval_results')
                dl_params_tp_save.pop('eval_measures')
                save_results_to_csv(start_time=start_time, SRs_amount=len(sr_objects),
                                    models_params=dl_params_tp_save, results=eval_results, saving_path=os.getcwd())
                print("Full modeling code has ended. Results are as follow: {}."
                      "\nThe process started at {} and finished at {}".format(eval_results, start_time,
                                                                              datetime.datetime.now()))

                # Analysis phase, saving the required data to a csv file
                res_summary_df = pd.DataFrame(all_test_data_pred, columns=['true_y', 'prediction_to_draw', 'sr_name'])
                res_summary_df.to_csv(data_path + 'dl_sentences_modeling_100_sent_length.csv', index=False)

            '''
            cur_test_sr_name_with_y = {sr_obj.name: sr_obj.trying_to_draw for sr_obj in cur_test_sr_objects}
            cur_test_results_summary = [(sent_score[1], sr_name, cur_test_sr_name_with_y[sr_name])
                                        for sent_score, sr_name, y in
                                        zip(dl_model_scores, cur_test_data_names, cur_y_test)]
            # converting it into pandas - much easier to work with
            cur_test_res_df = pd.DataFrame(cur_test_results_summary,
                                           columns=['prediction', 'sr_name', 'trying_to_draw'])
            cur_test_res_agg = cur_test_res_df.groupby(['sr_name'], as_index=False).agg(
                {'prediction': ['mean', 'max', 'min'], 'trying_to_draw': 'mean'})
            cur_test_res_agg.columns = ['sr_name', 'pred_mean', 'pred_max', 'pred_min', 'trying_to_draw']
            print("wow")
            '''


