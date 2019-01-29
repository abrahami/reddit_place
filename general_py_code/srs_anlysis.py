import pickle
import numpy as np
import sys
import os
import re
import collections
import pandas as pd
from collections import defaultdict
import datetime
import json

# function to sum up two defaultdict
def dsum(*dicts):
    ret = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            ret[k] += v
    return dict(ret)

def dsum_sets(*dicts):
    ret = defaultdict(set)
    for d in dicts:
        for k, v in d.items():
            ret[k] = ret[k].union(v)
    return dict(ret)

def dmin(*dicts):
    ret = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            #since the default dict returns 0 by default, we'll check if the key exists, since we need minumum (zero fucks it all)
            if k in ret:
                ret[k] = min(ret[k], v)
            else:
                ret[k] = v
    return dict(ret)

analysis = 'all_sr_stats'#'drawing_not_drawing_sr_stats'
start_time = datetime.datetime.now()
# 1. general sr statistics
if analysis == 'drawing_not_drawing_sr_stats':
    statistics = dict()
    data_path = '/data/home/orentsur/data/reddit_place/' if sys.platform == 'linux' \
        else 'C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\PhD\\reddit canvas\\data\\'
    sr_objects = pickle.load(open(data_path + "sr_objects_6_months_sampling_based_submission_30_12_2018.p", "rb"))

    drawing_sr_objects = [sr for sr in sr_objects if sr.trying_to_draw == 1]
    not_drawing_sr_objects = [sr for sr in sr_objects if sr.trying_to_draw == -1]

    statistics["communities"] = [len(drawing_sr_objects), len(not_drawing_sr_objects), len(sr_objects)]

    drawing_comm_avg_score = np.mean([sr.explanatory_features['comments_average_score'] for sr in drawing_sr_objects if 'comments_average_score' in sr.explanatory_features])
    not_drawing_comm_avg_score = np.mean([sr.explanatory_features['comments_average_score'] for sr in not_drawing_sr_objects if 'comments_average_score' in sr.explanatory_features])

    round(sum([d_sr.num_users for d_sr in drawing_sr_objects if type(d_sr.num_users) is not str]),2)
    round(np.mean([d_sr.num_users for d_sr in drawing_sr_objects if type(d_sr.num_users) is not str]), 2)
    round(np.std([d_sr.num_users for d_sr in drawing_sr_objects if type(d_sr.num_users) is not str]), 2)

    statistics["subscribers"] = [(round(sum([d_sr.num_users for d_sr in drawing_sr_objects if type(d_sr.num_users) is not str]), 2),
                                  round(np.mean([d_sr.num_users for d_sr in drawing_sr_objects if type(d_sr.num_users) is not str]), 2),
                                  round(np.median([d_sr.num_users for d_sr in drawing_sr_objects if type(d_sr.num_users) is not str]), 2),
                                  round(np.std([d_sr.num_users for d_sr in drawing_sr_objects if type(d_sr.num_users) is not str]), 2)),

                                 (round(sum([d_sr.num_users for d_sr in not_drawing_sr_objects if type(d_sr.num_users) is not str]),2),
                                  round(np.mean([d_sr.num_users for d_sr in not_drawing_sr_objects if type(d_sr.num_users) is not str]), 2),
                                  round(np.median([d_sr.num_users for d_sr in not_drawing_sr_objects if type(d_sr.num_users) is not str]), 2),
                                  round(np.std([d_sr.num_users for d_sr in not_drawing_sr_objects if type(d_sr.num_users) is not str]), 2)),

                                 (round(sum([d_sr.num_users for d_sr in sr_objects if type(d_sr.num_users) is not str]), 2),
                                  round(np.mean([d_sr.num_users for d_sr in sr_objects if type(d_sr.num_users) is not str]), 2),
                                  round(np.median([d_sr.num_users for d_sr in sr_objects if type(d_sr.num_users) is not str]), 2),
                                  round(np.std([d_sr.num_users for d_sr in sr_objects if type(d_sr.num_users) is not str]), 2))
                                 ]

    statistics["submissions"] = [(round(sum([len(d_sr.submissions_as_list) for d_sr in drawing_sr_objects]), 2),
                                 round(np.mean([len(d_sr.submissions_as_list) for d_sr in drawing_sr_objects]), 2),
                                 round(np.median([len(d_sr.submissions_as_list) for d_sr in drawing_sr_objects]), 2),
                                 round(np.std([len(d_sr.submissions_as_list) for d_sr in drawing_sr_objects]), 2)),

                                 (round(sum([len(d_sr.submissions_as_list) for d_sr in not_drawing_sr_objects]), 2),
                                 round(np.mean([len(d_sr.submissions_as_list) for d_sr in not_drawing_sr_objects]), 2),
                                 round(np.median([len(d_sr.submissions_as_list) for d_sr in not_drawing_sr_objects]), 2),
                                 round(np.std([len(d_sr.submissions_as_list) for d_sr in not_drawing_sr_objects]), 2)),

                                 (round(sum([len(d_sr.submissions_as_list) for d_sr in sr_objects]), 2),
                                 round(np.mean([len(d_sr.submissions_as_list) for d_sr in sr_objects]), 2),
                                 round(np.median([len(d_sr.submissions_as_list) for d_sr in sr_objects]), 2),
                                 round(np.std([len(d_sr.submissions_as_list) for d_sr in sr_objects]), 2))
                                 ]

    statistics["comments"] = [(round(sum([d_sr.explanatory_features['comments_submission_ratio'] for d_sr in drawing_sr_objects if 'comments_submission_ratio' in d_sr.explanatory_features]), 2),
                               round(np.mean([d_sr.explanatory_features['comments_submission_ratio'] for d_sr in drawing_sr_objects if 'comments_submission_ratio' in d_sr.explanatory_features]), 2),
                               round(np.median([d_sr.explanatory_features['comments_submission_ratio'] for d_sr in drawing_sr_objects if 'comments_submission_ratio' in d_sr.explanatory_features]), 2),
                               round(np.std([d_sr.explanatory_features['comments_submission_ratio'] for d_sr in drawing_sr_objects if 'comments_submission_ratio' in d_sr.explanatory_features]), 2)),

                              (round(sum([d_sr.explanatory_features['comments_submission_ratio'] for d_sr in not_drawing_sr_objects if 'comments_submission_ratio' in d_sr.explanatory_features]), 2),
                               round(np.mean([d_sr.explanatory_features['comments_submission_ratio'] for d_sr in not_drawing_sr_objects if 'comments_submission_ratio' in d_sr.explanatory_features]), 2),
                               round(np.median([d_sr.explanatory_features['comments_submission_ratio'] for d_sr in not_drawing_sr_objects if 'comments_submission_ratio' in d_sr.explanatory_features]), 2),
                               round(np.std([d_sr.explanatory_features['comments_submission_ratio'] for d_sr in not_drawing_sr_objects if 'comments_submission_ratio' in d_sr.explanatory_features]), 2)),

                              (round(sum([d_sr.explanatory_features['comments_submission_ratio'] for d_sr in sr_objects if 'comments_submission_ratio' in d_sr.explanatory_features]), 2),
                               round(np.mean([d_sr.explanatory_features['comments_submission_ratio'] for d_sr in sr_objects if 'comments_submission_ratio' in d_sr.explanatory_features]), 2),
                               round(np.median([d_sr.explanatory_features['comments_submission_ratio'] for d_sr in sr_objects if 'comments_submission_ratio' in d_sr.explanatory_features]), 2),
                               round(np.std([d_sr.explanatory_features['comments_submission_ratio'] for d_sr in sr_objects if 'comments_submission_ratio' in d_sr.explanatory_features]), 2))
                              ]

    statistics["Seniority"] = [(round(sum([d_sr.explanatory_features['days_pazam'] for d_sr in drawing_sr_objects if 'days_pazam' in d_sr.explanatory_features and d_sr.explanatory_features['days_pazam'] is not None]), 2),
                               round(np.mean([d_sr.explanatory_features['days_pazam'] for d_sr in drawing_sr_objects if 'days_pazam' in d_sr.explanatory_features and d_sr.explanatory_features['days_pazam'] is not None]), 2),
                               round(np.median([d_sr.explanatory_features['days_pazam'] for d_sr in drawing_sr_objects if 'days_pazam' in d_sr.explanatory_features and d_sr.explanatory_features['days_pazam'] is not None]), 2),
                               round(np.std([d_sr.explanatory_features['days_pazam'] for d_sr in drawing_sr_objects if 'days_pazam' in d_sr.explanatory_features and d_sr.explanatory_features['days_pazam'] is not None]), 2)),

                              (round(sum([d_sr.explanatory_features['days_pazam'] for d_sr in not_drawing_sr_objects if 'days_pazam' in d_sr.explanatory_features and d_sr.explanatory_features['days_pazam'] is not None]), 2),
                               round(np.mean([d_sr.explanatory_features['days_pazam'] for d_sr in not_drawing_sr_objects if 'days_pazam' in d_sr.explanatory_features and d_sr.explanatory_features['days_pazam'] is not None]), 2),
                               round(np.median([d_sr.explanatory_features['days_pazam'] for d_sr in not_drawing_sr_objects if 'days_pazam' in d_sr.explanatory_features and d_sr.explanatory_features['days_pazam'] is not None]), 2),
                               round(np.std([d_sr.explanatory_features['days_pazam'] for d_sr in not_drawing_sr_objects if 'days_pazam' in d_sr.explanatory_features and d_sr.explanatory_features['days_pazam'] is not None]), 2)),

                              (round(sum([d_sr.explanatory_features['days_pazam'] for d_sr in sr_objects if 'days_pazam' in d_sr.explanatory_features and d_sr.explanatory_features['days_pazam'] is not None]), 2),
                               round(np.mean([d_sr.explanatory_features['days_pazam'] for d_sr in sr_objects if 'days_pazam' in d_sr.explanatory_features and d_sr.explanatory_features['days_pazam'] is not None]), 2),
                               round(np.median([d_sr.explanatory_features['days_pazam'] for d_sr in sr_objects if 'days_pazam' in d_sr.explanatory_features and d_sr.explanatory_features['days_pazam'] is not None]), 2),
                               round(np.std([d_sr.explanatory_features['days_pazam'] for d_sr in sr_objects if 'days_pazam' in d_sr.explanatory_features and d_sr.explanatory_features['days_pazam'] is not None]), 2))
                              ]

elif analysis == 'all_sr_stats':
    start_peoriod = '2016-10'
    end_period = '2017-03'
    data_path = '/data/home/orentsur/data/reddit_place/' if sys.platform == 'linux' \
        else 'C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\PhD\\reddit canvas\\data\\'
    csv_path = \
        data_path + 'place_classifier_csvs/' if sys.platform == 'linux' else data_path + 'place_classifier_csvs\\'
    submission_files = [f for f in os.listdir(csv_path) if re.match(r'RS.*\.csv', f)]
    # taking only the submissions files from 10-2016 to 03-2017 by default
    submission_files = [i for i in submission_files if 'RS_'+start_peoriod + '.csv' <= i <= 'RS_'+end_period+'.csv']
    submission_files = sorted(submission_files)
    submm_amount = defaultdict(int)
    comm_amount = defaultdict(int)
    srs_users = defaultdict(set)
    idle_time = defaultdict(int)
    # iterating over each submission file
    for subm_idx, cur_submission_file in enumerate(submission_files):
        cur_submission_df = pd.read_csv(filepath_or_buffer=csv_path + cur_submission_file, encoding='utf-8')
        # aggregating data using 'group-by' operations
        cur_submission_df['days_since_rplace'] = datetime.datetime(2017, 4, 1) - pd.to_datetime(cur_submission_df['created_utc_as_date'])
        cur_submission_df['days_since_rplace'] = cur_submission_df['days_since_rplace'].apply(lambda x: x.days)
        group_by_obj = cur_submission_df.groupby('subreddit', as_index=False)
        cur_submm_amount = dict(group_by_obj.size())
        cur_comm_amount = group_by_obj.agg({'num_comments': 'sum'})
        cur_comm_amount = cur_comm_amount.set_index('subreddit').to_dict()['num_comments']
        cur_srs_users = group_by_obj.agg({"author": "unique"})
        cur_srs_users = cur_srs_users.set_index('subreddit').to_dict()['author']  # convert it to be a dict where key is the sr name
        cur_idle_time = group_by_obj.agg({'days_since_rplace': 'min'})
        cur_idle_time = cur_idle_time.set_index('subreddit').to_dict()['days_since_rplace']

        # convert each dictioanries to lowercase and the list of users to set of users
        cur_submm_amount = {key.lower(): value for key, value in cur_submm_amount.items()}
        cur_comm_amount = {key.lower(): value for key, value in cur_comm_amount.items()}
        cur_srs_users = {key.lower(): set(value) for key, value in cur_srs_users.items()}
        cur_idle_time = {key.lower(): value for key, value in cur_idle_time.items()}

        # summing up to the big dicts
        submm_amount = dsum(submm_amount, cur_submm_amount)
        comm_amount = dsum(comm_amount, cur_comm_amount)
        srs_users = dsum_sets(srs_users, cur_srs_users)
        idle_time = dmin(idle_time, cur_idle_time)

        duration = (datetime.datetime.now() - start_time).seconds
        print("Finished to work on {} file, this is loop # {}."
              "Time up to now: {}".format(cur_submission_file, subm_idx, duration))

    # now handeling the commnets
    comments_files = [f for f in os.listdir(csv_path) if re.match(r'RC.*\.csv', f)]
    comments_files = [i for i in comments_files if 'RC_' + start_peoriod + '.csv' <= i <= 'RC_' + end_period + '.csv']
    comments_files = sorted(comments_files)
    for comm_idx, cur_comments_file in enumerate(comments_files):
        if sys.platform == 'linux':
            cur_comments_df = pd.read_csv(filepath_or_buffer=csv_path + cur_comments_file)
        else:
            cur_comments_df = pd.read_csv(filepath_or_buffer=csv_path + cur_comments_file, encoding='latin-1')
        cur_comments_df['days_since_rplace'] = datetime.datetime(2017, 4, 1) - pd.to_datetime(cur_comments_df['created_utc_as_date'])
        cur_comments_df['days_since_rplace'] = cur_comments_df['days_since_rplace'].apply(lambda x: x.days)
        group_by_obj = cur_comments_df.groupby('subreddit', as_index=False)
        cur_srs_users = group_by_obj.agg({"author": "unique"})
        cur_srs_users = cur_srs_users.set_index('subreddit').to_dict()['author']  # convert it to be a dict where key is the sr name
        cur_idle_time = group_by_obj.agg({'days_since_rplace': 'min'})
        cur_idle_time = cur_idle_time.set_index('subreddit').to_dict()['days_since_rplace']

        # convert each dictioanry to lowercase and the list of users to set of users
        cur_srs_users = {key.lower(): set(value) for key, value in cur_srs_users.items()}
        cur_idle_time = {key.lower(): value for key, value in cur_idle_time.items()}

        srs_users = dsum_sets(srs_users, cur_srs_users)
        idle_time = dmin(idle_time, cur_idle_time)

        duration = (datetime.datetime.now() - start_time).seconds
        print("Finished to work on {} file, this is loop # {}."
              "Time up to now: {}".format(cur_comments_file, comm_idx, duration))

    # calculating the averge number of comments per submission, the total users amount and the period since last active day
    comm_per_submission = {key: value *1.0 /submm_amount[key] for key, value in comm_amount.items() if key in submm_amount}
    users_amount = {key: len(value) for key, value in srs_users.items()}
    duration_since_last_post = idle_time
    #duration_since_last_post = {key: (datetime.datetime(2017, 3, 31).date() - pd.to_datetime(value).date()).days
    #                            for key, value in idle_time.items()}
    all_srs = set(submm_amount.keys())
    srs_meta_data = defaultdict(list)
    with open(data_path + 'srs_meta_data_102016_to_032017.json') as f:
        for idx, line in enumerate(f):
            cur_line = json.loads(line)
            cur_sr_name = str(cur_line['display_name']).lower()
            # case we already 'met' this sr, we'll take the max out of all subscribers amount we see
            if cur_sr_name in srs_meta_data:
                srs_meta_data[cur_sr_name] = [max(cur_line['subscribers'], srs_meta_data[cur_sr_name][0]),
                                              cur_line['created']]

            elif cur_line['subscribers'] is not None:
                srs_meta_data[cur_sr_name] = [cur_line['subscribers'], cur_line['created']]
    sr_basic = pd.DataFrame.from_dict(data=srs_meta_data, orient='index')
    sr_basic.reset_index(inplace=True)
    sr_basic.columns = ['subreddit_name', 'number_of_subscribers', 'creation_epoch']
    sr_basic['number_of_subscribers'] = pd.to_numeric(sr_basic['number_of_subscribers'])

    # adding explicit timestamp to the data and filtering SRs which were created before r/place started
    sr_basic = sr_basic.assign(created_utc_as_date=pd.to_datetime(sr_basic['creation_epoch'], unit='s'))

    relevant_srs_meta_data = sr_basic[sr_basic['subreddit_name'].str.lower().isin(all_srs)][['subreddit_name',
                                                                                              'number_of_subscribers',
                                                                                              'created_utc_as_date']]
    subscribers = relevant_srs_meta_data[['subreddit_name','number_of_subscribers']]
    subscribers = subscribers.set_index('subreddit_name').to_dict()['number_of_subscribers']

    pazam = relevant_srs_meta_data[['subreddit_name','created_utc_as_date']]
    pazam = pazam.set_index('subreddit_name').to_dict()['created_utc_as_date']

    pazam = {key: (datetime.datetime(2017, 3, 31).date() - pd.to_datetime(value).date()).days
             for key, value in pazam.items()}

    #importing the drawing/not-drawing srs
    sr_objects = pickle.load(open(data_path + "sr_objects_6_months_sampling_based_submission_30_12_2018.p", "rb"))
    drawing_srs = set([sr.name for sr in sr_objects if sr.trying_to_draw == 1])
    not_drawing_srs = set([sr.name for sr in sr_objects if sr.trying_to_draw == -1])

    drawing_submm_amount = {key: value for key, value in submm_amount.items() if key in drawing_srs}
    not_drawing_submm_amount = {key: value for key, value in submm_amount.items() if key in not_drawing_srs}
    drawing_comm_per_submission = {key: value for key, value in comm_per_submission.items() if key in drawing_srs}
    not_drawing_comm_per_submission = {key: value for key, value in comm_per_submission.items() if key in not_drawing_srs}
    drawing_users_amount = {key: value for key, value in users_amount.items() if key in drawing_srs}
    not_drawing_users_amount = {key: value for key, value in users_amount.items() if key in not_drawing_srs}
    drawing_duration_since_last_post = {key: value for key, value in duration_since_last_post.items() if key in drawing_srs}
    not_drawing_duration_since_last_post = {key: value for key, value in duration_since_last_post.items() if key in not_drawing_srs}
    drawing_subscribers = {key: value for key, value in subscribers.items() if key in drawing_srs}
    not_drawing_subscribers = {key: value for key, value in subscribers.items() if key in not_drawing_srs}
    drawing_pazam = {key: value for key, value in pazam.items() if key in drawing_srs}
    not_drawing_pazam = {key: value for key, value in pazam.items() if key in not_drawing_srs}


    # now, calculating statistics based on the dictionaries we created
    # all srs in Reddit
    print("Here are the results for the S dataset (all active srs):")
    print("We have found: {} srs in submissions,"
          "{} srs in comments, {} srs in users_amount,"
          "{} srs in duration_since_last_post".format(len(submm_amount), len(comm_per_submission),
                                                      len(users_amount), len(duration_since_last_post)))
    print("Submissions sums: {} \nSubmissions avg: {} "
          "\nSubmissions median: {} \nSubmissions std: {}".format(round(np.sum([value for key, value in submm_amount.items()]),2),
                                                                  round(np.average([value for key, value in submm_amount.items()]),2),
                                                                  round(np.median([value for key, value in submm_amount.items()]),2),
                                                                  round(np.std([value for key, value in submm_amount.items()]),2)))

    print("Comments sums: {} \nComments avg: {} "
          "\nComments median: {} \nComments std: {}".format(round(np.sum([value for key, value in comm_per_submission.items()]),2),
                                                                  round(np.average([value for key, value in comm_per_submission.items()]),2),
                                                                  round(np.median([value for key, value in comm_per_submission.items()]),2),
                                                                  round(np.std([value for key, value in comm_per_submission.items()]),2)))


    print("users_amount sums: {} \nusers_amount avg: {} "
          "\nusers_amount median: {} \nusers_amount std: {}".format(round(np.sum([value for key, value in users_amount.items()]),2),
                                                                    round(np.average([value for key, value in users_amount.items()]),2),
                                                                    round(np.median([value for key, value in users_amount.items()]),2),
                                                                    round(np.std([value for key, value in users_amount.items()]),2)))

    print("duration_since_last_post avg: {} "
          "\nduration_since_last_post median: {} "
          "\nduration_since_last_post std: {}".format(round(np.average([value for key, value in duration_since_last_post.items()]),2),
                                                      round(np.median([value for key, value in duration_since_last_post.items()]),2),
                                                      round(np.std([value for key, value in duration_since_last_post.items()]),2)))
    print("subscribers avg: {} "
          "\nsubscribers median: {} "
          "\nsubscribers std: {}".format(round(np.average([value for key, value in subscribers.items()]),2),
                                                      round(np.median([value for key, value in subscribers.items()]),2),
                                                      round(np.std([value for key, value in subscribers.items()]),2)))

    print("pazam avg: {} "
          "\npazam median: {} "
          "\npazam std: {}".format(round(np.average([value for key, value in pazam.items()]),2),
                                   round(np.median([value for key, value in pazam.items()]),2),
                                   round(np.std([value for key, value in pazam.items()]),2)))

    # drawing_srs
    print("\n\n\nHere are the results for the S+ dataset (drawing srs):")
    print("We have found: {} srs in submissions,"
          "{} srs in comments, {} srs in users_amount,"
          "{} srs in duration_since_last_post".format(len(drawing_submm_amount), len(drawing_comm_per_submission),
                                                      len(drawing_users_amount), len(drawing_duration_since_last_post)))
    print("Submissions sums: {} \nSubmissions avg: {} "
          "\nSubmissions median: {} \nSubmissions std: {}".format(round(np.sum([value for key, value in drawing_submm_amount.items()]),2),
                                                                  round(np.average([value for key, value in drawing_submm_amount.items()]),2),
                                                                  round(np.median([value for key, value in drawing_submm_amount.items()]),2),
                                                                  round(np.std([value for key, value in drawing_submm_amount.items()]),2)))

    print("Comments sums: {} \nComments avg: {} "
          "\nComments median: {} \nComments std: {}".format(round(np.sum([value for key, value in drawing_comm_per_submission.items()]),2),
                                                                  round(np.average([value for key, value in drawing_comm_per_submission.items()]),2),
                                                                  round(np.median([value for key, value in drawing_comm_per_submission.items()]),2),
                                                                  round(np.std([value for key, value in drawing_comm_per_submission.items()]),2)))


    print("users_amount sums: {} \nusers_amount avg: {} "
          "\nusers_amount median: {} \nusers_amount std: {}".format(round(np.sum([value for key, value in drawing_users_amount.items()]),2),
                                                                    round(np.average([value for key, value in drawing_users_amount.items()]),2),
                                                                    round(np.median([value for key, value in drawing_users_amount.items()]),2),
                                                                    round(np.std([value for key, value in drawing_users_amount.items()]),2)))

    print("duration_since_last_post avg: {} "
          "\nduration_since_last_post median: {} "
          "\nduration_since_last_post std: {}".format(round(np.average([value for key, value in drawing_duration_since_last_post.items()]),2),
                                                      round(np.median([value for key, value in drawing_duration_since_last_post.items()]),2),
                                                      round(np.std([value for key, value in drawing_duration_since_last_post.items()]),2)))
    print("subscribers avg: {} "
          "\nsubscribers median: {} "
          "\nsubscribers std: {}".format(round(np.average([value for key, value in drawing_subscribers.items()]),2),
                                         round(np.median([value for key, value in drawing_subscribers.items()]),2),
                                         round(np.std([value for key, value in drawing_subscribers.items()]),2)))

    print("pazam avg: {} "
          "\npazam median: {} "
          "\npazam std: {}".format(round(np.average([value for key, value in drawing_pazam.items()]),2),
                                   round(np.median([value for key, value in drawing_pazam.items()]),2),
                                   round(np.std([value for key, value in drawing_pazam.items()]),2)))


    # not drawing_srs
    print("\n\n\nHere are the results for the S- dataset (not drawing srs):")
    print("We have found: {} srs in submissions,"
          "{} srs in comments, {} srs in users_amount,"
          "{} srs in duration_since_last_post".format(len(not_drawing_submm_amount), len(not_drawing_comm_per_submission),
                                                      len(not_drawing_users_amount), len(not_drawing_duration_since_last_post)))
    print("Submissions sums: {} \nSubmissions avg: {} "
          "\nSubmissions median: {} \nSubmissions std: {}".format(round(np.sum([value for key, value in not_drawing_submm_amount.items()]),2),
                                                                  round(np.average([value for key, value in not_drawing_submm_amount.items()]),2),
                                                                  round(np.median([value for key, value in not_drawing_submm_amount.items()]),2),
                                                                  round(np.std([value for key, value in not_drawing_submm_amount.items()]),2)))

    print("Comments sums: {} \nComments avg: {} "
          "\nComments median: {} \nComments std: {}".format(round(np.sum([value for key, value in not_drawing_comm_per_submission.items()]),2),
                                                                  round(np.average([value for key, value in not_drawing_comm_per_submission.items()]),2),
                                                                  round(np.median([value for key, value in not_drawing_comm_per_submission.items()]),2),
                                                                  round(np.std([value for key, value in not_drawing_comm_per_submission.items()]),2)))


    print("users_amount sums: {} \nusers_amount avg: {} "
          "\nusers_amount median: {} \nusers_amount std: {}".format(round(np.sum([value for key, value in not_drawing_users_amount.items()]),2),
                                                                    round(np.average([value for key, value in not_drawing_users_amount.items()]),2),
                                                                    round(np.median([value for key, value in not_drawing_users_amount.items()]),2),
                                                                    round(np.std([value for key, value in not_drawing_users_amount.items()]),2)))

    print("duration_since_last_post avg: {} "
          "\nduration_since_last_post median: {} "
          "\nduration_since_last_post std: {}".format(round(np.average([value for key, value in not_drawing_duration_since_last_post.items()]),2),
                                                      round(np.median([value for key, value in not_drawing_duration_since_last_post.items()]),2),
                                                      round(np.std([value for key, value in not_drawing_duration_since_last_post.items()]),2)))
    print("subscribers avg: {} "
          "\nsubscribers median: {} "
          "\nsubscribers std: {}".format(round(np.average([value for key, value in not_drawing_subscribers.items()]),2),
                                         round(np.median([value for key, value in not_drawing_subscribers.items()]),2),
                                         round(np.std([value for key, value in not_drawing_subscribers.items()]),2)))

    print("pazam avg: {} "
          "\npazam median: {} "
          "\npazam std: {}".format(round(np.average([value for key, value in not_drawing_pazam.items()]),2),
                                   round(np.median([value for key, value in not_drawing_pazam.items()]),2),
                                   round(np.std([value for key, value in not_drawing_pazam.items()]),2)))
    duration = (datetime.datetime.now() - start_time).seconds
    print("End of code, took us {} seconds". format(duration))







