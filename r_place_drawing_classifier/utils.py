# Authors: Avrahami Israeli (isabrah)
# Python version: 3.6
# Last update: 6.11.2018

import datetime
import pandas as pd
import re
import os
import collections
import pickle
import sys
import csv


def get_submissions_subset(files_path, srs_to_include):
    """
    pulling our subset of the submission data, which is related to the list of SRs given as input. This is very
    simple "where" statement
    :param files_path: string
        location of the files to be used (.csv ones)
    :param srs_to_include: list (maybe set will also work here)
        list with SR names to be included in the returned dataset. Expected to be lowe-case ones
    :return: pandas data-frame
        df including all relevant submission, related to the SRs given as input
    """
    start_time = datetime.datetime.now()
    # finding all the relevant zip files in the 'data_path' directory
    submission_files = [f for f in os.listdir(files_path) if re.match(r'RS.*\.csv', f)]
    # taking only the submissions files from 10-2016 to 03-2017
    submission_files = [i for i in submission_files if 'RS_2016-10.csv' <= i <= 'RS_2017-03.csv']
    submission_files = sorted(submission_files)
    submission_dfs = []
    # iterating over each submission file
    for subm_idx, cur_submission_file in enumerate(submission_files):
        cur_submission_df = pd.read_csv(filepath_or_buffer=files_path + cur_submission_file, encoding='utf-8')
        # filtering the data-frame based on the list of SRs we want to include and the date (before r/place started)
        cur_submission_df = cur_submission_df[cur_submission_df["subreddit"].str.lower().isin(srs_to_include)]
        cur_submission_df = cur_submission_df[cur_submission_df['created_utc_as_date'] < '2017-03-29 00:00:00']
        submission_dfs.append(cur_submission_df)
    full_submissions_df = pd.concat(submission_dfs)
    duration = (datetime.datetime.now() - start_time).seconds
    print("Function 'get_submission_subset_dataset' has ended. Took us : {} seconds. "
          "Submission data-frame shape created is {}".format(duration, full_submissions_df.shape))
    return full_submissions_df


def get_comments_subset(files_path, srs_to_include):
    """
    pulling our subset of the commnets data, which is related to the list of SRs given as input. This is very
    simple "where" statement
    :param files_path: string
        location of the files to be used (.csv ones)
    :param srs_to_include: list (maybe set will also work here)
        list with SR names to be included in the returned dataset. Expected to be lowe-case ones
    :return: pandas data-frame
        df including all relevant submission, related to the SRs given as input
    """
    start_time = datetime.datetime.now()
    comments_files = [f for f in os.listdir(files_path) if re.match(r'RC.*\.csv', f)]
    comments_files = [i for i in comments_files if 'RC_2016-10.csv' <= i <= 'RC_2017-03.csv']
    comments_files = sorted(comments_files)
    comments_dfs = []
    for comm_idx, cur_comments_file in enumerate(comments_files):
        if sys.platform == 'linux':
            cur_comments_df = pd.read_csv(filepath_or_buffer=files_path + cur_comments_file)
        else:
            cur_comments_df = pd.read_csv(filepath_or_buffer=files_path + cur_comments_file, encoding='latin-1')
        cur_comments_df = cur_comments_df[cur_comments_df["subreddit"].str.lower().isin(srs_to_include)]
        comments_dfs.append(cur_comments_df)
    full_comments_df = pd.concat(comments_dfs)
    duration = (datetime.datetime.now() - start_time).seconds
    print("Function 'get_comments_subset' has ended. Took us : {} seconds. "
          "Submission data-frame shape created is {}".format(duration, full_comments_df.shape))
    return full_comments_df


def calc_sr_statistics(files_path, included_years, saving_res_path=os.getcwd()):
    '''
    calculating relevant statistics to each sr found in the files given as input. This will be later used in order
    to filter SRs with no submission/very small amount of submissions
    :param files_path: string
        location of the files to be used (.csv ones)
    :param included_years: list
        list of years to include in the analysis
    :param saving_res_path: string
        location where to save results
    :return: dictionary
        dictionary with statistics to each SR. The function also saves the results as a pickle file

    Example:
    --------
    >>> data_path = "/home/isabrah/reddit_pycharm_proj_with_own_pc/data_loaders/r_place_classification/place_classifier_csvs/" if sys.platform == "linux" else "C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\PhD\\reddit canvas\\data\\place_classifier_csvs\\"
    >>> dict_res = calc_sr_statistics(files_path=data_path, saving_res_path=data_path, included_years=[2017, 2016], submission_only=True)
    '''
    start_time = datetime.datetime.now()
    submission_files = [f for f in os.listdir(files_path) if re.match(r'RS.*\.csv', f)]
    # taking only files which are in the 'included_years' subset
    submission_files = [sf for sf in submission_files if any(str(year) in sf for year in included_years)]
    # comments_files = [cf for cf in comments_files if any(str(year) in cf for year in included_years)]
    submission_files = sorted(submission_files)
    sr_statistics = collections.Counter()
    for subm_idx, cur_submission_file in enumerate(submission_files):
        cur_submission_df = pd.read_csv(filepath_or_buffer=files_path + cur_submission_file)
        cur_sr_statistics = collections.Counter(cur_submission_df["subreddit"])
        sr_statistics += cur_sr_statistics
        # writing status to screen
        duration = (datetime.datetime.now() - start_time).seconds
        print("Ended loop # {}, up to now took us {} seconds".format(subm_idx, duration))
    # saving the stats to a file
    pickle.dump(sr_statistics, open(saving_res_path + "submission_stats_102016_to_032017.p", "wb"))
    duration = (datetime.datetime.now() - start_time).seconds
    print("Function 'calc_sr_statistics' has ended. Took us : {} seconds. "
          "Final dictionary size is {}".format(duration, len(sr_statistics)))
    return sr_statistics


def save_results_to_csv(start_time, SRs_amount, models_params, results, saving_path=os.getcwd()):
    '''
    given inputs regarding a final run results - write these results into a file
    :param start_time: datetime
        time when the current result run started
    :param SRs_amount: int
        amount of SRs the run was based on, usually it is between 1000-2500
    :param models_params: dict
        dictionary holding all the model parameters. This is a complex list of features
    :param results: dict
        dictionary with all results. Currently it should contain the following keys: 'accuracy', 'precision', 'recall'
    :param saving_path: str
        location where results should be saved. It can be an exisitng/new csv file
    :return: None
        Nothing is returned, only saving to the file is being done
    '''
    results_file = saving_path + '/' if sys.platform == 'linux' else saving_path + '\\'
    results_file += 'place_drawing_classifier_results.csv'
    file_exists = os.path.isfile(results_file)
    rf = open(results_file, 'a', newline='')
    with rf as output_file:
        dict_writer = csv.DictWriter(output_file,
                                     fieldnames=['timestamp', 'start_time', 'machine', 'SRs_amount', 'cv_folds',
                                                 'models_params', 'accuracy', 'precision', 'recall'])
        # only in case the file doesn't exist, we'll add a header
        if not file_exists:
            dict_writer.writeheader()
        dict_writer.writerow({'timestamp': datetime.datetime.now(), 'start_time': start_time, 'SRs_amount': SRs_amount,
                              'machine':  os.environ['HOSTNAME'] if sys.platform == 'linux' else os.environ['COMPUTERNAME'],
                              'cv_folds': len(results['accuracy']), 'models_params': models_params,
                              'accuracy': results['accuracy'], 'precision': results['precision'],
                              'recall': results['recall']})
    rf.close()
