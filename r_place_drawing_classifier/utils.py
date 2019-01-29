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
import warnings
warnings.filterwarnings("ignore")


def get_submissions_subset(files_path, srs_to_include, start_month='2016-10', end_month='2017-03'):
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
    submission_files = [i for i in submission_files if
                        ''.join(['RS_', start_month, '.csv']) <= i <= ''.join(['RS_', end_month, '.csv'])]
    submission_files = sorted(submission_files)
    submission_dfs = []
    # iterating over each submission file
    for subm_idx, cur_submission_file in enumerate(submission_files):
        cur_submission_df = pd.read_csv(filepath_or_buffer=files_path + cur_submission_file, encoding='utf-8')
        # filtering the data-frame based on the list of SRs we want to include and the date (before r/place started)
        cur_submission_df = cur_submission_df[cur_submission_df["subreddit"].str.lower().isin(srs_to_include)]
        cur_submission_df = cur_submission_df[cur_submission_df['created_utc_as_date'] < '2017-03-29 00:00:00']
        submission_dfs.append(cur_submission_df)
    if len(submission_dfs) == 0:
        raise IOError("No submission file was found")

    full_submissions_df = pd.concat(submission_dfs)
    duration = (datetime.datetime.now() - start_time).seconds
    print("Function 'get_submission_subset_dataset' has ended. Took us : {} seconds. "
          "Submission data-frame shape created is {}".format(duration, full_submissions_df.shape))
    return full_submissions_df


def get_comments_subset(files_path, srs_to_include, start_month='2016-10', end_month='2017-03'):
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
    comments_files = [i for i in comments_files if
                      ''.join(['RC_', start_month, '.csv']) <= i <= ''.join(['RC_', end_month, '.csv'])]
    comments_files = sorted(comments_files)
    comments_dfs = []
    for comm_idx, cur_comments_file in enumerate(comments_files):
        if sys.platform == 'linux':
            cur_comments_df = pd.read_csv(filepath_or_buffer=files_path + cur_comments_file)
        else:
            cur_comments_df = pd.read_csv(filepath_or_buffer=files_path + cur_comments_file, encoding='latin-1')
        cur_comments_df = cur_comments_df[cur_comments_df["subreddit"].str.lower().isin(srs_to_include)]
        comments_dfs.append(cur_comments_df)
    if len(comments_dfs) == 0:
        raise IOError("No comments file were found")
    full_comments_df = pd.concat(comments_dfs)
    duration = (datetime.datetime.now() - start_time).seconds
    print("Function 'get_comments_subset' has ended. Took us : {} seconds. "
          "Comments data-frame shape created is {}".format(duration, full_comments_df.shape))
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


def save_results_to_csv(results_file, start_time, SRs_amount, config_dict, results, saving_path=os.getcwd()):
    '''
    given inputs regarding a final run results - write these results into a file
    :param results_file: str
        file of the csv where results should be placed
    :param start_time: datetime
        time when the current result run started
    :param SRs_amount: int
        amount of SRs the run was based on, usually it is between 1000-2500
    :param config_dict: dict
        dictionary holding all the configuration of the run, the one we get as input json
    :param results: dict
        dictionary with all results. Currently it should contain the following keys: 'accuracy', 'precision', 'recall'
    :param saving_path: str
        location where results should be saved. It can be an exisitng/new csv file
    :return: None
        Nothing is returned, only saving to the file is being done
    '''
    results_file = saving_path + '/' if sys.platform == 'linux' else saving_path + '\\'
    results_file += 'place_drawing_classifier_results_submission_based_sampling.csv'
    file_exists = os.path.isfile(results_file)
    rf = open(results_file, 'a', newline='')
    with rf as output_file:
        dict_writer = csv.DictWriter(output_file,
                                     fieldnames=['timestamp', 'start_time', 'machine', 'SRs_amount', 'cv_folds',
                                                 'configurations', 'accuracy', 'precision', 'recall'])
        # only in case the file doesn't exist, we'll add a header
        if not file_exists:
            dict_writer.writeheader()
        try:
            host_name = os.environ['HOSTNAME'] if sys.platform == 'linux' else os.environ['COMPUTERNAME']
        except KeyError:
            host_name = 'pycharm with this ssh: '+ os.environ['SSH_CONNECTION']
        dict_writer.writerow({'timestamp': datetime.datetime.now(), 'start_time': start_time, 'SRs_amount': SRs_amount,
                              'machine': host_name, 'cv_folds': len(results['accuracy']),
                              'configurations': config_dict, 'accuracy': results['accuracy'],
                              'precision': results['precision'], 'recall': results['recall']})
    rf.close()


def examine_word(sr_object, regex_required, tokenizer, saving_file=os.getcwd()):
    start_time = datetime.datetime.now()
    print("examine_word function has started")
    tot_cnt = 0
    if sys.platform == 'linux':
        explicit_file_name = saving_file + '/' + 'examine_word_res_regex_' + regex_required + '.txt'
    else:
        explicit_file_name = saving_file + '\\' + 'examine_word_res_regex_' + regex_required + '.txt'
    if os.path.exists(explicit_file_name):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    with open(explicit_file_name, append_write, encoding="utf-8") as text_file:
        text_file.write("\n\nHere are the relevant sentences with the regex {} in SR named {}. This sr is labeled as: "
                        "{}".format(regex_required, sr_object.name,
                                    'not drawing' if sr_object.trying_to_draw == -1 else 'drawing'))

        for subm in sr_object.submissions_as_list:
            normalized_text = []
            try:
                tokenized_txt_title = tokenizer(subm[1])
                normalized_text += tokenized_txt_title
            except TypeError:
                pass
            try:
                tokenized_txt_selftext = tokenizer(subm[2])
                normalized_text += tokenized_txt_selftext
            except TypeError:
                pass
            if regex_required in set(normalized_text):
                text_file.write('\n' + str(subm))
                tot_cnt += 1
        duration = (datetime.datetime.now() - start_time).seconds
        print("examine_word has ended, took us {} seconds."
              "Total of {} rows were written to a text file".format(duration, tot_cnt))


def remove_huge_srs(sr_objects, quantile=0.01):
    srs_summary = [(idx, cur_sr.name, cur_sr.trying_to_draw, len(cur_sr.submissions_as_list))
                   for idx, cur_sr in enumerate(sr_objects)]
    srs_summary.sort(key=lambda tup: tup[3], reverse=True)  # sorts in place according to the # of submissions
    amount_of_srs_to_remove = int(len(sr_objects)*quantile)
    srs_to_remove_summary = srs_summary[0:amount_of_srs_to_remove]
    drawing_removed_srs = sum([sr[2] for sr in srs_to_remove_summary if sr[2]==1])
    srs_to_remove = set([sr[1] for sr in srs_to_remove_summary])
    returned_list = [sr for sr in sr_objects if sr.name not in srs_to_remove]
    print("remove_huge_srs function has ended, {} srs have been removed,"
          " {} out of them are from class 1 (drawing)".format(len(srs_to_remove), drawing_removed_srs))
    return returned_list

