# Authors: Avrahami Israeli (isabrah)
# Python version: 3.6
# Last update: 12.8.2018

from relations_between_srs.sr_relations import SrRelations
import pandas as pd
import itertools
import pickle
import datetime
import os
import sys

###################################################### Configurations ##################################################
data_path = '/home/isabrah/reddit_data/' if sys.platform == 'linux' \
    else 'C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\PhD\\reddit canvas\\data\\sr_relations\\'
# next var should be one out of the following: 'build_objects', 'analyse_sr', 'info_extraction'
modeling_phase = 'info_extraction'
sr_to_analyse = 'idubbbz'  # 'leagueoflegends' # 'the_donald'#'dota2'#'brasil'#'israel'#'unitedkingdom'
not_interesting_sr = ['place', 'all', 'askreddit', 'rocketleagueexchange', 'globaloffensivetrade']
########################################################################################################################

if __name__ == "__main__":
    if modeling_phase == 'build_objects':
        saving_file_name = 'dict_results_all_posts.pickle'
    elif modeling_phase == 'analyse_sr':
        saving_file_name = sr_to_analyse + "_sr.txt"

    start_time = datetime.datetime.now()
    # reading the data again (case we load the data from the disk)
    if sys.platform == 'linux':
        submission_shrinked = pd.read_hdf(data_path + 'submission_shrinked.hdf')
        comments_shrinked = pd.read_hdf(data_path + 'comments_shrinked.hdf')
        subreddits_df = pd.read_csv(data_path + 'subreddits_revealed/all_subreddits_revealed.csv')
    else:
        submission_shrinked = pd.read_hdf(data_path + 'submission_shrinked.hdf')
        subreddits_df = pd.read_csv(data_path + 'all_subreddits_revealed.csv')
    subreddits_list = list(set(subreddits_df['sr']))
    subreddits_list = sorted(subreddits_list)
    submission = submission_shrinked[submission_shrinked['subreddit'].str.lower().isin(subreddits_list)]
    #comments = comments_place_related[comments_place_related['subreddit'].str.lower().isin(subreddits_list)]

    if modeling_phase == 'build_objects':
        # looping over each sr in the subreddits_list and finding its relations
        results_dict = dict()
        for idx, cur_sr in enumerate(subreddits_list):
            cur_sr_subm = submission[submission['subreddit'].str.lower() == cur_sr]
            if cur_sr in not_interesting_sr or (cur_sr_subm.shape[0] == 0): #and cur_sr_comm.shape[0] == 0):
                continue
            relation_obj = SrRelations(primary_sr=cur_sr, all_sr=set(subreddits_list))
            relation_obj.find_explicit_relations(reddit_data=cur_sr_subm, submission_data=True)
            #relation_obj.find_explicit_relations(reddit_data=cur_sr_comm, submission_data=False)
            results_dict[cur_sr] = relation_obj
            # printing status after each 100 cases
            if not idx % 100:
                duration = (datetime.datetime.now() - start_time).seconds
                print("Just finished to cover {} sr, out of {}."
                      "Aggregated time up to now is {}".format(idx, len(subreddits_list), duration))
        tot_explicit_matches = 0
        # finished the big loop, iterating over the results only to print some basic statistics
        for sr, res in results_dict.items():
            cur_ids = [ids for name, ids in res.subm_explicit_relations.items()]
            tot_explicit_matches += len(list(itertools.chain(*cur_ids)))
            cur_ids = [ids for name, ids in res.comm_explicit_relations.items()]
            tot_explicit_matches += len(list(itertools.chain(*cur_ids)))
        print("Total number of explicit matches we found is:{}".format(tot_explicit_matches))

        # saving file to disk
        with open(saving_file_name, 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif modeling_phase == 'analyse_sr':
        # removing the file if already exists
        try:
            os.remove(saving_file_name)
        except OSError:
            pass
        with open('dict_results_all_posts.pickle', 'rb') as handle:
            results_dict = pickle.load(handle)
        analysed_sr_obj = results_dict[sr_to_analyse]
        analysed_sr_obj.print_findings(reddit_data=submission, file_name=saving_file_name, submission_data=True)

    elif modeling_phase == 'info_extraction':
        with open('dict_results_all_posts.pickle', 'rb') as handle:
            results_dict = pickle.load(handle)
        analysed_sr_obj = results_dict[sr_to_analyse]
        cur_sr_subm = submission[submission['subreddit'].str.lower() == sr_to_analyse]
        analysed_sr_obj.information_extraction(reddit_data=cur_sr_subm, submission_data=True)
        summary_df = analysed_sr_obj.sentences_summary()
        summary_df.to_csv("sentences_summary_" + analysed_sr_obj.primary_sr + ".csv")

    duration = (datetime.datetime.now() - start_time).seconds
    print("Code has ended, total time it took us is: {} seconds".format(duration))
