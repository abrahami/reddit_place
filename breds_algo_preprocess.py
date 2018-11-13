from relations_between_srs.sr_relations import SrRelations
import pandas as pd

if __name__ == "__main__":
    cur_sr = 'the_donald'# ['leagueoflegends', 'the_donald', 'dota2', 'brasil', 'israel', 'unitedkingdom']
    submission_shrinked = pd.read_hdf('C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\'
                                      'PhD\\reddit canvas\\data\\sr_relations\\submission_shrinked.hdf')
    subreddits_df = pd.read_csv('C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\'
                                'PhD\\reddit canvas\\data\\sr_relations\\all_subreddits_revealed.csv')
    subreddits_list = list(set(subreddits_df['sr']))
    subreddits_list = sorted(subreddits_list)
    submission = submission_shrinked[submission_shrinked['subreddit'].str.lower().isin(subreddits_list)]
    cur_sr_subm = submission[submission['subreddit'].str.lower() == cur_sr]
    relation_obj = SrRelations(primary_sr=cur_sr, all_sr=set(subreddits_list))
    relation_obj.information_extraction(reddit_data=cur_sr_subm)