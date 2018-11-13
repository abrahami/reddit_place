# Authors: Avrahami Israeli (isabrah)
# Python version: 3.6
# Last update: 9.10.2018

import numpy as np
from collections import defaultdict, Counter
import datetime
from dateutil import parser

class SubReddit(object):
    """
    Class to handle the process of extracting the text relevant to each Sub-Reddit (SR).
    This is done usually as part of a pipline, where a transform function needs to be defined as part of the flow

    Parameters
    ----------
    meta_data: dict
        meta information about the sub-reddit. Must include: 'name'(str), 'end_state'(str), 'num_of_users'(int),
        'trying_to_draw'('Yes' or 'No' str only!)
    lang: str, default: 'en'
        the language used in the sub-reddit. This should be a shortcut same as used by spacy

    Attributes
    ----------
    name: str
        name of the SR
    creation_utc: date
        creation time of the SR (in UTC timestamp)
    end_state: str
        final status of the SR (e.g. 'alive')
    num_users: int
        number of useres subscribed to the SR (according to records we have for 2017)
    trying_to_draw: int
        -1 or 1 (1 means trying to draw, -1 means doesn't trying to
    lang: str
        main used language in the SR, in a shortcut used by spacy (e.g. en for English)
    submissions_as_list: list
        all submissions text organized in a list
    comments_as_list: list
        all comments text organized in a list
    explanatory_features: dict
        dictionary holding explanatory features of the SR (e.g. # of distinct users wrote a submission)
    urls_counter: dict
         dictionary holding how many occurrences from each url type were found. url type is defined by the
         urllib.parse.urlparse class (i.e. from this class we pull out the 'netloc' information and define it as a key)
    coordinates_counter: int
        number of occurrences where an indication to a coordinates was found (e.g. (200, 300) or [455, 87])
    """

    def __init__(self, meta_data, lang='en'):
        self.name = meta_data['SR']
        self.creation_utc = meta_data['creation_utc']
        self.end_state = meta_data['end_state']
        self.num_users = meta_data['num_of_users']
        self.trying_to_draw = 1 if meta_data['trying_to_draw'] == 'Yes' or meta_data['models_prediction'] >= 0.9 \
            else -1
        self.models_prediction = meta_data['models_prediction']
        self.lang = lang
        self.submissions = None
        self.comments = None
        self.submissions_as_list = None
        self.comments_as_list = None
        self.explanatory_features = defaultdict()
        # these two are being updated by the 'CleanTextTransformer' - since it handles the urls and coordinates findings
        self.urls_counter = None
        self.coordinates_counter = None

    def concat_and_weight_text(self, weight_policy = None):
        """
        joins together into one long string all the text found for the SR object. this is just a concatinations of all
        strings of the SR
        :param weight_policy: how to weight the text, since each post gets a score in Reddit. Defualt: None (no weight)
        :return: str
            long string of all the text, seperated with a '.' between each
        """
        concatenated_text = '.'.join([str(l[1]) + '. ' + str(l[2]) + '.'
                                       if type(l[2]) is str else str(l[1]) + '. ' for l in self.submissions_as_list ])
        # handling the comments data, if it exists
        if self.comments_as_list is not None:
            # comments have no 'header' so it is only the self-text
            concatenated_text = concatenated_text + '.'.join([str(l[1]) + '. ' for l in self.comments_as_list])
        '''
        concatenated_text = ''
        for t in self.submissions_as_list:
            if type(t[1]) is str and type(t[2]) is str:
                concatenated_text = concatenated_text + t[1] + '. ' + t[2] + '.'
            elif type(t[1]) is str and type(t[2]) is float:
                concatenated_text = concatenated_text + t[1] + '. '
            elif type(t[2]) is str and type(t[1]) is float:
                concatenated_text = concatenated_text + t[2] + '. '
            else:
                raise IOError("Problem along the 'concat_and_weight_text' function - input is illegal")
        if self.comments_as_list is not None:
            # comments have no 'header' so it is only the self-text
            for t in self.comments_as_list:
                concatenated_text = concatenated_text + t[1] + '. '  # + t[2] + '.'
        '''
        return concatenated_text

    def create_explanatory_features(self, submission_data, comments_data=None):
        """
        creation process of the explanatory features based on the input data sets
        additional explanatory features are created in other processes and are added to the 'explanatory_features' dict
        (not part of this function flow)
        :param submission_data: pandas df
            data-frame holding all the needed submission information
        :param comments_data: pandas df or None, default: None
            data-frame holding all the needed comments information
        :return:
            only updates the object's 'explanatory_features' dict
        """
        self.explanatory_features['submission_amount'] = submission_data.shape[0]
        try:
            self.explanatory_features['submission_amount_normalized'] = submission_data.shape[0] * 1.0 / \
                                                                        self.num_users * 1.0
        except TypeError:
            self.explanatory_features['submission_amount_normalized'] = None

        #if self.num_users is int or self.num_users is float:
        #    self.explanatory_features['submission_amount_normalized'] = submission_data.shape[0]*1.0 / \
        #                                                                self.num_users *1.0
        #else:
        #    self.explanatory_features['submission_amount_normalized'] = None
        # time related features
        try:
            sr_pazam = datetime.datetime(2017, 3, 29).date() - self.creation_utc.date()
            self.explanatory_features['days_pazam'] = sr_pazam.days
        except AttributeError:
            self.explanatory_features['days_pazam'] = None
        # starting calculating all relevant features to the submissions ones
        titles_len = []
        selftexts_len = []
        scores_values = []
        empty_selftext_cnt = 0
        deleted_or_removed_amount = 0
        submissions_writers = defaultdict(int)
        for cur_idx, cur_sub_row in submission_data.iterrows():
            try:
                titles_len.append(len(cur_sub_row['title']))
            except TypeError:
                pass
            # old way with if-else
            #if type(cur_sub_row['title']) is str:
            #    titles_len.append(len(cur_sub_row['title']))
            if type(cur_sub_row['score']) is int or type(cur_sub_row['score']) is float:
                scores_values.append(cur_sub_row['score'])
            if type(cur_sub_row['selftext']) is str and \
                    cur_sub_row['selftext'] != '[deleted]' and cur_sub_row['selftext'] != '[removed]':
                    selftexts_len.append(len(cur_sub_row['selftext']))
            if type(cur_sub_row['author']) is str and cur_sub_row['author'] != '[deleted]':
                submissions_writers[cur_sub_row['author']] += 1
            if type(cur_sub_row['selftext']) is float and np.isnan(cur_sub_row['selftext']):
                empty_selftext_cnt += 1
            if cur_sub_row['selftext'] in {'[deleted]', '[removed]'}:
                deleted_or_removed_amount += 1
            if type(cur_sub_row['author']) is str and cur_sub_row['author'] != '[deleted]':
                submissions_writers[cur_sub_row['author']] += 1
        # now, saving results to the sr object
        self.explanatory_features['submission_title_length'] = np.mean(titles_len) if len(titles_len) > 0 else 0
        self.explanatory_features['submission_selftext_length'] = np.mean(selftexts_len) if len(selftexts_len) > 0 else 0
        self.explanatory_features['submission_average_score'] = np.mean(scores_values) if len(scores_values) > 0 else 0
        # % of empty selftext exists in submissions
        self.explanatory_features['empty_selftext_ratio'] = empty_selftext_cnt * 1.0 / submission_data.shape[0] * 1.0
        # % of deleted or removed submissions
        self.explanatory_features['deleted_removed_submission_ratio'] = \
            deleted_or_removed_amount * 1.0 / submission_data.shape[0] * 1.0
        self.explanatory_features['submission_distinct_users'] = len(submissions_writers.keys())
        self.explanatory_features['submission_users_std'] = \
            np.std(list(submissions_writers.values())) if len(submissions_writers.values()) > 0 else None
        self.explanatory_features['average_submission_per_user'] = \
            (submission_data.shape[0] + 1) * 1.0 / (self.explanatory_features['submission_distinct_users'] + 1) * 1.0
        '''
        self.explanatory_features['submission_title_length'] = np.mean([len(title) for title in submission_data['title']])
        self.explanatory_features['submission_selftext_length'] = np.mean([len(selftext) for selftext in submission_data['selftext'] if type(selftext) is not float])
        self.explanatory_features['submission_average_score'] = 0 if submission_data.shape[0] == 0 else np.average(submission_data['score'])
        empty_amount = sum([1 if np.isnan(selftext) else 0 for selftext in submission_data['selftext']
                            if type(selftext) is float])
        self.explanatory_features['empty_selftext_ratio'] = empty_amount*1.0 / submission_data.shape[0]
        deleted_or_removed_amount = sum([1 if selftext == '[deleted]' or selftext == '[removed]' else 0
                                         for selftext in submission_data['selftext']])
        self.explanatory_features['deleted_removed_submission_ratio'] = deleted_or_removed_amount * 1.0 /\
                                                                        submission_data.shape[0]

        try:
            sr_pazam = datetime.datetime(2017, 3, 29).date() - self.creation_utc.date()
            self.explanatory_features['days_pazam'] = sr_pazam
        except AttributeError:
            print("Problem calculating the sr_pazam, self.creation_utc is: ".format(self.creation_utc))
            self.explanatory_features['days_pazam'] = None
            pass
        #distinct users amount feature
        submission_dist_users = Counter([name for name in submission_data['author'] if name != '[deleted]'])
        submission_users_std = np.std(list(submission_dist_users.values()))
        self.explanatory_features['submission_users_std'] = submission_users_std
        '''
        # comments related features - only relevant if we have comments data
        if comments_data is not None and comments_data.shape[0] > 0:
            body_len = []
            scores_values_comments = []
            deleted_or_removed_amount_comments = 0
            comments_writers = defaultdict(int)
            commented_comments = defaultdict(int)
            commented_submissions = defaultdict(int)
            commenting_a_submission_cnt = 0
            for cur_idx, cur_comm_row in comments_data.iterrows():
                try:
                    body_len.append(len(cur_comm_row['body']))
                except TypeError:
                    pass
                # old way with if-else
                #if type(cur_comm_row['body']) is str:
                #    body_len.append(len(cur_comm_row['body']))
                if type(cur_comm_row['score']) is int or type(cur_comm_row['score']) is float:
                    scores_values_comments.append(cur_comm_row['score'])
                if type(cur_comm_row['author']) is str and cur_comm_row['author'] != '[deleted]':
                    comments_writers[cur_comm_row['author']] += 1
                try:
                    if cur_comm_row['parent_id'].startswith('t3'):
                        commenting_a_submission_cnt += 1
                    elif cur_comm_row['parent_id'].startswith('t1'):
                        commented_comments[cur_comm_row['parent_id']] += 1
                except AttributeError:
                    pass
                # old way with if-else
                #if type(cur_comm_row['parent_id']) is str:
                #    if cur_comm_row['parent_id'].startswith('t3'):
                #        commenting_a_submission_cnt += 1
                #    elif cur_comm_row['parent_id'].startswith('t1'):
                #        commented_comments[cur_comm_row['parent_id']] += 1
                if type(cur_comm_row['link_id']) is str:
                    commented_submissions[cur_comm_row['link_id']] += 1
                if cur_comm_row['body'] in {'[deleted]', '[removed]'}:
                    deleted_or_removed_amount_comments += 1
            # adding 1 to both denominator and numerator to overcome division by zero errors
            self.explanatory_features['comments_submission_ratio'] = (comments_data.shape[0] + 1) * 1.0 /\
                                                                     (submission_data.shape[0] + 1)
            self.explanatory_features['submission_to_comments_users_ratio'] = \
                (len(comments_writers.keys()) + 1) * 1.0 / \
                (self.explanatory_features['submission_distinct_users'] + 1) * 1.0
            self.explanatory_features['distinct_comments_to_submission_ratio'] = \
                (len(commented_submissions.keys()) + 1) * 1.0 / (submission_data.shape[0] + 1) * 1.0
            self.explanatory_features['distinct_comments_to_comments_ratio'] = \
                (len(commented_comments.keys()) + 1) * 1.0 / (comments_data.shape[0] + 1) * 1.0
            self.explanatory_features['comments_users_std'] = \
                np.std(list(comments_writers.values())) if len(comments_writers.values()) > 0 else None
            self.explanatory_features['users_amount'] = \
                len(set(comments_writers.keys()).union(set(submissions_writers.keys())))
            '''
            self.explanatory_features['com_amount'] = comments_data.shape[0]
            #self.explanatory_features['com_amount_normalized'] = comments_data.shape[0] * 1.0 / self.num_users * 1.0
            # comments can be empty (no relevant comments), so we must handle it case the average is nan
            self.explanatory_features['average_comments_score'] = \
                0 if np.isnan(np.average(comments_data['score'])) else np.average(comments_data['score'])
            comments_dist_users = Counter([name for name in comments_data['author'] if name != '[deleted]'])
            comments_users_std = np.std(list(comments_dist_users.values()))
            self.explanatory_features['comments_users_std'] = comments_users_std
            dist_users = set(submissions_writers.keys()).union(set(comments_dist_users.keys()))
            self.explanatory_features['users_amount'] = len(dist_users)
            self.explanatory_features['comments_submission_ratio'] = (comments_data.shape[0] + 1) * 1.0 /\
                                                                     (submission_data.shape[0] + 1)
            self.explanatory_features['submission_to_comments_users_ratio'] = \
                len(set(submissions_writers.keys()))*1.0 / len(set(comments_dist_users.keys()))
            distinct_comments_to_tot_submission_ratio = len(set(comments_data['link_id'])) * 1.0 / \
                                                        submission_data.shape[0]
            self.explanatory_features['distinct_comments_to_submission_ratio'] = \
                distinct_comments_to_tot_submission_ratio
            distinct_comments_to_tot_comments_ratio = len(set(comments_data['link_id'])) * 1.0 / comments_data.shape[0]
            self.explanatory_features['distinct_comments_to_comments_ratio'] = distinct_comments_to_tot_comments_ratio
            
            commenting_upper_submission_ratio = np.mean([1 if p_id.startswith('t3') else 0 for p_id in
                                                         comments_data['parent_id']])
            self.explanatory_features['comment_to_upper_submission_ratio'] = commenting_upper_submission_ratio
            self.explanatory_features['comments_body_length'] = np.mean([len(body) for body in comments_data['body']
                                                                         if type(body) is not float])
             % of deleted or removed submissions
            deleted_or_removed_amount = sum([1 if body == '[deleted]' or body == '[removed]' else 0
                                             for body in comments_data['body']])
            self.explanatory_features['deleted_removed_comments_ratio'] = deleted_or_removed_amount * 1.0 / \
                                                                          comments_data.shape[0]
        '''
        # anyway, adding the users_amount feature, even if it based only on submissions data
        else:
            self.explanatory_features['users_amount'] = len(submissions_writers.keys())

    def update_urls(self, urls_found):
        """
        updates two attributes in the object related to the coordinates

        :param urls_found: dict
            dictionary with all the urls found and their counter (keys here are the netloc as defined by
            urllib.parse.urlparse class
        :return:
            only updates the instance parameters
        """
        self.urls_counter = urls_found
        self.explanatory_features["urls_ratio"] = sum(urls_found.values())*1.0/self.explanatory_features["submission_amount"]

    def update_coordinates(self, coordinates_found):
        """
        updates two attributes in the object related to the coordinates
        :param coordinates_found: int
            number of coordinates indication found
        :return:
            only updates the instance parameters
        """
        self.coordinates_counter = coordinates_found
        self.explanatory_features["coordinates_ratio"] = coordinates_found*1.0/self.explanatory_features["submission_amount"]
