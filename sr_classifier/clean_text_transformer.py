# Authors: Avrahami Israeli (isabrah)
# Python version: 3.6
# Last update: 9.10.2018

from sklearn.base import TransformerMixin
from reddit_data_preprocessing import RedditDataPrep as reddit_dp
import re
import datetime


class CleanTextTransformer(TransformerMixin):
    '''
    Class to handle the process of extracting the text relevant to each Sub-Reddit (SR).
    This is done usually as part of a pipline, where a transform function needs to be defined as part of the flow
    Parameters
    ----------
    marking_method: dict
        the way to mark interesting entities in the text of each SR. Currently options of replacement are:
        'replace' or 'add', according to the logic of the RedditDataPrep class functions

    Attributes
    ----------

    '''
    def __init__(self, marking_method={'urls': 'replace', 'coordinates': 'replace'}):
        self.marking_method = marking_method

    def transform(self, X, **transform_params):
        '''
        transforms the list of the input X (list of SR objects) to list texts.
        In addition, this function also updates the urls and coordinates found in the text (for each SR)
        The reason it is being done here, is due to the fact that here we have a full long text to each SR, so it is
        very easy to find all the interesting entities related to reddit r/place
        :param X: list
            the SR objects given as a list objects
        :param transform_params:
            all other parameters needed to be used along the transform function (currently nothing is passes here)
        :return: list of the clean text of each SR
        '''
        start_time = datetime.datetime.now()
        sr_text = []
        # looping over each SR in X
        for cur_sr_obj in X:
            # applying the built in functions within the SR object (e.g. concat_and_weight_text(), mark_urls())
            normalized_text = cur_sr_obj.concat_and_weight_text()
            normalized_text, urls_found = reddit_dp.mark_urls(text=normalized_text,
                                                              marking_method=self.marking_method['urls'])
            normalized_text, coordinates_found = reddit_dp.mark_coordinates(text=normalized_text,
                                                                            marking_method=self.marking_method['coordinates'])
            sr_text.append(normalized_text)
            # updaing the sr_object with the counters we have found
            cur_sr_obj.update_urls(urls_found=urls_found)
            cur_sr_obj.update_coordinates(coordinates_found=coordinates_found)
        # before we return the text, we add another cleaning, which removes redundant chars
        list_to_return = [self.clean_text(text) for text in sr_text]

        #duration = (datetime.datetime.now() - start_time).seconds
        #print("Finished running transform function inside the 'CleanTextTransformer' class over {} SRs"
        #      " Took us {} seconds".format(len(X), duration))
        return list_to_return

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

    @classmethod
    def clean_text(cls, text):
        '''
        Cleaning a given input string from redundant chars (e.g. new lines)
        :param text: str
            the string input to be replaced
        :return: str
            the cleaned string
        '''
        # removing multiple spaces at the beginning or end of the string
        text = text.strip()
        # removing extra new rows
        text = re.sub(pattern=r'[\r\n]+', repl=" ", string=text)
        # removing all extra spaces (cases when there are sequential spaces one after another)
        text = re.sub("\s\s+", " ", text)
        # converting all to lower-case letters
        text = text.lower()
        return text
