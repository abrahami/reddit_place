# Authors: Avrahami Israeli (isabrah)
# Python version: 3.6
# Last update: 9.10.2018

from sklearn.base import TransformerMixin
#import datetime


class MetaFeaturesExtractor(TransformerMixin):
    """
    Class to handle the process of extracting the meta features of a Sub-Reddit (SR) object
    This is done usually as part of a pipline, where a transform function needs to be defined as part of the flow
    """
    def transform(self, X, **transform_params):
        """
        transforms the list of the input X (list of SR objects) to list of dictionaries holding the explanatory features
        :param X: list
            the SR objects given as a list objects
        :param transform_params:
            all other parameters needed to be used along the transform function (currently nothing is passes here)
        :return: list of the explanatory features of each SR, as these are being hold within the SR object.
            Currently it is a list ob dictionaries, where each entry in the dictionary is an explanatory feature
        """

        #print("Finished running the transform function inside the class 'MetaFeaturesExtractor'"
        #      " The time now is {}".format(datetime.datetime.now()))
        return [cur_sr_obj.explanatory_features for cur_sr_obj in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

