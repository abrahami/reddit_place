# Authors: Avrahami Israeli (isabrah)
# Python version: 3.7
# Last update: 31.7.2019

import os
import re
import pickle
import itertools
from gensim.models.doc2vec import TaggedDocument
import datetime


class PostsYielder(object):
    """
    class to generate posts (submissions/comments) for further usage by the doc2vec algorithm.
    This is analogue to the sentences_yielder in utils.py, but since doc2vec works on documents with label, we need
    something a bit different.
    The concept is to yield information in a way not all data is loaded to memory.
    Idea and concept is taken from here: http://yaronvazana.com/2018/01/20/training-doc2vec-model-with-gensim/

    Parameters
    ----------
    config_dict: dict
        the configuration dictionary input. This is read as a json in the main file
    objects_limit: int or None
        limiting the number of objets files to handle
        Example: if value is 100, only the top 100 files will be taken
        In None - all files are taken into account
    verbose: bool, default: True
        whether to print progress to the screen


    Attributes
    ----------
    data_path: str
        location of the data, taken from the config_file information
    object_files: list
        sorted list of sr object files, they are sorted by name
    start_time: datetime
        time when the run started. Used later for verbose options
    """
    def __init__(self, config_dict, objects_limit=None, verbose=True):
        self.config_dict = config_dict
        self.verbose = verbose
        machine = self.config_dict['machine']
        self.data_path = self.config_dict["data_dir"][machine]
        objects_path = os.path.join(self.data_path, 'sr_objects')
        self.object_files = sorted([f for f in os.listdir(objects_path) if re.match(r'sr_obj_.*\.p', f)])
        if objects_limit is not None:
            self.object_files = self.object_files[0:objects_limit]
        self.start_time = datetime.datetime.now()

    # defining an iterator, so not all data will be loaded to memory, We do it with the yield function
    def __iter__(self):
        # looping over all sr object files found
        for file_idx, sr_obj_file in enumerate(self.object_files):
            if self.verbose and file_idx % 100 == 0:
                duration = (datetime.datetime.now() - self.start_time).seconds
                print("Handled {} files already. Duration up to now: {} seconds".format(file_idx, duration))
            subm_and_comments = []
            cur_sr = pickle.load(open(os.path.join(self.data_path, 'sr_objects', sr_obj_file), "rb"))
            cur_sr_name = cur_sr.name
            # looping all submission and joining together lists of the submission (this is due to the fact that
            # submissions_as_tokens breaks the data into sentences, but we want to have instance per submission, even
            # if contain multiple sentences)
            for st in cur_sr.submissions_as_tokens:
                if len(st) > 0:
                    subm_and_comments.append(list(itertools.chain.from_iterable(st)))
                else:
                    continue
            # looping all comments and joining together lists of the comments (this is due to the fact that
            # comments_as_tokens breaks the data into sentences, but we want to have instance per comments, even
            # if contain multiple sentences)

            for st in cur_sr.comments_as_tokens:
                if len(st) > 0:
                    subm_and_comments.append(list(itertools.chain.from_iterable(st)))
                else:
                    continue
            # yield the current submission/comment - this is critical, in order not to load all data to memory
            for post_idx, cur_post in enumerate(subm_and_comments):
                yield TaggedDocument(words=cur_post, tags=[cur_sr_name])
