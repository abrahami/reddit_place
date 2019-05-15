# Authors: Avrahami Israeli (isabrah)
# Python version: 3.7
# Last update: 24.4.2019

#! /usr/bin/env python
import os
import sys
if sys.platform == 'linux':
    sys.path.append('/data/home/isabrah/reddit_canvas/reddit_project_with_yalla_cluster/reddit-tools')
import datetime
import commentjson
import sys
import r_place_drawing_classifier.pytorch_cnn.utils as pytorch_cnn_utils
from srs_words_embedding.utils import sentences_yielder
import pickle
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from sr_classifier.reddit_data_preprocessing import RedditDataPrep
import re

###################################################### Configurations ##################################################
config_dict = commentjson.load(open(os.path.join(os.getcwd(), 'srs_words_emnedding_config.json')))
machine = 'yalla' if sys.platform == 'linux' else os.environ['COMPUTERNAME']
data_path = config_dict['data_dir'][machine]
########################################################################################################################
start_time = datetime.datetime.now()
pytorch_cnn_utils.set_random_seed(seed_value=config_dict["random_seed"])

if __name__ == "__main__":
    # update args of the configuration dictionary which can be known right as we start the run
    config_dict['machine'] = machine
    dp_obj = RedditDataPrep(is_submission_data=True, remove_stop_words=False, most_have_regex=None)
    sr_objects_path = os.path.join(data_path, 'sr_objects')
    sr_files = [f for f in os.listdir(sr_objects_path) if re.match(r'sr_obj.*\.p', f)]
    # looping over all files found in the directory
    for loop_idx, f_name in enumerate(sr_files):
        cur_sr = pickle.load(open(os.path.join(sr_objects_path, f_name), "rb"))
        start_time = datetime.datetime.now()
        if loop_idx % 10 == 0 and loop_idx != 0:
            duration = (datetime.datetime.now() - start_time).seconds
            print("Finished handling {} sr objects. Took us up to now: {} sec".format(loop_idx, duration))
        full_tok_text = sentences_yielder(dp_obj=dp_obj, sr_obj=cur_sr, config_dict=config_dict, verbose=True)
        if len(full_tok_text) < 50:
            print("sr {} has only {} sentences, skipping it".format(cur_sr.name, len(full_tok_text)))
            continue
        model = Word2Vec(full_tok_text, size=300, window=5, min_count=3, workers=20, sg=0)
        model.train(sentences=full_tok_text, total_examples=model.corpus_count, epochs=model.epochs)
        saving_file = os.path.join(config_dict['data_dir'][machine], "embedding", "embedding_per_sr")
        # check if the directory exists, if not - we'll create one
        if not os.path.isdir(saving_file):
            os.makedirs(saving_file)
        saving_file = os.path.join(saving_file, cur_sr.name + "_model_" + str(config_dict["model_version"] + ".model"))
        model.save(saving_file)
        duration = (datetime.datetime.now() - start_time).seconds
        print("Model built for sr {}, corpus size: {}, time: {} sec".format(cur_sr.name, len(model.wv.vocab), duration))
        # model.wv['i'] # will print the numpy vector representation of the word i
        # after going over all submissions, we add it to the object itself
    # case cv_splits were not initialized before - we'll do it here for the first time



