# Authors: Avrahami Israeli (isabrah)
# Python version: 3.7
# Last update: 31.7.2019

#! /usr/bin/env python
import os
import sys
if sys.platform == 'linux':
    sys.path.append('/data/home/isabrah/reddit_canvas/reddit_project_with_yalla_cluster/reddit-tools')
import datetime
import commentjson
import sys
import r_place_drawing_classifier.pytorch_cnn.utils as pytorch_cnn_utils
from srs_word_embeddings.embedding_models_creation.utils import sentences_yielder, analyse_vocab_size, is_sr_valid
import pickle
from gensim.models import Word2Vec, FastText
from sr_classifier.reddit_data_preprocessing import RedditDataPrep
import re
import pandas as pd

###################################################### Configurations ##################################################
config_dict = commentjson.load(open(os.path.join(os.path.dirname(os.getcwd()), 'configurations',
                                                 'srs_word_embeddings_config.json')))
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
    sr_files = [f for f in os.listdir(sr_objects_path) if re.match(r'sr_obj_.*\.p', f)]
    # sorting the files so we'll pass over them in a known order
    sr_files.sort()
    saving_path = os.path.join(config_dict['data_dir'][machine], "embedding",
                               "embedding_per_sr", config_dict["model_version"])
    # check if the directory exists, if not - we'll create one
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path)
    # case we don't want to re-write models, but rather take into account the existing models and build only new ones
    if not eval(config_dict["saving_options"]["overwrite_existing"]):
        existing_model_files = [f for f in os.listdir(saving_path)
                                if re.match(r'.*_model_{}.model$'.format(str(config_dict["model_version"])), f)]
        existing_srs_names = list(set([emf.partition("_model_")[0] for emf in existing_model_files]))
        sr_files_subset = [sf for sf in sr_files if re.sub(r'sr_obj_|_.p', '', sf) not in existing_srs_names]
        sr_files = sr_files_subset
        print("{} models were found, we are going to create {} models".format(len(existing_model_files), len(sr_files)))
    # saving the configuration file, so we'll know which configuration was used
    config_file_to_save = os.path.join(saving_path, 'config_file_model' + str(config_dict["model_version"]) + '.json')
    if os.path.exists(config_file_to_save):
        print("Config file for the model version provided already exists, we are overwriting it")
    with open(config_file_to_save, 'w') as fp:
        commentjson.dump(config_dict, fp, indent=2)
    # case we wish to boost some of the sentences, we'll load table holding relevant information
    if eval(config_dict["sentences_boosting"]["should_boost"]):
        trees_info_csv_path = config_dict["sentences_boosting"]["trees_information_csv_path"][machine]
        trees_information_df = pd.read_csv(filepath_or_buffer=trees_info_csv_path)
    else:
        trees_information_df = None
    # these 2 lines are used to analyse the vocabulary size distribution in order to filter out small SRs
    #analysis_res = analyse_vocab_size(sr_files=sr_files, sr_objects_path=sr_objects_path, filter_non_english=True)
    #print(analysis_res)
    skipped_srs = []
    # looping over all files found in the directory
    for loop_idx, f_name in enumerate(sr_files):
        cur_sr = pickle.load(open(os.path.join(sr_objects_path, f_name), "rb"))
        start_time = datetime.datetime.now()
        if loop_idx % 50 == 0 and loop_idx != 0:
            duration = (datetime.datetime.now() - start_time).seconds
            print("Finished handling {} sr objects. Took us up to now: {} sec".format(loop_idx, duration))
        if eval(config_dict["srs_filters"]["should_filter"]) and not is_sr_valid(sr=cur_sr, config_dict=config_dict):
            skipped_srs.append(cur_sr.name)
            continue
        trees_info_subset = trees_information_df[trees_information_df["sub_reddit"] == cur_sr.name] if trees_information_df is not None else None
        full_tok_text = sentences_yielder(dp_obj=dp_obj, sr_obj=cur_sr, config_dict=config_dict, verbose=True,
                                          trees_info=trees_info_subset)
        if len(full_tok_text) < 50:
            skipped_srs.append(cur_sr.name)
            print("sr {} has only {} sentences, skipping it".format(cur_sr.name, len(full_tok_text)))
            continue
        # creating a model object and running it
        if config_dict["model"]["type"] == "w2v":
            model = Word2Vec(full_tok_text, **config_dict["model"]["hyper_params"])
        elif config_dict["model"]["type"] == "fasttext":
            model = FastText(full_tok_text, **config_dict["model"]["hyper_params"])
        else:
            raise IOError("Invalid model name, has to be either 'w2v' or 'fasttext'")
        model.train(sentences=full_tok_text, total_examples=model.corpus_count, epochs=model.epochs)
        saving_file = os.path.join(saving_path, cur_sr.name + "_model_" + str(config_dict["model_version"] + ".model"))
        model.save(saving_file)
        duration = (datetime.datetime.now() - start_time).seconds
        print("Model built for sr {}, corpus size: {}, time: {} sec".format(cur_sr.name, len(model.wv.vocab), duration))
        # model.wv['i'] # will print the numpy vector representation of the word i
        # after going over all submissions, we add it to the object itself
    # case cv_splits were not initialized before - we'll do it here for the first time
    print("Code has ended, {} SRs models were not created due to configuration reasons".format(len(skipped_srs)))
    print("All models finished and saved in {}".format(saving_path))
