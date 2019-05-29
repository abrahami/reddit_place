# Authors: Shani Cohen (shanisa)
# Python version: 3.7
# Last update: 22.452019

#! /usr/bin/env python
import os
from os.path import join
import sys
import platform as p
from communities_comparison.utils import get_models_names, load_model
from communities_comparison.tf_idf import calc_tf_idf_all_models
import commentjson
import datetime
import sys
import pickle
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, FastText
import re
import config as c

#
# ###################################################### Configurations ##################################################
# config_dict = commentjson.load(open(os.path.join(os.getcwd(), 'srs_words_emnedding_config.json')))
# machine = 'yalla' if sys.platform == 'linux' else os.environ['COMPUTERNAME']
# data_path = config_dict['data_dir'][machine]
# ########################################################################################################################

# model_gameswap = Word2Vec.load(join(c.data_path, 'gameswap_model_1.01' + '.model'))
# model_babymetal = Word2Vec.load(join(c.data_path, 'babymetal_model_1.01' + '.model'))
# # temp - should be provided
# weights_gameswap = generate_weights(model=model_gameswap)
# weights_babymetal = generate_weights(model=model_babymetal)
#
# # todo- two elements to consider when comparing communities:
# #  1 - size of intersection
# #  2 - similarity based on intersection
# compare(model_1=model_gameswap, model_2=model_babymetal,
#         weights_1=weights_gameswap, weights_2=weights_babymetal)
#
# # tsne_plot(model_gameswap)
# # tsne_plot(model_babymetal)


if __name__ == "__main__":
    # update args of the configuration dictionary which can be known right as we start the run

    # config_dict['machine'] = machine
    # dp_obj = RedditDataPrep(is_submission_data=True, remove_stop_words=False, most_have_regex=None)
    # sr_objects_path = join(data_path, 'sr_objects')
    print(f"MODEL_TYPE = {c.MODEL_TYPE}")
    print(f"1 - CALC_TF_IDF -> {c.CALC_TF_IDF}")
    if c.CALC_TF_IDF:
        calc_tf_idf_all_models(m_type=c.MODEL_TYPE)

    print(f"2 - CALC_SCORES -> {c.CALC_SCORES}")
    # todo-pairwise comparisons
    m_names = get_models_names(path=c.data_path, m_type=c.MODEL_TYPE)
    for i, m_name in enumerate(m_names):
        curr_model = load_model(path=c.data_path, m_type=c.MODEL_TYPE, name=m_name)
        print()

    # model_files = [f for f in os.listdir(c.data_path) if re.match(r'sr_obj.*\.p', f)]
    # # looping over all files found in the directory
    # for loop_idx, f_name in enumerate(sr_files):
    #     cur_sr = pickle.load(open(os.path.join(sr_objects_path, f_name), "rb"))
    #     start_time = datetime.datetime.now()
    #     if loop_idx % 10 == 0 and loop_idx != 0:
    #         duration = (datetime.datetime.now() - start_time).seconds
    #         print("Finished handling {} sr objects. Took us up to now: {} sec".format(loop_idx, duration))
    #     full_tok_text = sentences_yielder(dp_obj=dp_obj, sr_obj=cur_sr, config_dict=config_dict, verbose=True)
    #     if len(full_tok_text) < 50:
    #         print("sr {} has only {} sentences, skipping it".format(cur_sr.name, len(full_tok_text)))
    #         continue
    #     model = Word2Vec(full_tok_text, size=300, window=5, min_count=3, workers=20, sg=0)
    #     model.train(sentences=full_tok_text, total_examples=model.corpus_count, epochs=model.epochs)
    #     saving_file = os.path.join(config_dict['data_dir'][machine], "embedding", "embedding_per_sr")
    #     # check if the directory exists, if not - we'll create one
    #     if not os.path.isdir(saving_file):
    #         os.makedirs(saving_file)
    #     saving_file = os.path.join(saving_file, cur_sr.name + "_model_" + str(config_dict["model_version"] + ".model"))
    #     model.save(saving_file)
    #     duration = (datetime.datetime.now() - start_time).seconds
    #     print("Model built for sr {}, corpus size: {}, time: {} sec".format(cur_sr.name, len(model.wv.vocab), duration))
    #     # model.wv['i'] # will print the numpy vector representation of the word i
    #     # after going over all submissions, we add it to the object itself
    # # case cv_splits were not initialized before - we'll do it here for the first time



