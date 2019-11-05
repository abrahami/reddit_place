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
from srs_word_embeddings.embedding_models_creation.posts_yielder import PostsYielder
from gensim.models import Doc2Vec
from sr_classifier.reddit_data_preprocessing import RedditDataPrep
import multiprocessing
import pickle
import itertools
import csv
import pandas as pd

###################################################### Configurations ##################################################
config_dict = commentjson.load(open(os.path.join(os.getcwd(), 'configurations', 'srs_word_embeddings_config.json')))
machine = 'yalla' if sys.platform == 'linux' else os.environ['COMPUTERNAME']
data_path = config_dict['data_dir'][machine]
########################################################################################################################
start_time = datetime.datetime.now()

if __name__ == "__main__":
    # update args of the configuration dictionary which can be known right as we start the run
    config_dict['machine'] = machine
    dp_obj = RedditDataPrep(is_submission_data=True, remove_stop_words=False, most_have_regex=None)
    sr_objects_path = os.path.join(data_path, 'sr_objects')
    saving_path = os.path.join(config_dict['data_dir'][machine], "embedding",
                               "embedding_per_sr", config_dict["model_version"])
    # check if the directory exists, if not - we'll create one
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path)

    model_saving_file = os.path.join(saving_path,
                                     "doc2vec_model_version_" + str(config_dict["model_version"] + ".model"))
    # deciding if we need to build a new gensim model or one already exists
    new_model_required = True if (not os.path.exists(model_saving_file)
                                  or eval(config_dict["doc2vec_config"]["overwrite_existing"])) else False
    # case building the model is required
    if new_model_required:
        # saving the configuration file, so we'll know which configuration was used
        config_file_to_save = os.path.join(saving_path, 'config_file_model' + str(config_dict["model_version"]) + '.json')
        with open(config_file_to_save, 'w') as fp:
            commentjson.dump(config_dict, fp, indent=2)
        # handling objects_limit - in case it is < 0, we'll replace it with None (this is the PostsYielder format)
        objects_limit = config_dict["doc2vec_config"]["objects_limit"]
        objects_limit = None if objects_limit < 0 else objects_limit
        posts_yielder_obj = PostsYielder(config_dict=config_dict, objects_limit=objects_limit, verbose=True)
        model_hyper_params = dict(config_dict["doc2vec_config"]["doc2vec_hyper_params"])
        # calling the Gensim Doc2Vec constructor
        model = Doc2Vec(**model_hyper_params, workers=int(multiprocessing.cpu_count() * 0.75 + 1))
        # building the vocabulary
        model.build_vocab(posts_yielder_obj)
        # training the model
        model.train(posts_yielder_obj, total_examples=model.corpus_count,
                    epochs=config_dict["doc2vec_config"]["epochs"])
        objects_vectors = {doctag_name: model.docvecs[doctag_name] for doctag_name, _ in model.docvecs.doctags.items()}
        # saving the model + the vector representation as a dict
        objects_vectors_saving_file = os.path.join(saving_path,
                                                   "doc2vec_dict_version_" + str(config_dict["model_version"] + ".p"))
        # case the model file doesn't exist, or we with to overwrite exiting one with newer one
        if not os.path.exists(model_saving_file) or eval(config_dict["doc2vec_config"]["overwrite_existing"]):
            model.save(model_saving_file)
            pickle.dump(objects_vectors, open(objects_vectors_saving_file, "wb"))
        duration = (datetime.datetime.now() - start_time).seconds
        print("Modeling has finished and saved in {}. Total time took us: {} seconds".format(saving_path, duration))
    else:
        doc2vec_model = Doc2Vec.load(model_saving_file)
        # as first step, we will save dictionary (of dictionaries) with full embeddings info
        communities_embeddings_dict = dict()
        community_names = list(doc2vec_model.docvecs.doctags.keys())
        for cur_community in community_names:
            cur_embedding = doc2vec_model.docvecs[cur_community]
            communities_embeddings_dict[cur_community] = {idx: value for idx, value  in enumerate(cur_embedding)}
        pickle.dump(communities_embeddings_dict, open(os.path.join(saving_path,
                                                                   "communities_doc2vec_model_5_11_2019_dict.p"), "wb"))
        # in case we wish to create all combinations of pairs similarity (used for further analysis later) - needed to
        # be done only once, this is why it is marked as a comment
        # loading the communities names (only part of them are relevant for the analysis)
        pairs_sim = pd.read_csv("/data/work/data/reddit_place/embedding/pairs_similarity_general_2.02_V3.csv")
        all_com_names = set(list(pairs_sim['name_m1']) + list(pairs_sim['name_m2']))
        pairs_similarity_res = list()
        community_names = list(doc2vec_model.docvecs.doctags.keys())
        communities_pairs = list(itertools.combinations(community_names, 2))
        for idx, cur_pair in enumerate(communities_pairs):
            if idx % 1000000 == 0:
                print(f"Passed over {idx} cases. Horray!")
            # case we should add the pair to the list
            if cur_pair[0] in all_com_names and cur_pair[1] in all_com_names:
                pairs_similarity_res.append({'name_m1': cur_pair[0], 'name_m2': cur_pair[1],
                                             'doc2vec_dist': doc2vec_model.docvecs.distance(cur_pair[0], cur_pair[1])})
        print(f"Total of {len(pairs_similarity_res)} pairs were found and calculated")
        keys = pairs_similarity_res[0].keys()
        with open(os.path.join(saving_path, 'pairs_similarity_doc2vec.csv'), 'w') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(pairs_similarity_res)

        # now for example we can get the most similar communities to 'babybumps'
        print(doc2vec_model.docvecs.most_similar('babybumps', topn=20))
        print(doc2vec_model.docvecs.similarity('babybumps', 'depression'))
        print(doc2vec_model.docvecs.most_similar('communism', topn=20))
        print(doc2vec_model.docvecs.similarity('babybumps', 'depression'))

        doc2vec_model.docvecs.most_similar('nyc', topn=20)
        doc2vec_model.docvecs.similarity('nyc', 'melbourne')

