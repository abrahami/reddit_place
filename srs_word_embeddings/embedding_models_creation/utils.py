# Authors: Avrahami Israeli (isabrah)
# Python version: 3.7
# Last update: 31.7.2019

import random
import os
import pickle
import datetime
import pandas as pd
import math
import numpy as np
from bisect import bisect_left


def sentences_yielder(dp_obj, sr_obj, config_dict, verbose=True, trees_info=None):
    """
    by given an SR object, returns all sentences of the SR object, broken into tokens after filtering all
    required data according to the configuration (e.g. vo
    :param dp_obj: data-prep object
        the object to be used in order to tokenize text and mark URLS
    :param sr_obj: 'sr_classifier' object
        the sub-reddit object to be used, it must have 'submissions_as_list' and 'comments_as_list'
    :param config_dict: dict
        configuration json which was converted into a dictionary and contains all configuration run settings
    :param verbose: bool, default: True
        whether or not to print information along the run
    :param trees_info: pandas df or None
        dataframe with information about each thread in the SR (this is recorder as tree_id and node_id
        if this is given as None, the boosting operation based on treed information is not used
    :return: list (of lists)
        each element in the list is a sentence and includes list of tokens.
        example: [['i', 'had', 'fun'], ['did', 'you', 'had', 'fun']]
    """
    config_dict_filters = config_dict["text_filters"]
    # in case we wish to use trees information and boost sentences based on this information
    if eval(config_dict["sentences_boosting"]["should_boost"]) and trees_info is not None:
        trees_info_dict = trees_info.set_index(['tree_id', 'node_id'])
        trees_info_dict = trees_info_dict.T.to_dict('list')

        # creating trees size distribution analysis, which is used later for an inner function
        # in this process we calculate only the limits of the distribution which interests us - this is in purpose
        # not to calculate the quartiles distribution of each call to the _sentence_factor_yielder function
        trees_size_dist = list(trees_info.groupby(by='tree_id')['tree_length'].max())
        max_boosting_factor = config_dict["sentences_boosting"]["max_boosting_factor"]
        percentiles_to_calc = [i / max_boosting_factor for i in range(1, max_boosting_factor+1)]
        if len(trees_size_dist):
            trees_size_quantiles = list(np.quantile(trees_size_dist, percentiles_to_calc))
        else:
            trees_size_quantiles = None

    subm_and_comments = []
    # looping over all submissions in the data
    for st in sr_obj.submissions_as_list:
        if eval(config_dict["sentences_boosting"]["should_boost"]) and trees_info is not None:
            # calculating the factor this submission should get (all sentences in the submission will be factored)
            cur_factor = \
                _sentence_factor_yielder(post_info=st, is_submissions=True, trees_info=trees_info_dict,
                                         trees_size_quantiles=trees_size_quantiles,
                                         max_boosting_factor=max_boosting_factor,
                                         trees_info_as_dict=True)
        else:
            cur_factor = 1
        # handling the header + selftext (it is non trivial in submissions, since some nan appear in header/selftext
        if type(st[2]) is str and type(st[1]) is str:
            for f in range(cur_factor):
                subm_and_comments.append((st[0], st[1] + ' . ' + st[2], st[5]))
        elif type(st[1]) is str:
            for f in range(cur_factor):
                subm_and_comments.append((st[0], st[1], st[5]))
        elif type(st[2]) is str:
            for f in range(cur_factor):
                subm_and_comments.append((st[0], st[2], st[5]))
        else:
            continue
    # looping over all comments in the data
    for st in sr_obj.comments_as_list:
        if eval(config_dict["sentences_boosting"]["should_boost"]) and trees_info is not None:
            # calculating the factor this comment should get (all sentences in the comment will be factored)
            cur_factor = \
                _sentence_factor_yielder(post_info=st, is_submissions=False, trees_info=trees_info_dict,
                                         trees_size_quantiles=trees_size_quantiles,
                                         max_boosting_factor=max_boosting_factor,
                                         trees_info_as_dict=True)
        else:
            cur_factor = 1
        if type(st[1]) is str:
            for f in range(cur_factor):
                subm_and_comments.append((st[0], st[1], st[5]))
        else:
            continue

    # first, in case we wish to filter data based on the score of each submission/comment
    if eval(config_dict_filters["sampling_rules"]["sample_data"]):
        subm_and_comments = list(filter(lambda tup: abs(tup[0]) >=
                                                    config_dict_filters["sampling_rules"]["min_abs_score"],
                                        subm_and_comments))
    # secondly, sorting the submissions/comments according to the logic required
    if config_dict_filters["ordering_logic"] == 'score':
        subm_and_comments.sort(key=lambda tup: tup[0], reverse=True)
    # case we want to sort the data by date - not an issue
    elif config_dict_filters["ordering_logic"] == 'date':
        subm_and_comments.sort(key=lambda tup: tup[2], reverse=True)
    # case we want to randomly sort the submissions
    elif config_dict_filters["ordering_logic"] == 'random':
        random.seed(config_dict["random_seed"])
        random.shuffle(subm_and_comments)
        random.shuffle(subm_and_comments)
    # lastly, if we wish to sample only part of the data
    if eval(config_dict_filters["sampling_rules"]["sample_data"]) and \
            config_dict_filters["sampling_rules"]["max_posts"] is not None:
        subm_and_comments = subm_and_comments[0:config_dict_filters["sampling_rules"]["max_posts"]]

    full_tok_text = []
    for st in subm_and_comments:
        normalized_text = dp_obj.mark_urls(st[1], marking_method='tag')[0]
        cur_tok_words = dp_obj.tokenize_text(sample=normalized_text, convert_to_lemmas=False, break_to_sents=True)
        full_tok_text.extend(cur_tok_words)
    if verbose:
        print("Finished handling sr {}. {} submissions/comments were processed, "
              "yielded {} sentences".format(sr_obj.name, len(subm_and_comments), len(full_tok_text)))
    return full_tok_text


def _sentence_factor_yielder(post_info, is_submissions, trees_info, trees_size_quantiles, max_boosting_factor=5,
                             trees_info_as_dict=True):
    """
    for a given pot (submission/comment) determins the factor required. This factor will be used later to boost
    sentences
    :param post_info: tuple
        tuple with full information about the post, which is:
        (0): the submission/comment score
        (1): the submission header or the comment body(depends which object we hold)
        (2): the submission/comment self-text (will be  '' for comments since the body of the comment is held in (1)
        (3): the submission/comment id. In case it is a comment a dictionary with the 3 relevant ids are held
             (this is the self id, the link_id and the parent_id)
        (4): the submission/comment author name
        (5): the submission/comment date (UTC time) saved as a date format
        Example of such tuple: (13, 'amazing work guys', '', '5lcgji', 'moshe' , 2017-01-01 00:00:03)
    :param is_submissions: bool
        whether the info is about submission or comment. This is important since data is pulled out differently in each
        case from the trees_info
    :param trees_info: pandas df
        dataframe with information about each thread in the SR (this is recorder as tree_id and node_id
    :param trees_size_quantiles: list
        list of length max_boosting_factor which holds the "border points" of the distribution (of trees size)
        example: [2, 5, 8, 22, 88] - which means that the 40% quantile of trees is 5 and the 80% quantile is 22
    :param max_boosting_factor: int
        maximum value for boosting to yield ,should be > 1
    :param trees_info_as_dict: bool
        whether the trees iforamtion was given as dictioanry. Other option is to get it as a pandas df, but dict option
        is much prefered due to performance issues of pandas df with "where" cluases

    :return:
    """
    if trees_info_as_dict:
        if is_submissions:
            # case no info about the tree was found, means it is a tree with a single node (?)
            try:
                post_info = trees_info[(post_info[3], post_info[3])]
            except KeyError:
                return 1
        # case we deal with a comment
        else:
            try:
                post_info = trees_info[(post_info[3]['parent_id'][3:], post_info[3]['id'])]
            except KeyError:
                return 1
        # case the size of the tree is smaller than the maximum boosting factor (in such case it doesn't make
        # sense to take the max_boosting_factor into account since the tree is very small)
        if post_info[3] <= max_boosting_factor:
            return 1
        # calculating a factor number for the tree size - as the tree is larger, number will be higher (anyway the number
        # should be in the [1, max_boosting_factor] range
        trees_size_score = bisect_left(a=trees_size_quantiles, x=post_info[3])
        # aliging the number (since it starts from zero) + making sure the number is not higher than max_boosting_factor
        trees_size_score = min(trees_size_score + 1, max_boosting_factor)
        # calculating a factor number for the height of the post in the tree - as the post is higher and has "more children"
        # it will get a higher number. Anyway the number will be in the [0, max_boosting_factor] range
        post_height_score = post_info[1] / post_info[3] * 1.0
        post_height_score = post_height_score * max_boosting_factor
        # final score is average between the 2 factors
        total_score = math.ceil(0.5*trees_size_score + 0.5*post_height_score)
        # making sure the final number is not below 1 and not above the maximum allowed
        return min(max(1, total_score), max_boosting_factor)
    # case the info we got is not in the format of a dict (NOT recommneded, very slow)
    else:
        if is_submissions:
            post_info = trees_info[(trees_info['tree_id'] == post_info[3]) & (trees_info['node_id'] == post_info[3])]
            # case no info about the tree was found, means it is a tree with a single node (?)
            if post_info.shape[0] == 0:
                return 1
            # case the size of the tree is smaller than the maximum boosting factor (in such case it doesn't make
            # sense to take the max_boosting_factor into account since the tree is very small)
            if post_info['tree_length'].values[0] <= max_boosting_factor:
                return post_info['tree_length'].values[0]
        else:
                post_info = trees_info[(trees_info['tree_id'] == post_info[3]['parent_id'][3:]) & (trees_info['node_id'] == post_info[3]['id'])]
        # calculating a factor number for the tree size - as the tree is larger, number will be higher (anyway the number
        # should be in the [1, max_boosting_factor] range
        trees_size_score = bisect_left(a=trees_size_quantiles, x=post_info['tree_length'].values[0])
        trees_size_score = min(trees_size_score + 1, max_boosting_factor)
        # calculating a factor number for the height of the post in the tree - as the post is higher and has "more children"
        # it will get a higher number. Anyway the number will be in the [0, max_boosting_factor] range
        post_height_score = post_info['subt_length'] / post_info['tree_length'] * 1.0
        post_height_score = max(post_height_score) * max_boosting_factor
        # final score is average between the 2 factors
        total_score = math.ceil(0.5*trees_size_score + 0.5*post_height_score)
        # making sure the final number is not below 1 and not above the maximum allowed
        return min(max(1, total_score), max_boosting_factor)


def analyse_vocab_size(sr_files, sr_objects_path, filter_non_english=True, verbose=True):
    '''
    Help function to analyse the distribution of the vocabulary size of all SRs. This is useful in order
    to decide upon a vocabulary size threshold which will help us to remove too small SRs
    :param sr_files: list
        list of file names (holding sr_objects)
    :param sr_objects_path: str
        path to the location of the files
    :param filter_non_english: bool, default: True
        whether to filter non English communities out of the corpus
    :param verbose: bool, default: True
        whether to print progress to screen
    :return: dict
       "counted_srs": number of communities took into account in the analysis
        "skipped_srs": number of communities filtered out (due to language)
        "distribution": the distribution of the communities took into account in terms of vocabulary size
    '''
    start_time = datetime.datetime.now()
    skipped_srs = 0
    vocabs_len = dict()
    # looping over each file
    for loop_idx, f_name in enumerate(sr_files):
        cur_sr = pickle.load(open(os.path.join(sr_objects_path, f_name), "rb"))
        # filtering sr in case needed and the language is not English
        if filter_non_english and not (cur_sr.lang == 'en' or cur_sr.lang is None):
            skipped_srs += 1
            continue
        subm_dict = dict(cur_sr.submissions_tokens_dict)
        comments_dict = dict(cur_sr.comments_tokens_dict)
        full_dict = {**subm_dict, **comments_dict}
        vocabs_len[cur_sr.name] = len(full_dict)

        if verbose and loop_idx % 50 == 0 and loop_idx != 0:
            duration = (datetime.datetime.now() - start_time).seconds
            print("Finished handling {} sr objects. Took us up to now: {} sec".format(loop_idx, duration))
    values_as_df = pd.DataFrame(list(vocabs_len.values()))
    return {"counted_srs": len(vocabs_len), "skipped_srs": skipped_srs,
            "distribution": values_as_df.quantile(q=[i/20.0 for i in list(range(0, 20))])}


def is_sr_valid(sr, config_dict):
    '''
    checking if the SR stands with the configurations limitations set (e.g. minimum number of vocabulary words)
    :param sr: sub-reddit object
        the sr to analyse
    :param config_dict: dict
        configuration dictionary. This is the json configuration current run is set with
    :return: bool
        True in case the SR is valid, False otherwise
    '''
    # if language is not English and configuration is set to remove non English communities
    if eval(config_dict["srs_filters"]["filter_non_english"]) and not (sr.lang == 'en' or sr.lang is None):
        return False
    subm_dict = dict(sr.submissions_tokens_dict)
    comments_dict = dict(sr.comments_tokens_dict)
    full_dict = {**subm_dict, **comments_dict}
    # if vocabulary size is too small
    if len(full_dict) < config_dict["srs_filters"]["minimum_dict_size"]:
        return False
    # all other options - we'll return True
    return True


def tensorboard_visualize(model, output_path):
    """
    Allows to create files for viewing the TB later as needed. This is used only in environments where TF exists
    :param model: gensim object
        Gensim trained model (word2vec)
    :param output_path: str
        location where files should be saved
    :return: Nothing

    Example how to run:
    1. Change the env. to be one which includes TF (maybe it is a virtual env) - in the settings
    2. In this code, change the folder to where the gensim model exists
    example:
    >>>os.chdir('C:\\Users\\avrahami\\Documents\\Private\\Uni\\BGU\\PhD\\reddit_canvas\\data\\embedding\\embedding_per_sr\\soccer')
    3. Load the file where the model exists
    example:
    >>>from gensim.models import Word2Vec
    >>>model = Word2Vec.load("soccer_model_2.02.model")
    4. Run this function (it should create new files in the folder
    >>>tensorboard_visualize(model, os.getcwd())

    After the above had been done, do the following in order to see the tensor-board
    1. Open anaconda prompt
    2. activate the tnesorflow env
    example:
    >>>activate tensorflow_cpu (the env should be changed)
    3. run the following code: tensorboard --logdir=NAME OF THE FOLDER
    example:
    >>>tensorboard --logdir=C:\\Users\\avrahami\\Documents\\Private\\Uni\\BGU\\PhD\\reddit_canvas\\data\\embedding\\embedding_per_sr\\soccer
    """
    import tensorflow as tf
    from tensorflow.python.framework import ops
    from tensorflow.contrib.tensorboard.plugins import projector
    meta_file = "w2x_metadata.tsv"
    placeholder = np.zeros((len(model.wv.index2word), model.vector_size))

    with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:
        for i, word in enumerate(model.wv.index2word):
            placeholder[i] = model[word]
            # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
            if word == '':
                print("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
                file_metadata.write("{0}".format('anything else').encode('utf-8') + b'\n')
            else:
                file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')
    ops.reset_default_graph()
    # define the model without training
    sess = tf.InteractiveSession()
    sess.close()
    sess = tf.InteractiveSession()

    embedding = tf.Variable(placeholder, trainable=False, name='w2x_metadata')
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'w2x_metadata'
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path, 'w2x_metadata.ckpt'))
    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))