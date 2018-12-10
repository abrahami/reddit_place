# Authors: Avrahami Israeli (isabrah)
# Python version: 3.7
# Last update: 21.11.2018

import dynet_config
# set random seed to have the same result each time
dynet_config.set(random_seed=1984)
import dynet as dy
from collections import defaultdict
import time
import random
import numpy as np
import pandas as pd
import pickle
import sys
from sr_classifier.reddit_data_preprocessing import RedditDataPrep
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import collections
import datetime
from r_place_drawing_classifier.utils import remove_huge_srs


class NNClassifier(object):
    """
    """

    def __init__(self, model_type='lstm', emb_size=100, hid_size=100, epochs=10, use_bilstm=False,
                 model_sentences = False, seed=1984):
        self.model_type = model_type
        self.use_bilstm = use_bilstm
        self.epochs = epochs
        # Functions to read in the corpus
        self.w2i = defaultdict(lambda: len(self.w2i))
        self.t2i = defaultdict(lambda: len(self.t2i))
        self.w2i["<unk>"]
        self.nwords = None
        self.ntags = None
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.model_sentences = model_sentences
        self.seed = seed
        random.seed(self.seed)
        self.eval_measures = dict()

    def get_reddit_sentences(self, sr_objects, tokenizer, maximum_sent_per_sr=5000):
        # looping over all sr objects
        for cur_sr in sr_objects:
            # pulling out the tag of the current sr
            tag = cur_sr.trying_to_draw
            if maximum_sent_per_sr is not None:
                cur_sr_sentences = [i[1] for idx, i in enumerate(cur_sr.submissions_as_list) if
                                    idx < maximum_sent_per_sr]
            else:
                cur_sr_sentences = [i[1] for i in cur_sr.submissions_as_list]

            if self.model_sentences:
                for sen in cur_sr_sentences:
                    sen_tokenized = tokenizer(sen)
                    if len(sen_tokenized) > 0:
                        yield ([self.w2i[x] for x in sen_tokenized], self.t2i[tag], cur_sr.name)
            else:
                cur_sr_sentences_as_int = []
                for sen in cur_sr_sentences:
                    sen_tokenized = tokenizer(sen)
                    if len(sen_tokenized) > 0:
                        cur_sr_sentences_as_int.append([self.w2i[x] for x in sen_tokenized])
                yield (cur_sr_sentences_as_int, self.t2i[tag], cur_sr.name)

        self.nwords = len(self.w2i)
        self.ntags = len(self.t2i)

    def read_dataset(self, filename, dataset='stanford'):
        if dataset == 'stanford':
            with open(filename, "r", encoding='utf-8') as f:
                for line in f:
                    # tag, words = line.lower().strip().split(" ||| ")
                    tag, words = line.lower().strip().split("\t")
                    tag = float(tag)
                    if 0.4 < tag < 0.6:
                        continue
                    else:
                        tag = 1 if tag > 0.6 else 0
                    yield ([self.w2i[x] for x in words.split(" ")], self.t2i[tag])
        elif dataset == 'kaggle':
            with open(filename, "r", encoding='utf-8') as f:
                for line in f:
                    # tag, words = line.lower().strip().split(" ||| ")
                    tag, words = line.lower().strip().split("\t")

    def data_prep(self, train_filename, test_filename):
        train = list(self.read_dataset(train_filename, dataset='stanford'))
        #w2i = defaultdict(lambda: UNK, w2i)
        dev = list(self.read_dataset(test_filename, dataset='stanford'))
        self.nwords = len(self.w2i)
        self.ntags = len(self.t2i)
        return train, dev

    # A function to calculate scores for one value
    def _calc_scores(self, words, W_emb, fwdLSTM, bwdLSTM, W_sm, b_sm, normalize_results=True):
        dy.renew_cg()
        word_embs = [dy.lookup(W_emb, x) for x in words]
        fwd_init = fwdLSTM.initial_state()
        fwd_embs = fwd_init.transduce(word_embs)
        if self.use_bilstm:
            bwd_init = bwdLSTM.initial_state()
            bwd_embs = bwd_init.transduce(reversed(word_embs))
            score_not_normalized = W_sm * dy.concatenate([fwd_embs[-1], bwd_embs[-1]]) + b_sm
        else:
            score_not_normalized = W_sm * dy.concatenate([fwd_embs[-1]]) + b_sm

        if normalize_results:
            return dy.softmax(score_not_normalized)
        else:
            return score_not_normalized

    def fit_model(self, train, test):
        # Start DyNet and define trainer
        model = dy.Model()
        trainer = dy.AdamTrainer(model)

        # Define the model
        W_emb = model.add_lookup_parameters((self.nwords, self.emb_size))   # Word embeddings
        fwdLSTM = dy.LSTMBuilder(1, self.emb_size, self.hid_size, model)    # Forward LSTM
        bwdLSTM = dy.LSTMBuilder(1, self.emb_size, self.hid_size, model)    # Backward LSTM
        if self.use_bilstm:
            W_sm = model.add_parameters((self.ntags, 2 * self.hid_size))      # Softmax weights
        else:
            W_sm = model.add_parameters((self.ntags, self.hid_size))          # Softmax weights
        b_sm = model.add_parameters(self.ntags)                               # Softmax bias
        for ITER in range(self.epochs):
            # Perform training
            random.shuffle(train)
            train_loss = 0.0
            start = time.time()
            for idx, (words, tag) in enumerate(train):
                my_loss = dy.pickneglogsoftmax(self._calc_scores(words=words, W_emb=W_emb, fwdLSTM=fwdLSTM,
                                                                 bwdLSTM=bwdLSTM, W_sm=W_sm, b_sm=b_sm,
                                                                 normalize_results=True), tag)
                train_loss += my_loss.value()
                my_loss.backward()
                trainer.update()
            print("iter %r: train loss/sent=%.4f, time=%.2fs" % (ITER, train_loss / len(train), time.time() - start))
            # Perform testing validation
            test_correct = 0.0
            for words, tag in test:
                scores = self._calc_scores(words=words, W_emb=W_emb, fwdLSTM=fwdLSTM,
                                           bwdLSTM=bwdLSTM, W_sm=W_sm, b_sm=b_sm).npvalue()
                predict = np.argmax(scores)
                if predict == tag:
                    test_correct += 1
            print("iter %r: test acc=%.4f" % (ITER, test_correct / len(test)))
        # Perform testing validation after all batches ended
        all_scores = []
        for words, tag in test:
            cur_score = self._calc_scores(words=words, W_emb=W_emb, fwdLSTM=fwdLSTM,
                                          bwdLSTM=bwdLSTM, W_sm=W_sm, b_sm=b_sm, normalize_results=True).npvalue()
            all_scores.append(cur_score)
        return all_scores

    def evaluate_model(self, dl_model_scores, sr_objects, ):
        sr_name_with_y = {sr.name: sr.trying_to_draw for sr in sr_objects}
        sr_names = [sr.name for sr in sr_objects]
        y_data = [sr.trying_to_draw for sr in sr_objects]
        results_summary = [(sent_score[1], sr_name, sr_name_with_y[sr_name]) for sent_score, sr_name, y in
                           zip(dl_model_scores, sr_names, y_data)]
        # converting it into pandas - much easier to work with
        results_df = pd.DataFrame(results_summary,
                                  columns=['prediction', 'sr_name', 'trying_to_draw'])
        results_agg = results_df.groupby(['sr_name'], as_index=False).agg(
            {'prediction': ['mean', 'median', 'max', 'min'], 'trying_to_draw': 'mean'})
        results_agg.columns = ['sr_name', 'pred_mean', 'pred_median', 'pred_max', 'pred_min', 'trying_to_draw']

        #evaluate the measrues
        scoring = {'acc': accuracy_score,
                   'precision': precision_score,
                   'recall': recall_score,
                   'auc': roc_auc_score}
        for measure, func in scoring.items():
            if measure != 'auc':
                cur_res_mean = scoring[measure](y_true=results_agg.trying_to_draw,
                                                y_pred=[1 if i > 0.5 else -1 for i in results_agg.pred_mean])
                cur_res_median = scoring[measure](y_true=results_agg.trying_to_draw,
                                                  y_pred=[1 if i > 0.5 else -1 for i in results_agg.pred_median])
                cur_res_max = scoring[measure](y_true=results_agg.trying_to_draw,
                                               y_pred=[1 if i > 0.8 else -1 for i in results_agg.pred_max])
                cur_res_min = scoring[measure](y_true=results_agg.trying_to_draw,
                                               y_pred=[1 if i > 0.2 else -1 for i in results_agg.pred_min])
            else:
                cur_res_mean = scoring[measure](y_true=results_agg.trying_to_draw,
                                                y_score=results_agg.pred_mean)
                cur_res_median = scoring[measure](y_true=results_agg.trying_to_draw,
                                                  y_score=results_agg.pred_median)
                cur_res_max = scoring[measure](y_true=results_agg.trying_to_draw,
                                               y_score=results_agg.pred_max)
                cur_res_min = scoring[measure](y_true=results_agg.trying_to_draw,
                                               y_score=results_agg.pred_min)
            self.eval_measures[measure + '_mean'] = cur_res_mean
            self.eval_measures[measure + '_median'] = cur_res_median
            self.eval_measures[measure + '_max'] = cur_res_max
            self.eval_measures[measure + '_min'] = cur_res_min

        return self.eval_measures


########################################################################################################################
    def fit_two_layers_model(self, train, test):
        # Start DyNet and define trainer
        model = dy.Model()
        trainer = dy.AdamTrainer(model)

        # Define the model
        W_emb = model.add_lookup_parameters((self.nwords, self.emb_size))      # Word embeddings
        first_lstm = dy.LSTMBuilder(1, self.emb_size, self.hid_size, model)    # Forward LSTM
        second_lstm = dy.LSTMBuilder(1, self.hid_size, self.hid_size, model)
        W_sm = model.add_parameters((self.ntags, self.hid_size))               # Softmax weights
        b_sm = model.add_parameters(self.ntags)                                # Softmax bias
        for ITER in range(self.epochs):
            # Perform training
            random.shuffle(train)
            train_loss = 0.0
            start = time.time()
            for idx, (sentences, tag) in enumerate(train):
                my_loss = dy.pickneglogsoftmax(self._calc_scores_two_layers(sentences=sentences, W_emb=W_emb,
                                                                            first_lstm=first_lstm,
                                                                            second_lstm=second_lstm, W_sm=W_sm,
                                                                            b_sm=b_sm, normalize_results=True), tag)
                train_loss += my_loss.value()
                my_loss.backward()
                trainer.update()
            print("iter %r: train loss/sent=%.4f, time=%.2fs" % (ITER, train_loss / len(train), time.time() - start))
            # Perform testing validation
            test_correct = 0.0
            for words, tag in test:
                scores = self._calc_scores_two_layers(sentences=words, W_emb=W_emb, first_lstm=first_lstm,
                                                      second_lstm=second_lstm, W_sm=W_sm, b_sm=b_sm).npvalue()
                predict = np.argmax(scores)
                if predict == tag:
                    test_correct += 1
            print("iter %r: test acc=%.4f" % (ITER, test_correct / len(test)))
        # Perform testing validation after all batches ended
        all_scores = []
        for words, tag in test:
            cur_score = self._calc_scores_two_layers(sentences=words, W_emb=W_emb, first_lstm=first_lstm,
                                                     second_lstm=second_lstm, W_sm=W_sm, b_sm=b_sm,
                                                     normalize_results=True).npvalue()
            all_scores.append(cur_score)
        return all_scores

    def _calc_scores_two_layers(self, sentences, W_emb, first_lstm, second_lstm, W_sm, b_sm, normalize_results=True):
        dy.renew_cg()
        sentences_len = len(sentences)
        word_embs = [[dy.lookup(W_emb, w) for w in words] for words in sentences]
        first_init = first_lstm.initial_state()
        first_embs=[]
        for wb in word_embs:
            first_embs.append(first_init.transduce(wb))
        last_comp_in_first_layer = [i[-1] for i in first_embs]
        second_init = second_lstm.initial_state()
        second_lstm_calc = second_init.transduce(last_comp_in_first_layer)
        score_not_normalized = W_sm * dy.concatenate([second_lstm_calc[-1]]) + b_sm
        if normalize_results:
            return dy.softmax(score_not_normalized)
        else:
            return score_not_normalized
        pass


if __name__ == "__main__":
    data_path = '/data/home/orentsur/data/reddit_place/' if sys.platform == 'linux' \
        else 'C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\PhD\\reddit canvas\\data\\'
    sr_objects = pickle.load(open(data_path + "sr_objects_102016_to_032017_sample.p", "rb"))
    # function to remove huge SRs, so parallalizem can be applied
    sr_objects = remove_huge_srs(sr_objects=sr_objects, quantile=0.05)
    # creating the y vector feature and printing status
    y_data = []
    for idx, cur_sr_obj in enumerate(sr_objects):
        y_data += [cur_sr_obj.trying_to_draw]

    print("Target feature distribution is: {}".format(collections.Counter(y_data)))
    # Modeling (learning phase)
    submission_dp_obj = RedditDataPrep(is_submission_data=True, remove_stop_words=False, most_have_regex=None)
    reddit_tokenizer = submission_dp_obj.tokenize_text
    dl_obj = NNClassifier(model_type='lstm', emb_size=100, hid_size=100, epochs=20, use_bilstm=False,
                          model_sentences=False, seed=1984)
    data_for_dynet = list(dl_obj.get_reddit_sentences(sr_objects=sr_objects, maximum_sent_per_sr=100,
                                                      tokenizer=reddit_tokenizer))
    data_names = [i[2] for i in data_for_dynet]
    data_for_dynet = [(i[0], i[1]) for i in data_for_dynet]
    #dl_model_scores = dl_obj.fit_model(train=data_for_dynet, test=data_for_dynet)
    dl_model_scores = dl_obj.fit_two_layers_model(train=data_for_dynet, test=data_for_dynet)
    cur_eval_measures = dl_obj.evaluate_model(dl_model_scores=dl_model_scores,
                                              sr_objects=sr_objects)
    #print("Fold # {} has ended, here are the results of this fold:".format(cv_idx))
    print(cur_eval_measures)
    #lstm_obj.fit_model(sr_objects=None, y_vector=None, train=train_data, dev=test_data)
