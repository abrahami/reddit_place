# Authors: Avrahami Israeli (isabrah)
# Python version: 3.7
# Last update: 21.11.2018


import dynet_config
# set random seed to have the same result each time
dynet_config.set(random_seed=1984, mem="4096")
import dynet as dy
from collections import defaultdict
import time
import random
import numpy as np
import pandas as pd
from sr_classifier.reddit_data_preprocessing import RedditDataPrep
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import collections
import datetime
from sklearn.preprocessing import Imputer, StandardScaler


class NNClassifier(object):
    """
    """

    def __init__(self, tokenizer, eval_measures, model_type='lstm', emb_size=100, hid_size=100, early_stopping=True,
                 epochs=10, use_bilstm=False, use_meta_features=True, model_sentences=False, maximum_sent_per_sr=10,
                 seed=1984):
        self.tokenizer = tokenizer
        self.eval_measures = eval_measures
        self.model_type = model_type
        self.use_bilstm = use_bilstm
        self.use_meta_features = use_meta_features
        self.epochs = epochs
        self.early_stopping = early_stopping
        # Functions to read in the corpus
        self.w2i = defaultdict(lambda: len(self.w2i))
        self.t2i = defaultdict(lambda: len(self.t2i))
        self.w2i["<unk>"]
        self.nwords = None
        self.ntags = None
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.model_sentences = model_sentences
        self.maximum_sent_per_sr = maximum_sent_per_sr
        self.seed = seed
        random.seed(self.seed)
        self.eval_results = defaultdict(list)

    def get_reddit_sentences(self, sr_objects):
        # looping over all sr objects
        for cur_sr in sr_objects:
            # pulling out the tag of the current sr
            tag = cur_sr.trying_to_draw
            if self.maximum_sent_per_sr is not None:
                cur_sr.submissions_as_list.sort(key=lambda tup: tup[0], reverse=True)
                cur_sr_sentences = [i[1] for idx, i in enumerate(cur_sr.submissions_as_list) if
                                             idx < self.maximum_sent_per_sr and type(i[1]) is str]
                #cur_sr_sentences = reversed([i[1] for idx, i in enumerate(reversed(cur_sr.submissions_as_list)) if
                #                             idx < maximum_sent_per_sr and type(i[1]) is str])
            else:
                cur_sr_sentences = [i[1] for i in cur_sr.submissions_as_list if type(i[1]) is str]

            if self.model_sentences:
                for sen in cur_sr_sentences:
                    sen_tokenized = self.tokenizer(sen)
                    if len(sen_tokenized) > 0:
                        yield ([self.w2i[x] for x in sen_tokenized], self.t2i[tag], cur_sr.name)
            else:
                cur_sr_sentences_as_int = []
                for sen in cur_sr_sentences:
                    sen_tokenized = self.tokenizer(sen)
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

    def data_prep_meta_features(self, train_data, test_data, update_objects=True):
        train_meta_features = {sr_obj.name: sr_obj.explanatory_features for sr_obj in train_data}
        test_meta_features = {sr_obj.name: sr_obj.explanatory_features for sr_obj in test_data}
        imp_obj = Imputer(strategy='mean', copy=False)
        normalize_obj = StandardScaler()
        train_as_df = pd.DataFrame.from_dict(train_meta_features, orient='index')
        test_as_df = pd.DataFrame.from_dict(test_meta_features, orient='index')
        #imputation phase
        train_as_df = pd.DataFrame(imp_obj.fit_transform(train_as_df), columns=train_as_df.columns, index=train_as_df.index)
        test_as_df = pd.DataFrame(imp_obj.transform(test_as_df), columns=test_as_df.columns, index=test_as_df.index)

        # normalization phase
        train_as_df = pd.DataFrame(normalize_obj.fit_transform(train_as_df), columns=train_as_df.columns, index=train_as_df.index)
        test_as_df = pd.DataFrame(normalize_obj.transform(test_as_df), columns=test_as_df.columns, index=test_as_df.index)

        train_meta_features_imputed = train_as_df.to_dict('index')
        test_meta_features_imputed = test_as_df.to_dict('index')
        updated_meta_features_dict_train = \
            {key: collections.defaultdict(None, value) for key, value in train_meta_features_imputed.items()}
        updated_meta_features_dict_test = \
            {key: collections.defaultdict(None, value) for key, value in test_meta_features_imputed.items()}
        if update_objects:
            # looping over all srs object and updating the meta data features
            for cur_sr in train_data:
                cur_sr.explanatory_features = updated_meta_features_dict_train[cur_sr.name]
            for cur_sr in test_data:
                cur_sr.explanatory_features = updated_meta_features_dict_test[cur_sr.name]
            return 0
        else:
            return updated_meta_features_dict_train, updated_meta_features_dict_test

    def build_embedding_matrix(self, embedding_file):
        start_time = datetime.datetime.now()
        embedding_matrix = np.random.normal(loc=0.0, scale=1.0, size=(self.nwords + 1, self.emb_size))
        embeddings_index = dict()
        found_words = 0
        #print("We are in the 'build_embedding_matrix' function")
        with open(embedding_file) as infile:
            for idx, line in enumerate(infile):
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
                # updating the embedding matrix according to the w2i dictionary
                if word in self.w2i:
                    embedding_matrix[self.w2i[word]] = coefs
                    found_words += 1
            infile.close()
        duration = (datetime.datetime.now() - start_time).seconds
        print("We have finished running the 'build_embedding_matrix' function. Took us {0:.2f} seconds. "
              "We have found {1:.1f}% of matching words "
              "compared to the embedding matrix.".format(duration, found_words * 100.0 / self.nwords))
        return embedding_matrix

    def calc_eval_measures(self, y_true, y_pred, nomalize_y=True):
        if nomalize_y:
            y_true = [1 if y > 0 else 0 for y in y_true]
            binary_y_pred = [1 if p > 0.5 else 0 for p in y_pred]
        else:
            binary_y_pred = [1 if p > 0.5 else -1 for p in y_pred]
        for name, func in self.eval_measures.items():
            self.eval_results[name].append(func(y_true, binary_y_pred))
        return self.eval_results

    def fit_simple_mlp(self, train_data, test_data):
        random.seed(self.seed)
        random.shuffle(train_data)
        self.data_prep_meta_features(train_data=train_data, test_data=test_data, update_objects=True)
        train_meta_data = [sr_obj.explanatory_features for sr_obj in train_data]
        test_meta_data = [sr_obj.explanatory_features for sr_obj in test_data]
        y_train = [sr_obj.trying_to_draw for sr_obj in train_data]
        y_test = [sr_obj.trying_to_draw for sr_obj in test_data]
        meta_data_dim = len(list(train_meta_data[0].keys()))
        # Start DyNet and define trainer
        model = dy.Model()
        trainer = dy.SimpleSGDTrainer(model)

        dy.renew_cg()
        W = model.add_parameters((self.hid_size, meta_data_dim))
        b = model.add_parameters(self.hid_size)
        V = model.add_parameters((1, self.hid_size))
        a = model.add_parameters(1)
        x = dy.vecInput(meta_data_dim)
        h = dy.tanh((W * x) + b)
        y = dy.scalarInput(0)
        y_pred = dy.logistic((V * h) + a)
        loss = dy.binary_log_loss(y_pred, y)
        mloss = [0.0, 0.0]  # we always save the current run loss and the prev one (for early stopping purposes
        for ITER in range(self.epochs):
            # checking the early stopping criterion
            if self.early_stopping and (ITER > self.epochs * 1.0 / 2) and (mloss[0]-mloss[1]) * 1.0 / mloss[0] <= 0.01:
                print("Early stopping has been applied since improvement was not greater than 1%")
                break
            # Perform training
            start = time.time()
            cur_mloss=0.0
            for idx, (cur_sr_dict, tag) in enumerate(zip(train_meta_data, y_train)):
                # create graph for computing loss
                cur_sr_values_ordered = [value for key, value in sorted(cur_sr_dict.items())]
                x.set(cur_sr_values_ordered)
                tag_normalized = 1 if tag == 1 else 0
                y.set(tag_normalized)
                # loss calc
                cur_mloss += loss.value()
                loss.backward()
                trainer.update()
            # updating the mloss for early stopping purposes
            mloss[0] = mloss[1]
            mloss[1] = cur_mloss
            print("iter %r: train loss/sr=%.4f, time=%.2fs" % (ITER, cur_mloss / len(y_train), time.time() - start))
            # Perform testing validation
            test_correct = 0.0
            y_pred = dy.logistic((V * h) + a)
            for idx, (cur_sr_dict, tag) in enumerate(zip(test_meta_data, y_test)):
                cur_sr_values_ordered = [value for key, value in sorted(cur_sr_dict.items())]
                x.set(cur_sr_values_ordered)
                y_pred_value = y_pred.value()
                if (y_pred_value >= .5 and tag == 1) or (y_pred_value <= .5 and tag == -1):
                    test_correct += 1
            print("iter %r: test acc=%.4f" % (ITER, test_correct / len(y_test)))
        # Perform testing validation after all batches ended
        test_predicitons = []
        test_correct = 0.0
        for idx, (cur_sr_dict, tag) in enumerate(zip(test_meta_data, y_test)):
            cur_sr_values_ordered = [value for key, value in sorted(cur_sr_dict.items())]
            x.set(cur_sr_values_ordered)
            y_pred_value = y_pred.value()
            test_predicitons.append(y_pred_value)
            if (y_pred_value >= .5 and tag == 1) or (y_pred_value <= .5 and tag == -1):
                test_correct += 1
        self.calc_eval_measures(y_true=y_test, y_pred=test_predicitons, nomalize_y=True)
        print("final test acc=%.4f" % (test_correct / len(y_test)))
        return self.eval_results, model, test_predicitons

    # A function to calculate scores for one value
    def _calc_scores_single_layer_lstm(self, words, W_emb, fwdLSTM, bwdLSTM, W_sm, b_sm, normalize_results=True):
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

    def fit_single_layer_lstm(self, train, test, embedding_file=None):
        # Start DyNet and define trainer
        model = dy.Model()
        trainer = dy.AdamTrainer(model)

        # Define the model
        # Word embeddings part
        if embedding_file is None:
            W_emb = model.add_lookup_parameters((self.nwords, self.emb_size))
        else:
            external_embedding = self.build_embedding_matrix(embedding_file)
            W_emb = model.add_lookup_parameters((self.nwords, self.emb_size), init=external_embedding)
        fwdLSTM = dy.LSTMBuilder(1, self.emb_size, self.hid_size, model)    # Forward LSTM
        bwdLSTM = dy.LSTMBuilder(1, self.emb_size, self.hid_size, model)    # Backward LSTM
        if self.use_bilstm:
            W_sm = model.add_parameters((self.ntags, 2 * self.hid_size))      # Softmax weights
        else:
            W_sm = model.add_parameters((self.ntags, self.hid_size))          # Softmax weights
        b_sm = model.add_parameters(self.ntags)                               # Softmax bias
        for ITER in range(self.epochs):
            # Perform training
            random.seed(self.seed)
            random.shuffle(train)
            train_loss = 0.0
            start = time.time()
            for idx, (words, tag) in enumerate(train):
                my_loss = dy.pickneglogsoftmax(self._calc_scores_single_layer_lstm(words=words, W_emb=W_emb, fwdLSTM=fwdLSTM,
                                                                                   bwdLSTM=bwdLSTM, W_sm=W_sm, b_sm=b_sm,
                                                                                   normalize_results=True), tag)
                train_loss += my_loss.value()
                my_loss.backward()
                trainer.update()
            print("iter %r: train loss/sent=%.4f, time=%.2fs" % (ITER, train_loss / len(train), time.time() - start))
            # Perform testing validation
            test_correct = 0.0
            for words, tag in test:
                scores = self._calc_scores_single_layer_lstm(words=words, W_emb=W_emb, fwdLSTM=fwdLSTM,
                                                             bwdLSTM=bwdLSTM, W_sm=W_sm, b_sm=b_sm).npvalue()
                predict = np.argmax(scores)
                if predict == tag:
                    test_correct += 1
            print("iter %r: test acc=%.4f" % (ITER, test_correct / len(test)))
        # Perform testing validation after all batches ended
        all_scores = []
        for words, tag in test:
            cur_score = self._calc_scores_single_layer_lstm(words=words, W_emb=W_emb, fwdLSTM=fwdLSTM,
                                                            bwdLSTM=bwdLSTM, W_sm=W_sm, b_sm=b_sm, normalize_results=True).npvalue()
            all_scores.append(cur_score)
        return all_scores

    def evaluate_model(self, dl_model_scores, sr_objects):
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
    def _calc_scores_two_layers(self, sentences, W_emb, first_lstm, second_lstm, W_sm, b_sm,
                                meta_data=None, normalize_results=True):
        dy.renew_cg()
        sentences_len = len(sentences)
        word_embs = [[dy.lookup(W_emb, w) for w in words] for words in sentences]
        first_init = first_lstm.initial_state()
        first_embs=[]
        for wb in word_embs:
            first_embs.append(first_init.transduce(wb))
        last_comp_in_first_layer = [i[-1] for i in first_embs]
        if second_lstm is not None:
            second_init = second_lstm.initial_state()
            second_lstm_calc = second_init.transduce(last_comp_in_first_layer)
            score_not_normalized = W_sm * dy.concatenate([second_lstm_calc[-1]]) + b_sm
            if normalize_results:
                return dy.softmax(score_not_normalized)
            else:
                return score_not_normalized
        # case we want to claculate the average of all the tensors instead of building another layer
        else:
            first_layer_avg = dy.average(last_comp_in_first_layer)
            if meta_data is None:
                score_not_normalized = W_sm * first_layer_avg + b_sm
            else:
                meta_data_ordered = [value for key, value in sorted(meta_data.items())]
                meta_data_vector = dy.inputVector(meta_data_ordered)
                first_layer_avg_and_meta_data = dy.concatenate([first_layer_avg, meta_data_vector])
                score_not_normalized = W_sm * first_layer_avg_and_meta_data + b_sm
            if normalize_results:
                return dy.softmax(score_not_normalized)
            else:
                return score_not_normalized

    def fit_two_layers_lstm(self, train_data, test_data, embedding_file=None, second_layer_as_lstm=True):

        # case we wish to use meta features along modeling, we need to prepare the SRs objects for this
        if self.use_meta_features:
            train_meta_data, test_meta_data = \
                self.data_prep_meta_features(train_data=train_data, test_data=test_data, update_objects=False)
            meta_data_dim = len(train_meta_data[list(train_meta_data.keys())[0]])
        else:
            train_meta_data = None
            test_meta_data = None
            meta_data_dim = 0
        # next we are creating the input for the algorithm. train_data_for_dynet will contain list of lists. Each inner
        # list contains the words index relevant to the specific sentence
        train_data_for_dynet = list(self.get_reddit_sentences(sr_objects=train_data))
        train_data_names = [i[2] for i in train_data_for_dynet]  # list of train sr names
        train_data_for_dynet = [(i[0], i[1]) for i in train_data_for_dynet]

        # test_data_for_dynet will contain list of lists. Each inner list contains the words index relevant to the
        # specific sentence
        test_data_for_dynet = list(self.get_reddit_sentences(sr_objects=test_data))
        test_data_names = [i[2] for i in test_data_for_dynet]   # list of test sr names
        ### Need to check here that the ordered is saved!!!!
        test_data_for_dynet = [(i[0], i[1]) for i in test_data_for_dynet]

        # Start DyNet and define trainer
        model = dy.Model()
        trainer = dy.AdamTrainer(model)
        # Define the model
        # Word embeddings part
        if embedding_file is None:
            W_emb = model.add_lookup_parameters((self.nwords, self.emb_size))
        else:
            external_embedding = self.build_embedding_matrix(embedding_file)
            W_emb = model.add_lookup_parameters((self.nwords, self.emb_size), init=external_embedding)
        first_lstm = dy.LSTMBuilder(1, self.emb_size, self.hid_size, model)    # Forward LSTM
        if second_layer_as_lstm:
            second_lstm = dy.LSTMBuilder(1, self.hid_size, self.hid_size, model)
        else:
            second_lstm = None
        # Last layer with softmax weights+bias, case we are not using meta data, meta_data_dim will be zero
        # and hence not relevant
        W_sm = model.add_parameters((self.ntags, self.hid_size + meta_data_dim))
        b_sm = model.add_parameters(self.ntags)
        for ITER in range(self.epochs):
            # Perform training
            #random.seed(self.seed)
            #random.shuffle(train_data_for_dynet)
            train_loss = 0.0
            start = time.time()
            for idx, (sentences, tag) in enumerate(train_data_for_dynet):
                cur_meta_data = train_meta_data[train_data_names[idx]] if self.use_meta_features else None
                my_loss =\
                    dy.pickneglogsoftmax(self._calc_scores_two_layers(sentences=sentences, W_emb=W_emb,
                                                                      first_lstm=first_lstm, second_lstm=second_lstm,
                                                                      W_sm=W_sm, b_sm=b_sm,
                                                                      meta_data=cur_meta_data,
                                                                      normalize_results=True), tag)
                train_loss += my_loss.value()
                my_loss.backward()
                trainer.update()
            print("iter %r: train loss/sr=%.4f, time=%.2fs" % (ITER, train_loss / len(train_data_for_dynet),
                                                               time.time() - start))
            # Perform testing validation
            test_correct = 0.0
            for idx, (words, tag) in enumerate(test_data_for_dynet):
                cur_meta_data = test_meta_data[test_data_names[idx]] if self.use_meta_features else None
                scores = self._calc_scores_two_layers(sentences=words, W_emb=W_emb, first_lstm=first_lstm,
                                                      second_lstm=second_lstm, W_sm=W_sm, b_sm=b_sm,
                                                      meta_data=cur_meta_data,
                                                      normalize_results=True).npvalue()
                predict = np.argmax(scores)
                if predict == tag:
                    test_correct += 1
            print("iter %r: test acc=%.4f" % (ITER, test_correct / len(test_data_for_dynet)))
        # Perform testing validation after all batches ended
        test_correct = 0.0
        test_predicitons = []
        for idx, (words, tag) in enumerate(test_data_for_dynet):
            cur_meta_data = test_meta_data[test_data_names[idx]] if self.use_meta_features else None
            cur_score = self._calc_scores_two_layers(sentences=words, W_emb=W_emb, first_lstm=first_lstm,
                                                     second_lstm=second_lstm, W_sm=W_sm, b_sm=b_sm,
                                                     meta_data=cur_meta_data,
                                                     normalize_results=True).npvalue()
            # adding the prediction of the sr to draw (to be label 1) and calculating the acc on the fly
            test_predicitons.append(cur_score[1])
            predict = np.argmax(cur_score)
            if predict == tag:
                test_correct += 1

        y_test = [a[1] for a in test_data_for_dynet]
        self.calc_eval_measures(y_true=y_test, y_pred=test_predicitons, nomalize_y=True)
        print("final test acc=%.4f" % (test_correct / len(y_test)))

        return self.eval_results, model, test_predicitons

'''
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
    #dl_model_scores = dl_obj.fit_single_layer_lstm(train=data_for_dynet, test=data_for_dynet)
    dl_model_scores = dl_obj.fit_two_layers_model(train=data_for_dynet, test=data_for_dynet)
    cur_eval_measures = dl_obj.evaluate_model(dl_model_scores=dl_model_scores,
                                              sr_objects=sr_objects)
    #print("Fold # {} has ended, here are the results of this fold:".format(cv_idx))
    print(cur_eval_measures)
    #lstm_obj.fit_single_layer_lstm(sr_objects=None, y_vector=None, train=train_data, dev=test_data)
'''