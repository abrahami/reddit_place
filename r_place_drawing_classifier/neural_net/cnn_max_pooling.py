# Authors: Avrahami Israeli (isabrah)
# Python version: 3.7
# Last update: 26.03.2019

# most code is taken from: https://github.com/neubig/nn4nlp-code/tree/c18372d8466bb12b8c603d4c85f881b0ceacb99c/05-cnn
from .nn_classifier import NNClassifier
#import dynet_config
# set random seed to have the same result each time
#dynet_config.set(random_seed=1984, mem='5000', autobatch=0)
#dynet_config.set_gpu()
import dynet as dy
import numpy as np
import time
import os
import pickle
import copy
from collections import defaultdict


class CnnMaxPooling(NNClassifier):
    """
    Parallel long-short term memory based model, using dynet package
    this class inherits from NNClassifier, and is a special case of a NN model. Model implemented here is a sequence of
    LSTMs models used before an MLP layer is applied. Each sentence in a sr goes through this sequence of LSTMs and only
    the last LSTM model output is taken into account for future usage. In order to aggregate all sentences output,
    we take the average of each node in the hidden layer across all LSTMs.

    :param use_bilstm: boolean
        whether or not to apply bi directional LSTM model to each sentence (reading the sentence from start to end
        as well as from end to start along modeling)    """

    def __init__(self, model, tokenizer, eval_measures, emb_size=100, early_stopping=True,
                 epochs=10, use_meta_features=True, batch_size=10, seed=1984, filter_size=8, win_size=3):
        super(CnnMaxPooling, self).__init__(model=model, tokenizer=tokenizer, eval_measures=eval_measures, emb_size=emb_size,
                                            early_stopping=early_stopping, epochs=epochs,
                                            use_meta_features=use_meta_features, batch_size=batch_size, seed=seed)
        self.filter_size = filter_size
        self.win_size = win_size
        # all these will be set along the execution
        self.W_emb = None
        self.W_cnn = None
        self.b_cnn = None
        self.W_mlp = None
        self.b_mlp = None
        self.V_mlp = None
        self.a_mlp = None
        #self.pc = self.model.add_subcollection()
        self.spec = (tokenizer, eval_measures, emb_size, early_stopping, epochs, use_meta_features,
                     batch_size, seed, filter_size, win_size)

    def calc_scores(self, sentences, meta_data=None, get_probability=True):
        """
        calculating the score for parallel LSTM network (in a specific state along learning phase)
        :param sentences: list
            list of lists of sentences (represented already as numbers and not letters)
        :param W_emb: model parameter (dynet obj). size:
            matrix holding weights of the mlp phase
        :param W_cnn: model parameter (dynet obj). size:
            vector holding weights of intercept for each hidden state
        :param b_cnn: model parameter (dynet obj). size:
            matrix holding weights of the logisitc regression phase. 2 is there due to the fact we are in a binary
            classification
        :param W_sm: model parameter (dynet obj). size:
            intercept value for the logistic regression phase
        :param b_sm: dict or None

        :return: dynet parameter. size: (2,)
            prediction of the instance to be a drawing one according to the model (vector of 2, first place is the
            probability to be a drawing team)
        """
        #dy.renew_cg()
        # padding with zeros in case sentences are too short
        for words in sentences:
            if len(words) < self.win_size:
                words += [0] * (self.win_size - len(words))

        # looping over each sentence, calculating the CNN max pooling and taking the average at the end
        pool_out_agg = []
        #for cur_sentences in sentences:
        for words in sentences:
            #cnn_in = dy.concatenate([dy.lookup(W_emb, x) for words in cur_sentences for x in words], d=1)
            cnn_in = dy.concatenate([dy.lookup(self.W_emb, x) for x in words], d=1)
            cnn_out = dy.conv2d_bias(cnn_in, self.W_cnn, self.b_cnn, stride=(1, 1), is_valid=False)
            pool_out = dy.max_dim(cnn_out, d=1)
            pool_out = dy.reshape(pool_out, (self.filter_size,))
            pool_out = dy.rectify(pool_out) # Relu function: max(x_i, 0)
            pool_out_agg.append(pool_out)
        pool_out_avg = dy.average(pool_out_agg)

        if meta_data is None:
            h = dy.tanh((self.W_mlp * pool_out_avg) + self.b_mlp)
            prediction = dy.logistic((self.V_mlp * h) + self.a_mlp)
            if get_probability:
                return prediction
            else:
                return pool_out_avg
        else:
            meta_data_ordered = [value for key, value in sorted(meta_data.items())]
            meta_data_vector = dy.inputVector(meta_data_ordered)
            first_layer_avg_and_meta_data = dy.concatenate([pool_out_avg, meta_data_vector])
            h = dy.tanh((self.W_mlp * first_layer_avg_and_meta_data) + self.b_mlp)
            prediction = dy.logistic((self.V_mlp * h) + self.a_mlp)
            if get_probability:
                return prediction
            else:
                return first_layer_avg_and_meta_data

    def fit_predict(self, train_data, test_data, embedding_file=None):
        """
        fits a parallel LSTM model
        :param train_data: list
            list of sr objects to be used as train set
        :param test_data: list
            list of sr objects to be used as train set
        :param embedding_file: str
            the path to the exact embedding file to be used. This should be a txt file, each row represents
            a word and it's embedding (separated by whitespace). Example can be taken from 'glove' pre-trained models
            If None, we build an embedding from random normal distribution
        :return: tuple
            tuple with 3 variables:
            self.eval_results, model, test_predicitons
            1. eval_results: dictionary with evaluation measures over the test set
            2. model: the MLP trained model which was used
            3. test_predicitons: list with predictions to each sr in the test dataset
        """
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
        # Need to check here that the order is saved!!!!
        test_data_for_dynet = [(i[0], i[1]) for i in test_data_for_dynet]

        # Start DyNet and define trainer
        #self.model = dy.Model()
        trainer = dy.AdamTrainer(self.model)
        # Define the model
        # Word embeddings part
        if embedding_file is None:
            self.W_emb = self.model.add_lookup_parameters((self.nwords, 1, 1, self.emb_size))
        else:
            external_embedding = self.build_embedding_matrix(embedding_file, add_extra_row=False)
            external_embedding = np.expand_dims(external_embedding, axis=0)
            external_embedding = np.expand_dims(external_embedding, axis=0)
            external_embedding_reshaped = external_embedding.reshape((self.nwords, 1, 1, self.emb_size))
            self.W_emb = self.model.add_lookup_parameters((self.nwords, 1, 1, self.emb_size), init=external_embedding_reshaped)
        self.W_cnn = self.model.add_parameters((1, self.win_size, self.emb_size, self.filter_size))  # cnn weights
        self.b_cnn = self.model.add_parameters((self.filter_size))  # cnn bias

        # Last layer with network is an MLP one, case we are not using meta data, meta_data_dim will be zero
        # and hence not relevant
        self.W_mlp = self.model.add_parameters((self.filter_size, self.filter_size + meta_data_dim))
        self.b_mlp = self.model.add_parameters(self.filter_size)
        self.V_mlp = self.model.add_parameters((self.ntags, self.filter_size))
        self.a_mlp = self.model.add_parameters(1)

        mloss = [0.0, 0.0]  # we always save the current run loss and the prev one (for early stopping purposes
        # looping over each epoch
        for ITER in range(self.epochs):
            # checking the early stopping criterion
            if self.early_stopping and (ITER >= (self.epochs * 1.0 / 2)) \
                    and ((mloss[0]-mloss[1]) * 1.0 / mloss[0]) <= 0.01:
                print("Early stopping has been applied since improvement was not greater than 1%")
                break
            # Perform training
            #random.seed(self.seed)
            #random.shuffle(train_data_for_dynet)
            start = time.time()
            cur_mloss = 0.0
            batches_starting_point = list(range(0, len(train_data_for_dynet), self.batch_size))
            # looping over each batch
            for cur_sp in batches_starting_point:
                dy.renew_cg()
                losses = []
                # looping over each SR (contains multiple sentences)
                for idx, (sentences, tag) in enumerate(train_data_for_dynet[cur_sp: cur_sp+self.batch_size]):
                    cur_meta_data = train_meta_data[train_data_names[cur_sp + idx]] if self.use_meta_features else None
                    scores = self.calc_scores(sentences, meta_data=cur_meta_data)
                    my_loss = dy.pickneglogsoftmax(scores, tag)
                    losses.append(my_loss)
                    cur_mloss += my_loss.value()

                batch_loss = dy.esum(losses) / (idx+1)
                batch_loss.backward()
                trainer.update()

            # updating the mloss for early stopping purposes
            mloss[0] = mloss[1]
            mloss[1] = cur_mloss
            print("iter %r: train loss/sr=%.4f, time=%.2fs" % (ITER, cur_mloss / len(train_data_for_dynet),
                                                               time.time() - start))
            # Perform testing validation (at the end of current epoch)
            test_correct = 0.0
            for idx, (sentences, tag) in enumerate(test_data_for_dynet):
                cur_meta_data = test_meta_data[test_data_names[idx]] if self.use_meta_features else None
                scores = self.calc_scores(sentences, meta_data=cur_meta_data).npvalue()
                predict = np.argmax(scores)
                if predict == tag:
                    test_correct += 1
            print("iter %r: test acc=%.4f" % (ITER, test_correct / len(test_data_for_dynet)))
        # Perform testing validation after all ephocs ended
        test_correct = 0.0
        test_predictions = []
        for idx, (words, tag) in enumerate(test_data_for_dynet):
            cur_meta_data = test_meta_data[test_data_names[idx]] if self.use_meta_features else None
            cur_score = self.calc_scores(words, meta_data=cur_meta_data).npvalue()
            # adding the prediction of the sr to draw (to be label 1) and calculating the acc on the fly
            test_predictions.append(cur_score[1])
            predict = np.argmax(cur_score)
            if predict == tag:
                test_correct += 1

        y_test = [a[1] for a in test_data_for_dynet]
        self.calc_eval_measures(y_true=y_test, y_pred=test_predictions, nomalize_y=True)
        print("final test acc=%.4f" % (test_correct / len(y_test)))

        return self.eval_results, self.model, test_predictions

    def predict(self, train_data, test_data, get_probability=True):
        if self.use_meta_features:
            _, test_meta_data = self.data_prep_meta_features(train_data=train_data, test_data=test_data,
                                                             update_objects=False)
        else:
            test_meta_data = None
        test_data_for_dynet = list(self.get_reddit_sentences(sr_objects=test_data))
        test_data_names = [i[2] for i in test_data_for_dynet]   # list of test sr names
        test_data_for_dynet = [(i[0], i[1]) for i in test_data_for_dynet]
        test_predictions = []
        for idx, (words, tag) in enumerate(test_data_for_dynet):
            cur_meta_data = test_meta_data[test_data_names[idx]] if self.use_meta_features else None
            if get_probability:
                cur_score = self.calc_scores(words, meta_data=cur_meta_data).npvalue()
                # adding the prediction of the sr to draw (to be label 1) and calculating the acc on the fly
                test_predictions.append(cur_score[1])
            else:
                cur_score = self.calc_scores(words, meta_data=cur_meta_data, get_probability=False).npvalue()
                # adding the prediction of the sr to draw (to be label 1) and calculating the acc on the fly
                test_predictions.append(cur_score[1])
        return test_predictions

    def save_model(self, path, model_version, fold=None):
        full_saving_path = os.path.join(path, "model_" + model_version)
        if not os.path.exists(full_saving_path):
            os.makedirs(full_saving_path)
        # list of all the vars we need to remove since these are dynet features (they will be saved soon separately
        nn_vars = ['W_emb', 'W_cnn', 'b_cnn', 'W_mlp', 'b_mlp', 'V_mlp', 'a_mlp']
        # saving these features (it will be saved in the desired directory under the model name)
        dy.save(os.path.join(full_saving_path, 'model_' + model_version + '_fold' + str(fold)), [getattr(self, i) for i in nn_vars])
        obj_to_save = copy.copy(self)
        # now setting these to None, since we cannot save them as is in a pickle type
        for n in nn_vars:
            setattr(obj_to_save, n, None)
        obj_to_save.model = None
        # converting default dict into dict, since pickle can only save dict objects and not defaultdict ones
        obj_to_save.w2i = dict(obj_to_save.w2i)
        obj_to_save.t2i = dict(obj_to_save.t2i)
        pickle.dump(obj_to_save,
                    open(os.path.join(full_saving_path, 'model_' + model_version + '_fold' + str(fold) + ".p"), "wb"))

    @staticmethod
    def load_model(path, model_version):
        full_saving_path = os.path.join(path, model_version)
        new_model_obj = pickle.load(open(full_saving_path + ".p", "rb"))
        model_to_load = dy.ParameterCollection()
        W_emb, W_cnn, b_cnn, W_mlp, b_mlp, V_mlp, a_mlp = dy.load(full_saving_path, model_to_load)
        new_model_obj.W_emb = W_emb
        new_model_obj.W_cnn = W_cnn
        new_model_obj.b_cnn = b_cnn
        new_model_obj.W_mlp = W_mlp
        new_model_obj.b_mlp = b_mlp
        new_model_obj.V_mlp = V_mlp
        new_model_obj.a_mlp = a_mlp
        # converting default dict into dict, since pickle can only save dict objects and not defaultdict ones
        new_model_obj.w2i = defaultdict(lambda: len(new_model_obj.w2i), new_model_obj.w2i)
        new_model_obj.t2i = defaultdict(lambda: len(new_model_obj.t2i), new_model_obj.t2i)
        new_model_obj.model = model_to_load
        return new_model_obj


    '''
    @staticmethod
    def from_spec(spec, model):
        tokenizer, eval_measures, emb_size, early_stopping, epochs, use_meta_features, batch_size, seed, filter_size, win_size = spec
        return CnnMaxPooling(model, tokenizer, eval_measures, emb_size, early_stopping, epochs, use_meta_features,
                             batch_size, seed, filter_size, win_size)

    # support saving:
    def param_collection(self):
        return self.pc
    '''
