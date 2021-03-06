# Authors: Avrahami Israeli (isabrah)
# Python version: 3.6
# Last update: 26.3.2019

import sys
if sys.platform == 'linux':
    sys.path.append('/data/home/isabrah/reddit_canvas/reddit_project_with_yalla_cluster/reddit-tools/sr_classifier')
from sr_classifier.meta_feautres_extractor import MetaFeaturesExtractor
from sr_classifier.clean_text_transformer import CleanTextTransformer
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import Imputer
import os
import pickle

STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
SEED = 1984


def print_n_most_informative(vectorizer, clf, N=10):
    """
    Prints out the N most important features in a model. Current accepted models are: logistic regression,
    gradient boosting trees and random forest
    :param vectorizer: list
        list of vactorizers used to build the model. If only one was used, it should be provided as a list of a single
        instance
    :param clf: object
        the model object used for modeling (this object should be after training part (e.g. post 'fit' operation)
    :param N: int, default=10
        the number of top features tp print to screen
    :return:
        returns nothing, only prints to screen
    """
    # pulling out the feature names, using all vecorizers given as input
    feature_names = [feature for vect in vectorizer for feature in vect.get_feature_names()]
    if type(clf) == LogisticRegression:
        coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
        top_class1 = coefs_with_fns[:N]
        top_class2 = coefs_with_fns[:-(N + 1):-1]
        print("Class 1 best n-grams (this is the -1 class - NOT drawing SRs): ")
        for feat in top_class1:
            print(feat)
        print("Class 2 best n-grams (this is the 1 class - DRAWING SRs): ")
        for feat in top_class2:
            print(feat)
    elif type(clf) == GradientBoostingClassifier or type(clf) == RandomForestClassifier or type(clf) == XGBClassifier:
        importance_with_fns = sorted(zip(clf.feature_importances_, feature_names), reverse=True)
        top_model_importance = importance_with_fns[0:N]
        print("The {} most important features accoridng to the ensamble model are: ".format(N))
        for feat in top_model_importance:
            print(feat)
    else:
        print("Current implementation of the 'print_n_most_informative' supports only"
              "LogisticRegression / GradientBoostingClassifier / RandomForestClassifier objects")


def examine_word(examined_word, vectorizer, train_corpus, N=5, verbose=True):
    """
    helps to find sentences where a specific word appeared in. This is useful in cases when we see some output of a
    vectorizer, and don't understand where a word came from after the tokanization process
    :param examined_word: str
        the word to look for - the one we should find sentences holding this word
    :param vectorizer: object
        the vectorizer object used for training. It must be trained, since we are pulling out from this vectorizer the
        dictionary it has been created
    :param train_corpus: list
        list of texts, which the tokenizer can be fitted on. It is recommended to provide here the same corpus as the
        tokenizer originally was trained on, in order to easier do the post-engineering about weired tokenizing word
    :param N: int
        number of texts containing the weired cases the function should print out
    :param verbose: bool
        whether to print out ot not the interesting cases
    :return: list
        list of cases (sentences) with the weired cases of tantalization
    """
    words = vectorizer.get_feature_names()
    # rel_idx should be a list with length 1 (or zero)
    rel_idx = [idx for idx, w in enumerate(words) if w == examined_word]
    if len(rel_idx) == 0:
        raise Exception("Seems like the word you provided doesn't exist in the vectorizer object")
    elif len(rel_idx) > 1:
        raise Exception("We found two cases of the specific word you provided in the vectorizer object, please check")
    else:
        rel_idx = rel_idx[0]
    # fitting the vectorizer again on the corpus, in order to see the tantalization result of the tokenizer
    vec_trained = vectorizer.fit_transform(train_corpus)
    posts_idx_with_rel_word = list(vec_trained[:, rel_idx].tocoo().row)
    if verbose:
        print([train_corpus[i] + "\n" for i in posts_idx_with_rel_word[0:N]])
    return [train_corpus[i] for i in posts_idx_with_rel_word[0:N]]


def fit_model(sr_objects, y_vector, tokenizer, use_two_vectorizers=True, clf_model=LogisticRegression, folds_amount=5,
              clf_parmas=None, stop_words=STOPLIST, ngram_size=2,
              vectorizers_general_params={'max_df': 0.8, 'min_df': 5},
              meta_features_only=False, return_predictions=True,
              saving_models_options={'path': os.getcwd(), 'model_version': '0.0.0'}):
    """
    Training a classification model, in order to distinguish between two types of sub-reddit (SR) groups - those that
    are trying to draw something in r/place and those that don't.
    The fitting process is done using pipeline object (sklearn)
    :param sr_objects: list
        list of SubReddit objects. These SRs should be used for training and are labeled as trying to draw or not
    :param y_vector: list
        list of [-1, 1] numbers. -1 means not trying to draw, 1 means trying to draw. Length of this list should be
        the same as 'sr_objects' one
    :param tokenizer: object
        tokenizer object, to be used by any vectorizer
    :param use_two_vectorizers: bool, default = True
        whether or not to use two vectorizer for modeling. If True - first vectorizer will be used of the 1-gram case
        (single words) and the other one for the n-grmas (n>1) cases. This is mainly useful when we want to remove
        stop-words from 1-gram words but not from 2-gram (or 3-gram) cases
    :param clf_model: object, default = LogisticRegression
        the classification object to be used for training. Assumed to be an sklearn classification object
    :param clf_parmas: dict or None, default = None
        hyper parameters to be used in the classification model. Will be passes with the ** operation into the clf
    :param stop_words: set, default = STOPLIST (defined on top of file)
        the set of stop-words which are passed to the Vectorizer
    :param ngram_size: int
        ngrams sequence to be used in the vectorizer.
    :param vectorizers_general_params: dict or None, default = {'max_df': 0.8, 'min_df': 5}
        hyper parameters to be used in the vectorizer object. Will be passes with the ** operation into the object
    :param meta_features_only: bool
        indicator whether or not to build a model which is built upon meta features alone. No lingustic features
        at all will be used in such case.
    :param return_predictions: bool , default=True
        an indicator whether to return predicted value of each instance
    :param saving_models_options: dict, default={'path': os.getcwd(), 'model_version': '0.0.0'}
        dictionary holding the needed information about saving the model parameters
        It must contain 2 keys:
        'path': the path where the model should be saved (string)
        'model_version': version name of the model (string)
    :return: tuple (with 3 elements)
        results - results of the model as returned by the 'cross_validate' object
        pipeline - the pipline object which was used for training (after the fit process)
        predictions - the prediciton to each sr, based on the cv we ran
    """
    start_time = datetime.datetime.now()
    # in case the clf_parmas dictionary holds the object of the classifier, we will remove it
    if 'clf' in clf_parmas.keys():
        clf_parmas.pop('clf')
    ctt = CleanTextTransformer(marking_method={'urls': 'replace', 'coordinates': 'replace'})
    # case we would like to split the vectorizer into 1-gram and n-gram (n>1) objects
    if use_two_vectorizers:
        single_word_vectorizer = TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1, 1), stop_words=stop_words,
                                                 **vectorizers_general_params)
        ngram_vectorizer = TfidfVectorizer(tokenizer=tokenizer, ngram_range=(2, ngram_size),
                                           **vectorizers_general_params)
        dict_vectorizer = DictVectorizer()
        clf = clf_model(**clf_parmas)

        # pipeline creation, with feature union
        pipeline = Pipeline([
            # Use FeatureUnion to combine the features from pure text and meta data
            ('union', FeatureUnion(
                transformer_list=[
                    # Pipeline for standard bag-of-words model the text
                    ('word_features', Pipeline([
                        ('cleanText', ctt),
                        ('vectorizer', single_word_vectorizer),
                    ])),

                    ('ngram_features', Pipeline([
                        ('cleanText', ctt),
                        ('vectorizer', ngram_vectorizer),
                    ])),

                    # Pipeline for handling numeric features (which are stored in each SR object
                    ('numeric_meta_features', Pipeline([
                        ('feature_extractor', MetaFeaturesExtractor()),
                        ('vect', dict_vectorizer),  # list of dicts -> feature matrix
                        ('imputer', Imputer(strategy="mean", axis=0))
                    ])),

                ],

                # weight components in FeatureUnion
                transformer_weights={
                    'text_features': 0.5,
                    'numeric_meta_features': 0.5,
                },
            )),
            # Use the defined classifier on the combined features
            ('clf', clf),
        ])
    elif meta_features_only:
        dict_vectorizer = DictVectorizer()
        if clf_parmas is None:
            clf = clf_model()
        else:
            clf = clf_model(**clf_parmas)

        # pipeline creation, with feature union
        # special pipeline in case we wish to run the model only with the meta-features
        pipeline = Pipeline([
            # Use FeatureUnion to combine the features from pure text and meta data
            ('union', FeatureUnion(
                transformer_list=[
                    # Pipeline for handling numeric features (which are stored in each SR object
                    ('numeric_meta_features', Pipeline([
                        ('feature_extractor', MetaFeaturesExtractor()),
                        ('vect', dict_vectorizer),  # list of dicts -> feature matrix
                        ('imputer', Imputer(strategy="mean", axis=0))
                    ])),
                ],

                # weight components in FeatureUnion
                transformer_weights={
                    'numeric_meta_features': 1.0
                },
            )),
            # Use the defined classifier on the combined features
            ('clf', clf),
        ])
    else:
        vectorizer = TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1, ngram_size), stop_words=stop_words,
                                     **vectorizers_general_params)
        dict_vectorizer = DictVectorizer()
        if clf_parmas is None:
            clf = clf_model()
        else:
            clf = clf_model(**clf_parmas)

        # pipeline creation, with feature union
        pipeline = Pipeline([
            # Use FeatureUnion to combine the features from pure text and meta data
            ('union', FeatureUnion(
                transformer_list=[
                    # Pipeline for standard bag-of-words model the text
                    ('ngram_features', Pipeline([
                        ('cleanText', ctt),
                        ('vectorizer', vectorizer),
                    ])),

                    # Pipeline for handling numeric features (which are stored in each SR object
                    ('numeric_meta_features', Pipeline([
                        ('feature_extractor', MetaFeaturesExtractor()),
                        ('vect', dict_vectorizer),  # list of dicts -> feature matrix
                        ('imputer', Imputer(strategy="mean", axis=0))
                    ])),
                ],

                # weight components in FeatureUnion
                transformer_weights={
                    'text_features': 1.0,
                    'numeric_meta_features': 1.0,
                },
            )),
            # Use the defined classifier on the combined features
            ('clf', clf),
        ])
    # k-fold CV using stratified strategy
    cv_obj = StratifiedKFold(n_splits=folds_amount, random_state=SEED)

    # train and printing CV results
    scoring = {'acc': 'accuracy',
               'precision': 'precision',
               'recall': 'recall'}
    cv_results = cross_validate(estimator=pipeline, X=sr_objects, y=y_vector, #n_jobs=cv_obj.n_splits,
                                scoring=scoring, cv=cv_obj, return_train_score=False, verbose=100, return_estimator=True)
    # calling the cross_val_predict only in case it is needed - it is an extra call to the fit process!!!
    cross_val_predictions = None
    if return_predictions:
        cross_val_predictions = cross_val_predict(estimator=pipeline, X=sr_objects, y=y_vector,
                                                  cv=cv_obj, method='predict_proba')#, n_jobs=cv_obj.n_splits)
    # saving the models as pickle
    if saving_models_options is not None:
        cur_folder_name = os.path.join(saving_models_options['path'],
                                       "model_" + saving_models_options['model_version'])
        for fold_number, cur_model in enumerate(cv_results['estimator']):
            cur_file_name = saving_models_options['model_version'] + "_fold" + str(fold_number) + ".p"
            # create directory for the model if it doesn't exist
            if not os.path.exists(cur_folder_name):
                os.makedirs(cur_folder_name)
            pickle.dump(obj=cur_model, file=open(os.path.join(cur_folder_name, cur_file_name), "wb"))
        print("{} models have been saved to the directory {}".format(len(cv_results['estimator']), cur_folder_name))

    # pulling out relevant results and giving it proper names
    results = {'accuracy': list(cv_results['test_acc']), 'precision': list(cv_results['test_precision']),
               'recall': list(cv_results['test_recall'])}

    duration = (datetime.datetime.now() - start_time).seconds
    print("End of fit model function. This function run took us : {} seconds".format(duration))
    return results, pipeline, cross_val_predictions
    

def predict_model(pipeline, sr_objects, predict_proba=True):
    """
    runs a prediction process over SR objects, using on a pipeline modeling process. It assumes the pipeline is already
    fitted
    :param pipeline: object
         sklearn pipeline obejct, traind using the 'fit_predict' function
    :param sr_objects: list
        list of SubReddit objects. These SRs should be used for training and are labeled as trying to draw or not
    :param predict_proba: bool, default = True
        whether a probability vector should be returned (in the [0,1] range) or a prediction vector (with 0/1 values)
        only should be returned
    :return: list
        list of tuples with the name of the SR next to it's probability/prediction
    """
    if predict_proba:
        prediction = pipeline.predict_proba(sr_objects)
    else:
        prediction = pipeline.predict(sr_objects)
    sr_names = [sr.name for sr in sr_objects]
    return list(zip(sr_names, prediction[:, 1]))

