# Authors: Avrahami Israeli (isabrah)
# Python version: 3.7
# Last update: 31.7.2019

import copy
import ast
import pandas as pd
import os
import datetime

pd.set_option('display.expand_frame_repr', False)
#data_path = 'C:\\Users\\avrahami\\Documents\\Private\\Uni\\BGU\\PhD\\courses\\advanced NLP\\project\\data'
data_path = '/data/work/data/reddit_place/embedding/'
nohup_file_path = os.path.join(data_path, 'model_2.03', 'nohup_model_2.03_N_1565_intersection_four')
pairs_sim_file_path = os.path.join(data_path, 'pairs_similarity_general_2.02_V3.csv')
drawing_and_not_drawing_srs = os.path.join(data_path, 'drawing_and_not_drawing_srs.xlsx')


def extract_info(text_line):
    """
    help function to pull out information from a log file along the 3com algorithm run (Shani code)
    This function is used by the 2nd main function below
    :param text_line: str
        single line of information taken from the log file
    :return: dict
        dictionary with information needed (important keys are 'name_m1' and 'name_m2')
    """
    # removing redundnet info from the text line
    cur_text_info = text_line.split(', ', 1)[1]
    cur_text_info = cur_text_info.split(", elapsed time")[0]
    cur_text_info_as_dict = ast.literal_eval(cur_text_info)
    # changing the name of the key to be only the SR name
    cur_text_info_as_dict['name_m1'] = cur_text_info_as_dict['name_m1'].split('_model_')[0]
    cur_text_info_as_dict['name_m2'] = cur_text_info_as_dict['name_m2'].split('_model_')[0]
    return cur_text_info_as_dict


def k_nn_classifier(sr_analysed, pairs_similarity_filtered, drawing_srs, k=9):
    """
    simple implementation of the k-nn algorithm, for a single instance. We find most similar communities to the given
    community in focus (sr_analysed) and check how many our the similar ones are from the same type (drawing/no-drawing)
    as the one in focus (sr_analysed)
    This function is used by the 1st main function below
    :param sr_analysed: str
        name of the community in focus - the one which we need to return the results for
    :param pairs_similarity_filtered: pandas df
        sorted data-frame with pairs of SRs (one of them is the community in focus, sr_analysed). It is sorted based
        on distance between communities
    :param drawing_srs: list or set
        names of all drawing communities
    :param k: int, default: 9
        number of communities to compare with for the accuracy measure
    :return: float
        accuracy of the given sr_analysed according to the k-nn algorithm
    """
    srs_found = set()
    i = 0
    true_classification_in_k = 0
    # defining the class of the SR in focus (1 if drawing, 0 if not drawing)
    cur_sr_class = 1 if sr_analysed in drawing_srs else 0
    # looping untill we find enough similar SRs
    while len(srs_found) < k:
        cur_raw = pairs_similarity_filtered.iloc[i]
        cur_pairs = {cur_raw['name_m1'], cur_raw['name_m2']}
        # stayin only with the second SR (candidate one) and converting it to a string
        cur_pairs.remove(sr_analysed)
        candidate_sr = list(cur_pairs)[0]
        # only in case the cur sr was analysed already (since there are duplications in pairs_similarity)
        if candidate_sr not in srs_found:
            srs_found.add(candidate_sr)
            if cur_sr_class:
                true_classification_in_k += candidate_sr in drawing_srs
            else:
                true_classification_in_k += candidate_sr not in drawing_srs
        i += 1
    # returning the accuracy
    return (true_classification_in_k * 1.0) / (k * 1.0)


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    pairs_similarity = pd.read_csv(pairs_sim_file_path)
    drawing_srs = list(pd.read_excel(drawing_and_not_drawing_srs, sheet_name='drawing', header=None)[0])
    pairs_similarity_srs = list(set.union(set(pairs_similarity['name_m1']), set(pairs_similarity['name_m2'])))
    drawing_srs_in_pairs_sim = [ds for ds in drawing_srs if ds in pairs_similarity_srs]
    not_drawing_srs_in_pairs_sim = [pss for pss in pairs_similarity_srs if pss not in drawing_srs]
    drawing_res = dict()
    not_drawing_res = dict()

    # now checking if the logic works in a way (accuracy calculation for the not-drawing ones)
    for cur_sr in not_drawing_srs_in_pairs_sim:
        # taking only rows which contain the cur_sr name in one of the columns
        pairs_similarity_filtered = copy.copy(pairs_similarity[(pairs_similarity['name_m1'] == cur_sr) |
                                                               (pairs_similarity['name_m2'] == cur_sr)])
        # sorting the list based on score
        pairs_similarity_filtered.sort_values(by='score', inplace=True)
        cur_proba = k_nn_classifier(sr_analysed=cur_sr, pairs_similarity_filtered=pairs_similarity_filtered,
                                    drawing_srs=drawing_srs, k=9)
        not_drawing_res[cur_sr] = cur_proba
    true_pos = [key for key, value in not_drawing_res.items() if value >= 0.5]
    precision = len(true_pos) / len(not_drawing_srs_in_pairs_sim) * 1.0
    print("Not drawing precision value is: {}({}\\{})".format(precision, len(true_pos),
                                                              len(not_drawing_srs_in_pairs_sim)))

    # now checking if the logic works in a way (precision calculation for the drawing ones)
    for cur_sr in drawing_srs_in_pairs_sim:
        # taking only rows which contain the cur_sr name in one of the columns
        pairs_similarity_filtered = copy.copy(pairs_similarity[(pairs_similarity['name_m1'] == cur_sr) |
                                                               (pairs_similarity['name_m2'] == cur_sr)])
        # sorting the list based on score
        pairs_similarity_filtered.sort_values(by='score', inplace=True)
        cur_proba = k_nn_classifier(sr_analysed=cur_sr, pairs_similarity_filtered=pairs_similarity_filtered,
                                    drawing_srs=drawing_srs, k=9)
        drawing_res[cur_sr] = cur_proba
    true_pos = [key for key, value in drawing_res.items() if value > 0.5]
    precision = len(true_pos) / len(drawing_srs_in_pairs_sim) * 1.0
    print("Drawing precision value is: {} ({}\\{})".format(precision, len(true_pos),
                                                           len(drawing_srs_in_pairs_sim)))

"""

if __name__ == "__main__":
    # main file for extracting info from the nohup file Shani created
    start_time = datetime.datetime.now()
    pairs_found = []
    full_pairs_info = []
    # opening the nohup log file
    with open(nohup_file_path, 'r') as nohup_file:
        # passing over each line there
        for idx, line in enumerate(nohup_file):
            if not line.startswith('iteration'):
                continue
            else:
                # extracting the information
                cur_info_as_dict = extract_info(text_line=line)
                # adding the pairs found according to the order of the SR name - this wat we make sure the first SR in
                # the tuple is "smaller" than the second one
                if cur_info_as_dict['name_m1'] < cur_info_as_dict['name_m2']:
                    pairs_found.append((cur_info_as_dict['name_m1'], cur_info_as_dict['name_m2']))
                else:
                    pairs_found.append((cur_info_as_dict['name_m2'], cur_info_as_dict['name_m1']))
                full_pairs_info.append(cur_info_as_dict)
    # sorting the pairs found according to the first name and then second name
    pairs_found.sort(key=lambda tup: (tup[0], tup[1]))
    # save results
    # first the pairs found
    with open(os.path.join(data_path, 'pairs_found.txt'), 'w') as fp:
        fp.write('\n'.join(str(x) for x in pairs_found))
    # secondly, the pairs with the information about the pair
    full_pairs_info_df = pd.DataFrame(full_pairs_info)
    full_pairs_info_df.to_csv(path_or_buf=os.path.join(data_path, "pairs_similarity_results.csv"), index=False)

    duration = (datetime.datetime.now() - start_time).seconds
    print("Code ended. {} rows were recorded, took us {} seconds".format(full_pairs_info_df.shape[0], duration))
"""