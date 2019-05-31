from os.path import join, sep
from os import getcwd
import os
import platform as p

# region Params PATH
if p.system() == 'Windows':
    if 'ssarusi' in getcwd():
        data_path = join('C:', sep, 'Users', 'ssarusi', 'Desktop', 'second_degree', 'semester_d', 'NLP')
        TOP = 2
        tf_idf_path = join('C:', sep, 'Users', 'ssarusi', 'Desktop', 'second_degree', 'semester_d', 'NLP', 'tf_idf')
        scores_path = join('C:', sep, 'Users', 'ssarusi', 'Desktop', 'second_degree', 'semester_d', 'NLP', 'scores')
else:
    data_path = join('C:', sep, 'data', 'home', 'orentsur', 'data', 'reddit_place', 'embedding', 'embedding_per_sr')
    TOP = 100
    tf_idf_path = join('C:', sep, 'data', 'home', 'shanisa', 'project_data', 'tf_idf', 'top_' + str(TOP))
    scores_path = join('C:', sep, 'data', 'home', 'shanisa', 'project_data', 'scores', 'top_' + str(TOP))
    dis_path = join('C:', sep, 'data', 'home', 'shanisa', 'project_data', 'distances', 'top_' + str(TOP))
    for p in [tf_idf_path, scores_path, dis_path]:
        if not os.path.exists(p):
            os.makedirs(p)
# endregion

CALC_TF_IDF = False
CALC_SCORES = True
VISUALIZATION = False

SEED = 7
# Word2Vec
# MODEL_TYPE, METHOD, CALC_DIS = '1', 'intersection', False

# FastText
MODEL_TYPE, METHOD, CALC_DIS = '2', 'union', True
# MODEL_TYPE, METHOD, CALC_DIS = '2', 'intersection', False

