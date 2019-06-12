from os.path import join, sep
from os import getcwd
import os
import platform as p

# region Params PATH
if p.system() == 'Windows':
    if 'ssarusi' in getcwd():
        data_path = join('C:', sep, 'Users', 'ssarusi', 'Desktop', 'second_degree', 'semester_d', 'NLP')
        N = 2
        tf_idf_path = join('C:', sep, 'Users', 'ssarusi', 'Desktop', 'second_degree', 'semester_d', 'NLP', 'tf_idf')
        scores_path = join('C:', sep, 'Users', 'ssarusi', 'Desktop', 'second_degree', 'semester_d', 'NLP', 'scores')
        dis_path = join('C:', sep, 'Users', 'ssarusi', 'Desktop', 'second_degree', 'semester_d', 'NLP', 'distances')
else:
    # data_path = join('C:', sep, 'data', 'home', 'orentsur', 'data', 'reddit_place', 'embedding', 'embedding_per_sr')
    data_path = join('C:', sep, 'data', 'work', 'data', 'reddit_place', 'embedding', 'embedding_per_sr')
    N = 200
    tf_idf_path = join('C:', sep, 'data', 'home', 'shanisa', 'project_data', 'tf_idf', 'N_' + str(N))
    scores_path = join('C:', sep, 'data', 'home', 'shanisa', 'project_data', 'scores', 'N_' + str(N))
    dis_path = join('C:', sep, 'data', 'home', 'shanisa', 'project_data', 'distances', 'N_' + str(N))
    for p in [tf_idf_path, scores_path, dis_path]:
        if not os.path.exists(p):
            os.makedirs(p)
# endregion

CALC_TF_IDF = True
CALC_SCORES = True

SEED = 7
CPU_COUNT = 50  # mp.cpu_count()
# Word2Vec
# MODEL_TYPE, METHOD, CALC_DIS = '1', 'intersection', False

# FastText
# MODEL_TYPE, METHOD, CALC_DIS = '2', 'union', True
MODEL_TYPE, METHOD, CALC_DIS = '2', 'intersection', True

