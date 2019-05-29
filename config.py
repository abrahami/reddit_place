from os.path import join, sep
from os import getcwd
import platform as p

# region Params PATH
if p.system() == 'Windows':
    if 'ssarusi' in getcwd():
        data_path = join('C:', sep, 'Users', 'ssarusi', 'Desktop', 'second_degree', 'semester_d', 'NLP')
        tf_idf_path = join('C:', sep, 'Users', 'ssarusi', 'Desktop', 'second_degree', 'semester_d', 'NLP', 'tf_idf')
else:
    data_path = join('C:', sep, 'data', 'home', 'orentsur', 'data', 'reddit_place', 'embedding', 'embedding_per_sr')
    tf_idf_path = join('C:', sep, 'data', 'home', 'shanisa')

# endregion

CALC_TF_IDF = True
CALC_SCORES = True

SEED = 7
MODEL_TYPE = '1'  # Word2Vec
# MODEL_TYPE = '2'  # Word2Vec
