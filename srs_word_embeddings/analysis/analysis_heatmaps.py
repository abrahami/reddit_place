import sys
if sys.platform == 'linux':
    sys.path.append('/data/home/shanisa/reddit_place')
print(f"\nCurrent platform: {sys.platform}")
from os.path import join, sep
from scipy.spatial.distance import squareform
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import commentjson
import os


def load_dis_matrices(n_m1, n_m2, method, partial, config_dict):
    '''

    :param n_m1:
    :param n_m2:
    :param method:
    :param partial:
    :return:
    '''
    dis_path = config_dict['dis_path'][config_dict['machine']]
    name_1 = n_m1 + '_' + method + '_' + n_m2
    name_2 = n_m2 + '_' + method + '_' + n_m1
    print(f"load {name_1}")
    with open(join(dis_path, name_1 + '.pickle'), 'rb') as handle:
        dis_1 = pickle.load(handle)
    print(f"load {name_2}")
    with open(join(dis_path, name_2 + '.pickle'), 'rb') as handle:
        dis_2 = pickle.load(handle)
    if partial:
        dis_1 = dis_1[:120]
        dis_2 = dis_2[:120]
    return squareform(dis_1), squareform(dis_2)


def generate_heatmaps(n_m1, n_m2, method, config_dict, partial=False):
    '''

    :param n_m1:
    :param n_m2:
    :param method:
    :param partial:
    :return:
    '''
    DPI = 100

    dis_1, dis_2 = load_dis_matrices(n_m1, n_m2, method, partial, config_dict=config_dict)

    title = "Cosine similarity between " + method + " words"
    dis = [dis_1, dis_2]
    # t1 = n_m1.replace(r"_model_.*", "")
    # t2 = n_m2.replace("_model_2.01.model", "")
    community = [n_m1, n_m2]

    print("generate_heatmaps")
    cols = 2
    fig, ax = plt.subplots(ncols=cols, figsize=(24, 12))
    fig.suptitle(title, fontsize=30)
    for j in range(cols):
        ax[j].set_title(community[j], fontsize=30)
        ax[j].get_xaxis().set_visible(False)
        ax[j].get_yaxis().set_visible(False)
        mask = np.zeros_like(dis[j])
        mask[np.triu_indices_from(mask)] = True
        print(f"heatmap {j +1}")
        curr_ax = sns.heatmap(dis[j], mask=mask, annot=False, cmap="Blues", square=True, ax=ax[j], vmin=-1, vmax=1)
        cbar = curr_ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        if j == 1:
            cbar.ax.set_visible(False)

    f_name = n_m1 + '_' + n_m2 + '_' + method + '_' + config_dict['general_config']['model_type']
    if partial:
        f_name = f_name + '_partial'
    fig.savefig(join(config_dict['dis_path'][config_dict['machine']], f_name + '.png'), bbox_inches="tight", dpi=DPI)
    # plt.show()
    print("DONE\n")


###################################################### Configurations ##################################################
config_dict = commentjson.load(open(os.path.join(os.path.dirname(os.getcwd()), 'configurations',
                                                 'analysis_config.json')))
machine = 'yalla' if sys.platform == 'linux' else os.environ['COMPUTERNAME']
config_dict['machine'] = machine
# adjusting the condif dict to current run (e.g., model_type to paths)
config_dict['data_path'][machine] = join(config_dict['data_path'][machine], config_dict['general_config']['model_type'])
config_dict['vocab_distr_path'][machine] = join(config_dict['vocab_distr_path'][machine],
                                                config_dict['general_config']['model_type'])
config_dict['tf_idf_path'][machine] = join(config_dict['tf_idf_path'][machine],
                                           config_dict['general_config']['model_type'])
config_dict['scores_path'][machine] = join(config_dict['scores_path'][machine],
                                           config_dict['general_config']['model_type'])
config_dict['dis_path'][machine] = join(config_dict['dis_path'][machine], config_dict['general_config']['model_type'])
########################################################################################################################

if __name__ == "__main__":
    method_ = 'intersection'
    # # American Football  vs  American Football  --->  score = 0.110
    # generate_heatmaps('patriots', 'greenbaypackers', method_, config_dict=config_dict, partial=True)
    # # American Football  vs  News/Politics  --->  score = 0.962
    # generate_heatmaps('patriots', 'womenfortrump', method_, config_dict=config_dict)

    # guitar  vs  music  --->  score = 0.130
    # intersection/union = 0.382
    generate_heatmaps('guitar', 'music', method_, config_dict=config_dict, partial=True)
    # guitar  vs  spongebob (TV)  --->  score = 0.940
    # intersection/union = 0.084
    generate_heatmaps('guitar', 'spongebob', method_, config_dict=config_dict)


