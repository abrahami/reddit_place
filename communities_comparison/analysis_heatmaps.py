import sys
if sys.platform == 'linux':
    sys.path.append('/data/home/shanisa/reddit_place')
print(f"\nCurrent platform: {sys.platform}")
import config as c
from os.path import join, sep
from scipy.spatial.distance import squareform
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_dis_matrices(n_m1, n_m2, method, partial):
    name_1 = n_m1 + '_' + method + '_' + n_m2
    name_2 = n_m2 + '_' + method + '_' + n_m1
    print(f"load {name_1}")
    with open(join(c.dis_path, name_1 + '.pickle'), 'rb') as handle:
        dis_1 = pickle.load(handle)
    print(f"load {name_2}")
    with open(join(c.dis_path, name_2 + '.pickle'), 'rb') as handle:
        dis_2 = pickle.load(handle)
    if partial:
        dis_1 = dis_1[:120]
        dis_2 = dis_2[:120]
    return squareform(dis_1), squareform(dis_2)


def generate_heatmaps(n_m1, n_m2, method, partial=False):
    DPI = 100

    dis_1, dis_2 = load_dis_matrices(n_m1, n_m2, method, partial)

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

    f_name = n_m1 + '_' + n_m2 + '_' + method + '_' + c.MODEL_TYPE
    if partial:
        f_name = f_name + '_partial'
    fig.savefig(join(c.dis_path, f_name + '.png'), bbox_inches="tight", dpi=DPI)
    # plt.show()
    print("DONE\n")


method_ = 'intersection'
# # American Football  vs  American Football  --->  score = 0.110
generate_heatmaps('patriots', 'greenbaypackers', method_, partial=True)
# # American Football  vs  News/Politics  --->  score = 0.962
generate_heatmaps('patriots', 'womenfortrump', method_)

# Music  vs  Music  --->  score = 0.130
generate_heatmaps('guitar', 'music', method_, partial=True)
generate_heatmaps('guitar', 'spongebob', method_)


