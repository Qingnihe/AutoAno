from matplotlib import pyplot as plt
import numpy as np
import pickle
from joblib import Parallel, delayed
from data_config import *
import pandas as pd
data_num = 200
index_num = global_z_dim
score_dir = exp_dir / f'result'
pic_dir = exp_dir / f'pic'

font_size = 30


def plot(cluster_index, data_index):  
    fig = plt.figure(figsize=(40, 30))
    pic_path = pic_dir / str(cluster_index) / f'{data_index}_z.png'
    pic_path.parent.mkdir(parents=True, exist_ok=True)
    z = np.load(score_dir/ f'{data_index}/z.npy', allow_pickle=True)
    # print('z.shape', z.shape)
    for row in range(index_num):
        ax = fig.add_subplot(index_num, 1, index_num-row)
        ax.tick_params(labelsize=font_size)
        ax.set_ylabel(f"{row}", fontsize=font_size)
        ax.plot(range(z.shape[0]), z[:, row], color='black', linewidth=5)


    plt.subplots_adjust(wspace=0, hspace=0.2)
    plt.savefig(pic_path, bbox_inches="tight")
    plt.close(fig)
    # plt.clf()
    print(f"cluster:{cluster_index} pic:{data_index}")


if __name__ == "__main__":
    # with open(cluster_file, 'r') as file:
    #     clusters = json.load(file)
    data_cluster_list = [(cluster['label'], cluster['test']) for cluster in clusters]
    cluster_index_list, di_list = [], []
    for cluster_index, data_index_list in data_cluster_list:
        cluster_index_list.extend([cluster_index]*len(data_index_list))
        di_list.extend(data_index_list)
    Parallel(n_jobs=30)(delayed(plot)(cluster_index, data_index) for cluster_index, data_index in zip(cluster_index_list, di_list))
