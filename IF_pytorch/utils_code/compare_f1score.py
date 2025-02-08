# 比较不同算法对应数据f1score等结果差异
import sys
sys.path.append('/home/chengdaguo/code/Compared_zhujun/USAD')

import pandas as pd
import pathlib
from matplotlib import pyplot as plt
import numpy as np
from data_config import *

if dataset_type is None:
    data_num = 200
    # filter_index_list = np.loadtxt('/home/liangminghan/code/Compared_zhujun/exp_data/config/filter_data_index.txt')
    filter_index_list =  np.array([])
elif dataset_type == 'data2':
    data_num = 316
    filter_index_list = np.array([])
else:
    data_num = 41
    filter_index_list = np.array([])


path_list = [
    pathlib.Path('/home/chengdaguo/code/Compared_zhujun/USAD/out/use_center_2021_7daytrain/evaluation_result/bf_machine_best_f1.csv'),
    pathlib.Path('/home/chengdaguo/code/Compared_zhujun/USAD/out/finetune_freeze_01_4_2021_7daytrain/evaluation_result/bf_machine_best_f1.csv'),
    ]
df_list = []
for path in path_list:
    df_list.append(pd.read_csv(path))
col_index_list = [1,2,3,6]

fig = plt.figure(figsize=(30, 10))
ax_list = []
for index, _ in enumerate(col_index_list):
    ax_list.append(fig.add_subplot(len(col_index_list), 1, index+1))
# ax_list.append(fig.add_subplot(len(col_index_list)+1, 1, len(col_index_list)+1))

for df_index, df in enumerate(df_list):
    for index, col in enumerate(col_index_list):
        for i in filter_index_list:
            df.iloc[int(i), col] = 0
        ax_list[index].plot(list(range(data_num)), df.values[:, col], label=path_list[df_index].parent.parent.name)
        ax_list[index].set_ylabel(f"{df.columns[col]}")
        ax_list[index].set_xticks(list(range(0, data_num, 5)))

plt.legend(loc='best')

save_name = ''
for p in path_list:
    save_name += p.parent.parent.name +'_'
save_name = save_name[:50]
plt.savefig(f"/home/chengdaguo/code/Compared_zhujun/USAD/utils_code/{save_name}.png", bbox_inches="tight")

df_merge =  pd.DataFrame()
for index, col in enumerate(col_index_list):
    for df_index, df in enumerate(df_list):
        if index == 0:
            tp = np.sum(df['tp'].values)
            fp = np.sum(df['fp'].values)
            fn = np.sum(df['fn'].values)
            p = tp / (tp+fp)
            r = tp / (tp+fn)
            f1 = 2*p*r/(p+r)
            print(f"{path_list[df_index].parent.parent.name} tp:{tp} fp:{fp} fn:{fn} p:{round(p, 4)} r:{round(r, 4)} f1:{round(f1, 4)}")
        columns = df.columns
        for i in filter_index_list:
            df.iloc[int(i), col] = 0
        if df_merge.empty:
            df_merge['machine_id'] = df['machine_id']
        # print(df[columns[col]].shape)
        new_col = f"{columns[col]}_{df_index}"
        df_merge[new_col] = df[columns[col]]

df_merge.to_csv(f"/home/chengdaguo/code/Compared_zhujun/USAD/utils_code/{save_name}.csv", index=False)