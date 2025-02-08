# 根据簇内每个数据距离簇中心距离确定距离阈值，结合不同的迁移策略
import sys
sys.path.append('/home/chengdaguo/code/Compared_zhujun/USAD')
from data_config import *
import numpy as np
import pandas as pd

# far_train_type = 'finetune_freeze_01234_'
far_train_type = 'finetune_freeze_0123_234'
# near_train_type = 'finetune_freeze_4_01'
near_train_type = 'finetune_freeze_01234_'

if dataset_type == None:
    
    max_thres = 1.0
    min_thres = 0
    step = 0.01

elif dataset_type == 'data2':
    near_train_type += f"_{dataset_type}"
    far_train_type += f"_{dataset_type}"
    
    max_thres = 8.0
    min_thres = 0
    step = 0.01

thres = min_thres
best_thres = 0
best_f1 = 0


# search
for i in range(int((max_thres - min_thres)/step)):
    thres = thres + step
    # clear
    near_center_machines = []
    far_center_machines = []
    for cluster in clusters:
        near_idxs = np.where(np.array(cluster['distance'])<=thres)[0]
        far_idxs = np.where(np.array(cluster['distance'])>thres)[0]
        machines = np.array(cluster['test'])
        # 根据和簇中心的距离划分
        near_center_machines.extend(machines[near_idxs])
        far_center_machines.extend(machines[far_idxs])

    best_df1 = pd.read_csv(project_path / 'out' / f'{near_train_type}_earlystop25' / 'evaluation_result/bf_machine_best_f1.csv')
    best_df2 = pd.read_csv(project_path / 'out' / f'{far_train_type}_earlystop25' / 'evaluation_result/bf_machine_best_f1.csv')
    tp = np.sum(best_df1['tp'].values[near_center_machines]) + np.sum(best_df2['tp'].values[far_center_machines])
    fp = np.sum(best_df1['fp'].values[near_center_machines]) + np.sum(best_df2['fp'].values[far_center_machines])
    fn = np.sum(best_df1['fn'].values[near_center_machines]) + np.sum(best_df2['fn'].values[far_center_machines])
    p = tp / (tp+fp)
    r = tp / (tp+fn)
    f1 = 2*p*r / (p+r)

    if f1 > best_f1:
        best_f1 = f1
        best_thres = thres

# with open('best_distance_thres.txt', 'a+') as f:
#     f.write(f'near train type: {near_train_type}, far train type: {far_train_type}\n\
# best distance threshold: {best_thres}, best f1: {best_f1}\n')


print(f'near train type: {near_train_type}, far train type: {far_train_type} best distance threshold: {round(best_thres, 2)}, best f1: {round(best_f1, 4)}')