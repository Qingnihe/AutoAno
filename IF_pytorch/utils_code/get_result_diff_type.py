# 获取不同种类的数据的结果

# 共六种数据
"""
0 无异常
1 平稳短时间
2 平稳长时间
3 周期长时间
4 周期短时间
5 概念漂移，阻碍训练效果
"""
# 获取每种类型的数据index
with open('/home/liangminghan/code/Compared_zhujun/exp_data/config/data_type.txt') as f:
    lines = list(f.readlines())
res_dict = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
}
for t in range(6):
    t_str = str(t)
    for index, l in enumerate(lines):
        if t_str in l:
            res_dict[t].append(index)


# 计算每种类型的结果
import sys
sys.path.append('/home/liangminghan/code/Compared_zhujun/USAD')
import pandas as pd
import numpy as np
from data_config import *
bf_res_df = pd.read_csv(project_path / f"exp_data/{exp_key}/evaluation_p2p/best_f1.csv")
for k, v in res_dict.items():
    # print(k, v)
    item_df = bf_res_df.iloc[v]
    tp = np.sum(item_df['TP'].values)
    fp = np.sum(item_df['FP'].values)
    fn = np.sum(item_df['FN'].values)
    p = tp / (tp+fp)
    r = tp / (tp+fn)
    f1 = 2*p*r/(p+r)
    print(f"{k}--f1:{f1} p:{p} r:{r} TP:{tp} FP:{fp} FN:{fn}")