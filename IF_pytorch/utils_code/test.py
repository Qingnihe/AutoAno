import pathlib

import numpy as np


result_dir = pathlib.Path('/home/chengdaguo/code/Compared_zhujun/IF_pytorch/out_ano/data2/finetune_all_2021_2daytrain_modeldim500_weight10/result')
test_list = []
for i in range(316):
    if not (result_dir / f"{i}/test_score.npy").exists():
        test_list.append(i)

print(test_list)
print(len(test_list))

# s = np.load('/home/chengdaguo/code/Compared_zhujun/IF_pytorch/out_ano/noshare_nopnf_data2_2021_5daytrain_modeldim500/result/314/test_score.npy')
# print(s.shape)
