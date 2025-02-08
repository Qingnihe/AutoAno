import pickle

# res = pickle.load(open('/home/liangminghan/code/Compared_zhujun/USAD/exp_data/cluster_10_Euclidean_alpha_0.1_beta_0.9/result/0/test_score.pkl', mode='rb'))
# print(res.shape)

import numpy as np
# recon = np.load('/home/liangminghan/code/Compared_zhujun/USAD/exp_data/cluster_10_Euclidean_alpha_0.1_beta_0.9/result/0/recon_G.npy')
# print(recon.shape)
# res = np.load('/home/liangminghan/code/Compared_zhujun/USAD/out/alpha_0.5_beta_0.5_use_center_ws20_ELU/result/0/test_score.npy')
# print(res.shape)

import os
l = os.listdir('/home/chengdaguo/code/Compared_zhujun/USAD/out/alpha_0.5_beta_0.5_noshare_ws20_ELU_ctf_earlystop10/result')
print(len(l))
