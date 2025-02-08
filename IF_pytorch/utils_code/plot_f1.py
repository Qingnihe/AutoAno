# 画出各个参数对比的f1score图
import pathlib
import os
import pandas as pd
exp_data_path = pathlib.Path('/home/liangminghan/code/Compared_zhujun/USAD/exp_data')
# print(os.listdir(exp_data_path))
dir_list = [
    # 'cluster_10_Euclidean_alpha_0.1_beta_0.9_transfer_mean-std_ws10_nofreeze_encoder', 
    # 'cluster_10_Euclidean_alpha_0.1_beta_0.9_transfer_mean-std_ws10', 
    'cluster_10_Euclidean_alpha_0.1_beta_0.9_noshare_min-max_ws10', 
    'cluster_10_Euclidean_alpha_0.1_beta_0.9_noshare_min-max_ws10remove_drift', 
    # 'cluster_10_Euclidean_alpha_0.1_beta_0.9_k_20_noshare_ELU', 
    # 'cluster_10_Euclidean_alpha_0.1_beta_0.9_sharemany_mean-std_ws60', 
    # 'cluster_10_Euclidean_alpha_0.1_beta_0.9_noshare_mean-std_ws10', 
    # 'cluster_10_Euclidean_alpha_0.1_beta_0.9_sharemany_mean-std_ws10'
    ]
f1score_list = []
item_list = []
for dir_name in dir_list:
    f1score = [0]*200
    path_item = exp_data_path / dir_name
    for data_index in range(200):
        if (path_item/f"evaluation_p2p/best_f1_result/{data_index}/best_f1_machine.csv").exists():
            df = pd.read_csv(path_item/f"evaluation_p2p/best_f1_result/{data_index}/best_f1_machine.csv")
            for machine_id, f1 in zip(df['machine_id'].values, df['f1'].values):
                f1score[int(machine_id)] = f1
        
    f1score_list.append(f1score)
    item_list.append(dir_name[40:])

res_dict = {}
for item, f1 in zip(item_list, f1score_list):
    res_dict[item] = f1

pd.DataFrame(res_dict).to_csv('/home/liangminghan/code/Compared_zhujun/USAD/f1_compare.csv', index=True)




# from matplotlib import pyplot as plt

# from matplotlib.pyplot import MultipleLocator

# fig = plt.figure(figsize=(50, 10))
# for row, (item_name, f1score) in enumerate(zip(item_list, f1score_list)):
#     # ax = fig.add_subplot(len(item_list), 1, row+1)
#     # ax.scatter(list(range(200)), f1score, marker='*')
#     # plt.scatter(list(range(200)), f1score, label=item_name, linewidths=2)
#     plt.plot(list(range(200)), f1score, label=item_name, linewidth=2)
#     # ax.set_ylim(-0.02, 1.02)
#     # ax.set_ylabel(item_name)
# plt.vlines(list(range(200)), 0, 1, colors='black', linewidths=1)
# plt.ylim(-0.02, 1.02)
# plt.legend(fontsize=25)
# plt.tick_params(axis='both',which='major',labelsize=25)

# x_major_locator=MultipleLocator(5)
# #把x轴的刻度间隔设置为1，并存在变量里
# y_major_locator=MultipleLocator(0.1)
# #把y轴的刻度间隔设置为10，并存在变量里
# ax=plt.gca()
# #ax为两条坐标轴的实例
# ax.xaxis.set_major_locator(x_major_locator)
# #把x轴的主刻度设置为1的倍数
# ax.yaxis.set_major_locator(y_major_locator)


# plt.savefig('/home/liangminghan/code/Compared_zhujun/USAD/f1_compare.png', bbox_inches="tight") 
# plt.close(fig)

