# 将txt数据文件切分为一天天的时候使用滑动平均
# 将14天数据文件切分为每天的文件
import pathlib
import os
import numpy as np
import pandas as pd

def move_avg(data_1d):
    df = pd.Series(data_1d)
    # 以当前点为中心的方法有问题，线上检测的时候需要等的时间较长
    moving_avg = df.rolling(window=4, min_periods=1, center=False).median()
    moving_avg = moving_avg.values.flatten()
    return moving_avg

def main():
    # txt_root = pathlib.Path('/home/liangminghan/code/CTF_yidong/data/CTF_data_txt_2000')
    # ctf_data_root = pathlib.Path('/home/liangminghan/code/CTF_yidong/data/CTF_data_smooth_2000')
    # txt_root_smooth = pathlib.Path('/home/liangminghan/code/CTF_yidong/data/CTF_data_txt_smooth_2000')
    # txt_list = os.listdir(str(txt_root))
    # for txt_file in txt_list:
    #     print(index)
    #     index = int(txt_file.split('.')[0])
    #     # if index > 199:
    #     #     continue
    #     txt_data = np.genfromtxt(txt_root / txt_file, delimiter=',')
    #     # smooth
    #     smooth_txt_data = np.zeros_like(txt_data)
    #     for col in range(txt_data.shape[1]):
    #         smooth_txt_data[:, col] = move_avg(txt_data[:, col])
    #     np.savetxt(txt_root_smooth / f"{index}.txt", smooth_txt_data,  delimiter=',', fmt='%.2f')
    #     for day in range(14):
    #         day_data = smooth_txt_data[day*96:(day + 1)*96]
    #         np.savetxt(ctf_data_root / f"{index}_{day + 1}.txt", day_data, delimiter=',', fmt='%.2f')
    raw_data = np.load('/home/liangminghan/code/Compared_zhujun/exp_data/data_raw/raw-data.npy')
    smooth_data = np.zeros_like(raw_data)
    for data_index in range(raw_data.shape[0]):
        print(data_index)
        for col in range(raw_data.shape[1]):
            smooth_data[data_index, col, :] = move_avg(smooth_data[data_index, col, :])
    np.save('/home/liangminghan/code/Compared_zhujun/exp_data/data_raw/smooth-data.npy', smooth_data)
if __name__ == '__main__':
    main()