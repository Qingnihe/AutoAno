import os
import matplotlib.pyplot as plt
from data_config import *
import numpy as np


def plot_test(file):
    print(dataset_type)
    chosed_index = [3]
    entity_test = np.array([1,2,3])
    entity = 0
    for i,(tr,te,lab) in enumerate(zip(train_data,test_data,label)):
        if i+1 not in chosed_index:
            continue
        tr=np.array(tr)
        te=np.array(te)
        entity = i
        _, entity_test = preprocess(tr.T, te.T)
        entity_test_transpose = entity_test.transpose()
        print('test.shape',entity_test.shape)
        print("entity:", i+1)

    file = file + f"/result/{entity}"

    recon_mean = np.load(file + "/recon_mean.npy")
    recon_mean_transpose = recon_mean.transpose()

    test_score = np.load(file + "/test_score.npy")
    test_score_transpose = test_score.transpose()
    
    fig = plt.figure(1, figsize=(16,19))
    
    for metric in range(len(entity_test_transpose)):
        ax=plt.subplot(len(entity_test_transpose),1,metric+1)  
        plt.plot(entity_test_transpose[metric])

        zeros_array = np.zeros(63)
        fommat_recon = np.insert(zeros_array, -1, recon_mean_transpose[metric])
        plt.plot(fommat_recon)

        # plt.plot(test_score_transpose[metric])
        
    
    plt.savefig("imgs/recon/origin_recon_yd.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    
def plot_recon_diff(file1, file2):
    print(dataset_type)
    chosed_index = [9]
    entity_test = np.array([1,2,3])
    entity = 0
    for i,(tr,te,lab) in enumerate(zip(train_data,test_data,label)):
        if i+1 not in chosed_index:
            continue
        tr=np.array(tr)
        te=np.array(te)
        entity = i
        _, entity_test = preprocess(tr.T, te.T)
        entity_test_transpose = entity_test.transpose()
        print('test.shape',entity_test.shape)
        print("entity:", i+1)

    file1 = file1 + f"/result/{entity}"
    file2 = file2 + f"/result/{entity}"

    recon_mean = np.load(file1 + "/recon_mean.npy")
    recon_mean_transpose = recon_mean.transpose()

    recon_mean2 = np.load(file2 + "/recon_mean.npy")
    recon_mean_transpose2 = recon_mean2.transpose()

    test_score = np.load(file1 + "/test_score.npy")
    test_score_transpose = test_score.transpose()

    test_score2 = np.load(file2 + "/test_score.npy")
    test_score_transpose2 = test_score2.transpose()
    
    fig = plt.figure(1, figsize=(16,19))
    
    for metric in range(len(entity_test_transpose)):
        ax=plt.subplot(len(entity_test_transpose),1,metric+1)  
        # plt.plot(entity_test_transpose[metric], label='origin')

        # zeros_array = np.zeros(63)
        # fommat_recon = np.insert(zeros_array, -1, recon_mean_transpose[metric])
        # plt.plot(fommat_recon, linestyle='-',label='minloss')

        # fommat_recon2 = np.insert(zeros_array, -1, recon_mean_transpose2[metric])
        # plt.plot(fommat_recon2, linestyle='-', label='maxf1')

        plt.plot(test_score_transpose[metric],label='minloss')
        plt.plot(test_score_transpose2[metric],label='maxf1')
        
    plt.legend()
    plt.savefig("imgs/recon/sdf_f1loss_asd9_score_diff.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)

def main():
    # plot_test(file="/home/zhangshenglin/chenshiqi/SDFVAE_pytorch/out_asd_9_w300/application-server-dataset/noshare_2020_bs50_T5_s4_d15_md100_lr0.001_ws300")
    plot_test(file="/home/sunyongqian/chenshiqi/SDFVAE_pytorch/0522/yidong-22/noshare_-3clip_2020_model100_s8_d10_T5_lr0.001_epoch250_windows60")
    # plot_recon_diff(file1="/home/zhangshenglin/chenshiqi/SDFVAE_pytorch/out_asd_9_w300/application-server-dataset/noshare_2020_bs50_T5_s4_d15_md100_lr0.001_ws300",
    # file2="/home/zhangshenglin/chenshiqi/SDFVAE_pytorch/out_asd_9_w300/application-server-dataset/noshare_2020_bs50_T5_s12_d5_md100_lr0.001_ws300")
    


if __name__ == '__main__':
    main()
