import pathlib
import json
import numpy as np
import os
import argparse
import torch
from sklearn.preprocessing import MinMaxScaler


def preprocess_meanstd(df_train, df_test):
    """returns normalized and standardized data.
    """
    # print('meanstd', end=' ')
    df_train = np.asarray(df_train, dtype=np.float32)

    if len(df_train.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df_train)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df_train = np.nan_to_num(df_train)
    
    k = 5
    e = 1e-3
    mean_array = np.mean(df_train, axis=0, keepdims=True)
    std_array = np.std(df_train, axis=0, keepdims=True)
    std_array[np.where(std_array==0)] = e
    df_train = np.where(df_train > mean_array + k * std_array, mean_array + k * std_array, df_train)
    df_train = np.where(df_train < mean_array - k * std_array, mean_array - k * std_array, df_train)
    
    train_mean_array = np.mean(df_train, axis=0, keepdims=True)
    train_std_array = np.std(df_train, axis=0, keepdims=True)
    train_std_array[np.where(train_std_array==0)] = e
    
    df_train_new = (df_train - train_mean_array) / train_std_array
    
    df_test = np.where(df_test > train_mean_array + k * train_std_array, train_mean_array + k * train_std_array, df_test)
    df_test = np.where(df_test < train_mean_array - k * train_std_array, train_mean_array - k * train_std_array, df_test)
    df_test_new = (df_test - train_mean_array) / train_std_array

    return df_train_new, df_test_new

def preprocess_minmax(df_train, df_test):
    scaler = MinMaxScaler().fit(df_train)
    train = scaler.transform(df_train)
    test = scaler.transform(df_test)
    test = np.clip(test, a_min=-3.0, a_max=3.0)
    return train, test

def load_data_from_json(s):
    json_file = json.load(open(s))
    data = json_file['data']
    label = json_file['label']
    return data,label

def global_ELBO_loss(self, x, z, z_flowed, q_zx, p_z, p_xz, flow_log_det: torch.Tensor, test_id=None):

        log_p_xz = torch.sum(p_xz.log_prob(x), dim=-1)

        log_q_zx = torch.sum(q_zx.log_prob(z), dim=-1) - flow_log_det.sum(dim=-1)
        log_p_z = p_z.log_prob(z_flowed)
        # return loss, recon, kl
        return -torch.mean(log_p_xz+log_p_z-log_q_zx), -torch.mean(log_p_xz)

project_path = pathlib.Path(os.path.abspath(__file__)).parent

parser = argparse.ArgumentParser()
# GPU option
parser.add_argument('--gpu_id', type=int, default=0)
# dataset
parser.add_argument('--dataset_type', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--batch_size', type=int, default=250)
parser.add_argument('--base_model_dir', type=str, default=None)

# model
parser.add_argument('--z_dim', type=int, default=3)
parser.add_argument('--model_dim', type=int, default=500)
parser.add_argument('--window_size', type=int, default=60)

# training
parser.add_argument('--epochs', type=int, default= 10) 
parser.add_argument('--index_weight', type=int, default= 10) # 不存在此参数
parser.add_argument('--lr', type=float, default= 1e-3)
parser.add_argument('--seed', type=int, default=None)  # 409，2021，2022
parser.add_argument('--train_type', type=str)
parser.add_argument('--valid_epoch', type=int, default=5) 

parser.add_argument('--min_std', type=float, default=0) # 限制标准差最小值


args = parser.parse_args()


out_dir = 'out/' + args.out_dir
GPU_index = str(args.gpu_id)
global_device = torch.device(f'cuda:{GPU_index}')

dataset_type = args.dataset_type
global_epochs = args.epochs
seed = args.seed

global_z_dim= args.z_dim
global_rnn_dims= args.model_dim
global_dense_dims= args.model_dim
global_batch_size= args.batch_size
global_learning_rate = args.lr

train_type = args.train_type

if_freeze_seq = True 

global_min_std = args.min_std

exp_key = train_type
exp_key += f"_{global_min_std}clip"
exp_key += f"_{seed}"
exp_key += f"_{global_epochs}epoch"
exp_key += f"_{args.window_size}ws"
exp_key += f"_{args.lr}lr"
exp_key += f"_{args.model_dim}model_dim"


exp_dir = project_path /out_dir/ dataset_type / exp_key


base_model_dir = args.base_model_dir


# 学习率衰减
learning_rate_decay_by_epoch = 10
learning_rate_decay_factor = 1
global_valid_epoch_freq = args.valid_epoch

dataset_root = pathlib.Path(f"../data/{dataset_type}")
dataset_root = pathlib.Path(f"/home/sunyongqian/chenshiqi/Dataset/{dataset_type}")

train_data_json = dataset_root / f"{dataset_type}-train.json"
test_data_json = dataset_root / f"{dataset_type}-test.json"


train_data,_ = load_data_from_json(train_data_json)
test_data,label = load_data_from_json(test_data_json)

min_std = global_min_std
# min_std = 0
global_window_size = args.window_size

bf_search_min = 0
bf_search_max = 1000
bf_search_step_size = 1
# eval_item_length = 96*7 - (global_window_size - 1)
# noshare_save_dir = project_path / base_model_dir

#yidong
if dataset_type=='yidong-22':
    # chosed_index = [9]
    chosed_index = [29,33,35,49,57,94,105,4,16,69,77,80,93,8,16,20,21,24,30,53,54,55,58,70,75,85,93,101,102,99,100]
    train_data = [train_data[i] for i in range(len(train_data)) if sum(label[i])!=0]
    test_data = [test_data[i] for i in range(len(test_data)) if sum(label[i])!=0]
    label = [label[i] for i in range(len(label)) if sum(label[i])!=0]

#smd
if dataset_type == 'server-machine-dataset':
    chosed_index = [1]
#smap
if dataset_type == 'soil-moisture-active-passive':
    chosed_index = [1]
# msl
if dataset_type == 'mars-science-laboratory':
    chosed_index = [1]
# asd
if dataset_type == 'application-server-dataset':
    chosed_index = [5]
# ctf
if dataset_type == 'CTF_OmniClusterSelected_th48_26cluster':
    chosed_index = [2]
# SWaT
if dataset_type == 'secure-water-treatment':
    chosed_index = [1]
# WADI
if dataset_type == 'water-distribution':
    chosed_index = [1]







