import pathlib
import json
import numpy as np
import os
import argparse
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from nni.utils import merge_parameter
import nni
import logging

def load_data_from_json(s):
    json_file = json.load(open(s))
    data = json_file['data']
    label = json_file['label']
    return data,label

def preprocess_meanstd(df_train, df_test):
    """returns normalized and standardized data.
    """

    df_train = np.asarray(df_train, dtype=np.float32)

    if len(df_train.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df_train)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df_train = np.nan_to_num(df_train)

    # normalize data
    # df = MinMaxScaler().fit_transform(df)
    mean_array = np.mean(df_train, axis=0, keepdims=True)
    std_array = np.std(df_train, axis=0, keepdims=True)
    # df_train = np.where(df_train > mean_array + k * std_array, mean_array + k * std_array, df_train)
    # df_train = np.where(df_train < mean_array - k * std_array, mean_array - k * std_array, df_train)
    df_train_new = (df_train - np.mean(df_train, axis=0, keepdims=True)) / (
        np.std(df_train, axis=0, keepdims=True) + 1e-3)

    # df_test = np.where(df_test > mean_array + k * std_array, mean_array + k * std_array, df_test)
    # df_test = np.where(df_test < mean_array - k * std_array, mean_array - k * std_array, df_test)
    df_test_new = (df_test - np.mean(df_train, axis=0, keepdims=True)) / (np.std(df_train, axis=0, keepdims=True) + 1e-3)

    return df_train_new, df_test_new



def get_params():
    parser = argparse.ArgumentParser()
    # GPU option
    parser.add_argument('--gpu_id', type=int, default=0)
    # dataset
    parser.add_argument('--dataset_type', type=str, default='application-server-dataset')
    parser.add_argument('--out_dir', type=str, default='out_nni')
    parser.add_argument('--base_model_dir', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=64)

    # model
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--z_dim', type=int, default=3)
    parser.add_argument('--window_size', type=int, default=60)

    # training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=2020)  # 409，2021，2022
    parser.add_argument('--train_type', type=str, default='noshare')
    # parser.add_argument('--training_period', type=int, default="None") #训练时间长度 天数
    parser.add_argument('--valid_step', type=int, default=200) 
    parser.add_argument('--valid_epoch', type=int, default=5) 
    parser.add_argument('--entity', type=int, default=1)

    return parser.parse_known_args()[0]

logger = logging.getLogger('usad_AutoML')
project_path = pathlib.Path(os.path.abspath(__file__)).parent

if 'args' not in globals():
    tuner_params= nni.get_next_parameter()
    logger.debug("tuner_params:", tuner_params)
    # args = vars(merge_parameter(get_params(), tuner_params))
    args = vars(get_params())
    logger.debug("updated args:", args)

single_score_th = 10000
out_dir = "out/" + args['out_dir']
# 注意202只有1块gpu，只能设置为0
GPU_index = str(args['gpu_id'])
global_device = torch.device(f'cuda:{GPU_index}')

dataset_type = args['dataset_type']
global_epochs = args['epochs']
seed = args['seed']
global_window_size = args['window_size']
# training_period = args['training_period']

# USAD参数
global_alpha = args['alpha']
global_beta = args['beta']
global_z_dim= args['z_dim']
global_batch_size= args['batch_size']
# learning rate
global_lr = args['lr']

train_type = args['train_type']
base_model_dir = args['base_model_dir']
global_valid_step_freq = args['valid_step']
global_valid_epoch_freq = args['valid_epoch']

exp_key = train_type
exp_key += f"_{seed}"
# exp_key +=f"_{training_period}daytrain"
exp_key +=f"_{global_lr}lr"
exp_key +=f"_{global_epochs}epoch"
exp_key +=f"_{args['window_size']}ws"
exp_dir = project_path / out_dir / dataset_type / exp_key

learning_rate_decay_by_step = 10000000
learning_rate_decay_factor = 1
# 读取数据
dataset_root = pathlib.Path(f"/home/sunyongqian/chenshiqi/Dataset/{dataset_type}")
train_data_json = dataset_root / f"{dataset_type}-train.json"
test_data_json = dataset_root / f"{dataset_type}-test.json"

train_data,_ = load_data_from_json(train_data_json)
test_data,label = load_data_from_json(test_data_json)

# 实验参数
bf_search_min = 0
bf_search_max = 1000
bf_search_step_size = 1
noshare_save_dir = project_path / base_model_dir

#yidong
if dataset_type=='yidong-22':
    chosed_index = [9]
#smd
if dataset_type == 'server-machine-dataset':
    chosed_index = [1]
#smap
if dataset_type == 'soil-moisture-active-passive':
    chosed_index = [19,31,35,38,51]
# msl
if dataset_type == 'mars-science-laboratory':
    chosed_index = [26]
# asd
if dataset_type == 'application-server-dataset':
    chosed_index = [args['entity']]
# ctf
if dataset_type == 'CTF_OmniClusterSelected_th48_26cluster':
    chosed_index = [5,11,12,13,16,17,19,23,25,26]

# SWaT
if dataset_type == 'secure-water-treatment':
    chosed_index = [1]
# WADI
if dataset_type == 'water-distribution':
    chosed_index = [1]






