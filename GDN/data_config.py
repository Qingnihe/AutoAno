import pathlib
import json
import numpy as np
import os
import argparse
import torch
from sklearn.preprocessing import MinMaxScaler

def preprocess_minmax(df_train, df_test):
    scaler = MinMaxScaler().fit(df_train)
    train = scaler.transform(df_train)
    test = scaler.transform(df_test)
    test = np.clip(test, a_min=-3.0, a_max=3.0)
    return train, test

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

    k = 3
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

def wrap_data(data, window_size):
    data = np.concatenate([data[-(window_size-1):], data], axis=0)
    return data

def preprocess(train_data_item, test_data_item):
    train_data_item, test_data_item = preprocess_meanstd(train_data_item, test_data_item)
    train_data_item_wrap = wrap_data(train_data_item, global_window_size)
    return train_data_item, test_data_item, train_data_item_wrap

def load_data_from_json(s):
    json_file = json.load(open(s))
    data = json_file['data']
    label = json_file['label']
    return data,label

global_loss_fn = torch.nn.MSELoss(reduction='mean')

project_path = pathlib.Path(os.path.abspath(__file__)).parent
parser = argparse.ArgumentParser()
# GPU option
parser.add_argument('--gpu_id', type=int, default=0)
# dataset
parser.add_argument('--dataset_type', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--base_model_dir', type=str, default=None)

# model
parser.add_argument('--embed_dim', type=int, default=64)
parser.add_argument('--topk', type=int, default=20)
parser.add_argument('--window_size', type=int, default=60)

# training
parser.add_argument('--epochs', type=int, default= 10)
# parser.add_argument('--index_weight', type=int, default= 10)
parser.add_argument('--lr', type=float, default= 1e-3)
parser.add_argument('--seed', type=int, default=None)  # 409，2021，2022
parser.add_argument('--train_type', type=str)
# parser.add_argument('--training_period', type=int, default=None) 
parser.add_argument('--valid_epoch_freq', type=int, default=5) 
parser.add_argument('--entity', type=int, default=1) 

# parser.add_argument('--dataset_path',type=str)
# parser.add_argument('--train_num',type=int,default=5)
# parser.add_argument('--index_weight_index',type=int,default=1)

args = parser.parse_known_args()[0]

single_score_th = 10000
out_dir = "out/" + args.out_dir

GPU_index = str(args.gpu_id)
global_device = torch.device(f'cuda:{GPU_index}')

dataset_type = args.dataset_type
global_val_ratio = 0.3
global_epochs = args.epochs
seed = args.seed
# training_period = args.training_period

global_embed_dim= args.embed_dim
global_batch_size= args.batch_size
global_out_layer_num = 1
global_out_layer_inter_dim = 256
global_topk = args.topk
# learning rate
global_learning_rate = args.lr
# circle_loss_weight = args.index_weight

# global_min_std = args.min_std
# dataset_path = args.dataset_path
# index_weight_index = args.index_weight_index
# train_num = args.train_num
if_freeze_seq = True
train_type = args.train_type

exp_key = train_type
# exp_key += f"_{train_num}nodes"
# exp_key += f"_{index_weight_index}iwi"
# exp_key += f"_{global_min_std}clip"
exp_key += f"_{seed}"
# exp_key +=f"_{training_period}daytrain"
exp_key += f"_{global_epochs}epoch"
exp_key += f"_{args.window_size}ws"
exp_key += f"_{args.lr}lr"
exp_dir = project_path /out_dir/ dataset_type / exp_key
feature_dim=22
if 'initr' in exp_key:
    global_learning_rate = 5e-3
base_model_dir = args.base_model_dir

learning_rate_decay_by_epoch = 0

# learning_rate_decay_factor = 1
global_valid_epoch_freq = args.valid_epoch_freq

# dataset_root = pathlib.Path(f"../data/{dataset_type}")
dataset_root = pathlib.Path(f"/home/sunyongqian/chenshiqi/Dataset/{dataset_type}")

train_data_json = dataset_root / f"{dataset_type}-train.json"
test_data_json = dataset_root / f"{dataset_type}-test.json"

global_window_size = args.window_size
global_slide_stride = 1
   
bf_search_min = 0
bf_search_max = 1000
bf_search_step_size = 1



train_data,_ = load_data_from_json(train_data_json)
test_data,label = load_data_from_json(test_data_json)
# noshare_save_dir = project_path / base_model_dir
#yidong
if dataset_type=='yidong-22':
    chosed_index = [9]
    # chosed_index = [1, 3, 6, 7, 8, 11, 12, 13, 14, 15, 17, 21, 24, 25, 26, 27, 29, 30, 31, 32, 35, 36, 37, 48, 49, 50, 51, 52, 54, 55, 59, 60, 66, 68, 70, 71, 75, 76, 77, 78, 79, 80, 83, 88, 89, 90, 94, 95, 96, 98, 99, 101, 102, 103, 107]
    # chosed_index = chosed_index[::3]
    # train_data = [train_data[i] for i in range(len(train_data)) if sum(label[i])!=0]
    # test_data = [test_data[i] for i in range(len(test_data)) if sum(label[i])!=0]
    # label = [label[i] for i in range(len(label)) if sum(label[i])!=0]
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
    chosed_index = [2]
# ctf
if dataset_type == 'CTF_OmniClusterSelected_th48_26cluster':
    chosed_index = [4,5,11,12,13,16,17,18,19,23,25,26]
# SWaT
if dataset_type == 'secure-water-treatment':
    chosed_index = [1]
# WADI
if dataset_type == 'water-distribution':
    chosed_index = [1]

# PSM
if dataset_type == 'PSM':
    chosed_index = [1]

if dataset_type == 'hai-23':
    chosed_index = [1,2,3,4]
    
if dataset_type == 'hai-22':
    chosed_index = [1,2,3,4,5,6]
    
if dataset_type == 'hai-21':
    chosed_index = [1,2,3,4,5]