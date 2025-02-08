import os
import numpy as np

import torch

# from omnianomaly.model_lgssm_concat import OmniAnomaly
from omnianomaly.nni_model import OmniAnomaly

from get_eval_result import *
import requests

import time, random

from data_config import *

# os.environ['CUDA_VISIBLE_DEVICES'] = GPU_index


class Config:
    x_dims = ''
    z_dims = global_z_dim
    max_epochs = global_epochs
    batch_size = global_batch_size
    window_size = global_window_size

    exp_dir = exp_dir
    save_dir = exp_dir/ 'model'
    result_dir = exp_dir/'result'


def end_to_end(i, config, train_data_item, test_data_item):

    total_train_time = 0
    feature_dim = train_data_item.shape[1]
    config.x_dims = feature_dim
    model = OmniAnomaly(x_dims=config.x_dims,
                z_dims=config.z_dims,
                max_epochs=config.max_epochs,
                batch_size=config.batch_size,
                window_size=config.window_size,
                learning_rate=global_learning_rate)

    # if if_freeze_seq:
    #     model.freeze_layers(freeze_layer='seq')
    
    # train
    print(f'-------training for cluster {i}---------')
    x_train_list = []
    train_id = i
    print(f'---------machine index: {train_id}---------')
    x_train, _ = preprocess_meanstd(train_data_item, test_data_item)
    x_train_list.append(x_train)
    (config.save_dir/ f'cluster_{i}').mkdir(parents=True, exist_ok=True)
    save_path = config.save_dir/ f'cluster_{i}'/ 'model.pkl'

    fw_time = open(exp_dir/'time.txt','w')
    fw_time.write("global_batch_size:{}\n\n".format(global_batch_size))
    train_start = time.time()
    model.fit(x_train_list, save_path, valid_portion=0.05)
    train_end = time.time()
    total_train_time += train_end - train_start
    train_time = train_end - train_start
    fw_time.write('train time: {}\n\n'.format(train_time))

    # test
    print(f'-------testing for cluster {i}---------')     

    
    save_path = config.save_dir/ f'cluster_{i}'/ 'model.pkl'
    model.restore(save_path)
    
    x_train, x_test = preprocess_meanstd(train_data_item, test_data_item)

    (config.result_dir/f'{i}').mkdir(parents=True, exist_ok=True) 

    test_start_time = time.time()
    score, recon_mean, recon_std, z = model.predict(x_test, save_path)
    test_end_time = time.time()
    test_time = test_end_time-test_start_time
    fw_time.write('test time: {}\n'.format(test_time))
    fw_time.close()
    if score is not None:
        np.save(config.result_dir/f'{i}/test_score.npy', -score)
        np.save(config.result_dir/f'{i}/recon_mean.npy', recon_mean)
        np.save(config.result_dir/f'{i}/recon_std.npy', recon_std)
        np.save(config.result_dir/f'{i}/z.npy', z)
    return total_train_time


def torch_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def register_210():
    pid = os.getpid()

    url = 'http://10.10.1.210/api/v1/job/create'
    d = {'student_id': '2012661', 
        'password': '210csq',
        'description': 'asd-2-w300', 
        'server_ip': '10.10.1.219',
        'duration': '一小时内', 
        'pid': pid,
        'server_user': 'sunyongqian',
        'command': '', 
        'use_gpu': 1,
        }
    r = requests.post(url, data=d)
    print("注册完成")

def by_entity():
    torch_seed()
    total_train_time = 0
    for i,(tr,te,lab) in enumerate(zip(train_data,test_data,label)):
        if i+1 not in chosed_index:
            continue
        tr=np.array(tr)
        te=np.array(te)
        train_time=end_to_end(i, config, tr.T, te.T)
        print(i,train_time,'s')
        total_train_time+=train_time
    print(f'exp:{dataset_type}{exp_key}total train time: {total_train_time}')
    get_exp_result()

def by_dataset():
    torch_seed()
    total_train_time = 0
    for i,(tr,te,lab) in enumerate(zip(train_data,test_data,label)):
        tr=np.array(tr)
        te=np.array(te)
        train_time=end_to_end(i, config, tr.T, te.T)
        print(i,train_time,'s')
        total_train_time+=train_time
    print(f'exp:{dataset_type}{exp_key}total train time: {total_train_time}')

def main():
    register_210()
    if train_type == "noshare":
        by_entity()
    else:
        by_dataset()

if __name__ == '__main__':
    # get config
    config = Config()
    print(config.save_dir)
    main()

