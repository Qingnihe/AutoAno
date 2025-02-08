
import os
import numpy as np
import torch
from util.evaluate import get_err_scores
import time, random
from get_eval_result import *
from data_config import *
# from models.nni_model import TrainGDN
import requests

# 不使用 nni 
from models.TrainGDN import TrainGDN

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_index

def torch_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Config:
    train_config = {
        'batch': global_batch_size,
        'epoch': global_epochs,
        'slide_win': global_window_size,
        'dim': global_embed_dim,
        'slide_stride': global_slide_stride,
        'comment': '',
        'seed': seed,
        'out_layer_num': global_out_layer_num,
        'out_layer_inter_dim': global_out_layer_inter_dim,
        'decay': learning_rate_decay_by_epoch,
        'val_ratio': global_val_ratio,
        'topk': global_topk,
        'loss_fn': global_loss_fn
    }

    env_config={
        'save_path': exp_dir,
        'device': global_device,
        'load_model_path': exp_dir,

        'dataset': 'None',
        'report': 'best'
    }


def end_to_end(i, train_data_item, test_data_item):
    # return 0

    feature_dim = train_data_item.shape[1]
    config.train_config["feature_dim"] = feature_dim
    print(train_data_item.shape,test_data_item.shape)
    # train_data_item, test_data_item = preprocess_meanstd(train_data_item, test_data_item)
    # train_data_item_wrap = wrap_data(train_data_item, global_window_size)
    train_data_item, test_data_item, train_data_item_wrap = preprocess(train_data_item, test_data_item)
    print(train_data_item.shape, train_data_item_wrap.shape)
    data_index = i
    total_train_time = 0
    model = TrainGDN(config.train_config, config.env_config)
    model_save_dir = exp_dir / f'model/{data_index}'
    result_path = exp_dir / f"result/{data_index}"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    result_path.mkdir(parents=True, exist_ok=True)
    model_save_path = model_save_dir / f"model.pth"
    # train
    fw_time = open(exp_dir/'time.txt','w')
    fw_time.write("global_batch_size:{}\n\n".format(global_batch_size))
    train_start = time.time()
    model.fit(model_save_path, train_values=train_data_item_wrap)
    fit_one_time = time.time() - train_start
    fw_time.write('train time: {}\n\n'.format(fit_one_time))
    print(f"fit {data_index} cost:{fit_one_time}s")
    # test
    model.restore(model_save_path)
    test_start_time = time.time()
    avg_loss, test_predicted_list, test_ground_list = model.predict(test_values=test_data_item)
    test_end_time = time.time()
    test_time = test_end_time-test_start_time
    fw_time.write('test time: {}\n'.format(test_time))
    fw_time.close()
    score = get_err_scores(test_predicted_list, test_ground_list)
    print(score.shape, test_predicted_list.shape, test_ground_list.shape)
    np.save(result_path/f"recon_mean.npy", test_predicted_list)
    np.save(result_path/f"test_score.npy", score)
    total_train_time += fit_one_time
    return total_train_time



def by_entity():
    total_train_time = 0
    for i,(tr,te,lab) in enumerate(zip(train_data,test_data,label)):
        if i+1 not in chosed_index:
            continue
        torch_seed()
        tr=np.array(tr)
        te=np.array(te)
        train_time=end_to_end(i, tr.T, te.T)
        print(f"{i}--{train_time}s")
        total_train_time+=train_time
    print(f'exp:{dataset_type}{exp_key}total train time: {total_train_time}')
    get_exp_result()

def by_dataset():
    total_train_time = 0
    for i,(tr,te,lab) in enumerate(zip(train_data,test_data,label)):
        torch_seed()
        tr=np.array(tr)
        te=np.array(te)
        train_time=end_to_end(i, tr.T, te.T)
        print(f"{i}--{train_time}s")
        total_train_time+=train_time
    print(f'exp:{dataset_type}{exp_key}total train time: {total_train_time}')

def main():

    if train_type == "noshare":
        by_entity()
    else:
        by_entity()


if __name__ == '__main__':
    # get config
    config = Config()
    main()

