import numpy as np
import torch
from data_config import *
# from usad.model_v3 import USAD
from usad.nni_model import USAD
# from usad.model_fcn import USAD
import time, random
from get_eval_result import *
import requests
from usad.utils import get_data, ConfigHandler, merge_data_to_csv, get_threshold, get_best_f1


class Config:
    batch_size = global_batch_size
    window_size = global_window_size
    z_dims = global_z_dim
    exp_dir = exp_dir
    save_dir = exp_dir/ 'model'
    result_dir = exp_dir/'result'

def end_to_end(i, config, train_data_item, test_data_item):
    total_train_time = 0
    feature_dim = train_data_item.shape[1]
    model = USAD(x_dims=feature_dim, batch_size=config.batch_size, z_dims=config.z_dims, window_size=config.window_size)
    print(f'-------training for cluster {i}---------')
    x_train_list = []
    train_id = i
    print(f'---------machine index: {train_id}---------')
    x_train, x_test = preprocess_meanstd(train_data_item, test_data_item)
    x_train_list.append(x_train)
    model_save_dir = config.save_dir / f'{i}'
    model_save_dir.mkdir(parents=True, exist_ok=True)
    train_start = time.time()
    model.fit(x_train_list, model_save_dir, local_epoch=global_epochs)
    train_end = time.time()
    total_train_time += train_end - train_start
    # test
    print(f'-------testing for cluster {i}---------')       
    # # restore model
    model.restore(model_save_dir)
    (config.result_dir/f'{i}').mkdir(parents=True, exist_ok=True)       
    test_score = np.array(model.predict(x_test))
    np.save(config.result_dir/f'{i}/test_score.npy', test_score)

    # threshold = get_threshold(label[i], test_score)
    f1 = get_best_f1(np.array([-x for x in test_score]), np.array(label[i]))
    with open(os.path.join(config.result_dir, f'{i}/threshold_f1.txt'), 'w') as file:
        file.write(str(f1))

    return total_train_time


def torch_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def by_entity():
    total_train_time = 0
    # ts_list = Parallel(n_jobs=1)(delayed(end_to_end)(cluster, config, train_data[cluster['center']].T, test_data[cluster['center']].T) for cluster in clusters)
    # total_train_time += sum(ts_list)

    for i,(tr,te,lab) in enumerate(zip(train_data,test_data,label)):
        torch_seed()
        if i+1 not in chosed_index:
            continue
        tr=np.array(tr)
        te=np.array(te)
        print('tr.shape',tr.shape)
        print('te.shape',te.shape)
        # print(feature_dim)
        # train_data_item, test_data_item = get_data_by_index(dataset_type, cluster['center'], training_period)
        train_time=end_to_end(i, config, tr.T, te.T)
        print(i,train_time,'s')
        total_train_time+=train_time
        # break

    print(f'exp:{dataset_type}{exp_key}total train time: {total_train_time}')
    # get_exp_result()


def main():
    if train_type == "noshare":
        by_entity()
    else:
        by_entity()



if __name__ == '__main__':
    config = Config()
    print(config.save_dir)
    main()

