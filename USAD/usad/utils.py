# -*- coding: utf-8 -*-
import os
import yaml
import pickle
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

prefix = "processed"


class ConfigHandler:

    def __init__(self):
        # load default config
        dir_ = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(dir_, 'config.yml')
        with open(config_path, 'r') as f:
            self._config_dict = yaml.load(f.read(), Loader=yaml.FullLoader)

        # update config according to executing parameters
        parser = argparse.ArgumentParser()
        for field, value in self._config_dict.items():
            parser.add_argument(f'--{field}', default=value)
        self._config = parser.parse_args()

        # complete config
        self._trans_format()
        self._complete_dirs()

    def _trans_format(self):
        """
        convert invalid formats of config to valid ones
        """
        config_dict = vars(self._config)
        for item, value in config_dict.items():
            if value == 'None':
                config_dict[item] = None
            elif isinstance(value, str) and is_number(value):
                if value.isdigit():
                    value = int(value)
                else:
                    value = float(value)
                config_dict[item] = value

    def _complete_dirs(self):
        """
        complete dirs in config
        """
        if self._config.save_dir:
            self._config.save_dir = self._make_dir(self._config.save_dir)

        if self._config.result_dir:
            self._config.result_dir = self._make_dir(self._config.result_dir)

    def _make_dir(self, dir_):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        par_dir = os.path.dirname(cur_dir)
        dir_ = os.path.join(par_dir, dir_)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        this_dir = os.path.join(dir_, f'{self._config.dataset}_ws{self._config.window_size[self._config.dataset]}_zd{self._config.z_dims}_me{self._config.max_epochs[self._config.dataset]}'
                                      f'_alpha{self._config.alpha}_beta{self._config.beta}')
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
        return this_dir

    @property
    def config(self):
        return self._config


def get_data_dim(dataset):
    if dataset == 'SMAP':
        return 25
    elif dataset == 'MSL':
        return 55
    elif str(dataset).startswith('machine'):
        return 38
    elif str(dataset).startswith('service'):
        return 3
    elif str(dataset).startswith('explore'):
        return 24
    elif str(dataset).startswith('new'):
        return 17
    else:
        raise ValueError('unknown dataset '+str(dataset))


def merge_data_to_csv(train_data, test_data, score, path):
    """
    merge train data, test_data, and their scores to a csv file
    """
    data = np.vstack((train_data, test_data))
    timestamp = np.arange(data.shape[0])
    df = {'timestamp': timestamp}
    for i in range(data.shape[1]):
        df[f'value{i}'] = data[:, i]
    df['tag'] = score
    df = pd.DataFrame(df)
    df.to_csv(path, index=False)


def get_data(dataset, max_train_size=None, max_test_size=None, do_preprocess=True, train_start=0,
             test_start=0):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print('load data of:', dataset)
    print("train: ", train_start, train_end)
    print("test: ", test_start, test_end)
    x_dim = get_data_dim(dataset)
    f = open(os.path.join(prefix, dataset + '_train.pkl'), "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
    f.close()
    try:
        f = open(os.path.join(prefix, dataset + '_test.pkl'), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:test_end]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    if do_preprocess:
        train_data, test_data = preprocess(train_data, test_data)
    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    return (train_data, None), (test_data, test_label)


def preprocess(df_train, df_test):
    """
    normalize raw data
    """
    df_train = np.asarray(df_train, dtype=np.float32)
    df_test = np.asarray(df_test, dtype=np.float32)
    if len(df_train.shape) == 1 or len(df_test.shape) == 1:
        raise ValueError('Data must be a 2-D array')
    if np.any(sum(np.isnan(df_train)) != 0):
        print('train data contains null values. Will be replaced with 0')
        df_train = np.nan_to_num()
    if np.any(sum(np.isnan(df_test)) != 0):
        print('test data contains null values. Will be replaced with 0')
        df_test = np.nan_to_num()
    scaler = MinMaxScaler()
    scaler = scaler.fit(df_train)
    df_train = scaler.transform(df_train)
    df_test = scaler.transform(df_test)
    return df_train, df_test


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False


def get_threshold(label, test_score):
    f1 = 0
    ret = 0
    for i in test_score:
        y_pred = []
        for j in test_score:
            if j >= i:
                y_pred.append(1)
            else:
                y_pred.append(0)
        if f1_score(label, y_pred) > f1:
            f1 = f1_score(label, y_pred)
            ret = i
    return ret, f1


def get_best_f1(score, label):
    '''
    :param score: 1-D array, input score, tot_length
    :param label: 1-D array, standard label for anomaly
    :return: list for results, threshold
    '''

    assert score.shape == label.shape
    print('***computing best f1***')
    search_set = []
    tot_anomaly = 0
    for i in range(label.shape[0]):
        tot_anomaly += (label[i] > 0.5)
    flag = 0
    cur_anomaly_len = 0
    cur_min_anomaly_score = 1e5
    for i in range(label.shape[0]):
        if label[i] > 0.5:
            # here for an anomaly
            if flag == 1:
                cur_anomaly_len += 1
                cur_min_anomaly_score = score[i] if score[i] < cur_min_anomaly_score else cur_min_anomaly_score
            else:
                flag = 1
                cur_anomaly_len = 1
                cur_min_anomaly_score = score[i]
        else:
            # here for normal points
            if flag == 1:
                flag = 0
                search_set.append((cur_min_anomaly_score, cur_anomaly_len, True))
                search_set.append((score[i], 1, False))
            else:
                search_set.append((score[i], 1, False))
    if flag == 1:
        search_set.append((cur_min_anomaly_score, cur_anomaly_len, True))
    search_set.sort(key=lambda x: x[0])
    best_f1_res = - 1
    threshold = 1
    P = 0
    TP = 0
    best_P = 0
    best_TP = 0
    for i in range(len(search_set)):
        P += search_set[i][1]
        if search_set[i][2]:  # for an anomaly point
            TP += search_set[i][1]
        precision = TP / (P + 1e-5)
        recall = TP / (tot_anomaly + 1e-5)
        f1 = 2 * precision * recall / (precision + recall + 1e-5)
        if f1 > best_f1_res:
            best_f1_res = f1
            threshold = search_set[i][0]
            best_P = P
            best_TP = TP

    print('***  best_f1  ***: ', best_f1_res)
    print('*** threshold ***: ', threshold)
    return (best_f1_res,
            best_TP / (best_P + 1e-5),
            best_TP / (tot_anomaly + 1e-5),
            best_TP,
            score.shape[0] - best_P - tot_anomaly + best_TP,
            best_P - best_TP,
            tot_anomaly - best_TP), threshold