# -*- coding: utf-8 -*-

import datetime
import multiprocessing as mp
import numpy as np
import os
import pickle
import random
import warnings

from sklearn.model_selection import RepeatedKFold

from filepath import *
from config import *

from data.origin_data import data_loader
from function.data_tools import construct_data, process_data
from function.function import get_logger, get_metrics, process_result
from model import Model

warnings.filterwarnings("ignore")


##### Experimental initialization
# Get logger
LOGGER = get_logger(version, f"{log_dir}/main.txt")


def do_exp(task_id, indice):
    LOGGER.info(f"Run task {task_id:>2d}, pid is {os.getpid()}...")
    e_s = datetime.datetime.now()

    # Get data
    # logger.info("Get data...")
    with open(f"{exp_dir}/exp_data.pkl", 'rb') as f:
        exp_data = pickle.load(f)

    x, view_info = construct_data(exp_data)
    y = exp_data['Label']
    x_train, y_train = x[indice[0]], y[indice[0]]
    x_test, y_test = x[indice[1]], y[indice[1]]

    model = Model(task_id, view_info)
    model.fit(x_train, y_train)

    with open(f"{mod_dir}/model_{task_id}.dat", 'wb') as f:
        pickle.dump(model, f)

    y_pred = model.predict(x_test)

    e_e = datetime.datetime.now()
    output_str = f"Task {task_id:>2d}, finished! Cost time: {e_e-e_s}\t"

    perf_dict = get_metrics(metric_names, y_test, y_pred, decimals)
    for met, perf in perf_dict.items():
        output_str += f"{met}: {perf:.4f}\t"
    LOGGER.info(output_str)

    return list(perf_dict.values())

if __name__ == '__main__':
    start = datetime.datetime.now()
    LOGGER.info("Process start at {}".format(start))

    # Set seed
    random.seed(random_state)
    np.random.seed(random_state)

    # Load data
    LOGGER.info("Load data...")
    if do_load_data:
        view_data = data_loader.load_data(ori_dir, feature_types, cell_line_name, score_type)
        with open(f"{src_dir}/view_data.pkl", 'wb') as f:
            pickle.dump(view_data, f)
    else:
        with open(f"{src_dir}/view_data.pkl", 'rb') as f:
            view_data = pickle.load(f)

    # Split data
    LOGGER.info("Split data...")
    if do_split_data:
        rkf = RepeatedKFold(n_splits=n_fold, n_repeats=n_repeat, random_state=random_state)
        indices = [(train_id, test_id) for train_id, test_id in rkf.split(np.arange(len(view_data['Label'])))]
        with open(f"{exp_dir}/split_indices.pkl", 'wb') as f:
            pickle.dump(indices, f)
    else:
        with open(f"{exp_dir}/split_indices.pkl", 'rb') as f:
            indices = pickle.load(f)

    # Feature Preprocessing
    LOGGER.info("Feature Preprocessing...")
    if do_fea_pro:
        exp_data = process_data(view_data, feature_types, operator_dict)
        exp_data['Label'] = view_data['Label']
        with open(f"{exp_dir}/exp_data.pkl", 'wb') as f:
            pickle.dump(exp_data, f)
    else:
        with open(f"{exp_dir}/exp_data.pkl", 'rb') as f:
            exp_data = pickle.load(f)

    if exp_mp:
        # Multi-process
        LOGGER.info("Run tasks in parallel...")
        pool = mp.Pool()
        tasks = [pool.apply_async(do_exp, args=(task_id, indices[task_id])) for task_id in range(n_process)]
        pool.close()
        pool.join()
        results = [task.get() for task in tasks]
    else:
        # Single-process
        LOGGER.info("Run tasks in sequence...")
        results = [do_exp(task_id, indices[task_id]) for task_id in range(n_process)]

    perf_s = []
    for task_id in range(n_process):
        perf = results[task_id]
        perf_s.append(perf)

    end = datetime.datetime.now()
    LOGGER.info(f"All tasks finished! Cost time: {end - start}")
    process_result(LOGGER, perf_s, metric_names, res_dir)













