
import csv
import logging
import os
import sys
import xgboost

import numpy as np

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_metric(metric, y_true, y_pred):
    if metric == 'MAE':
        return mean_absolute_error(y_true, y_pred)
    elif metric == 'MSE':
        return mean_squared_error(y_true, y_pred)
    elif metric == 'RMSE':
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric == 'R2':
        return r2_score(y_true, y_pred)
    elif metric == 'PCC':
        return pearsonr(y_true, y_pred)[0]
    else:
        raise TypeError(f"Unknown metric: {metric}!")

def get_metrics(metrics, y_true, y_pred, decimals=None):
    perf_dict = {met: get_metric(met, y_true, y_pred) for met in metrics}

    if decimals is not None:
        for met in perf_dict:
            perf_dict[met] = np.round(perf_dict[met], decimals)

    return perf_dict

def get_logger(name, filepath, write=True, output=True, mode='a'):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s -  %(message)s')

    if write:
        file_handler = logging.FileHandler(filepath, mode=mode)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if output:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def get_learner(name, params):
    if name == 'XGB':
        return xgboost.XGBRegressor(**params)
    else:
        raise ValueError(f"Invalid name parameter: {name}!")

def process_result(logger, perf_s, metric_names, res_path):
    # output result
    logger.info("5-fold cross validation results: ")
    perf_mean = np.round(np.mean(np.array(perf_s), axis=0), 4)
    perf_std = np.round(np.std(np.array(perf_s), axis=0), 4)
    perf_ms = np.array(['{:.4f}\u00B1{:.4f}'.format(perf_mean[i], perf_std[i]) for i in range(len(perf_mean))])
    for i in range(len(metric_names)):
        logger.info("{}\tmean:{:.4f}\tstd:{:.4f}".format(metric_names[i], perf_mean[i], perf_std[i]))

    # save result
    row_name = np.array([[''] + [str(task_id + 1) for task_id in range(len(perf_s))] +
                         ['mean', 'std', 'mean\u00B1std']]).T
    col_name = np.array([metric_names])
    save_data = np.hstack((row_name, np.vstack((col_name, np.vstack((perf_s, perf_mean, perf_std, perf_ms))))))
    with open(f"{res_path}/result.csv", mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(save_data)


