# -*- coding:gbk -*-

import numpy as np

from sklearn.model_selection import RepeatedKFold, KFold

def process_data(view_data, feature_types, operator_dict):
    exp_data = {}
    for feature_type in feature_types:
        if feature_type in ['MF', 'PP', 'GEP']:
            view_a, view_b = view_data[f"{feature_type}_A"], view_data[f"{feature_type}_B"]
            if feature_type == 'MF':
                for op in operator_dict['logical']:
                    exp_data[f"{feature_type}_{op}"] = calculate_x(op, view_a, view_b)
            else:
                for op in operator_dict['mathematic']:
                    exp_data[f"{feature_type}_{op}"] = calculate_x(op, view_a, view_b)
                for op in operator_dict['comparison']:
                    exp_data[f"{feature_type}_{op}"] = calculate_x(op, view_a, view_b)
        else:
            exp_data[feature_type] = view_data[feature_type]
    return exp_data

def calculate_x(op, x_a, x_b):
    if op == 'AND':
        cal_x = np.logical_and(x_a, x_b).astype(np.int)
    elif op == 'OR':
        cal_x = np.logical_or(x_a, x_b).astype(np.int)
    elif op == 'XOR':
        cal_x = np.logical_xor(x_a, x_b).astype(np.int)
    elif op == 'SUM':
        cal_x = x_a + x_b
    elif op == 'DIF':
        cal_x = abs(x_a - x_b)
    elif op == 'MAX':
        cal_x = np.where(x_a > x_b, x_a, x_b)
    elif op == 'MIN':
        cal_x = np.where(x_a < x_b, x_a, x_b)
    else:
        raise ValueError(f"Invalid operator: {op}")
    return cal_x

def construct_data(data_dict):
    items, view_info = [], {}
    s, e = None, 0
    for key, value in data_dict.items():
        if not key == 'Label':
            items.append(value)

            s = e
            e += value.shape[1]
            view_info[key] = np.arange(s, e)
    data = np.concatenate(items, axis=1)
    return data, view_info

def kfold_sampling(x, y, n_fold=5, random_state=0, n_repeat=1):
    # rkf = RepeatedKFold(n_splits=n_fold, n_repeats=n_repeat, random_state=random_state)
    rkf = KFold(n_splits=n_fold, shuffle=True, random_state=random_state)
    x_train_s, y_train_s, x_test_s, y_test_s = [], [], [], []
    for train_id, test_id in rkf.split(x, y):
        x_train_s.append(x[train_id])
        y_train_s.append(y[train_id])
        x_test_s.append(x[test_id])
        y_test_s.append(y[test_id])
    return x_train_s, y_train_s, x_test_s, y_test_s



