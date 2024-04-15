
import pickle

import numpy as np

from sklearn.model_selection import RepeatedKFold

from data.origin_data import data_loader
from function.data_tools import process_data

from filepath import *
from config import *

if __name__ == '__main__':
    # Load data
    view_data = data_loader.load_data(ori_dir, feature_types, cell_line_name, score_type)
    n_sample = len(view_data['Label'])
    with open(f"{src_dir}/view_data.pkl", 'wb') as f:
        pickle.dump(view_data, f)

    # Process data
    exp_data = process_data(view_data, feature_types, operator_dict)
    exp_data['Label'] = view_data['Label']
    with open(f"{exp_dir}/exp_data.pkl", 'wb') as f:
        pickle.dump(exp_data, f)

    # Split data
    rkf = RepeatedKFold(n_splits=n_fold, n_repeats=n_repeat, random_state=random_state)
    split_indices = [(train_id, test_id) for train_id, test_id in rkf.split(np.arange(n_sample))]
    with open(f"{exp_dir}/split_indices.pkl", 'wb') as f:
        pickle.dump(split_indices, f)


