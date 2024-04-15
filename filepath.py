
from function.function import make_dir
from config import *

##### File directories
### Data
dat_dir = make_dir('data')
ori_dir = make_dir(f"{dat_dir}/origin_data")
src_dir = make_dir(f"{dat_dir}/source_data")
exp_dir = make_dir(f"{dat_dir}/exp_data")
### Results
res_dir = make_dir(version)
log_dir = make_dir(f"{res_dir}/log")
mod_dir = make_dir(f"{res_dir}/model")
fsc_dir = make_dir(f"{res_dir}/feature_score")
ind_dir = make_dir(f"{res_dir}/ind")
est_dir = make_dir(f"{res_dir}/est")


