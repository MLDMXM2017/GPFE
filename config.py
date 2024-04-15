


##### Experimental Setting
version = 'main'
### Parameters for experiment
exp_mp = True       # Whether to use multi-process for experiments
do_load_data = True
do_split_data = True
do_fea_pro = True
do_fea_sel = True
do_fea_eva = True
do_fea_ext = True
show_pop_info = True

n_repeat = 3        # Number of repeat
n_fold = 5          # Number of fold
n_process = n_repeat * n_fold       # Number of processes for the multi-process
random_state = 0    # Random seed
# GPU Parameter
n_gpu = 3           # Number of GPU deployed
gpu_index = 0
# Analysis Parameter
metric_names = ['RMSE', 'MAE', 'R2', 'PCC']
decimals = 4

### Parameters for data
feature_types = ['MF', 'PP', 'GEP', 'CL']    # Feature types
cell_line_name = 'all'                      # Cell lines
score_type = 'S'                            # Label type

### Parameters for the 'Feature Pairs Preprocessing' module
# Operators
logical_operators = ['AND', 'OR', 'XOR']        # Logical operators     -- AND, OR, XOR
mathematic_operators = ['SUM', 'DIF']           # Mathematic operators  -- SUM, DIF
comparison_operators = ['MAX', 'MIN']           # Comparison operators  -- MAX, MIN
operator_dict = {
    'logical': logical_operators,
    'mathematic': mathematic_operators,
    'comparison': comparison_operators
}
# Synthetic Features
feature_operators = ['MF_AND', 'MF_OR', 'MF_XOR',
                     'PP_SUM', 'PP_DIF', 'PP_MAX', 'PP_MIN',
                     'VGEP_SUM', 'VGEP_DIF', 'VGEP_MAX', 'VGEP_MIN',
                     'CL']                  # Feature _ Operator
feature_nums = [881, 881, 881,
                55, 55, 55, 55,
                978, 978, 978, 978,
                978]                        # The number of each synthetic feature types

### Parameters for the 'Feature Preselection' module
fs_methods = ['PCC', 'SCC', 'KCC', 'RF', 'XGB', 'LXGB']
p_feature = 0.1         # The proportion of features retained
min_select_num = 30     # minimum number of features to be retained for each view

### Parameters for the 'Feature Extraction' module
n_primitive = len(feature_operators) * len(fs_methods)
# GP Parameters
n_generation = 20   # Number of generation, default=20
s_population = 50   # Size of population, default=50
fitness_type = 'MSE'    # Fitness type: MSE, R2
w_fitness = -1.0    # Fitness weight, set it to -1.0 for reverse order
p_crossover = 1     # Crossover probability, default=1
p_mutation = 0.2    # Mutation probability, default=0.2
elite_reserve = True    # Whether to reserve the elite of population
# Base learner Parameters
learner_type = 'XGB'    # The type of base learner
n_estimators = 25
learner_param = {
    'XGB': {
        # est param
        'n_estimators': n_estimators,
        'max_depth': 8,
        'subsample': 0.9,
        # exp param
        'random_state': random_state,
        'n_jobs': -1,
        'tree_method': 'gpu_hist',
        'gpu_id': 0     # default=0
    }
}

### Parameters for the 'Ensemble Learning' module
min_simi_r = 0.6


