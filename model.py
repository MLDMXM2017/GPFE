# -*- coding: utf-8 -*-

import datetime
import numpy as np
import operator
import pickle
import random
import warnings

from deap import algorithms, base, creator, gp, tools
from function.data_tools import kfold_sampling
from function.gp_tools import evaluate, output

from filepath import *
from config import *

from function.fs_tools import get_filter, get_embedded
from function.function import get_learner, get_logger
from sklearn.feature_selection import SelectPercentile

warnings.filterwarnings("ignore")

### GP
# Terminal set
pset = gp.PrimitiveSet('MAIN', n_primitive)
# Function set
pset.addPrimitive(np.logical_and, 2)
pset.addPrimitive(np.logical_or, 2)
pset.addPrimitive(np.logical_xor, 2)
# Fitness
if fitness_type in ['R2']:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Default case
elif fitness_type in ['MAE', 'MSE', 'RMSE']:
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,)) # Reverse case
# Individual
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset, output=output)
# Toolbox
toolbox = base.Toolbox()
# Register
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)              # Initial population construction
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr) # Individual register
toolbox.register("population", tools.initRepeat, list, toolbox.individual)          # Population register
toolbox.register("compile", gp.compile, pset = pset)                                # Compile register
toolbox.register("select", tools.selTournament, tournsize = 2)                      # Tournament param
toolbox.register("mate", gp.cxOnePoint)                                             # Crossover param
toolbox.register("expr_mut", gp.genGrow, min_ = 1, max_ = 3)
toolbox.register("mutate", gp.mutUniform, expr = toolbox.expr_mut, pset = pset)     # Mutation param
toolbox.register("evaluate", evaluate)                                              # Fitness evaluation func
# Decorate -- Limit max height of GP trees
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
# Statistics -- Experimental statistics
stats_fitness = tools.Statistics(lambda ind: ind.fitness.values)
stats_height = tools.Statistics(lambda ind: ind.height)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fitness, height=stats_height, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)
# GP info
verbose = __debug__


class Model:
    def __init__(self, task_id=0, view_info=None, random_seed=None):
        self.task_id = task_id
        self.logger = get_logger(f"log_{task_id}", f"{log_dir}/log_{task_id}.txt", output=False)
        # self.gpu_id = task_id % n_gpu + gpu_index
        self.view_info = view_info
        self.random_state = random_state if random_seed is None else random_seed

        self.base_learner_type = learner_type
        self.base_learner_param = learner_param[learner_type]
        self.base_learner_param['gpu_id'] = task_id % n_gpu + gpu_index

        self.feature_subsets = None
        self.estimators = None
        self.ind_weights = None

        random.seed(self.random_state)
        np.random.seed(self.random_state)

    def fit(self, x_train, y_train):
        f_s = datetime.datetime.now()

        # Feature Preselection
        self.logger.info("\tFeature Preselection...")
        if do_fea_sel:
            fea_list_s = self.feature_preselection(x_train, y_train)
            with open(f"{fsc_dir}/select_{self.task_id}.pkl", 'wb') as f:
                pickle.dump(fea_list_s, f)
        else:
            with open(f"{fsc_dir}/select_{self.task_id}.pkl", 'rb') as f:
                fea_list_s = pickle.load(f)

        # Get k-fold data
        self.logger.info("\tGet k-fold data...")
        x_train_s, y_train_s, x_valid_s, y_valid_s = kfold_sampling(x_train, y_train, n_fold, random_state)

        # Feature Extraction
        self.logger.info("\tFeature Extraction...")
        if do_fea_ext:
            population = self.feature_extraction(x_train_s, y_train_s, x_valid_s, y_valid_s, fea_list_s)
            with open(f"{ind_dir}/population.pkl", 'wb') as f:
                pickle.dump(population, f)
        else:
            with open(f"{ind_dir}/population.pkl", 'rb') as f:
                population = pickle.load(f)

        # Ensemble Learning
        self.logger.info("\tEnsemble Learning...")
        self.ensemble_learning(fea_list_s, population, x_train, y_train)

        f_e = datetime.datetime.now()
        self.logger.info(f"Task finished! Cost time: {f_e - f_s}")

    def feature_preselection(self, x_train, y_train):
        n_feature = x_train.shape[1]
        fea_list_s = np.full((len(self.view_info.keys()) * len(fs_methods), n_feature), False)

        idx = -1
        for v_name, v_indice in self.view_info.items():
            view_x, view_y = x_train[:, v_indice], y_train

            for method in fs_methods:
                idx += 1
                if do_fea_eva:
                    if method in ['PCC', 'SCC', 'KCC']:
                        model = get_filter(method)
                        selector = SelectPercentile(model, percentile=100)
                        selector.fit(view_x, view_y)
                        score = selector.scores_
                        score[np.isnan(score)] = -1 * np.float("inf")
                        score[np.isinf(score)] = -1 * np.float("inf")
                    elif method in ['RF', 'XGB', 'LXGB']:
                        clf = get_embedded(method, self.random_state, n_estimators)
                        clf.fit(view_x, view_y)
                        score = clf.feature_importances_
                    else:
                        raise ValueError(f"Invalid method parameter: {method}")

                    with open(f"{fsc_dir}/score_{self.task_id}_{v_name}_{method}.pkl", 'wb') as f:
                        pickle.dump(score, f)
                else:
                    with open(f"{fsc_dir}/score_{self.task_id}_{v_name}_{method}.pkl", 'rb') as f:
                        score = pickle.load(f)

                n_select = max(int(len(v_indice) * p_feature), min_select_num)
                sel_ind = np.argsort(score)[::-1][:n_select]
                fea_list_s[idx, sel_ind + v_indice[0]] = True
        return fea_list_s

    def feature_extraction(self, x_train_s, y_train_s, x_valid_s, y_valid_s, fea_list_s):
        fitness_dict = {}                   # Global dictionary: record individuals' fitness  ==> Reduce computing cost
        population_dict = {}
        elite_ind = tools.HallOfFame(1)

        # # Define logbook
        # logbook = tools.Logbook()
        # logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])

        # Generate population
        population = toolbox.population(n=s_population)

        # Calculate fitness for each inds
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = []
        for ind in invalid_ind:
            str_fea_set = ''.join(ind.output(fea_list_s).astype(int).astype(str))
            if str_fea_set in fitness_dict:
                fitnesses.append(fitness_dict[str_fea_set])
            else:
                fitness = toolbox.evaluate(ind, fitness_type, self.base_learner_type, self.base_learner_param,
                                           x_train_s, y_train_s, x_valid_s, y_valid_s, fea_list_s)
                fitnesses.append(fitness)
                fitness_dict[str_fea_set] = fitness
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update elite ind with the best individual from the current population
        elite_ind.update(population)

        # # Output information to Logbook
        # record = mstats.compile(population) if mstats else {}
        # logbook.record(gen=0, nevals=len(invalid_ind), **record)

        # if verbose:
        #     line = logbook.stream
        #     self.logger.info(line)
        if show_pop_info:
            fits = np.array([ind.fitness.values[0] for ind in population])
            population_dict[0] = {
                'Num': len(fits),
                'Min': min(fits),
                'Max': max(fits),
                'Avg': np.mean(fits),
                'Std': np.std(fits)
            }
            output_str = f"Gen 0\t"
            for key, value in population_dict[0].items():
                output_str += f"{key}: {value:.2f}\t"
            self.logger.info(output_str)

        # Begin the generational process
        for gen in range(1, n_generation + 1):
            # Select the next generation individuals
            offspring = toolbox.select(population, s_population)

            # Vary the pool of individuals by crossover and mutation operation
            offspring = algorithms.varAnd(offspring, toolbox, p_crossover, p_mutation)

            # Evaluate individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = []
            for ind in invalid_ind:
                str_fea_set = ''.join(ind.output(fea_list_s).astype(int).astype(str))
                if str_fea_set in fitness_dict:
                    fitnesses.append(fitness_dict[str_fea_set])
                else:
                    fitness = toolbox.evaluate(ind, fitness_type, self.base_learner_type, self.base_learner_param,
                                               x_train_s, y_train_s, x_valid_s, y_valid_s, fea_list_s)
                    fitnesses.append(fitness)
                    fitness_dict[str_fea_set] = fitness
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace the current population by the offspring
            population[:] = offspring

            # Replace the worst ind in the current population with the elite ind
            if elite_ind[0] not in population:
                worst_individual = algorithms.tools.selWorst(population, 1)[0]
                index = population.index(worst_individual)
                population[index] = toolbox.clone(elite_ind[0])
            # 2024-4-7 换成population试试
            # elite_ind.update(offspring)
            elite_ind.update(population)

            # Output the current generation statistics to the logbook
            # record = mstats.compile(population) if mstats else {}
            # logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            # if verbose:
            #     line = logbook.stream
            #     self.logger.info(line)
            if show_pop_info:
                fits = np.array([ind.fitness.values[0] for ind in population])
                population_dict[gen] = {
                    'Num': len(fits),
                    'Min': min(fits),
                    'Max': max(fits),
                    'Avg': np.mean(fits),
                    'Std': np.std(fits)
                }
                output_str = f"Gen {gen}\t"
                for key, value in population_dict[gen].items():
                    output_str += f"{key}: {value:.2f}\t"
                self.logger.info(output_str)
        return population

    def ensemble_learning(self, fea_list_s, population, x_train, y_train):
        n_ind = len(population)

        feature_subsets = []
        performances = []
        for ind in population:
            feature_subsets.append(ind.output(fea_list_s))
            performances.append(ind.fitness.values[0])
        feature_subsets = np.array(feature_subsets)
        performances = np.array(performances)

        select_sign = np.full(n_ind, True)
        perf_index = np.argsort(performances)
        for i in range(n_ind):
            p_ind = perf_index[i]
            if not select_sign[p_ind]:
                continue
            for j in range(i + 1, n_ind):
                q_ind = perf_index[j]
                if not select_sign[q_ind]:
                    continue

                fea_set_p = feature_subsets[p_ind]
                fea_set_q = feature_subsets[q_ind]
                simi_r = len(np.where(fea_set_p & fea_set_q)[0]) / len(np.where(fea_set_p | fea_set_q)[0])
                if simi_r > min_simi_r:
                    select_sign[q_ind] = False
            if performances[p_ind] == 300:
                select_sign[p_ind] = False

        sel_feature_subsets = feature_subsets[select_sign]
        self.feature_subsets = sel_feature_subsets
        sel_performances = performances[select_sign]

        estimators = []
        for feature_subset in sel_feature_subsets:
            learner = get_learner(self.base_learner_type, self.base_learner_param)
            learner.fit(x_train[:, feature_subset], y_train)
            estimators.append(learner)
        self.estimators = estimators

        ind_weights = []
        for sel_perf in sel_performances:
            ind_weights.append(1.0 / sel_perf)
        ind_weights = np.array(ind_weights)
        ind_weights = ind_weights / sum(ind_weights)
        self.ind_weights = ind_weights

        with open(f"{est_dir}/estimator.pkl", 'wb') as f:
            pickle.dump(estimators, f)
        with open(f"{est_dir}/weight.pkl", 'wb') as f:
            pickle.dump(ind_weights, f)

    def predict(self, x):
        y_pred = np.zeros(x.shape[0])
        for sel_i in range(len(self.estimators)):
            y_pred += self.estimators[sel_i].predict(x[:, self.feature_subsets[sel_i]]) * self.ind_weights[sel_i]
        return y_pred











