
import numpy as np

from deap import gp
from function.function import get_metric, get_learner


def evaluate(individual, fitness_type, learner_type, learner_param,
             x_train_s, y_train_s, x_valid_s, y_valid_s, fea_list_s):
    fea_set = individual.output(fea_list_s)
    fea_num = len(np.where(fea_set)[0])

    # Set the Fitness for the invalid case
    if fea_num <= 2:
        if fitness_type in ['R2']:
            return 0,
        elif fitness_type in ['MAE', 'MSE', 'RMSE']:
            return 300,

    # Assessment of individual fitness by n-fold cross validation
    fitness_s = []
    for i in range(len(x_train_s)):
        learner = get_learner(learner_type, learner_param)

        x_train, y_train = x_train_s[i][:, fea_set], y_train_s[i]
        x_valid, y_valid = x_valid_s[i][:, fea_set], y_valid_s[i]

        learner.fit(x_train, y_train)
        y_pred = learner.predict(x_valid)

        fitness = get_metric(fitness_type, y_valid, y_pred)
        fitness_s.append(fitness)
    fitness_avg = np.mean(np.array(fitness_s))
    return fitness_avg,


# Convert an individual from a tree structure to a one-dimensional array with binary value
def output(self, fea_list_s):
    func = gp.compile(self, pset = self.pset)
    fea_set = func(*fea_list_s)

    if np.sum(fea_set) <= 1:
        fea_set[0] = True
    self.fea_set = fea_set
    return fea_set


