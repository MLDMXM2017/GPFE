
import lightgbm
import numpy as np
import warnings
import xgboost

from minepy import MINE
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

def distcorr(X, Y):
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

def distcorr_X_y(X, Y):
    return np.array(list(map(lambda x: distcorr(x, Y), X.T))).T

def kendalltau_X_y(X, Y):
    return np.array(list(map(lambda x: kendalltau(x, Y), X.T))).T[0]

def mic_X_y(X, Y):
    def mic(x,y):
        mine = MINE()
        mine.compute_score(x,y)
        return (mine.mic(),0.5)
    return np.array(list(map(lambda x: mic(x, Y), X.T))).T[0]

def pearson_X_y(X, Y):
    return np.array(list(map(lambda x: pearsonr(x, Y), X.T))).T[0]

def spearman_X_y(X, Y):
    return np.array(list(map(lambda x: spearmanr(x, Y), X.T))).T[0]

def get_filter(method):
    if method == 'PCC':
        return pearson_X_y
    elif method == 'SCC':
        return spearman_X_y
    elif method == 'KCC':
        return kendalltau_X_y
    else:
        raise ValueError(f"Invalid method parameter: {method}")

def get_embedded(method, random_state, n_estimators=100):
    if method == 'RF':
        return RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, criterion='mse')
    elif method == 'XGB':
        return xgboost.XGBRegressor(n_estimators=n_estimators, random_state=random_state, objective='reg:squarederror')
    elif method == 'LXGB':
        return lightgbm.LGBMRegressor(n_estimators=n_estimators, random_state=random_state, verbose=0)
    else:
        raise ValueError(f"Invalid method parameter: {method}")







