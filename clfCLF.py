import numpy as np
import matplotlib.pyplot as plt
import GPyOpt
from torchvision import datasets, transforms, utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, cross_val_score
from scipy.stats import uniform
import random
from load_MNIST import load_MNIST
from sklearn.ensemble import AdaBoostClassifier
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn import preprocessing
from torchvision import datasets, transforms, utils

np.random.seed(32)
random.seed(32)

## we can load the training set and test set
Xtrain, ytrain, Xtest, ytest = load_MNIST()

## we use a mask to selects those subsets
train_filter = np.isin(ytrain, [3, 5, 8])
test_filter = np.isin(ytest, [3, 5, 8])

# apply the mask to the entire dataset
Xtrain, ytrain = Xtrain[train_filter], ytrain[train_filter]
Xtest, ytest = Xtest[test_filter], ytest[test_filter]
print(np.shape(Xtrain))
print(np.shape(ytrain))

model = AdaBoostClassifier(n_estimators=100)
model.fit(Xtrain, ytrain)
print(model)
# hyperparams dictionary
domain = dict(n_estimators=range(5, 95, 5), learning_rate=range(1, 2), algorithm=["SAMME", "SAMME.R"],
              random_state=[32, None])
# rs = RandomizedSearchCV(model, param_distributions=domain, cv=3, verbose =2, n_iter=10)
# rs.fit(Xtrain, ytrain)

# create the ParameterSampler
param_list = list(ParameterSampler(domain, n_iter=20, random_state=32))
print('Param list')
print(param_list)

current_best_score = 0
iteration_best_score = 0
max_score_per_iteration = []
i = 0

import time

for params in param_list:
    print(i)
    print(params)
    model = AdaBoostClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'],
                               algorithm=params['algorithm'],random_state=params['random_state'])

    start = time.time()
    model.fit(Xtrain, ytrain)
    end = time.time()
    model_score = model.score(Xtest,ytest)
    print('Score found:', model_score)
    if model_score > current_best_score:
        current_best_score = model_score
        iteration_best_score = i

    max_score_per_iteration.append(current_best_score)
    i += 1
    print(f'It took {end - start} seconds')
"""
## define the domain of the considered parameters
n_estimators = tuple(np.arange(1, 101, 1, dtype=np.int))
# print(n_estimators)
max_depth = tuple(np.arange(10, 110, 10, dtype=np.int))
# max_features = ('log2', 'sqrt', None)
max_features = (0, 1)
# criterion = ('gini', 'entropy')
criterion = (0, 1)

# define the dictionary for GPyOpt
domain = [{'name': 'n_estimators', 'type': 'discrete', 'domain': n_estimators},
          {'name': 'max_depth', 'type': 'discrete', 'domain': max_depth},
          {'name': 'max_features', 'type': 'categorical', 'domain': max_features},
          {'name': 'criterion', 'type': 'categorical', 'domain': criterion}]


## we have to define the function we want to maximize --> validation accuracy,
## note it should take a 2D ndarray but it is ok that it assumes only one point
## in this setting
def objective_function(x):
    # print(x)
    # we have to handle the categorical variables that is convert 0/1 to labels
    # log2/sqrt and gini/entropy
    param = x[0]
    print(param)
    # we have to handle the categorical variables
    if param[2] == 0:
        max_f = 'log2'
    elif param[2] == 1:
        max_f = 'sqrt'
    else:
        max_f = None

    if param[3] == 0:
        crit = 'gini'
    else:
        crit = 'entropy'

    # create the model
    model = RandomForestClassifier(n_estimators=int(param[0]), max_depth=int(param[1]), max_features=max_f,
                                   criterion=crit, oob_score=True, n_jobs=-1)

    # fit the model
    model.fit(Xtrain, ytrain)
    print(model.oob_score_)
    return - model.oob_score_
"""
