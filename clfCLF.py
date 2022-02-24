import numpy as np
import matplotlib.pyplot as plt
import GPyOpt
from torchvision import datasets, transforms, utils
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

<<<<<<< HEAD
ran = np.random.randint(0,17402,6000)
Xtrain, ytrain = Xtrain[train_filter][ran], ytrain[train_filter][ran]

# apply the mask to the entire dataset
# Xtrain, ytrain = Xtrain[train_filter], ytrain[train_filter]
=======
ran = np.random.randint(0,17402,3000)
Xtrain, ytrain = Xtrain[train_filter][ran], ytrain[train_filter][ran]

# apply the mask to the entire dataset
#Xtrain, ytrain = Xtrain[train_filter], ytrain[train_filter]
>>>>>>> 80dab346ddd4271f4642f7978e8da7d466b24cfa
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

## define the domain of the considered parameters
n_estimators = tuple(np.arange(5, 95, 5, dtype=np.int))
# print(n_estimators)
learning_rate=np.linspace(1e-5,1e-2,6)
# algorithm = ("SAMME", "SAMME.R")
algorithm = (0, 1)
# random_state = (32, None)
random_state = (0, 1)

# define the dictionary for GPyOpt
domain = [{'name': 'n_estimators', 'type': 'discrete', 'domain': n_estimators},
          {'name': 'learning_rate', 'type': 'discrete', 'domain': learning_rate},
          {'name': 'algorithm', 'type': 'categorical', 'domain': algorithm},
          {'name': 'random_state', 'type': 'categorical', 'domain': random_state}]

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
        algorithm = "SAMME"
    elif param[2] == 1:
        algorithm = "SAMME.R"

    if param[3] == 0:
        random_state = 32
    else:
        random_state = None

    # create the model

<<<<<<< HEAD
    model = AdaBoostClassifier(n_estimators=int(param[0]), learning_rate=param[1],
=======
    model = AdaBoostClassifier(n_estimators=param[0], learning_rate=param[1],
>>>>>>> 80dab346ddd4271f4642f7978e8da7d466b24cfa
                               algorithm=algorithm, random_state=random_state)

    # fit the model
    model.fit(Xtrain, ytrain)
    print(model.score(Xtest,ytest))
    return - model.score(Xtest,ytest)

opt = GPyOpt.methods.BayesianOptimization(f = objective_function,   # function to optimize
                                              domain = domain,         # box-constrains of the problem
                                              acquisition_type = 'EI' ,      # Select acquisition function MPI, EI, LCB
                                             )
opt.acquisition.exploration_weight=0.5

opt.run_optimization(max_iter = 15)

x_best = opt.X[np.argmin(opt.Y)]
print("The best parameters obtained: n_estimators=" + str(x_best[0]) + ", learning_rate=" + str(x_best[1]) + ", algorithm=" + str(
    x_best[2])  + ", random_state=" + str(
    x_best[3]))

## comparison between random search and bayesian optimization
## we can plot the maximum oob per iteration of the sequence

# collect the maximum each iteration of BO, note that it is also provided by GPOpt in Y_Best
y_bo = np.maximum.accumulate(-opt.Y).ravel()
# define iteration number
xs = np.arange(1,21,1)

plt.plot(xs, max_score_per_iteration, 'o-', color = 'red', label='Random Search')
plt.plot(xs, y_bo, 'o-', color = 'blue', label='Bayesian Optimization')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Out of bag error')
plt.title('Comparison between Random Search and Bayesian Optimization')
plt.show()

#We start by taking a look a the model subclass of our Bayesian optimization object
print(opt.model.input_dim)
#so there is 6 dimensions, that is 1 for each of n_estimators and depth which are discrete variables.
# the remaining 4 is because one-out-of-K encoding is using for the categorical variables max_features and criterion
#we can also look at kernel parameters
print(opt.model.get_model_parameters_names())
#and get the current fitted values
print(opt.model.get_model_parameters())
#To get a plot of the acquisition function we use the function opt.acquisition.acquisition_function
#first we define a sensible grid for the first to parameters
#indexing='ij' ensures that x/y axes are not flipped (which is default):
#we also add two extra axes for the categorical varibles and here fix these to 0 ('log2' and 'gini')
#note that the acqusition function can actually take any value not only integers as it lives in the GP space (here 0.5 intervals)
#and it is quite fast to evaluate - here in 40000 points
n_feat = np.arange(1,101,0.5)
max_d = np.arange(10,110,0.5)
pgrid = np.array(np.meshgrid(n_feat, max_d,[1],[0],[1],[0],indexing='ij'))
print(pgrid.reshape(6,-1).T.shape)
#we then unfold the 4D array and simply pass it to the acqusition function
acq_img = opt.acquisition.acquisition_function(pgrid.reshape(6,-1).T)
#it is typical to scale this between 0 and 1:
acq_img = (-acq_img - np.min(-acq_img))/(np.max(-acq_img - np.min(-acq_img)))
#then fold it back into an image and plot
acq_img = acq_img.reshape(pgrid[0].shape[:2])
plt.figure()
plt.imshow(acq_img.T, origin='lower',extent=[n_feat[0],n_feat[-1],max_d[0],max_d[-1]])
plt.colorbar()
plt.xlabel('n_features')
plt.ylabel('max_depth')
plt.title('Acquisition function')