import numpy as np
import matplotlib.pyplot as plt
import GPyOpt
from sklearn.linear_model import LogisticRegression
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

ran = np.random.randint(0,17402,200)
Xtrain, ytrain = Xtrain[train_filter][ran], ytrain[train_filter][ran]

# apply the mask to the entire dataset
# Xtrain, ytrain = Xtrain[train_filter], ytrain[train_filter]
Xtest, ytest = Xtest[test_filter], ytest[test_filter]
print(np.shape(Xtrain))
print(np.shape(ytrain))

model = LogisticRegression()
model.fit(Xtrain, ytrain)
print(model)
# hyperparams dictionary
domain = dict(penalty=["l2", "none"], C=range(10, 60, 10), solver=["newton-cg", "lbfgs", "sag", "saga"],
              fit_intercept=[True, False], max_iter = range(100,1000,100))
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
    model = LogisticRegression(fit_intercept=params["fit_intercept"],
                        penalty=params["penalty"],
                        solver=params["solver"],
                        max_iter=params["max_iter"],
                        C=params["C"])

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
C = np.linspace(1e-10,1e10,20)
# print(n_estimators)
max_iter = np.linspace(100,1000,100)

# define the dictionary for GPyOpt
domain = [{'name': 'penalty', 'type': 'categorical', 'domain': (0,1)},
          {'name': 'C', 'type': 'discrete', 'domain': C},
          {'name': 'fit_intercept', 'type': 'categorical', 'domain': (0,1)},
          {'name': 'max_iter', 'type': 'discrete', 'domain': max_iter},
          {'name': 'solver', 'type': 'categorical', 'domain': (0, 1, 2, 3)}]

## we have to define the function we want to maximize --> validation accuracy,
## note it should take a 2D ndarray but it is ok that it assumes only one point
## in this setting
def objective_function(x):
    # print(x)
    # we have to handle the categorical variables that is convert 0/1 to labels
    # log2/sqrt and gini/entropy
    param = x[0]
    # we have to handle the categorical variables
    if param[0] == 0:
        penalty = "l2"
    elif param[0] == 1:
        penalty = "none"

    if param[2] == 0:
        fit_intercept = True
    else:
        fit_intercept = False

    if param[4] == 0:
        solver = "newton-cg"
    elif param[4] == 1:
        solver = "lbfgs"
    elif param[4] == 2:
        solver = "sag"
    elif param[4] == 3:
        solver = "saga"

    # create the model
    model = LogisticRegression(fit_intercept=fit_intercept,
                        penalty=penalty,
                        solver=solver,
                        max_iter=param[3]//1,
                        C=param[1]//1)

    # fit the model
    model.fit(Xtrain, ytrain)
    print(model.score(Xtest,ytest))
    return - model.score(Xtest,ytest)

opt = GPyOpt.methods.BayesianOptimization(f = objective_function,   # function to optimize
                                              domain = domain,         # box-constrains of the problem
                                              acquisition_type = 'EI' ,      # Select acquisition function MPI, EI, LCB
                                             )
opt.acquisition.exploration_weight=0.5

# opt.run_optimization(max_iter = 20)
opt.run_optimization(max_iter=15, eps = 1e-12)

x_best = opt.X[np.argmin(opt.Y)]
# print("The best parameters obtained: n_estimators=" + str(x_best[0]) + ", learning_rate=" + str(x_best[1]) + ", algorithm=" + str(
#     x_best[2])  + ", random_state=" + str(
#     x_best[3]))

## comparison between random search and bayesian optimization
## we can plot the maximum oob per iteration of the sequence

# collect the maximum each iteration of BO, note that it is also provided by GPOpt in Y_Best
print(-opt.Y)
print(max_score_per_iteration)


opt.plot_convergence()

y_bo = np.maximum.accumulate(-opt.Y).ravel()
# define iteration number
xs = np.arange(1,21,1)

plt.plot(xs, max_score_per_iteration, 'o-', color = 'red', label='Random Search')
plt.plot(xs, y_bo, 'o-', color = 'blue', label='Bayesian Optimization')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Accuracy Score')
plt.title('Random Search Accuracy Score over iterations')
plt.show()

exit()

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