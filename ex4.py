import numpy as np
import matplotlib.pyplot as plt
import GPyOpt
from torchvision import datasets, transforms, utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, cross_val_score
from scipy.stats import uniform
import random

np.random.seed(32)
random.seed(32)

def load_MNIST():
    '''
    Function to load the MNIST training and test set with corresponding labels.

    :return: training_examples, training_labels, test_examples, test_labels
    '''

    # we want to flat the examples

    training_set = datasets.MNIST(root='./data', train=True, download=True, transform= None)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform= None)

    Xtrain = training_set.data.numpy().reshape(-1,28*28)
    Xtest = test_set.data.numpy().reshape(-1,28*28)

    ytrain = training_set.targets.numpy()
    ytest = test_set.targets.numpy()

    return Xtrain, ytrain, Xtest, ytest
  
  ## we can load the training set and test set
Xtrain, ytrain, Xtest, ytest = load_MNIST()

## we use a mask to selects those subsets
#train_filter = np.isin(ytrain, [3, 5, 8, 9])
#test_filter = np.isin(ytest, [3, 5, 8, 9])

# apply the mask to the entire dataset
#Xtrain, ytrain = Xtrain[train_filter], ytrain[train_filter]
#Xtest, ytest = Xtest[test_filter], ytest[test_filter]

# print some information
print('Information about the new datasets')
print('Training set shape:', Xtrain.shape)
print('Test set shape', Xtest.shape)

model = RandomForestClassifier(oob_score=True)
model.fit(Xtrain, ytrain)
print('Default model out of bag error', model.oob_score_)
print('Default model test accuracy', model.score(Xtest, ytest))

import time 

# hyperparams dictionary 
domain = {"n_estimators": range(1, 101), 
          "criterion": ['gini', 'entropy'],
          "max_depth": range(10, 60, 5),
          "max_features": ['sqrt', 'log2']}
#rs = RandomizedSearchCV(model, param_distributions=domain, cv=3, verbose =2, n_iter=10)
#rs.fit(Xtrain, ytrain)

# create the ParameterSampler
param_list = list(ParameterSampler(domain, n_iter=20, random_state=32))
print('Param list')
print(param_list)
#rounded_list = [dict((k,v) for (k, v) in d.items()) for d in param_list]

#print('Random parameters we are going to consider')
#print(rounded_list)

## now we can train the random forest using these parameters tuple, and for
## each iteration we store the best value of the oob

current_best_oob = 0
iteration_best_oob = 0 
max_oob_per_iteration = []
i = 0
for params in param_list:
    print(i)
    print(params)
    model = RandomForestClassifier(n_estimators= params['n_estimators'], criterion = params['criterion'], 
                           max_depth = params['max_depth'], max_features = params['max_features'], oob_score=True, n_jobs=-1)

    start = time.time()
    model.fit(Xtrain, ytrain)
    end = time.time()
    model_oob = model.oob_score_
    print('OOB found:', model_oob)
    if model_oob > current_best_oob:
        current_best_oob = model_oob
        iteration_best_oob = i
    
    max_oob_per_iteration.append(current_best_oob)
    i += 1
    print(f'It took {end - start} seconds')
    
## define the domain of the considered parameters
n_estimators = tuple(np.arange(1,101,1, dtype= np.int))
# print(n_estimators)
max_depth = tuple(np.arange(10,110,10, dtype= np.int))
# max_features = ('log2', 'sqrt', None)
max_features = (0, 1)
# criterion = ('gini', 'entropy')
criterion = (0, 1)


# define the dictionary for GPyOpt
domain = [{'name': 'n_estimators', 'type': 'discrete', 'domain':n_estimators},
          {'name': 'max_depth', 'type': 'discrete', 'domain': max_depth},
          {'name': 'max_features', 'type': 'categorical', 'domain': max_features},
          {'name': 'criterion', 'type': 'categorical', 'domain': criterion}]


## we have to define the function we want to maximize --> validation accuracy, 
## note it should take a 2D ndarray but it is ok that it assumes only one point
## in this setting
def objective_function(x): 
    #print(x)
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

    #create the model
    model = RandomForestClassifier(n_estimators = int(param[0]), max_depth = int(param[1]), max_features= max_f, criterion = crit, oob_score=True, n_jobs=-1)
    
    # fit the model 
    model.fit(Xtrain, ytrain)
    print(model.oob_score_)
    return - model.oob_score_


opt = GPyOpt.methods.BayesianOptimization(f = objective_function,   # function to optimize
                                              domain = domain,         # box-constrains of the problem
                                              acquisition_type = 'EI' ,      # Select acquisition function MPI, EI, LCB
                                             )
opt.acquisition.exploration_weight=0.5

opt.run_optimization(max_iter = 15) 

x_best = opt.X[np.argmin(opt.Y)]
print("The best parameters obtained: n_estimators=" + str(x_best[0]) + ", max_depth=" + str(x_best[1]) + ", max_features=" + str(
    x_best[2])  + ", criterion=" + str(
    x_best[3]))

## comparison between random search and bayesian optimization
## we can plot the maximum oob per iteration of the sequence

# collect the maximum each iteration of BO, note that it is also provided by GPOpt in Y_Best
y_bo = np.maximum.accumulate(-opt.Y).ravel()
# define iteration number
xs = np.arange(1,21,1)

plt.plot(xs, max_oob_per_iteration, 'o-', color = 'red', label='Random Search')
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
plt.title('Acquisition function');
