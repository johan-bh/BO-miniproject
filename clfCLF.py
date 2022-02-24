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
domain = {"n_estimators": range(1, 101, 5),
          "learning_rate": range(1, 2),
          "algorithm": ["SAMME", "SAMME.R"],
          "random_state": ['32', 'None']}
# rs = RandomizedSearchCV(model, param_distributions=domain, cv=3, verbose =2, n_iter=10)
# rs.fit(Xtrain, ytrain)

# create the ParameterSampler
param_list = list(ParameterSampler(domain, n_iter=20, random_state=32))
print('Param list')
print(param_list)

