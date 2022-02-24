import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import GPyOpt
from load_MNIST import load_MNIST
from torchvision import datasets
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn import preprocessing

# Load data from keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()
## we use a mask to selects those subsets
train_filter = np.isin(y_train, [3, 5, 8, 9])
test_filter = np.isin(y_test, [3, 5, 8, 9])

# apply the mask to the entire dataset
Xtrain, ytrain = X_train[train_filter], y_train[train_filter]
Xtest, ytest = X_test[test_filter], y_test[test_filter]
print(np.shape(Xtrain))
# Image size = input dims
input_dim =  X_train.shape[1] * X_train.shape[2]

# Reshape data to match NN
X_train = X_train.reshape(X_train.shape[0], input_dim).astype('float32')
X_test = X_test.reshape(X_test.shape[0], input_dim).astype('float32')

# Normalize train and test data
X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)

# One-out-of-K encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_feature = y_test.shape[1]


#Xtrain, ytrain, Xtest, ytest = load_MNIST()
## define the domain of the considered parameters
n_neighbors = tuple(np.arange(1,10,1, dtype= np.int))
# print(n_estimators)
weights = (0, 1)
# max_features = ('log2', 'sqrt', None)
algorithm = (0, 1, 2)
# criterion = ('gini', 'entropy')
p = tuple(np.arange(1,10,1, dtype = np.int))

domain = [{'name': 'n_neighbors', 'type': 'discrete', 'domain':n_neighbors},
          {'name': 'weights', 'type': 'categorical', 'domain': weights},
          {'name': 'algorithm', 'type': 'categorical', 'domain': algorithm},
          {'name': 'p', 'type': 'discrete', 'domain': p}]

print("Domain defined")

def objective_function(x):
    # print(x)
    # we have to handle the categorical variables that is convert 0/1 to labels
    # log2/sqrt and gini/entropy
    param = x[0]
    # print(param)
    # we have to handle the categorical variables
    if param[1] == 0:
        weight = 'uniform'
    elif param[1] == 1:
        weight = 'distance'

    if param[2] == 0:
        algorithm = 'ball_tree'
    if param[2] == 1:
        algorithm = 'kd_tree'
    if param[2] == 2:
        algorithm = 'brute'

    # create the model
    model = KNeighborsClassifier(n_neighbors=int(param[0]), weights=weight, algorithm=algorithm, p=int(param[3]))

    # fit the model
    model.fit(X_train, y_train)
    print(model.score(X_train, y_train))
    return - model.score(X_train, y_train)

opt = GPyOpt.methods.BayesianOptimization(f = objective_function,   # function to optimize
                                              domain = domain,         # box-constrains of the problem
                                              acquisition_type = 'EI' ,      # Select acquisition function MPI, EI, LCB
                                             )
print("Opt defined")
#set weight
opt.acquisition.exploration_weight=0.5
print("exploration weight set")
opt.run_optimization(max_iter = 15)
print("Optimization done")
x_best = opt.X[np.argmin(opt.Y)]
print("The best parameters obtained: n_neighbours =" + str(x_best[0]) + ", weight =" + str(x_best[1]) + ", algorithm =" + str(
    x_best[2])  + ", p =" + str(
    x_best[3]))

## comparison between random search and bayesian optimization
## we can plot the maximum oob per iteration of the sequence

# collect the maximum each iteration of BO, note that it is also provided by GPOpt in Y_Best
y_bo = np.maximum.accumulate(-opt.Y).ravel()
# define iteration number
xs = np.arange(1,21,1)

plt.plot(xs, max_score, 'o-', color = 'red', label='Random Search')
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