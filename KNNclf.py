from sklearn.neighbors import KNeighborsClassifier
## define the domain of the considered parameters
n_neighbours = tuple(np.arange(1,10,1, dtype= np.int))
# print(n_estimators)
weights = (0, 1)
# max_features = ('log2', 'sqrt', None)
algorithm = (0, 1, 2)
# criterion = ('gini', 'entropy')
p = tuple(np.arange(1,10,1, dtype = np.int))

domain = [{'name': 'n_neighbours', 'type': 'discrete', 'domain':n_neighbours},
          {'name': 'weights', 'type': 'categorical', 'domain': weights},
          {'name': 'algorithm', 'type': 'categorical', 'domain': algorithm},
          {'name': 'p', 'type': 'discrete', 'domain': p}]


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
    model = KNeighborsClassifier(n_neigbours=int(param[0]), weights=weight, algorithm=algorithm, p=int(param[3]))

    # fit the model
    model.fit(Xtrain, ytrain)
    print(model.score)
    return - model.score

opt = GPyOpt.methods.BayesianOptimization(f = objective_function,   # function to optimize
                                              domain = domain,         # box-constrains of the problem
                                              acquisition_type = 'EI' ,      # Select acquisition function MPI, EI, LCB
                                             )

opt.acquisition.exploration_weight=0.5

opt.run_optimization(max_iter = 15)

x_best = opt.X[np.argmin(opt.Y)]
print("The best parameters obtained: n_neighbours =" + str(x_best[0]) + ", weight =" + str(x_best[1]) + ", algorithm =" + str(
    x_best[2])  + ", p =" + str(
    x_best[3]))