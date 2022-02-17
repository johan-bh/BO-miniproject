def load_MNIST():
    '''
    Function to load the MNIST training and test set with corresponding labels.

    :return: training_examples, training_labels, test_examples, test_labels
    '''

    # we want to flat the examples
    from torchvision import datasets
    import numpy

    training_set = datasets.MNIST(root='./data', train=True, download=True, transform= None)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform= None)

    Xtrain = training_set.data.numpy().reshape(-1,28*28)
    Xtest = test_set.data.numpy().reshape(-1,28*28)

    ytrain = training_set.targets.numpy()
    ytest = test_set.targets.numpy()

    return Xtrain, ytrain, Xtest, ytest

# LOL