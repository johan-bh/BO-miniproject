from keras.datasets import mnist
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

# Load data from keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()
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

#
def Simple_NN():
    model = Sequential()
    model.add(Dense(input_dim, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_feature, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

baseline = Simple_NN()
baseline.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=200, verbose=0)
scores = baseline.evaluate(X_test, y_test, verbose=0)
print(f"Baseline Error:{(100-scores[1]*100)}")

# Use Bayesian Optimization to optimize hyperparameters