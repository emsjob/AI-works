'''
Emil Sjöberg

Solving mnist
'''

import numpy as np
import matplotlib.pyplot as plt
import tabulate
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.datasets import load_iris
import itertools
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Activation

from keras.datasets import mnist

import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# Softmax activation function let's us view the outputs as probabilities
def softmax(Z):
	Z = [np.exp(z) for z in Z]
	S = np.sum(Z)
	Z /= S
	return Z

# Load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = 10

# See how the data is packaged
print("{} train samples and {} test samples".format(x_train.shape[0], x_test.shape[0]))
print("Training data shape: ", x_train.shape, y_train.shape)
print("Test data shape: ", x_test.shape, y_test.shape)

samples = np.concatenate([np.concatenate([x_train[i] for i in [int(random.random()*len(x_train)) for i in range(16)]],
			axis=1) for i in range(4)], axis=0)
plt.figure(figsize=(16,4))
plt.imshow(samples, cmap='grey')


# Reshape input vectors
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Make float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize to number in 0 to 1
x_train /= 255
x_test /= 255

print("First sample of y_train before one-hot vectorization", y_train[0])

# Change the output form from a number to a vector representing the numbers as true or false
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print("First sample of y_train after one-hot vectorization", y_train[0])

# Create a neural network for MNIST with 2 layers with 100 neurons each, with sigmoid activation
# The output layer goes through softmax activation
model = Sequential()
model.add(Dense(100, activation='sigmoid', input_dim = 784))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))
#model.summary()

# Compile model to optimize for the categorical cross-entropy loss
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Train network for 20 epochs with batch size of 100
model.fit(x_train, y_train, batch_size=100, epochs=20, verbose=1, validation_data=(x_test, y_test))

# Train network again for another 20 epochs
model.fit(x_train, y_train, batch_size=100, epochs=20, verbose=1, validation_data=(x_test, y_test))

# Evaluate performance of network
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])


# Try Cifar-10 dataset
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10

# Reshape input vectors
x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

# Make float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize to 0 to 1
x_train /= 255
x_test /= 255

# Convert output to one-hot vectors
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Use the same model again
model = Sequential()
model.add(Dense(100, activation='sigmoid', input_dim = 3072))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Train model for 60 epochs
model.fit(x_train, y_train, batch_size=100, epochs=60, verbose=1, validation_data=(x_test, y_test))


