'''
EMIL SJÖBERG

Convolutional neural network to solve fashion mnist
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation
from keras.utils import to_categorical

from tensorflow.keras.datasets import fashion_mnist

# Load datset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
num_classes = 10

# Split training data into training and validation data
x_validate = x_train[:10000]
y_validate = y_train[:10000]
x_train = x_train[10000:]
y_train = y_train[10000:]

# Shuffle data
shuffle_train = np.random.permutation(len(y_train))
shuffle_validate = np.random.permutation(len(y_validate))
shuffle_test = np.random.permutation(len(y_test))

x_train = x_train[shuffle_train, :]
y_train = y_train[shuffle_train]

x_validate = x_validate[shuffle_validate, :]
y_validate = y_validate[shuffle_validate]

x_test = x_test[shuffle_test, :]
y_test = y_test[shuffle_test]

# Make float32 and normalize
x_train = x_train.astype('float32') / 255.0
x_validate = x_validate.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape image dataset to 3D tensors
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_validate = np.reshape(x_validate, (x_validate.shape[0], x_validate.shape[1], x_validate.shape[2], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

# Reshape output vectors
y_train = to_categorical(y_train)
y_validate = to_categorical(y_validate)
y_test = to_categorical(y_test)

# Build model
model = Sequential()

# Add layers
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, batch_size=32, epochs=3, validation_data=(x_validate, y_validate))

# Evaluate model
_, acc = model.evaluate(x_test, y_test)
print("Accuracy = ", acc)

